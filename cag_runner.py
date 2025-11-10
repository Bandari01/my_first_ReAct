# cag_runner.py
# -*- coding: utf-8 -*-
"""
CAG Runner (Integrated, safe boolean checks)
- LLM-enabled planner (new OpenAI SDK first, fallback to old; else heuristic)
- Op alias resolver so LLM "nicknames" still execute
- Optional AUX merge for store-sales data (stores/transactions/oil/holidays)
- Two baselines:
    * Store-sales DoW cascade mean
    * Generic ML-lite (HistGradientBoosting), fallback to grouped-mean
- Flexible sample mapping (any first-two-column id/label names)
- Outputs:
    * submission.csv
    * outputs["prediction_mean"]
    * detailed logs (including fallback/normalization info)
"""
from __future__ import annotations
import os, json, traceback, re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# ==================== Utilities ====================
def summarize_csv(path: str, nrows: int = 2000) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": path, "exists": bool(path and os.path.exists(path))}
    if not info["exists"]:
        return info
    try:
        df = pd.read_csv(path, nrows=nrows)
        info.update({
            "rows_scanned": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "na_rate": {c: float(df[c].isna().mean()) for c in df.columns},
            "preview": df.head(5).to_dict(orient="records"),
        })
    except Exception as e:
        info["error"] = f"read_error: {e}"
    return info

def _read_safe(p: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        if p and os.path.exists(p):
            return pd.read_csv(p)
    except Exception:
        return None
    return None

# ==================== Planner ====================
def _heuristic_plan(dataset_name: str, summaries: Dict[str, Any], llm_error: Optional[str]=None) -> Dict[str, Any]:
    train_cols = set(summaries.get("train", {}).get("columns", []) or [])
    test_cols  = set(summaries.get("test", {}).get("columns", []) or [])

    if {"date","store_nbr","family","sales"}.issubset(train_cols) and {"date","store_nbr","family"}.issubset(test_cols):
        plan = {
            "dataset": dataset_name,
            "goal": "Store sales forecast baseline",
            "metrics": ["SMAPE"],
            "graph": [
                {"id":"prep","op":"prepare_store_sales","inputs":["train","test"],"outputs":["train_prep","test_prep"]},
                {"id":"baseline","op":"dow_mean_forecast","inputs":["train_prep","test_prep"],"outputs":["submission"]},
                {"id":"format","op":"align_sample_submission","inputs":["submission","sample"],"outputs":["submission_aligned"]},
            ],
        }
    else:
        plan = {
            "dataset": dataset_name,
            "goal": "Generic tabular ML-lite baseline",
            "metrics": [],
            "graph": [
                {"id":"prep_tab","op":"prepare_tabular_generic","inputs":["train","test","sample"],"outputs":["train_g","test_g","id_col","target_col","id_name","label_name"]},
                {"id":"ml_pred","op":"tabular_ml_predict","inputs":["train_g","test_g","id_col","target_col"],"outputs":["submission"]},
                {"id":"format","op":"align_sample_submission","inputs":["submission","sample"],"outputs":["submission_aligned"]},
            ],
        }
    if llm_error:
        plan["_note"] = f"LLM disabled or failed: {llm_error}"
    return plan


# ==================== Planner (LLM with schema + repair) ====================
ALLOWED_OPS = [
    "prepare_store_sales",
    "merge_aux",
    "dow_mean_forecast",
    "prepare_tabular_generic",
    "tabular_ml_predict",
    "align_sample_submission",
    "noop",
]

PLAN_INSTRUCTION = (
    "Return STRICT JSON for a DAG plan with fields: {\"dataset\": str, \"goal\": str, \"metrics\": list, "
    "\"graph\": [{\"id\": str, \"op\": str, \"inputs\": list, \"outputs\": list} ...]}. "
    "Allowed ops (use ONLY these exact names): "
    + ", ".join(ALLOWED_OPS) + ". "
    "Choose EITHER the store-sales path: prepare_store_sales (+ optional merge_aux) -> dow_mean_forecast -> align_sample_submission; "
    "OR the generic tabular path: prepare_tabular_generic -> tabular_ml_predict -> align_sample_submission. "
    "Use concise node ids like 'prep','merge','baseline','format'. No extra commentary; JSON only."
)

def _llm_call_json(payload: dict, model: str) -> dict:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise planning agent. Respond with strict JSON only."},
                {"role":"user","content":json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1, max_tokens=800, response_format={"type":"json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as e_new:
        import openai  # type: ignore
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a precise planning agent. Respond with strict JSON only."},
                {"role":"user","content":json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1, max_tokens=800,
        )
        text = resp.choices[0].message["content"].strip()
        if "```" in text:
            import re as _re
            m = _re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text) or _re.search(r"```\s*(\{[\s\S]*?\})\s*```", text)
            if m: text = m.group(1).strip()
        return json.loads(text)

def planner_llm(dataset_name: str, summaries: Dict[str, Any]) -> Dict[str, Any]:
    use_llm = os.environ.get("USE_LLM", "0") == "1" and bool(os.environ.get("OPENAI_API_KEY"))
    if not use_llm:
        return _heuristic_plan(dataset_name, summaries)
    model = os.environ.get("OPENAI_MODEL","gpt-4o-mini")
    payload = {"dataset": dataset_name, "summaries": summaries, "instruction": PLAN_INSTRUCTION}
    try:
        plan = _llm_call_json(payload, model=model)
    except Exception as e:
        return _heuristic_plan(dataset_name, summaries, llm_error=str(e))
    # normalize and validate; if unrecognized ops, try a repair round via LLM
    graph = (plan.get("graph") or [])
    bad = []
    for node in graph:
        node["op"] = _normalize_op(node.get("op",""))
        if node["op"] not in ALLOWED_OPS:
            bad.append(node["op"])
    if bad:
        fix_payload = {
            "dataset": dataset_name,
            "summaries": summaries,
            "bad_ops": bad,
            "note": "Your previous plan used unknown ops. Regenerate using only: " + ", ".join(ALLOWED_OPS),
            "instruction": PLAN_INSTRUCTION,
        }
        try:
            plan2 = _llm_call_json(fix_payload, model=model)
            plan = plan2
            for node in (plan.get("graph") or []):
                node["op"] = _normalize_op(node.get("op",""))
        except Exception:
            pass
    return plan

# ====== LLM op alias resolver ======
OP_ALIASES = {
    # sales prep/forecast
    "prepare-sales": "prepare_store_sales",
    "prepare_store_sales": "prepare_store_sales",
    "prep_sales": "prepare_store_sales",
    "prep_store_sales": "prepare_store_sales",
    "prepare_store": "prepare_store_sales",

    "dow-mean-forecast": "dow_mean_forecast",
    "dow_mean_forecast": "dow_mean_forecast",
    "forecast_store_sales": "dow_mean_forecast",
    "baseline_forecast": "dow_mean_forecast",
    "sales_baseline": "dow_mean_forecast",

    # generic tabular
    "prepare_generic": "prepare_tabular_generic",
    "prepare_tabular": "prepare_tabular_generic",
    "tabular_prepare": "prepare_tabular_generic",

    "ml_predict": "tabular_ml_predict",
    "predict_tabular": "tabular_ml_predict",
    "merge_aux_then_predict": "merge_aux",

    # align
    "align_to_sample": "align_sample_submission",
    "align_sample": "align_sample_submission",
    "format_submission": "align_sample_submission",

    # aux merge
    "merge_aux": "merge_aux",
    "join_aux": "merge_aux",
    "feature_union": "merge_aux",

    # noop
    "noop": "noop",
}

def _normalize_op(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return OP_ALIASES.get(key, key)

def _is_known_op(name: str) -> bool:
    return _normalize_op(name) in {
        "prepare_store_sales",
        "dow_mean_forecast",
        "prepare_tabular_generic",
        "tabular_ml_predict",
        "align_sample_submission",
        "merge_aux",
        "noop",
    }

# ==================== Store-sales baseline ====================
def prepare_store_sales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["dow"] = out["date"].dt.dayofweek
    if "onpromotion" in out.columns:
        out["onpromotion"] = pd.to_numeric(out["onpromotion"], errors="coerce").fillna(0).astype(int)
    return out

def dow_mean_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    tr = prepare_store_sales(train)
    te = prepare_store_sales(test)
    tr1 = tr.groupby(["store_nbr","family","dow"], as_index=False)["sales"].mean().rename(columns={"sales":"m1"})
    tr2 = tr.groupby(["store_nbr","family"], as_index=False)["sales"].mean().rename(columns={"sales":"m2"})
    tr3 = tr.groupby(["family","dow"], as_index=False)["sales"].mean().rename(columns={"sales":"m3"})
    tr4 = tr.groupby(["family"], as_index=False)["sales"].mean().rename(columns={"sales":"m4"})
    gmean = float(tr["sales"].mean()) if len(tr) else 0.0
    pred = te.copy()
    for t, on in [(tr1, ["store_nbr","family","dow"]), (tr2, ["store_nbr","family"]), (tr3, ["family","dow"]), (tr4, ["family"])]:
        pred = pred.merge(t, on=on, how="left")
    pred["sales_pred"] = pred["m1"].fillna(pred["m2"]).fillna(pred["m3"]).fillna(pred["m4"]).fillna(gmean)
    if "id" in pred.columns:
        sub = pred[["id"]].copy()
        sub["sales"] = pred["sales_pred"].clip(lower=0.0).astype(float)
    else:
        sub = pd.DataFrame({"id": np.arange(len(pred)), "sales": pred["sales_pred"].clip(lower=0.0).astype(float)})
    return sub

# ==================== Generic tabular ML-lite ====================
COMMON_TARGETS = ["TARGET","target","label","y","loss"]

def prepare_tabular_generic(train_df: pd.DataFrame, test_df: pd.DataFrame, sample_path: Optional[str]):
    id_name, label_name = None, None
    if sample_path and os.path.exists(sample_path):
        try:
            sm = pd.read_csv(sample_path, nrows=1)
            if len(sm.columns) >= 2:
                id_name, label_name = sm.columns[:2].tolist()
        except Exception:
            pass

    target_col = next((c for c in COMMON_TARGETS if c in train_df.columns), None)
    if target_col is None:
        only_in_train = [c for c in train_df.columns if c not in test_df.columns]
        num_in_train  = [c for c in only_in_train if pd.api.types.is_numeric_dtype(train_df[c])]
        target_col = num_in_train[0] if num_in_train else None

    id_col = id_name if (id_name and id_name in test_df.columns) else None
    if id_col is None:
        common = [c for c in test_df.columns if c in train_df.columns]
        if common:
            id_col = max(common, key=lambda c: test_df[c].nunique(dropna=True))

    return train_df.copy(), test_df.copy(), id_col, target_col, id_name, label_name

def _split_features(df: pd.DataFrame, exclude: List[str]):
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def _build_ml_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, id_col: Optional[str]):
    exclude = [target_col]
    if id_col and id_col in train_df.columns:
        exclude.append(id_col)

    num_cols, cat_cols = _split_features(train_df, exclude)

    # Impute numerics
    for c in num_cols:
        med = pd.to_numeric(train_df[c], errors="coerce").median()
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(med)
        test_df[c]  = pd.to_numeric(test_df[c], errors="coerce").fillna(med)

    # Ordinal encode categoricals
    for c in cat_cols:
        trc = train_df[c].astype(str).fillna("NaN")
        tec = test_df[c].astype(str).fillna("NaN")
        cats = pd.Index(trc.unique())
        mapping = {k:i for i,k in enumerate(cats)}
        train_df[c] = trc.map(mapping).fillna(-1).astype(int)
        test_df[c]  = tec.map(mapping).fillna(-1).astype(int)

    X_train = train_df[num_cols + cat_cols].copy()
    X_test  = test_df[num_cols + cat_cols].copy()
    y = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).values
    return X_train, X_test, y

def tabular_grouped_mean_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: Optional[str], target_col: Optional[str]):
    if target_col is None or target_col not in train_df.columns:
        gmean = 0.0
    else:
        gmean = float(pd.to_numeric(train_df[target_col], errors="coerce").dropna().mean())
    n = len(test_df)
    out = pd.DataFrame({"id": test_df[id_col] if (id_col and id_col in test_df.columns) else np.arange(n)})
    out["sales"] = gmean
    return out

def tabular_ml_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: Optional[str], target_col: Optional[str]):
    if target_col is None or target_col not in train_df.columns:
        return tabular_grouped_mean_predict(train_df, test_df, id_col, target_col)

    max_rows = 200_000
    if len(train_df) > max_rows:
        train_df = train_df.sample(n=max_rows, random_state=42)

    X_train, X_test, y = _build_ml_matrix(train_df.copy(), test_df.copy(), target_col, id_col)

    y_unique = np.unique(y[~np.isnan(y)])
    is_binary = (len(y_unique) <= 2) or (np.array_equal(np.unique(y.astype(int)), np.array([0,1])))

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    except Exception:
        return tabular_grouped_mean_predict(train_df, test_df, id_col, target_col)

    try:
        if is_binary:
            model = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42)
            model.fit(X_train, y.astype(int))
            proba = model.predict_proba(X_test)[:, 1]
            preds = proba.astype(float)
        else:
            model = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42)
            model.fit(X_train, y.astype(float))
            preds = model.predict(X_test).astype(float)
    except Exception:
        return tabular_grouped_mean_predict(train_df, test_df, id_col, target_col)

    n = len(test_df)
    out = pd.DataFrame({"id": test_df[id_col] if (id_col and id_col in test_df.columns) else np.arange(n)})
    out["sales"] = preds
    return out

# ==================== AUX merge for sales ====================
def _find_aux_path(aux_list: List[str], key: str) -> Optional[str]:
    key = key.lower()
    for p in aux_list:
        if p and key in os.path.basename(p).lower():
            return p
    return None

def merge_aux_sales(train_df: pd.DataFrame, test_df: pd.DataFrame, aux_paths: List[str], logs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_df.copy(), test_df.copy()

    def _merge(df_left: pd.DataFrame, df_right: Optional[pd.DataFrame], on, how="left"):
        if df_right is None:
            return df_left
        try:
            return df_left.merge(df_right, on=on, how=how)
        except Exception as e:
            logs.append(f"merge on {on} failed: {e}")
            return df_left

    # stores.csv by store_nbr
    stores_p = _find_aux_path(aux_paths, "stores")
    stores = _read_safe(stores_p)
    if stores is not None and not tr.empty and "store_nbr" in tr.columns:
        tr = _merge(tr, stores, on="store_nbr")
        te = _merge(te, stores, on="store_nbr")
        logs.append("Merged stores.csv on store_nbr.")

    # transactions.csv by date, store_nbr
    trans_p = _find_aux_path(aux_paths, "transactions")
    trans = _read_safe(trans_p)
    if trans is not None and not tr.empty and {"date","store_nbr"}.issubset(tr.columns):
        for df in (tr, te, trans):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        tr = _merge(tr, trans, on=["date","store_nbr"])
        te = _merge(te, trans, on=["date","store_nbr"])
        logs.append("Merged transactions.csv on [date, store_nbr].")

    # oil.csv by date (ffill)
    oil_p = _find_aux_path(aux_paths, "oil")
    oil = _read_safe(oil_p)
    if oil is not None and not tr.empty and "date" in tr.columns:
        oil["date"] = pd.to_datetime(oil["date"], errors="coerce")
        oil = oil.sort_values("date").ffill()
        tr = _merge(tr, oil, on="date")
        te = _merge(te, oil, on="date")
        logs.append("Merged oil.csv on date (ffill).")

    # holidays_events.csv by date (selected cols)
    hol_p = _find_aux_path(aux_paths, "holidays") or _find_aux_path(aux_paths, "holiday")
    hol = _read_safe(hol_p)
    if hol is not None and not tr.empty and "date" in tr.columns:
        hol["date"] = pd.to_datetime(hol["date"], errors="coerce")
        keep_cols = [c for c in ["date","type","locale","transferred"] if c in hol.columns]
        tr = _merge(tr, hol[keep_cols], on="date")
        te = _merge(te, hol[keep_cols], on="date")
        logs.append("Merged holidays_events.csv on date (selected cols).")

    return tr, te

# ==================== Align to sample ====================
def align_sample_submission(sub: pd.DataFrame, sample_path: Optional[str]) -> pd.DataFrame:
    if sample_path is None or not os.path.exists(sample_path):
        return sub
    try:
        sm = pd.read_csv(sample_path)
        # Only align if we actually have "sales" predictions or a mappable prediction column
        if len(sm.columns) >= 2 and ("sales" in sub.columns or "id" in sub.columns):
            id_name, label_name = sm.columns[:2].tolist()
            # Prefer mapping by name if possible
            pred_col = "sales" if "sales" in sub.columns else None
            if pred_col is None:
                return sub  # no known prediction column, skip alignment
            # Use id column from sub if it matches id_name, otherwise try 'id'
            if id_name in sub.columns:
                pred_map = dict(zip(sub[id_name], sub[pred_col]))
            else:
                pred_map = dict(zip(sub["id"], sub[pred_col]))
            out = sm.copy()
            out[label_name] = out[id_name].map(pred_map)
            return out
    except Exception:
        pass
    return sub

# ==================== Entry / Executor ====================
def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = os.path.basename(cfg.get("base_dir", "")).strip() or "dataset"
    summaries = {
        "train": summarize_csv(cfg.get("train")) if cfg.get("train") else {"exists": False},
        "test": summarize_csv(cfg.get("test")) if cfg.get("test") else {"exists": False},
        "sample": summarize_csv(cfg.get("sample_submission")) if cfg.get("sample_submission") else {"exists": False},
        "aux": [summarize_csv(p) for p in (cfg.get("aux") or [])],
    }

    logs: List[str] = []
    outputs: Dict[str, Any] = {}
    label_for_mean = None

    # Build plan (LLM or heuristic)
    try:
        plan = planner_llm(dataset_name, summaries)
    except Exception as e:
        plan = _heuristic_plan(dataset_name, summaries, llm_error=str(e))

    # Normalize ops
    graph = plan.get("graph", []) or []
    for node in graph:
        node["op"] = _normalize_op(node.get("op", ""))

    if graph:
        unknown = [n["op"] for n in graph if not _is_known_op(n["op"])]
        if unknown:
            logs.append(f"ℹ️ normalized ops; unknown after mapping (ignored if any): {unknown}")

    # If none known ops, fallback
    if not any(_is_known_op(n["op"]) and n["op"] != "noop" for n in graph):
        logs.append("⚠️ LLM plan had no executable ops; synthesizing a canonical plan.")
        plan = _heuristic_plan(dataset_name, summaries, llm_error=plan.get("_note"))
        graph = plan.get("graph", []) or []
        for node in graph:
            node["op"] = _normalize_op(node.get("op", ""))

    # Load base data once
    tmp_tr = _read_safe(cfg.get("train"))
    train_df = tmp_tr if tmp_tr is not None else pd.DataFrame()
    tmp_te  = _read_safe(cfg.get("test"))
    test_df  = tmp_te if tmp_te is not None else pd.DataFrame()

    # State through pipeline
    cur_train, cur_test = train_df, test_df
    id_col = None; target_col = None; id_name = None; label_name = None
    submission = None

    try:
        for node in graph:
            op = node["op"]
            if op == "noop":
                logs.append(f"Skipped noop for node {node.get('id','')}")
                continue

            if op == "prepare_store_sales":
                if not cur_train.empty:
                    cur_train = prepare_store_sales(cur_train)
                if not cur_test.empty:
                    cur_test  = prepare_store_sales(cur_test)
                logs.append("Prepared store-sales features (date→dow, onpromotion cast).")

            elif op == "merge_aux":
                aux_list = cfg.get("aux") or []
                cur_train, cur_test = merge_aux_sales(cur_train, cur_test, aux_list, logs)

            elif op == "dow_mean_forecast":
                base_tr = cur_train if not cur_train.empty else train_df
                base_te = cur_test if not cur_test.empty else test_df
                if base_tr.empty or base_te.empty:
                    logs.append("dow_mean_forecast skipped: empty train/test")
                else:
                    submission = dow_mean_forecast(base_tr, base_te)
                    logs.append("Ran DoW mean forecast baseline.")

            elif op == "prepare_tabular_generic":
                base_tr = cur_train if not cur_train.empty else train_df
                base_te = cur_test if not cur_test.empty else test_df
                cur_train, cur_test, id_col, target_col, id_name, label_name = prepare_tabular_generic(
                    base_tr, base_te, cfg.get("sample_submission")
                )
                logs.append(f"Prepared generic tabular (id_col={id_col}, target_col={target_col}, sample_id={id_name}, label={label_name}).")

            elif op == "tabular_ml_predict":
                base_tr = cur_train if not cur_train.empty else train_df
                base_te = cur_test if not cur_test.empty else test_df
                if base_te.empty:
                    logs.append("tabular_ml_predict skipped: empty test")
                else:
                    submission = tabular_ml_predict(base_tr, base_te, id_col, target_col)
                    # carry sample id col if present for align
                    if id_name and id_name in base_te.columns and id_name not in submission.columns:
                        submission[id_name] = base_te[id_name].values
                    logs.append("Ran ML-lite predictor for generic tabular.")

            elif op == "align_sample_submission":
                # we align at the end outside the loop; mark intention
                logs.append("Will align to sample submission at the end.")

            else:
                logs.append(f"Unknown op encountered (ignored): {op}")

        # Align & save
        if submission is not None and not submission.empty:
            before_align_mean = float(pd.to_numeric(
                submission["sales"] if "sales" in submission.columns else submission.iloc[:, -1],
                errors="coerce"
            ).mean())

            aligned = align_sample_submission(submission, cfg.get("sample_submission"))

            # derive label col name for mean print
            if cfg.get("sample_submission") and os.path.exists(cfg["sample_submission"]):
                sm = pd.read_csv(cfg["sample_submission"], nrows=1)
                if len(sm.columns) >= 2:
                    label_for_mean = sm.columns[1]

            # fill NaNs using pre-align mean
            if label_for_mean and label_for_mean in aligned.columns:
                aligned[label_for_mean] = pd.to_numeric(aligned[label_for_mean], errors="coerce").fillna(before_align_mean)
                mean_val = float(aligned[label_for_mean].mean())
            else:
                # fall back to 'sales' if exists
                if "sales" in aligned.columns:
                    aligned["sales"] = pd.to_numeric(aligned["sales"], errors="coerce").fillna(before_align_mean)
                    mean_val = float(aligned["sales"].mean())
                else:
                    # last column as prediction column
                    col = aligned.columns[-1]
                    aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(before_align_mean)
                    mean_val = float(aligned[col].mean())

            out_path = os.path.join(cfg["base_dir"], "submission.csv")
            aligned.to_csv(out_path, index=False)
            outputs["submission_path"] = out_path
            outputs["prediction_mean"] = mean_val
            logs.append(f"Mean prediction ({label_for_mean or 'prediction'}) = {mean_val:.6f}")
            logs.append(f"Saved submission to: {out_path}")
        else:
            logs.append("No submission was produced by the executed graph.")
    except Exception as e:
        logs.append("Execution error: " + str(e))
        logs.append(traceback.format_exc())

    return {"dataset": dataset_name, "plan": plan, "summaries": summaries, "outputs": outputs, "logs": logs}

def run_cag(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return run(cfg)
