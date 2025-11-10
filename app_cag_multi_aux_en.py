
# app_cag_multi_aux.py
# -*- coding: utf-8 -*-
import os
import io
import json
from typing import List, Optional, Dict, Any

import streamlit as st

import pandas as pd

# Local runner
try:
    from cag_runner import run as cag_run
except Exception:
    import importlib
    cag_run = importlib.import_module("cag_runner").run  # type: ignore


st.set_page_config(page_title="CAG · Multi-dataset Runner", layout="wide")

# ===================== Sidebar: LLM Controls =====================
def _apply_llm_env(openai_key: str, model: str, enable: bool):
    """Write environment variables into the current process only."""
    if enable and openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
        os.environ["USE_LLM"] = "1"
        os.environ["OPENAI_MODEL"] = model.strip()
    else:
        os.environ["USE_LLM"] = "0"

with st.sidebar:
    st.markdown("### 🤖 LLM Planner")
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "openai_enabled" not in st.session_state:
        st.session_state.openai_enabled = False
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4o-mini"

    key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_key,
        placeholder="sk-************************",
        help="，。",
    )
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"],
        index=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"].index(st.session_state.openai_model)
        if st.session_state.openai_model in ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"] else 0,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Enable", use_container_width=True):
            st.session_state.openai_key = key_input
            st.session_state.openai_model = model
            st.session_state.openai_enabled = bool(key_input.strip())
            _apply_llm_env(key_input, model, enable=st.session_state.openai_enabled)
            if st.session_state.openai_enabled:
                st.success("Enable LLM Planner")
            else:
                st.warning("API Key，Disable。")
    with col_b:
        if st.button("Disable", use_container_width=True):
            st.session_state.openai_enabled = False
            _apply_llm_env("", model, enable=False)
            st.info("Disable LLM Planner")

    if st.session_state.openai_enabled:
        st.caption(f"✅ ：{st.session_state.openai_model}")
    else:
        st.caption("⛔ LLM Planner Disable（/）")


# ===================== Dataset Section (Manual selectors always shown) =====================
st.markdown("## 📦 Dataset")
st.caption("train/test/sample（） CSV ；。")

base_dir = st.text_input("Dataset folder", value=os.getcwd())

def _list_csvs(folder: str) -> List[str]:
    try:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
        return files
    except Exception:
        return []

train_file = test_file = sample_file = ""
aux_files: List[str] = []

if os.path.isdir(base_dir):
    csvs = _list_csvs(base_dir)
    st.write("CSV files in this folder:", csvs)

    # autodetect defaults
    for name in csvs:
        low = name.lower()
        if "train" in low and not train_file:
            train_file = name
        elif "test" in low and not test_file:
            test_file = name
        elif "sample" in low and "submission" in low and not sample_file:
            sample_file = name
    default_aux = [name for name in csvs if name not in {train_file, test_file, sample_file}]

    # manual selectors (always visible)
    opts = [""] + csvs
    def _idx(val: str) -> int:
        try:
            return opts.index(val) if val in opts else 0
        except Exception:
            return 0

    c1, c2, c3 = st.columns(3)
    with c1:
        train_file = st.selectbox("Select train.csv", options=opts, index=_idx(train_file))
    with c2:
        test_file = st.selectbox("Select test.csv", options=opts, index=_idx(test_file))
    with c3:
        sample_file = st.selectbox("Select sample_submission.csv (optional)", options=opts, index=_idx(sample_file))

    aux_files = st.multiselect("Select auxiliary CSVs (multi)", options=csvs, default=default_aux)

else:
    st.warning("Dataset folder。")

col_run1, col_run2 = st.columns([1,1])
with col_run1:
    run_btn = st.button("🚀 Send to CAG", type="primary", use_container_width=True)
with col_run2:
    clear_btn = st.button("Clear Results", use_container_width=True)
    if clear_btn:
        st.session_state.pop("report", None)
        st.experimental_rerun()


# ===================== Execute =====================
if run_btn:
    cfg: Dict[str, Any] = {
        "base_dir": base_dir,
        "train": os.path.join(base_dir, train_file) if train_file else None,
        "test": os.path.join(base_dir, test_file) if test_file else None,
        "sample_submission": os.path.join(base_dir, sample_file) if sample_file else None,
        "aux": [os.path.join(base_dir, x) for x in aux_files] if aux_files else [],
    }
    try:
        report = cag_run(cfg)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        report = {"outputs": {}, "logs": [f"Front error: {e}", tb]}
    st.session_state.report = report

report = st.session_state.get("report")

# ===================== Outputs / Visualization =====================
if report:
    st.markdown("---")
    st.markdown("## 📊 Results")

    # 预测均值卡片（居中展示）
    outputs = report.get("outputs", {})
    pred_mean = outputs.get("prediction_mean", None)
    if pred_mean is not None:
        mean_card = f"""<div style='display:flex; justify-content:center; width:100%;'>
          <div style='text-align:center; background:#F6F8FA; border-radius:12px; padding:16px 18px; width:380px;'>
            <div style='font-size:14px;color:#6b7280;'>🧮 </div>
            <div style='font-size:30px; font-weight:800; color:#2B8A3E;'>{float(pred_mean):.6f}</div>
          </div>
        </div>"""
        st.markdown(mean_card, unsafe_allow_html=True)

    # submission 预览 + 下载
    sub_path = outputs.get("submission_path")
    if sub_path and os.path.exists(sub_path):
        try:
            sub_df = pd.read_csv(sub_path)
            st.markdown("#### 📝 Submission (first 20 rows)")
            st.dataframe(sub_df.head(20), use_container_width=True)

            # Download button
            csv_buf = io.StringIO()
            sub_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download submission.csv",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"submission ：{e}")
    else:
        st.info("No submission.csv was generated")

    # Fallback / 规范化提示
    logs: List[str] = report.get("logs", [])
    if any("falling back" in x.lower() for x in logs):
        st.info("🧠 LLM Planner ，。")
    if any("normalized ops" in x.lower() for x in logs):
        st.caption("ℹ️  LLM （op alias）。")

    # 展示计划 & 汇总
    with st.expander("📋 Plan", expanded=False):
        st.json(report.get("plan", {}))
    with st.expander("📄 Summaries", expanded=False):
        st.json(report.get("summaries", {}))
    with st.expander("🪵 Logs", expanded=True):
        for ln in logs:
            st.text(ln)

else:
    st.caption("。「Send to CAG」。")
