"""
ReAct Agent实现（时间序列优化版）

本文件已整合原 `base_agent.py` 的核心功能，使 `ReactAgent` 可以独立运行。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json
import time
from typing import Dict, Any
import asyncio

from backend.config import config
from backend.llm.llm_client import LLMClient
from backend.executor.code_executor import GeneratedCodeExecutor
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReactResult:
    """简化的 React 运行结果对象"""
    success: bool
    generated_code: Optional[str] = None
    submission_path: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    llm_calls: int = 0
    code_generation_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0
    observations: List[str] = field(default_factory=list)


class ReactAgent:
    """Minimal ReactAgent: accepts data, asks LLM for a runnable script, and delegates execution+fix to GeneratedCodeExecutor."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: int = 300,
        competition_name: str = "",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.llm = LLMClient(provider="openai", model=llm_model, temperature=temperature, max_tokens=max_tokens)
        self.executor = GeneratedCodeExecutor(timeout=timeout)
        self.competition_name = competition_name
        # prepare output dir
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.generated_code_dir / f"{competition_name}_{timestamp}"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger

    def _save_code(self, code: str) -> Path:
        code_file = self.output_dir / "generated_solution.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        self._logger.info(f"代码已保存: {code_file}")
        return code_file

    async def run(self, problem_description: str, data_info: Dict[str, Any]) -> ReactResult:
        """End-to-end: prepare prompt, call LLM to generate a complete Python script, save it, and hand to executor for run+autofix."""
        start = time.time()
        result = ReactResult(success=False)

        # Build prompt
        file_summary = "\n".join([f"- {k}: {v.get('columns', [])}" for k, v in data_info.get("all_files_info", {}).items()])
        system_prompt = (
            "You are a Kaggle time-series advisor. Produce a complete, runnable Python script that performs the full end-to-end pipeline for a Kaggle-style time-series forecasting problem.\n"
            "Return only the Python source code."
        )
        prompt = f"Task description:\n{problem_description}\n\nDetected files and basic schema:\n{file_summary}\n"

        try:
            # Generate code
            gen_start = time.time()
            response = self.llm.generate(prompt, system_prompt, temperature=0.3, max_tokens=4000)
            result.llm_calls += 1
            code = response.content if hasattr(response, 'content') else str(response)
            code = code.strip()
            result.code_generation_time = time.time() - gen_start
            result.generated_code = code
            self._save_code(code)

            # Execute with auto-fix
            exec_start = time.time()
            exec_res = await asyncio.to_thread(self.executor.execute_and_autofix, code, self.output_dir)
            result.execution_time = time.time() - exec_start

            if getattr(exec_res, 'success', False):
                result.success = True
                result.output = getattr(exec_res, 'output', '')
                result.submission_path = str(self.output_dir / 'submission.csv') if (self.output_dir / 'submission.csv').exists() else None
            else:
                result.success = False
                result.error = getattr(exec_res, 'error', 'Execution failed')
                result.output = getattr(exec_res, 'output', '')

            # observations
            if hasattr(exec_res, 'error') and exec_res.error:
                result.observations.append(f"Execution error: {exec_res.error}")
            if hasattr(exec_res, 'attempts'):
                result.observations.append(f"Fix attempts: {getattr(exec_res, 'attempts')}")

        except Exception as e:
            self._logger.exception("ReactAgent.run failed")
            result.success = False
            result.error = str(e)

        finally:
            result.total_time = time.time() - start

        return result

