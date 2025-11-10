"""
简化的代码执行器

功能：
- 执行生成的 Python 代码（以子进程隔离）
- 若执行失败（返回码非 0 或出现异常），将 traceback 与原始代码发给项目的 LLM 客户端，请求修正后的代码
- 将 LLM 返回的修正代码抽取并重试执行，最多重试若干次

注意：此模块依赖 `backend.llm.llm_client.LLMClient` 的 `generate(prompt)` 方法，返回值的 `.content` 包含 LLM 生成的修正代码（最好包含 ```python ``` 代码块）。
"""
from dataclasses import dataclass
import subprocess
import sys
import tempfile
import time
import os
import re
from pathlib import Path
from typing import Optional

from backend.utils.logger import get_logger
from backend.llm.llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    return_code: int = 0


class GeneratedCodeExecutor:
    """只执行生成代码并在错误时请求 LLM 修正的执行器"""

    def __init__(self, timeout: int = 300, max_fix_attempts: int = 3):
        self.timeout = timeout
        self.max_fix_attempts = max_fix_attempts
        # 初始化 LLM 客户端（使用默认环境配置）
        try:
            self.llm = LLMClient()
        except Exception as e:
            logger.warning(f"无法初始化 LLMClient: {e}")
            self.llm = None

    def _run_subprocess(self, code: str, working_dir: Optional[Path] = None, env: Optional[dict] = None) -> ExecutionResult:
        start = time.time()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            tmp_path = Path(f.name)
            f.write(code)

        cwd = str(working_dir) if working_dir else None
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)

        try:
            proc = subprocess.Popen([sys.executable, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env_vars, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return ExecutionResult(False, stdout, error=f"Timeout after {self.timeout}s", execution_time=time.time()-start, return_code=-1)

            return ExecutionResult(proc.returncode == 0, stdout, error=stderr if stderr else None, execution_time=time.time()-start, return_code=proc.returncode)
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _extract_code_from_response(self, text: str) -> str:
        # 首先尝试抓取 ```python ... ``` 区块
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # 回退：尝试提取第一个 def/class 或以 import 开头的代码段
        return text.strip()

    def _ask_llm_to_fix(self, original_code: str, traceback_text: str) -> Optional[str]:
        if not self.llm:
            logger.error("LLM client 未初始化，无法请求修复")
            return None

        prompt = (
            "你的任务：修复下面的 Python 代码。\n"
            "要求：返回 **仅** 修正后的完整 Python 代码（如果可能，请用 ```python ``` 包裹）。\n\n"
            "=== 原始代码 ===\n"
            f"{original_code}\n\n"
            "=== 发生的错误/Traceback ===\n"
            f"{traceback_text}\n\n"
            "请直接返回修正后的代码，不要返回解释或额外文本。"
        )

        try:
            resp = self.llm.generate(prompt)
            return self._extract_code_from_response(resp.content)
        except Exception as e:
            logger.error(f"请求 LLM 修复失败：{e}")
            return None

    def execute_and_autofix(self, code: str, working_dir: Optional[Path] = None, env: Optional[dict] = None) -> ExecutionResult:
        """执行代码，若失败则调用 LLM 修复并重试（最多 self.max_fix_attempts 次）。"""
        attempt = 0
        current_code = code

        while True:
            attempt += 1
            logger.info(f"执行代码，尝试 #{attempt}")
            result = self._run_subprocess(current_code, working_dir, env)

            if result.success:
                logger.info(f"代码执行成功 (attempt={attempt}, time={result.execution_time:.2f}s)")
                return result

            # 执行失败
            tb = result.error or ""
            error_summary = tb.strip()
            error_message = f"代码执行失败 (return_code={result.return_code})" + (f"\n错误信息: {error_summary[:400]}..." if error_summary else "")
            logger.warning(f"{error_message}，准备请求 LLM 修复")

            if attempt > self.max_fix_attempts:
                logger.error(f"已达到最大修复尝试次数 ({self.max_fix_attempts})，停止重试")
                return result

            fixed = self._ask_llm_to_fix(current_code, tb or "No stderr captured")
            if fixed:
                logger.info("LLM 返回修正代码，准备下一次尝试...")
            if not fixed:
                logger.error("LLM 未返回修正代码或请求失败，停止重试")
                return result

            logger.info("LLM 返回修正代码，进行下一次尝试")
            current_code = fixed


