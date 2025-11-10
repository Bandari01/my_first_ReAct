"""
LLM集成模块

提供统一的LLM接口，支持OpenAI、Ollama等
"""
from .llm_client import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse"]

