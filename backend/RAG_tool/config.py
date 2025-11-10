"""
RAG configuration - moved under backend.RAG_tool
"""
import os
from pathlib import Path


class RAGConfig:
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        max_retries: int = 1,
        llm_timeout: int = 60,
        knowledge_base_path: str | None = None,
        top_k_retrieval: int = 5,
        similarity_threshold: float = 0.1,
        enable_validation: bool = True,
        openai_api_key: str | None = None,
    ):
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.llm_timeout = llm_timeout
        # Default knowledge base path relative to this file
        if knowledge_base_path is None:
            kb_dir = Path(__file__).resolve().parent / "knowledge_base"
            self.knowledge_base_path = str(kb_dir)
        else:
            self.knowledge_base_path = knowledge_base_path

        self.top_k_retrieval = top_k_retrieval
        self.similarity_threshold = similarity_threshold
        self.enable_validation = enable_validation
        self.openai_api_key = openai_api_key
