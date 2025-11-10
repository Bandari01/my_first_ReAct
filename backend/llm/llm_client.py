"""
LLM客户端

统一的LLM调用接口，支持OpenAI和Ollama
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from openai import OpenAI

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    
    def __str__(self):
        return self.content


class LLMClient:
    """
    LLM客户端
    
    支持OpenAI和Ollama，提供统一接口
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        初始化LLM客户端
        
        Args:
            provider: 提供商 ("openai" 或 "ollama")
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
            
            if not self.api_key:
                raise ValueError("OpenAI API密钥未设置")
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None
            )
            logger.info(f"初始化OpenAI客户端: model={self.model}")
            
        elif provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            # Ollama使用OpenAI兼容的API
            self.client = OpenAI(
                api_key="ollama",  # Ollama不需要真实的key
                base_url=f"{self.base_url}/v1"
            )
            logger.info(f"初始化Ollama客户端: model={self.model}")
        else:
            raise ValueError(f"不支持的提供商: {provider}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大token数（覆盖默认值）
            
        Returns:
            LLM响应
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason
            
            logger.info(f"LLM调用成功: tokens={tokens_used}, reason={finish_reason}")
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        生成文本（简化接口）
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            LLM响应
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens)

