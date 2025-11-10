"""
全局配置管理
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
COMPETITIONS_DIR = DATA_DIR / "competitions"
GENERATED_CODE_DIR = DATA_DIR / "generated_code"

# 创建必要的目录
for directory in [DATA_DIR, COMPETITIONS_DIR, GENERATED_CODE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Kaggle配置
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Ollama配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# 执行配置
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME", "600"))  # 秒（增加到10分钟）
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "2048"))
ENABLE_DOCKER_SANDBOX = os.getenv("ENABLE_DOCKER_SANDBOX", "false").lower() == "true"

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# LLM配置
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096

class Config:
    """配置类"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.competitions_dir = COMPETITIONS_DIR
        self.generated_code_dir = GENERATED_CODE_DIR
        
        self.kaggle_username = KAGGLE_USERNAME
        self.kaggle_key = KAGGLE_KEY
        
        self.ollama_base_url = OLLAMA_BASE_URL
        self.ollama_model = OLLAMA_MODEL
        
        self.max_execution_time = MAX_EXECUTION_TIME
        self.max_memory_mb = MAX_MEMORY_MB
        self.enable_docker_sandbox = ENABLE_DOCKER_SANDBOX
        
        self.log_level = LOG_LEVEL
        
        self.default_temperature = DEFAULT_TEMPERATURE
        self.default_max_tokens = DEFAULT_MAX_TOKENS
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """验证配置是否完整"""
        if not self.kaggle_username or not self.kaggle_key:
            return False, "Kaggle API凭证未配置"
        
        return True, None

# 全局配置实例
config = Config()
