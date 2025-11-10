"""
日志工具
"""
import sys
from pathlib import Path
from loguru import logger
from backend.config import LOG_LEVEL, PROJECT_ROOT

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 移除默认处理器
logger.remove()

# 添加控制台输出
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# 添加文件输出
logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="00:00",  # 每天轮换
    retention="30 days",  # 保留30天
    compression="zip"  # 压缩旧日志
)

def get_logger(name: str):
    """获取命名logger"""
    return logger.bind(name=name)

__all__ = ["logger", "get_logger"]

