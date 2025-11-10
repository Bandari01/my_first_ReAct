"""
Kaggle集成模块

提供Kaggle竞赛数据获取、问题解析等功能
"""
from .data_fetcher import KaggleDataFetcher, CompetitionInfo
from .submission_validator import SubmissionValidator

__all__ = ["KaggleDataFetcher", "CompetitionInfo", "SubmissionValidator"]

