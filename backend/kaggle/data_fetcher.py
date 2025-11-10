
"""
Kaggle数据获取器

负责从Kaggle下载竞赛数据、解析问题描述、获取评估指标等
"""
import os
import re
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CompetitionInfo:
    """
    竞赛信息类
    
    包含从Kaggle获取的所有竞赛相关信息
    """
    # 基本信息
    competition_id: str  # 竞赛ID（从URL提取）
    competition_name: str  # 竞赛名称
    competition_url: str  # 竞赛URL
    
    # 问题信息
    title: str = ""
    description: str = ""
    evaluation_metric: str = ""
    problem_type: str = ""  # classification, regression, time_series, etc.
    
    # 数据信息
    data_path: Optional[Path] = None
    train_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    sample_submission_file: Optional[str] = None
    
    # 数据统计
    train_shape: Optional[tuple] = None
    test_shape: Optional[tuple] = None
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    
    # 额外信息
    deadline: Optional[str] = None
    reward: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)  # 存储所有文件的详细信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "competition_id": self.competition_id,
            "competition_name": self.competition_name,
            "competition_url": self.competition_url,
            "title": self.title,
            "description": self.description,
            "evaluation_metric": self.evaluation_metric,
            "problem_type": self.problem_type,
            "data_path": str(self.data_path) if self.data_path else None,
            "train_files": self.train_files,
            "test_files": self.test_files,
            "sample_submission_file": self.sample_submission_file,
            "train_shape": self.train_shape,
            "test_shape": self.test_shape,
            "columns": self.columns,
            "column_types": self.column_types,
            "deadline": self.deadline,
            "reward": self.reward,
            "tags": self.tags,
            "extra_info": self.extra_info
        }
    
    def save(self, path: Path):
        """保存竞赛信息到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class KaggleDataFetcher:
    """
    Kaggle数据获取器
    
    功能：
    1. 从Kaggle URL提取竞赛ID
    2. 下载竞赛数据
    3. 解析竞赛描述和规则
    4. 分析数据集结构
    5. 识别训练/测试文件
    """
    
    def __init__(self):
        """初始化Kaggle API"""
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            logger.info("Kaggle API认证成功")
        except Exception as e:
            logger.error(f"Kaggle API认证失败: {e}")
            logger.info("请确保已配置 ~/.kaggle/kaggle.json 或环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY")
            raise
    
    @staticmethod
    def extract_competition_id(url: str) -> str:
        """
        从Kaggle URL提取竞赛ID
        
        支持的URL格式：
        - https://www.kaggle.com/competitions/store-sales-time-series-forecasting
        - https://www.kaggle.com/c/store-sales-time-series-forecasting
        - store-sales-time-series-forecasting
        
        Args:
            url: Kaggle竞赛URL或ID
            
        Returns:
            竞赛ID
        """
        # 如果已经是ID格式
        if not url.startswith("http"):
            return url
        
        # 提取ID
        patterns = [
            r'kaggle\.com/competitions/([^/\?]+)',
            r'kaggle\.com/c/([^/\?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"无法从URL提取竞赛ID: {url}")
    
    def fetch_competition_info(self, competition_url: str) -> CompetitionInfo:
        """
        获取竞赛基本信息
        
        Args:
            competition_url: 竞赛URL或ID
            
        Returns:
            竞赛信息对象
        """
        competition_id = self.extract_competition_id(competition_url)
        logger.info(f"获取竞赛信息: {competition_id}")
        
        try:
            # 获取竞赛详情
            competition = self.api.competition_list_cli(competition_id)
            
            # 创建信息对象
            info = CompetitionInfo(
                competition_id=competition_id,
                competition_name=competition.title or competition_id,
                competition_url=f"https://www.kaggle.com/competitions/{competition_id}",
                title=competition.title or "",
                description=competition.description or "",
                evaluation_metric=competition.evaluationMetric or "",
                deadline=str(competition.deadline) if competition.deadline else None,
                reward=competition.reward or None,
                tags=competition.tags or []
            )
            
            # 推断问题类型
            info.problem_type = self._infer_problem_type(info)
            
            logger.info(f"✓ 竞赛信息获取成功: {info.competition_name}")
            return info
            
        except Exception as e:
            logger.error(f"获取竞赛信息失败: {e}")
            # 返回基本信息
            return CompetitionInfo(
                competition_id=competition_id,
                competition_name=competition_id,
                competition_url=f"https://www.kaggle.com/competitions/{competition_id}"
            )
    
    def download_data(
        self,
        competition_id: str,
        download_path: Optional[Path] = None,
        force: bool = False
    ) -> Path:
        """
        下载竞赛数据
        
        Args:
            competition_id: 竞赛ID
            download_path: 下载路径（默认为data/competitions/{competition_id}）
            force: 是否强制重新下载
            
        Returns:
            数据目录路径
        """
        # 设置下载路径
        if download_path is None:
            download_path = config.competitions_dir / competition_id
        
        download_path.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已下载
        if not force and list(download_path.glob("*.csv")):
            logger.info(f"数据已存在，跳过下载: {download_path}")
            return download_path
        
        logger.info(f"开始下载竞赛数据: {competition_id}")
        logger.info(f"下载路径: {download_path}")
        
        try:
            # 下载所有文件
            self.api.competition_download_files(
                competition_id,
                path=str(download_path),
                quiet=False
            )
            
            # 解压zip文件
            self._extract_zip_files(download_path)
            
            logger.info(f"✓ 数据下载完成: {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"下载数据失败: {e}")
            raise
    
    def _extract_zip_files(self, directory: Path):
        """解压目录中的所有zip文件"""
        for zip_file in directory.glob("*.zip"):
            try:
                logger.info(f"解压: {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(directory)
                # 删除zip文件
                zip_file.unlink()
                logger.info(f"✓ 解压完成: {zip_file.name}")
            except Exception as e:
                logger.warning(f"解压失败 {zip_file.name}: {e}")
    
    def analyze_data(self, data_path: Path, info: CompetitionInfo) -> CompetitionInfo:
        """
        分析数据集结构
        
        Args:
            data_path: 数据目录路径
            info: 竞赛信息对象（会被更新）
            
        Returns:
            更新后的竞赛信息
        """
        logger.info(f"开始分析数据集: {data_path}")
        
        info.data_path = data_path
        
        # 识别所有CSV文件
        csv_files = list(data_path.glob("*.csv"))
        logger.info(f"发现 {len(csv_files)} 个CSV文件")
        
        # 存储所有文件的详细信息
        all_files_info = {}
        
        for csv_file in csv_files:
            filename = csv_file.name.lower()
            
            # 分类文件
            if 'train' in filename:
                info.train_files.append(csv_file.name)
            elif 'test' in filename:
                info.test_files.append(csv_file.name)
            elif 'sample' in filename or 'submission' in filename:
                info.sample_submission_file = csv_file.name
            
            # 分析每个CSV文件的结构
            try:
                logger.info(f"分析文件: {csv_file.name}")
                df = pd.read_csv(csv_file, nrows=100)  # 读取前100行了解结构
                
                file_info = {
                    "filename": csv_file.name,
                    "rows_sample": len(df),
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "sample_data": df.head(3).to_dict('records')  # 前3行样例数据
                }
                
                all_files_info[csv_file.name] = file_info
                logger.info(f"  ✓ {csv_file.name}: {len(df.columns)} 列")
                
            except Exception as e:
                logger.warning(f"  ✗ 分析 {csv_file.name} 失败: {e}")
        
        # 保存所有文件信息到extra字段
        if not hasattr(info, 'extra_info'):
            info.extra_info = {}
        info.extra_info['all_files'] = all_files_info
        
        # 分析训练数据（主要数据）
        if info.train_files:
            train_file = data_path / info.train_files[0]
            try:
                logger.info(f"详细分析训练数据: {train_file.name}")
                df = pd.read_csv(train_file, nrows=1000)
                
                info.train_shape = (len(df), len(df.columns))
                info.columns = df.columns.tolist()
                info.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                logger.info(f"✓ 训练数据形状: {info.train_shape}")
                logger.info(f"✓ 列数: {len(info.columns)}")
                
            except Exception as e:
                logger.warning(f"分析训练数据失败: {e}")
        
        # 分析测试数据
        if info.test_files:
            test_file = data_path / info.test_files[0]
            try:
                logger.info(f"详细分析测试数据: {test_file.name}")
                df = pd.read_csv(test_file, nrows=100)
                info.test_shape = (len(df), len(df.columns))
                logger.info(f"✓ 测试数据形状: {info.test_shape}")
            except Exception as e:
                logger.warning(f"分析测试数据失败: {e}")
        
        logger.info("✓ 数据分析完成")
        return info
    
    def fetch_complete_info(
        self,
        competition_url: str,
        download_data: bool = True,
        force_download: bool = False
    ) -> CompetitionInfo:
        """
        获取完整的竞赛信息（一站式方法）
        
        Args:
            competition_url: 竞赛URL或ID
            download_data: 是否下载数据
            force_download: 是否强制重新下载
            
        Returns:
            完整的竞赛信息
        """
        logger.info("=" * 60)
        logger.info("开始获取完整竞赛信息")
        logger.info("=" * 60)
        
        # 1. 获取基本信息
        info = self.fetch_competition_info(competition_url)
        
        # 2. 下载数据
        if download_data:
            data_path = self.download_data(
                info.competition_id,
                force=force_download
            )
            
            # 3. 分析数据
            info = self.analyze_data(data_path, info)
        
        # 4. 保存信息
        if info.data_path:
            info_file = info.data_path / "competition_info.json"
            info.save(info_file)
            logger.info(f"竞赛信息已保存: {info_file}")
        
        logger.info("=" * 60)
        logger.info("✓ 完整竞赛信息获取成功")
        logger.info("=" * 60)
        
        return info
    
    def _infer_problem_type(self, info: CompetitionInfo) -> str:
        """
        根据描述和指标推断问题类型
        
        Returns:
            问题类型：classification, regression, time_series, ranking, etc.
        """
        text = (info.description + " " + info.evaluation_metric + " " + info.title).lower()
        
        # 时间序列
        if any(keyword in text for keyword in ['time series', 'forecasting', 'forecast', 'temporal']):
            return "time_series_forecasting"
        
        # 分类
        if any(keyword in text for keyword in ['classification', 'classify', 'class', 'accuracy', 'f1', 'auc', 'roc']):
            return "classification"
        
        # 回归
        if any(keyword in text for keyword in ['regression', 'predict', 'rmse', 'mae', 'mse', 'r2']):
            return "regression"
        
        # 排序
        if any(keyword in text for keyword in ['ranking', 'recommend', 'retrieval']):
            return "ranking"
        
        # 聚类
        if any(keyword in text for keyword in ['clustering', 'cluster', 'segmentation']):
            return "clustering"
        
        # NLP
        if any(keyword in text for keyword in ['nlp', 'text', 'sentiment', 'language']):
            return "nlp"
        
        # 计算机视觉
        if any(keyword in text for keyword in ['image', 'vision', 'detection', 'segmentation', 'object']):
            return "computer_vision"
        
        return "unknown"
    
    def get_data_summary(self, info: CompetitionInfo) -> str:
        """
        生成数据摘要（用于传给LLM）
        
        Args:
            info: 竞赛信息
            
        Returns:
            格式化的数据摘要文本
        """
        summary = []
        summary.append(f"# {info.title or info.competition_name}")
        summary.append("")
        summary.append("## 竞赛信息")
        summary.append(f"- 竞赛ID: {info.competition_id}")
        summary.append(f"- 问题类型: {info.problem_type}")
        summary.append(f"- 评估指标: {info.evaluation_metric}")
        
        if info.description:
            summary.append("")
            summary.append("## 问题描述")
            summary.append(info.description[:500] + "..." if len(info.description) > 500 else info.description)
        
        summary.append("")
        summary.append("## 主要数据文件")
        summary.append(f"- 训练文件: {', '.join(info.train_files)}")
        summary.append(f"- 测试文件: {', '.join(info.test_files)}")
        if info.sample_submission_file:
            summary.append(f"- 提交样例: {info.sample_submission_file}")
        
        # 添加所有文件的详细信息
        if info.extra_info and 'all_files' in info.extra_info:
            summary.append("")
            summary.append("## 所有可用数据文件详情")
            all_files = info.extra_info['all_files']
            
            for filename, file_info in all_files.items():
                summary.append(f"\n### {filename}")
                summary.append(f"- 列: {', '.join(file_info['columns'])}")
                summary.append(f"- 数据类型:")
                for col, dtype in file_info['dtypes'].items():
                    summary.append(f"  - {col}: {dtype}")
                
                # 添加样例数据
                if file_info.get('sample_data'):
                    summary.append(f"- 样例数据（前3行）:")
                    for i, row in enumerate(file_info['sample_data'][:3], 1):
                        summary.append(f"  行{i}: {row}")
        
        if info.train_shape:
            summary.append("")
            summary.append("## 数据规模")
            summary.append(f"- 训练集形状: {info.train_shape}")
            if info.test_shape:
                summary.append(f"- 测试集形状: {info.test_shape}")
        
        if info.columns:
            summary.append("")
            summary.append("## 训练数据列详情")
            summary.append(f"共 {len(info.columns)} 列:")
            for col in info.columns[:20]:
                col_type = info.column_types.get(col, "unknown")
                summary.append(f"  - {col}: {col_type}")
            if len(info.columns) > 20:
                summary.append(f"  ... 还有 {len(info.columns) - 20} 列")
        
        summary.append("")
        summary.append("## 重要提示")
        summary.append("- 请充分利用所有可用的数据文件来构建特征")
        summary.append("- 辅助数据文件（如stores, oil, holidays等）可能包含重要的预测特征")
        summary.append("- 使用适当的join/merge操作合并数据")
        
        return "\n".join(summary)
