"""
提交文件验证器

验证生成的submission.csv是否符合Kaggle要求
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SubmissionValidator:
    """
    提交文件验证器
    
    功能：
    1. 验证submission.csv格式
    2. 检查列名和列数
    3. 验证ID完整性
    4. 检查预测值范围
    """
    
    def __init__(self, sample_submission_path: Optional[Path] = None):
        """
        初始化验证器
        
        Args:
            sample_submission_path: 样例提交文件路径
        """
        self.sample_submission_path = sample_submission_path
        self.sample_df: Optional[pd.DataFrame] = None
        
        if sample_submission_path and sample_submission_path.exists():
            try:
                self.sample_df = pd.read_csv(sample_submission_path)
                logger.info(f"加载样例提交文件: {sample_submission_path}")
                logger.info(f"样例形状: {self.sample_df.shape}")
                logger.info(f"样例列: {list(self.sample_df.columns)}")
            except Exception as e:
                logger.warning(f"加载样例提交文件失败: {e}")
    
    def validate(self, submission_path: Path) -> Tuple[bool, List[str]]:
        """
        验证提交文件
        
        Args:
            submission_path: 提交文件路径
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 1. 检查文件是否存在
        if not submission_path.exists():
            errors.append(f"提交文件不存在: {submission_path}")
            return False, errors
        
        try:
            # 2. 尝试读取文件
            submission_df = pd.read_csv(submission_path)
            logger.info(f"提交文件形状: {submission_df.shape}")
            logger.info(f"提交文件列: {list(submission_df.columns)}")
            
        except Exception as e:
            errors.append(f"无法读取提交文件: {e}")
            return False, errors
        
        # 3. 如果有样例文件，进行对比验证
        if self.sample_df is not None:
            errors.extend(self._validate_against_sample(submission_df))
        else:
            # 4. 基本验证（没有样例文件时）
            errors.extend(self._basic_validation(submission_df))
        
        # 5. 检查是否有缺失值
        if submission_df.isnull().any().any():
            null_cols = submission_df.columns[submission_df.isnull().any()].tolist()
            errors.append(f"存在缺失值的列: {null_cols}")
        
        # 6. 检查是否有无穷值
        numeric_cols = submission_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if (submission_df[col] == float('inf')).any() or (submission_df[col] == float('-inf')).any():
                errors.append(f"列 '{col}' 包含无穷值")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("✓ 提交文件验证通过")
        else:
            logger.warning(f"✗ 提交文件验证失败，发现 {len(errors)} 个问题")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    def _validate_against_sample(self, submission_df: pd.DataFrame) -> List[str]:
        """对比样例文件进行验证"""
        errors = []
        
        # 检查列名
        expected_cols = list(self.sample_df.columns)
        actual_cols = list(submission_df.columns)
        
        if expected_cols != actual_cols:
            errors.append(
                f"列名不匹配。期望: {expected_cols}, 实际: {actual_cols}"
            )
        
        # 检查行数
        expected_rows = len(self.sample_df)
        actual_rows = len(submission_df)
        
        if expected_rows != actual_rows:
            errors.append(
                f"行数不匹配。期望: {expected_rows}, 实际: {actual_rows}"
            )
        
        # 检查ID列（通常第一列是ID）
        if len(expected_cols) > 0 and len(actual_cols) > 0:
            id_col = expected_cols[0]
            
            if id_col in submission_df.columns and id_col in self.sample_df.columns:
                expected_ids = set(self.sample_df[id_col])
                actual_ids = set(submission_df[id_col])
                
                missing_ids = expected_ids - actual_ids
                extra_ids = actual_ids - expected_ids
                
                if missing_ids:
                    errors.append(f"缺少 {len(missing_ids)} 个ID")
                if extra_ids:
                    errors.append(f"多出 {len(extra_ids)} 个ID")
        
        # 检查数据类型
        for col in expected_cols:
            if col in actual_cols:
                expected_dtype = self.sample_df[col].dtype
                actual_dtype = submission_df[col].dtype
                
                # 允许int/float的兼容
                if not self._dtypes_compatible(expected_dtype, actual_dtype):
                    errors.append(
                        f"列 '{col}' 数据类型不匹配。期望: {expected_dtype}, 实际: {actual_dtype}"
                    )
        
        return errors
    
    def _basic_validation(self, submission_df: pd.DataFrame) -> List[str]:
        """基本验证（无样例文件时）"""
        errors = []
        
        # 检查是否为空
        if len(submission_df) == 0:
            errors.append("提交文件为空")
        
        # 检查列数
        if len(submission_df.columns) < 2:
            errors.append(f"列数过少: {len(submission_df.columns)}，通常至少需要ID列和预测列")
        
        # 检查是否有ID列（常见的ID列名）
        id_col_names = ['id', 'Id', 'ID', 'index']
        has_id_col = any(col in submission_df.columns for col in id_col_names)
        
        if not has_id_col:
            logger.warning("未找到明确的ID列，请确认第一列是否为ID")
        
        return errors
    
    @staticmethod
    def _dtypes_compatible(dtype1, dtype2) -> bool:
        """检查两个数据类型是否兼容"""
        # 数值类型之间兼容
        numeric_types = ['int64', 'int32', 'float64', 'float32']
        if str(dtype1) in numeric_types and str(dtype2) in numeric_types:
            return True
        
        # 对象类型和字符串类型兼容
        string_types = ['object', 'string']
        if str(dtype1) in string_types and str(dtype2) in string_types:
            return True
        
        # 完全相同
        return dtype1 == dtype2
    
    def get_submission_summary(self, submission_path: Path) -> Dict:
        """
        获取提交文件摘要
        
        Args:
            submission_path: 提交文件路径
            
        Returns:
            摘要信息字典
        """
        try:
            df = pd.read_csv(submission_path)
            
            summary = {
                "file_path": str(submission_path),
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "has_null": df.isnull().any().any(),
                "null_counts": df.isnull().sum().to_dict(),
            }
            
            # 数值列的统计
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                summary["numeric_stats"] = {}
                for col in numeric_cols:
                    summary["numeric_stats"][col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std())
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取提交文件摘要失败: {e}")
            return {"error": str(e)}
    
    def fix_common_issues(
        self,
        submission_path: Path,
        output_path: Optional[Path] = None
    ) -> Tuple[bool, Path]:
        """
        尝试修复常见问题
        
        Args:
            submission_path: 提交文件路径
            output_path: 输出路径（默认覆盖原文件）
            
        Returns:
            (是否成功, 输出文件路径)
        """
        if output_path is None:
            output_path = submission_path
        
        try:
            df = pd.read_csv(submission_path)
            modified = False
            
            # 1. 填充缺失值（用0或中位数）
            if df.isnull().any().any():
                logger.info("修复: 填充缺失值")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                
                # 对象类型用空字符串填充
                object_cols = df.select_dtypes(include=['object']).columns
                for col in object_cols:
                    if df[col].isnull().any():
                        df[col].fillna("", inplace=True)
                
                modified = True
            
            # 2. 替换无穷值
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if (df[col] == float('inf')).any():
                    logger.info(f"修复: 替换列 '{col}' 中的正无穷值")
                    df[col].replace(float('inf'), df[col][df[col] != float('inf')].max(), inplace=True)
                    modified = True
                
                if (df[col] == float('-inf')).any():
                    logger.info(f"修复: 替换列 '{col}' 中的负无穷值")
                    df[col].replace(float('-inf'), df[col][df[col] != float('-inf')].min(), inplace=True)
                    modified = True
            
            # 3. 如果有样例文件，对齐列顺序
            if self.sample_df is not None:
                expected_cols = list(self.sample_df.columns)
                if list(df.columns) != expected_cols:
                    logger.info("修复: 对齐列顺序")
                    # 只保留期望的列，并按期望的顺序
                    df = df[[col for col in expected_cols if col in df.columns]]
                    modified = True
            
            # 保存修复后的文件
            if modified:
                df.to_csv(output_path, index=False)
                logger.info(f"✓ 修复后的文件已保存: {output_path}")
                return True, output_path
            else:
                logger.info("无需修复")
                return True, submission_path
            
        except Exception as e:
            logger.error(f"修复失败: {e}")
            return False, submission_path

