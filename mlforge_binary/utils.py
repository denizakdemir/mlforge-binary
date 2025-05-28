"""
Utility functions for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import yaml
import time
import joblib


def check_binary_target(y: Union[np.ndarray, pd.Series]) -> bool:
    """Check if target is binary classification."""
    unique_values = np.unique(y)
    return len(unique_values) == 2


def detect_imbalance(y: Union[np.ndarray, pd.Series], threshold: float = 0.1) -> bool:
    """Detect if dataset is imbalanced."""
    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    class_counts = y_series.value_counts()
    minority_ratio = class_counts.min() / class_counts.sum()
    return minority_ratio < threshold


def detect_missing_pattern(X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """Analyze missing value patterns in the data."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    missing_info = {}
    missing_info['total_missing'] = X.isnull().sum().sum()
    missing_info['missing_per_column'] = X.isnull().sum()
    missing_info['missing_percentage'] = (X.isnull().sum() / len(X) * 100)
    missing_info['columns_with_missing'] = X.columns[X.isnull().any()].tolist()
    missing_info['missing_pattern'] = X.isnull().sum(axis=1).value_counts().sort_index()
    
    # Classify missing pattern
    missing_cols = len(missing_info['columns_with_missing'])
    total_cols = len(X.columns)
    
    if missing_cols == 0:
        missing_info['pattern_type'] = 'none'
    elif missing_cols / total_cols < 0.1:
        missing_info['pattern_type'] = 'sporadic'
    elif missing_cols / total_cols < 0.5:
        missing_info['pattern_type'] = 'moderate'
    else:
        missing_info['pattern_type'] = 'extensive'
    
    return missing_info


def detect_categorical_columns(X: Union[np.ndarray, pd.DataFrame], 
                              categorical_threshold: int = 20) -> List[str]:
    """Detect categorical columns in the data."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    categorical_columns = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical_columns.append(col)
        elif X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # Check if numeric column might be categorical
            unique_values = X[col].nunique()
            if unique_values <= categorical_threshold and unique_values > 1:
                categorical_columns.append(col)
    
    return categorical_columns


def detect_high_cardinality_columns(X: Union[np.ndarray, pd.DataFrame], 
                                   threshold: int = 50) -> List[str]:
    """Detect high cardinality categorical columns."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    categorical_cols = detect_categorical_columns(X)
    high_cardinality_cols = []
    
    for col in categorical_cols:
        if X[col].nunique() > threshold:
            high_cardinality_cols.append(col)
    
    return high_cardinality_cols


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def validate_input_data(X: Union[np.ndarray, pd.DataFrame], 
                       y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Validate and convert input data to consistent format."""
    # Convert X to DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be numpy array or pandas DataFrame, got {type(X)}")
    
    # Convert y to Series if provided
    if y is not None:
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        elif not isinstance(y, pd.Series):
            raise ValueError(f"y must be numpy array or pandas Series, got {type(y)}")
        
        # Check if y is binary
        if not check_binary_target(y):
            raise ValueError("Target variable must be binary (exactly 2 unique values)")
        
        # Ensure X and y have same length
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples. X: {len(X)}, y: {len(y)}")
    
    return X, y


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Save model to file using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Union[str, Path]) -> Any:
    """Load model from file using joblib."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    return joblib.load(filepath)


def calculate_feature_importance_consensus(importance_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate consensus feature importance from multiple methods."""
    methods = list(importance_dict.keys())
    n_features = len(importance_dict[methods[0]])
    
    # Normalize each importance array to [0, 1]
    normalized_importances = {}
    for method, importance in importance_dict.items():
        importance = np.abs(importance)  # Take absolute values
        if importance.max() > 0:
            normalized_importances[method] = importance / importance.max()
        else:
            normalized_importances[method] = importance
    
    # Calculate average importance
    consensus_importance = np.zeros(n_features)
    for importance in normalized_importances.values():
        consensus_importance += importance
    
    consensus_importance /= len(methods)
    return consensus_importance


def format_number(number: float, decimals: int = 3) -> str:
    """Format number for display with appropriate precision."""
    if abs(number) >= 1000:
        return f"{number:.0f}"
    elif abs(number) >= 1:
        return f"{number:.{decimals}f}"
    else:
        return f"{number:.{decimals}f}"


def get_memory_usage(obj: Any) -> str:
    """Get memory usage of an object in human readable format."""
    try:
        import sys
        size_bytes = sys.getsizeof(obj)
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.1f} MB"
        else:
            return f"{size_bytes/1024**3:.1f} GB"
    except:
        return "Unknown"


def suppress_warnings():
    """Suppress common warnings that clutter output."""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


class Timer:
    """Simple timer context manager."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time