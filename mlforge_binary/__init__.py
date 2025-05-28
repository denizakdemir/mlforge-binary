"""
MLForge-Binary: Simple ML Pipeline for Binary Classification

A scikit-learn style library for binary classification that handles common 
real-world challenges automatically. Tunes the entire pipeline (preprocessing + model) 
as one cohesive system.
"""

# Suppress common warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')
warnings.filterwarnings('ignore', message='.*but LGBMClassifier was fitted with feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

__version__ = "0.1.0"
__author__ = "MLForge Team"

from .classifier import BinaryClassifier
from .automl import AutoML, compare_models, quick_experiment

__all__ = [
    "BinaryClassifier",
    "AutoML", 
    "compare_models",
    "quick_experiment"
]