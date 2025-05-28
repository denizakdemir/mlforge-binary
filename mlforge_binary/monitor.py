"""
Model monitoring and drift detection for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats
from datetime import datetime


class DriftDetector:
    """Detect data drift in production."""
    
    def __init__(self, reference_data: np.ndarray):
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data)
    
    def _calculate_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for drift detection."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect drift using statistical tests."""
        drift_results = {
            'is_drifted': False,
            'p_values': [],
            'drifted_features': [],
            'timestamp': datetime.now()
        }
        
        for i in range(new_data.shape[1]):
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_feature, new_feature)
            
            drift_results['p_values'].append(p_value)
            
            if p_value < threshold:
                drift_results['is_drifted'] = True
                drift_results['drifted_features'].append(i)
        
        return drift_results


class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, model):
        self.model = model
        self.prediction_log = []
        self.drift_detector = None
    
    def log_prediction(self, X: np.ndarray, prediction: np.ndarray, 
                      actual: Optional[np.ndarray] = None):
        """Log prediction for monitoring."""
        log_entry = {
            'timestamp': datetime.now(),
            'features': X,
            'prediction': prediction,
            'actual': actual
        }
        
        self.prediction_log.append(log_entry)
    
    def set_reference_data(self, X_reference: np.ndarray):
        """Set reference data for drift detection."""
        self.drift_detector = DriftDetector(X_reference)
    
    def check_drift(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Check for data drift."""
        if self.drift_detector is None:
            return {'error': 'No reference data set'}
        
        return self.drift_detector.detect_drift(X_new)
