"""
Fairness and bias detection for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

try:
    import fairlearn.metrics as fl_metrics
    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False


class FairnessChecker:
    """Check model fairness and detect bias."""
    
    def __init__(self):
        pass
    
    def check_fairness(self, 
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sensitive_features: Dict[str, np.ndarray],
                      metrics: List[str] = None) -> Dict[str, Any]:
        """Check fairness across sensitive features."""
        
        if not HAS_FAIRLEARN:
            return {'error': 'fairlearn not available'}
        
        if metrics is None:
            metrics = ['demographic_parity', 'equalized_odds']
        
        results = {}
        
        for feature_name, feature_values in sensitive_features.items():
            feature_results = {}
            
            try:
                if 'demographic_parity' in metrics:
                    dp_diff = fl_metrics.demographic_parity_difference(
                        y_true, y_pred, sensitive_features=feature_values
                    )
                    feature_results['demographic_parity_difference'] = dp_diff
                
                if 'equalized_odds' in metrics:
                    eo_diff = fl_metrics.equalized_odds_difference(
                        y_true, y_pred, sensitive_features=feature_values
                    )
                    feature_results['equalized_odds_difference'] = eo_diff
                
            except Exception as e:
                feature_results['error'] = str(e)
            
            results[feature_name] = feature_results
        
        return results
