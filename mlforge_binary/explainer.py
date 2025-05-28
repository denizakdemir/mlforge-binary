"""
Model explanation functionality for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


class ModelExplainer:
    """Model explanation using SHAP and LIME."""
    
    def __init__(self, model, X_train: Optional[np.ndarray] = None):
        self.model = model
        self.X_train = X_train
        self.shap_explainer = None
        self.lime_explainer = None
        
    def explain_global(self, X: np.ndarray, max_display: int = 20) -> Dict[str, Any]:
        """Generate global explanations."""
        explanations = {}
        
        if HAS_SHAP:
            try:
                # Create SHAP explainer
                if self.shap_explainer is None:
                    self.shap_explainer = shap.Explainer(self.model, self.X_train)
                
                # Get SHAP values
                shap_values = self.shap_explainer(X)
                
                # Global feature importance
                feature_importance = np.abs(shap_values.values).mean(0)
                
                explanations['shap_feature_importance'] = feature_importance
                explanations['shap_values'] = shap_values
                
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        return explanations
    
    def explain_instance(self, instance: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Explain a single prediction."""
        explanations = {}
        
        if HAS_LIME and self.X_train is not None:
            try:
                # Create LIME explainer
                if self.lime_explainer is None:
                    self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        self.X_train,
                        mode='classification',
                        feature_names=feature_names
                    )
                
                # Get LIME explanation
                lime_explanation = self.lime_explainer.explain_instance(
                    instance, self.model.predict_proba
                )
                
                explanations['lime_explanation'] = lime_explanation
                
            except Exception as e:
                print(f"LIME explanation failed: {e}")
        
        return explanations
