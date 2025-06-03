"""
Model explanation functionality for MLForge-Binary

This module provides SHAP and LIME-based explanations for models
trained with the BinaryClassifier. It includes both global feature
importance explanations and local instance-level explanations.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

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
    """Model explanation using SHAP and LIME.
    
    This class provides methods for explaining model predictions using
    SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable
    Model-agnostic Explanations). It supports both global feature importance
    and individual prediction explanations.
    
    Attributes:
        model: The trained model to explain (must have predict_proba method)
        X_train: Training data used for initializing explainers
        feature_names: Names of the features (optional)
        categorical_features: Indices of categorical features (optional)
        shap_explainer: Initialized SHAP explainer
        lime_explainer: Initialized LIME explainer
        categorical_names: Dictionary mapping categorical feature indices to their values
    """
    
    def __init__(self, 
                model, 
                X_train: Optional[np.ndarray] = None,
                feature_names: Optional[List[str]] = None,
                categorical_features: Optional[List[int]] = None,
                categorical_names: Optional[Dict[int, List[str]]] = None,
                class_names: Optional[List[str]] = None,
                verbose: bool = False):
        """Initialize the model explainer.
        
        Args:
            model: The trained model to explain
            X_train: Training data used for initializing explainers
            feature_names: Names of the features
            categorical_features: Indices of categorical features
            categorical_names: Dictionary mapping categorical feature indices to their values
            class_names: Names of the target classes
            verbose: Whether to print verbose output
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.verbose = verbose
        
        # Initialize explainers to None
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Check if model has predict_proba method
        if not hasattr(self.model, 'predict_proba'):
            warnings.warn("Model does not have predict_proba method, explanations may be limited")
    
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ModelExplainer] {message}")
            
    def _ensure_2d(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Ensure X is a 2D array."""
        if isinstance(X, pd.DataFrame):
            return X.values
            
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            return X_array.reshape(1, -1)
        return X_array
        
    def _create_shap_explainer(self) -> None:
        """Create SHAP explainer based on model type."""
        if not HAS_SHAP:
            self._log("SHAP not available. Install with: pip install shap")
            return
            
        try:
            self._log("Creating SHAP explainer...")
            
            # Create prediction function wrapper
            def predict_proba_wrapper(X):
                return self.model.predict_proba(X)
                
            # Create SHAP explainer
            # For efficiency, use a subset of training data if it's large
            X_background = self.X_train
            if X_background is not None and len(X_background) > 100:
                # Use random sample of 100 instances as background
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(len(X_background), 100, replace=False)
                X_background = X_background[indices]
                
            self.shap_explainer = shap.Explainer(predict_proba_wrapper, X_background)
            self._log("SHAP explainer created successfully")
            
        except Exception as e:
            self._log(f"Failed to create SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _create_lime_explainer(self) -> None:
        """Create LIME explainer for tabular data."""
        if not HAS_LIME:
            self._log("LIME not available. Install with: pip install lime")
            return
            
        if self.X_train is None:
            self._log("Training data required for LIME explainer")
            return
            
        try:
            self._log("Creating LIME explainer...")
            
            # Create LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                categorical_features=self.categorical_features,
                categorical_names=self.categorical_names,
                class_names=self.class_names,
                mode='classification'
            )
            
            self._log("LIME explainer created successfully")
            
        except Exception as e:
            self._log(f"Failed to create LIME explainer: {e}")
            self.lime_explainer = None
    
    def explain_global(self, X: np.ndarray, max_display: int = 20) -> Dict[str, Any]:
        """Generate global explanations.
        
        Args:
            X: Data to explain
            max_display: Maximum number of features to include in explanations
            
        Returns:
            Dictionary containing various explanations
        """
        explanations = {}
        
        # Ensure X is 2D
        X = self._ensure_2d(X)
        
        # Extract basic feature importance if available
        self._extract_model_feature_importance(explanations)
        
        # Add SHAP explanations if available
        self._add_shap_explanations(X, explanations, max_display)
        
        return explanations
    
    def _extract_model_feature_importance(self, explanations: Dict[str, Any]) -> None:
        """Extract feature importance directly from the model if available."""
        try:
            # For tree-based models
            if hasattr(self.model, 'feature_importances_'):
                explanations['model_feature_importance'] = self.model.feature_importances_
                
            # For linear models
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_
                if coef.ndim > 1:
                    # For multi-class, take the first class or average
                    coef = coef[0]
                explanations['model_feature_importance'] = np.abs(coef)
                
        except Exception as e:
            self._log(f"Could not extract model feature importance: {e}")
    
    def _add_shap_explanations(self, X: np.ndarray, explanations: Dict[str, Any], max_display: int = 20) -> None:
        """Add SHAP-based explanations."""
        if not HAS_SHAP:
            return
            
        try:
            # Create SHAP explainer if not already created
            if self.shap_explainer is None:
                self._create_shap_explainer()
                
            if self.shap_explainer is not None:
                # Get SHAP values
                self._log("Calculating SHAP values...")
                
                # Limit to a reasonable number of samples for efficiency
                X_sample = X[:min(50, len(X))]
                shap_values = self.shap_explainer(X_sample)
                
                # Extract SHAP values based on format returned
                if hasattr(shap_values, 'values'):
                    # For newer SHAP versions
                    shap_array = shap_values.values
                else:
                    # For older versions
                    shap_array = shap_values
                
                # Handle different output shapes
                if shap_array.ndim == 3:
                    # Binary classification [samples, features, classes]
                    # Use positive class (index 1)
                    shap_array_to_use = shap_array[:, :, 1]
                elif shap_array.ndim == 2:
                    # Already for single class [samples, features]
                    shap_array_to_use = shap_array
                else:
                    self._log(f"Unexpected SHAP values shape: {shap_array.shape}")
                    return
                
                # Calculate global feature importance from SHAP values
                feature_importance = np.abs(shap_array_to_use).mean(0)
                
                explanations['shap_feature_importance'] = feature_importance
                explanations['shap_values'] = shap_values
                
        except Exception as e:
            self._log(f"SHAP explanation failed: {e}")
    
    def explain_instance(self, 
                        instance: np.ndarray, 
                        feature_names: List[str] = None,
                        num_features: int = 10) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            instance: Instance to explain
            feature_names: Names of features (if not provided during initialization)
            num_features: Number of features to include in LIME explanation
            
        Returns:
            Dictionary containing explanations for the instance
        """
        explanations = {}
        
        # Get prediction for the instance
        instance_2d = instance.reshape(1, -1) if instance.ndim == 1 else instance
        try:
            prediction = self.model.predict(instance_2d)[0]
            prediction_proba = self.model.predict_proba(instance_2d)[0]
            
            explanations['prediction'] = prediction
            explanations['prediction_proba'] = prediction_proba
        except Exception as e:
            self._log(f"Could not get prediction: {e}")
        
        # Add LIME explanation if available
        self._add_lime_explanation(instance, explanations, feature_names, num_features)
        
        # Add SHAP explanation for this instance if explainer is available
        self._add_instance_shap_explanation(instance, explanations)
        
        return explanations
    
    def _add_lime_explanation(self, 
                             instance: np.ndarray, 
                             explanations: Dict[str, Any],
                             feature_names: List[str] = None,
                             num_features: int = 10) -> None:
        """Add LIME explanation for an instance."""
        if not HAS_LIME or self.X_train is None:
            return
            
        try:
            # Create LIME explainer if not already created
            if self.lime_explainer is None:
                # Use feature names from parameters or instance
                if feature_names is not None:
                    self.feature_names = feature_names
                self._create_lime_explainer()
                
            if self.lime_explainer is not None:
                self._log("Generating LIME explanation...")
                
                # Create wrapper for predict_proba
                def predict_proba_wrapper(X):
                    return self.model.predict_proba(X)
                
                # Get LIME explanation
                lime_exp = self.lime_explainer.explain_instance(
                    instance, 
                    predict_proba_wrapper,
                    num_features=num_features
                )
                
                # Extract explanation as list and add to results
                explanations['lime_explanation'] = lime_exp
                explanations['lime_features'] = lime_exp.as_list()
                
        except Exception as e:
            self._log(f"LIME explanation failed: {e}")
    
    def _add_instance_shap_explanation(self, instance: np.ndarray, explanations: Dict[str, Any]) -> None:
        """Add SHAP explanation for a single instance."""
        if not HAS_SHAP or self.shap_explainer is None:
            return
            
        try:
            # Reshape instance to 2D if needed
            instance_2d = instance.reshape(1, -1) if instance.ndim == 1 else instance
            
            # Get SHAP values for this instance
            self._log("Calculating SHAP values for instance...")
            shap_values = self.shap_explainer(instance_2d)
            
            # Extract actual SHAP values based on format
            if hasattr(shap_values, 'values'):
                shap_array = shap_values.values
            else:
                shap_array = shap_values
                
            # Handle different shapes
            if shap_array.ndim == 3:
                # For binary classification [1, features, classes]
                instance_values = shap_array[0, :, 1]  # Positive class
            elif shap_array.ndim == 2:
                # Already for single instance [1, features]
                instance_values = shap_array[0]
            else:
                self._log(f"Unexpected SHAP values shape for instance: {shap_array.shape}")
                return
                
            # Add to explanations
            explanations['shap_values_instance'] = instance_values
            
            # If feature names available, create feature-value pairs
            if self.feature_names is not None:
                shap_features = list(zip(self.feature_names, instance_values))
                # Sort by absolute contribution
                shap_features_sorted = sorted(shap_features, key=lambda x: abs(x[1]), reverse=True)
                explanations['shap_features'] = shap_features_sorted
                
        except Exception as e:
            self._log(f"SHAP instance explanation failed: {e}")

    def create_waterfall_plot(self, 
                             instance: np.ndarray, 
                             feature_names: List[str] = None,
                             feature_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create data for a waterfall plot explaining a single prediction.
        
        Args:
            instance: Instance to explain
            feature_names: Names of the features
            feature_values: Original feature values (for display)
            
        Returns:
            Dictionary with waterfall plot data
        """
        if not HAS_SHAP:
            self._log("SHAP not available. Install with: pip install shap")
            return {}
            
        # Create SHAP explainer if not already created
        if self.shap_explainer is None:
            self._create_shap_explainer()
            
        if self.shap_explainer is None:
            self._log("SHAP explainer could not be created")
            return {}
            
        try:
            # Reshape instance to 2D if needed
            instance_2d = instance.reshape(1, -1) if instance.ndim == 1 else instance
            
            # Get prediction
            prediction = self.model.predict(instance_2d)[0]
            prediction_proba = self.model.predict_proba(instance_2d)[0]
            
            # Get SHAP values
            shap_values = self.shap_explainer(instance_2d)
            
            # Extract SHAP values based on format
            if hasattr(shap_values, 'values'):
                shap_array = shap_values.values
            else:
                shap_array = shap_values
                
            # Handle different shapes
            if shap_array.ndim == 3:
                # For binary classification [1, features, classes]
                instance_values = shap_array[0, :, 1]  # Positive class
            elif shap_array.ndim == 2:
                # Already for single instance [1, features]
                instance_values = shap_array[0]
            else:
                self._log(f"Unexpected SHAP values shape for waterfall: {shap_array.shape}")
                return {}
            
            # Get base value (expected value)
            if hasattr(shap_values, 'base_values'):
                base_value = shap_values.base_values
                # Handle different formats
                if isinstance(base_value, np.ndarray):
                    if base_value.ndim > 1:
                        base_value = base_value[0, 1]  # For binary classification
                    else:
                        base_value = base_value[0]
            else:
                # Fallback: use average prediction from training data
                base_value = 0.5  # Default for binary classification
                
            # Use provided feature names or default
            names = feature_names or self.feature_names
            if names is None:
                names = [f"Feature {i}" for i in range(len(instance_values))]
                
            # Create feature contribution list
            features = []
            for i, (name, value) in enumerate(zip(names, instance_values)):
                feature_display_value = None
                if feature_values and name in feature_values:
                    feature_display_value = feature_values[name]
                    
                features.append({
                    'name': name,
                    'contribution': float(value),
                    'original_value': feature_display_value
                })
                
            # Sort by absolute contribution
            features_sorted = sorted(features, key=lambda x: abs(x['contribution']), reverse=True)
            
            # Create waterfall data
            waterfall_data = {
                'features': features_sorted,
                'base_value': float(base_value),
                'prediction': int(prediction),
                'prediction_proba': float(prediction_proba[1]) if prediction_proba.ndim > 0 else float(prediction_proba),
                'total_contribution': float(np.sum(instance_values))
            }
            
            return waterfall_data
            
        except Exception as e:
            self._log(f"Waterfall plot creation failed: {e}")
            return {}
