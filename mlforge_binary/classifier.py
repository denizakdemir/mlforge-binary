"""
Main BinaryClassifier class for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
from pathlib import Path
import warnings
import joblib

from .preprocessing import AutoPreprocessor
from .models import ModelWrapper, AutoModelSelector, EnsembleClassifier, get_default_params
from .evaluation import ComprehensiveEvaluator, MetricsCalculator, VisualizationGenerator
from .explainer import ModelExplainer
from .utils import validate_input_data, save_model, load_model, Timer, suppress_warnings


class BinaryClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible binary classifier with production features."""
    
    def __init__(self,
                 model: str = "auto",
                 models_to_try: Optional[List[str]] = None,
                 ensemble_models: Optional[List[str]] = None,
                 ensemble_method: str = "stacking",
                 meta_learner: str = "logistic",
                 handle_missing: str = "auto",
                 handle_categorical: str = "auto",
                 handle_scaling: str = "auto",
                 handle_imbalanced: str = "auto",
                 optimize_threshold: bool = True,
                 calibrate: bool = True,
                 feature_selection: bool = False,
                 feature_engineering: bool = False,
                 explain: bool = True,
                 tune_hyperparameters: bool = False,
                 tuning_budget: int = 100,
                 cv_folds: int = 5,
                 scoring: str = 'roc_auc',
                 random_state: int = 42,
                 verbose: bool = True):
        
        # Model configuration
        self.model = model
        self.models_to_try = models_to_try
        self.ensemble_models = ensemble_models
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner
        
        # Preprocessing configuration
        self.handle_missing = handle_missing
        self.handle_categorical = handle_categorical
        self.handle_scaling = handle_scaling
        self.handle_imbalanced = handle_imbalanced
        
        # Model optimization
        self.optimize_threshold = optimize_threshold
        self.calibrate = calibrate
        self.feature_selection = feature_selection
        self.feature_engineering = feature_engineering
        self.explain_enabled = explain
        self.tune_hyperparameters = tune_hyperparameters
        self.tuning_budget = tuning_budget
        
        # Evaluation configuration
        self.cv_folds = cv_folds
        self.scoring = scoring
        
        # General configuration
        self.random_state = random_state
        self.verbose = verbose
        
        # Internal state
        self.preprocessor_ = None
        self.model_ = None
        self.is_fitted_ = False
        self.optimal_threshold_ = 0.5
        self.feature_names_in_ = None
        self.classes_ = None
        self.training_metrics_ = None
        self.explainer_ = None
        
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[MLForge] {message}")
    
    def _create_preprocessor(self) -> AutoPreprocessor:
        """Create preprocessing pipeline."""
        return AutoPreprocessor(
            handle_missing=self.handle_missing,
            handle_categorical=self.handle_categorical,
            handle_scaling=self.handle_scaling,
            handle_imbalanced=self.handle_imbalanced,
            feature_selection=self.feature_selection,
            feature_engineering=self.feature_engineering,
            random_state=self.random_state
        )
    
    def _select_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Select best model based on configuration."""
        if self.model == "auto":
            self._log("Auto-selecting best model...")
            
            selector = AutoModelSelector(
                models_to_try=self.models_to_try,
                cv_folds=self.cv_folds,
                scoring=self.scoring,
                random_state=self.random_state
            )
            
            best_model_type = selector.select_best_model(X, y)
            self._log(f"Selected model: {best_model_type}")
            
            # Store model selection results
            self.model_selection_results_ = selector.model_scores_
            
            # Create best model with default parameters
            params = get_default_params(best_model_type, self.random_state)
            return ModelWrapper(best_model_type, **params)
            
        elif self.model == "ensemble":
            self._log("Creating ensemble model...")
            
            models = self.ensemble_models or ['logistic', 'random_forest']
            return EnsembleClassifier(
                models=models,
                ensemble_method=self.ensemble_method,
                meta_learner=self.meta_learner,
                random_state=self.random_state
            )
            
        else:
            # Use specific model
            self._log(f"Using {self.model} model...")
            params = get_default_params(self.model, self.random_state)
            return ModelWrapper(self.model, **params)
    
    def _optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal classification threshold."""
        self._log("Optimizing classification threshold...")
        
        # Get probabilities on training data
        y_proba = self.model_.predict_proba(X)[:, 1]
        
        # Find threshold that maximizes F1 score
        from sklearn.metrics import f1_score
        
        thresholds = np.linspace(0.1, 0.9, 17)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1 = f1_score(y, y_pred_thresh)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self._log(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series]):
        """Fit the binary classifier."""
        
        if not self.verbose:
            suppress_warnings()
        
        with Timer() as timer:
            self._log("Starting model training...")
            
            # Validate and convert input
            X, y = validate_input_data(X, y)
            self.feature_names_in_ = X.columns.tolist()
            self.classes_ = np.unique(y)
            
            # Create and fit preprocessor
            self._log("Fitting preprocessing pipeline...")
            self.preprocessor_ = self._create_preprocessor()
            
            # Fit and transform data
            if self.handle_imbalanced != "none":
                result = self.preprocessor_.fit_transform(X, y)
                if isinstance(result, tuple):
                    X_processed, y_processed = result
                else:
                    X_processed = result
                    y_processed = y.values
            else:
                X_processed = self.preprocessor_.fit_transform(X, y)
                y_processed = y.values
            
            self._log(f"Processed data shape: {X_processed.shape}")
            
            # Select and create model
            self.model_ = self._select_model(X_processed, y_processed)
            
            # Fit model
            self._log("Training model...")
            self.model_.fit(X_processed, y_processed)
            
            # Calibrate probabilities if requested
            if self.calibrate:
                self._log("Calibrating probabilities...")
                # If using ModelWrapper, calibrate the underlying model
                if hasattr(self.model_, 'model_') and hasattr(self.model_, 'model_type'):
                    base_model = self.model_.model_
                else:
                    base_model = self.model_
                
                self.model_ = CalibratedClassifierCV(
                    base_model, 
                    method='isotonic', 
                    cv=3
                )
                self.model_.fit(X_processed, y_processed)
            
            # Set fitted flag before calculating metrics and optimizing threshold
            self.is_fitted_ = True
            
            # Optimize threshold if requested
            if self.optimize_threshold:
                self.optimal_threshold_ = self._optimize_threshold(X_processed, y_processed)
            
            # Calculate training metrics
            self._log("Calculating training metrics...")
            self._calculate_training_metrics(X_processed, y_processed)
            
            # Initialize explainer if explain is enabled
            if self.explain_enabled:
                self._log("Initializing model explainer...")
                # Get feature names after preprocessing if available
                try:
                    feature_names = self.get_feature_names_out()
                except Exception:
                    feature_names = None
                    
                try:
                    self.explainer_ = ModelExplainer(
                        model=self.model_, 
                        X_train=X_processed,
                        feature_names=feature_names,
                        verbose=self.verbose
                    )
                except Exception:
                    self.explainer_ = None
            
        self._log(f"Training completed in {timer.elapsed_time:.2f} seconds")
        return self
    
    def _calculate_training_metrics(self, X_processed: np.ndarray, y: np.ndarray) -> None:
        """Calculate metrics on training data using processed features."""
        try:
            # Check if model has predict_proba method
            if hasattr(self.model_, 'predict_proba'):
                y_proba = self.model_.predict_proba(X_processed)
                if y_proba.shape[1] == 2:
                    y_pred = (y_proba[:, 1] >= self.optimal_threshold_).astype(int)
                else:
                    # For multi-class, just take argmax
                    y_pred = np.argmax(y_proba, axis=1)
            else:
                # Fallback to predict method
                y_pred = self.model_.predict(X_processed)
                y_proba = None
            
            from .evaluation import MetricsCalculator
            metrics_calc = MetricsCalculator()
            self.training_metrics_ = metrics_calc.calculate_all_metrics(
                y, y_pred, y_proba
            )
        except Exception as e:
            self._log(f"Warning: Could not calculate training metrics: {e}")
            self.training_metrics_ = {}
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get probabilities and apply threshold
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= self.optimal_threshold_).astype(int)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Validate and convert input
        X, _ = validate_input_data(X)
        
        # Preprocess
        X_processed = self.preprocessor_.transform(X)
        
        # Predict
        return self.model_.predict_proba(X_processed)
    
    def evaluate(self, 
                X_test: Union[np.ndarray, pd.DataFrame],
                y_test: Union[np.ndarray, pd.Series],
                generate_report: bool = False,
                report_path: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Validate input
        X_test, y_test = validate_input_data(X_test, y_test)
        
        # Use comprehensive evaluator
        evaluator = ComprehensiveEvaluator()
        
        model_name = f"MLForge-{self.model}" if self.model != "auto" else "MLForge-Auto"
        
        return evaluator.evaluate(
            model=self,
            X_test=X_test,
            y_test=y_test.values,
            model_name=model_name,
            generate_report=generate_report,
            report_path=report_path
        )
    
    def explain(self, 
               X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
               top_features: int = 20) -> Dict[str, Any]:
        """Explain model predictions and feature importance.
        
        Args:
            X: Data to explain (for SHAP and LIME explanations). If None, only global
               explanations based on the model are provided.
            top_features: Number of top features to include in feature importance.
            
        Returns:
            Dictionary containing various explanations.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Initialize explanations dictionary with model metadata
        explanations = {
            'model_type': self.model,
            'optimal_threshold': self.optimal_threshold_,
        }
        
        if hasattr(self, 'training_metrics_') and self.training_metrics_:
            explanations['training_performance'] = self.training_metrics_
        
        # If explainer is not initialized but explain is enabled, try to create it
        if not hasattr(self, 'explainer_') or self.explainer_ is None:
            if hasattr(self, 'explain_enabled') and self.explain_enabled:
                self._log("Model explainer not initialized. Trying to create one.")
                try:
                    # Get processed data for background
                    X_sample = X if X is not None else pd.DataFrame({col: [0] for col in self.feature_names_in_})
                    X_processed = self.preprocessor_.transform(X_sample)
                    
                    # Get feature names
                    try:
                        feature_names = self.get_feature_names_out()
                    except:
                        feature_names = None
                    
                    # Create explainer
                    from .explainer import ModelExplainer
                    self.explainer_ = ModelExplainer(
                        model=self.model_,
                        X_train=X_processed,
                        feature_names=feature_names,
                        verbose=self.verbose
                    )
                except Exception as e:
                    self._log(f"Failed to create explainer: {e}")
                    self.explainer_ = None
            
            # If explainer still not available, use basic feature importance
            if not hasattr(self, 'explainer_') or self.explainer_ is None:
                self._log("Using basic feature importance extraction.")
                self._extract_basic_feature_importance(explanations, top_features)
                return explanations
        
        # Use the ModelExplainer for advanced explanations
        try:
            # Get global explanations (feature importance, SHAP values, etc.)
            if X is not None:
                # Preprocess the data if needed
                X_processed = self.preprocessor_.transform(X)
                
                # Get global explanations
                global_explanations = self.explainer_.explain_global(X_processed)
                explanations.update(global_explanations)
                
                # If we have SHAP values, extract feature importance from them
                if 'shap_values' in global_explanations:
                    self._extract_shap_feature_importance(explanations, global_explanations, top_features)
            
            # If no global explanations or SHAP feature importance, fall back to basic extraction
            if 'feature_importance' not in explanations:
                self._extract_basic_feature_importance(explanations, top_features)
                
        except Exception as e:
            self._log(f"Warning: Could not generate explanations with ModelExplainer: {e}")
            # Fall back to basic feature importance
            self._extract_basic_feature_importance(explanations, top_features)
        
        return explanations
        
    def _extract_basic_feature_importance(self, explanations: Dict[str, Any], top_features: int = 20) -> None:
        """Extract basic feature importance directly from the model."""
        try:
            # Feature importance (if available)
            if hasattr(self.model_, 'feature_importances_'):
                feature_names = self.get_feature_names_out()
                importance_values = self.model_.feature_importances_
                
                # Create feature importance dictionary
                feature_importance = dict(zip(feature_names, importance_values))
                
                # Sort by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                explanations['feature_importance'] = dict(sorted_features[:top_features])
                
            elif hasattr(self.model_, 'coef_'):
                # Linear model coefficients
                feature_names = self.get_feature_names_out()
                coef_values = self.model_.coef_[0] if self.model_.coef_.ndim > 1 else self.model_.coef_
                
                feature_importance = dict(zip(feature_names, np.abs(coef_values)))
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                explanations['feature_importance'] = dict(sorted_features[:top_features])
        except Exception as e:
            self._log(f"Warning: Could not extract basic feature importance: {e}")
            
    def _extract_shap_feature_importance(self, explanations: Dict[str, Any], 
                                        global_explanations: Dict[str, Any],
                                        top_features: int = 20) -> None:
        """Extract feature importance from SHAP values."""
        try:
            if 'shap_feature_importance' in global_explanations:
                feature_names = self.get_feature_names_out()
                importance_values = global_explanations['shap_feature_importance']
                
                # Create feature importance dictionary
                feature_importance = dict(zip(feature_names, importance_values))
                
                # Sort by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                explanations['feature_importance'] = dict(sorted_features[:top_features])
        except Exception as e:
            self._log(f"Warning: Could not extract SHAP feature importance: {e}")
    
    def explain_instance(self, instance: Union[np.ndarray, pd.DataFrame, pd.Series], 
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            instance: Single instance to explain (as array, DataFrame row, or Series)
            feature_names: Feature names for the instance (used for LIME)
            
        Returns:
            Dictionary containing explanations for the instance.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Convert instance to appropriate format
        if isinstance(instance, pd.DataFrame):
            if len(instance) != 1:
                raise ValueError("Instance must be a single example (one row)")
            instance_array = instance.values[0]
        elif isinstance(instance, pd.Series):
            instance_array = instance.values
        else:
            instance_array = instance
            
        # Preprocess the instance if needed
        if hasattr(self, 'preprocessor_') and self.preprocessor_ is not None:
            # Create a DataFrame with one row for preprocessing
            if isinstance(instance, (pd.DataFrame, pd.Series)):
                X_instance = instance.to_frame().T if isinstance(instance, pd.Series) else instance
            else:
                X_instance = pd.DataFrame([instance], columns=self.feature_names_in_)
                
            # Preprocess
            instance_processed = self.preprocessor_.transform(X_instance)
            
            # Get the processed instance as array
            if isinstance(instance_processed, np.ndarray):
                instance_array = instance_processed[0]
            else:
                # Handle other return types if necessary
                instance_array = instance_processed.values[0] if hasattr(instance_processed, 'values') else instance_processed[0]
        
        # Use explainer to explain the instance
        explanations = {}
        
        if self.explainer_ is not None:
            # Get feature names for processed data if available
            if feature_names is None and hasattr(self, 'get_feature_names_out'):
                try:
                    feature_names = self.get_feature_names_out()
                except:
                    pass
                    
            # Get local explanations
            local_explanations = self.explainer_.explain_instance(instance_array, feature_names)
            explanations.update(local_explanations)
            
        return explanations
    
    def explain_instance(self, instance: Union[np.ndarray, pd.DataFrame, pd.Series], 
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            instance: Single instance to explain (as array, DataFrame row, or Series)
            feature_names: Feature names for the instance (used for LIME)
            
        Returns:
            Dictionary containing explanations for the instance.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Convert instance to appropriate format
        if isinstance(instance, pd.DataFrame):
            if len(instance) != 1:
                raise ValueError("Instance must be a single example (one row)")
            instance_array = instance.values[0]
        elif isinstance(instance, pd.Series):
            instance_array = instance.values
        else:
            instance_array = instance
            
        # Preprocess the instance if needed
        if hasattr(self, 'preprocessor_') and self.preprocessor_ is not None:
            # Create a DataFrame with one row for preprocessing
            if isinstance(instance, (pd.DataFrame, pd.Series)):
                X_instance = instance.to_frame().T if isinstance(instance, pd.Series) else instance
            else:
                X_instance = pd.DataFrame([instance], columns=self.feature_names_in_)
                
            # Preprocess
            instance_processed = self.preprocessor_.transform(X_instance)
            
            # Get the processed instance as array
            if isinstance(instance_processed, np.ndarray):
                instance_array = instance_processed[0]
            else:
                # Handle other return types if necessary
                instance_array = instance_processed.values[0] if hasattr(instance_processed, 'values') else instance_processed[0]
        
        # Use explainer to explain the instance
        explanations = {}
        
        # If explainer doesn't exist, try to create one
        if not hasattr(self, 'explainer_') or self.explainer_ is None:
            # Try to initialize explainer
            self._log("Explainer not initialized, trying to create one for instance explanation")
            try:
                # Create a sample for background data
                X_sample = pd.DataFrame({col: [0] for col in self.feature_names_in_})
                X_processed = self.preprocessor_.transform(X_sample)
                
                # Get feature names
                try:
                    feature_names_out = self.get_feature_names_out()
                except:
                    feature_names_out = None
                
                # Create explainer
                from .explainer import ModelExplainer
                self.explainer_ = ModelExplainer(
                    model=self.model_,
                    X_train=X_processed,
                    feature_names=feature_names_out,
                    verbose=self.verbose
                )
            except Exception as e:
                self._log(f"Failed to create explainer: {e}")
                self.explainer_ = None
        
        # Get direct prediction results
        try:
            # Create a DataFrame with one row
            if isinstance(instance, (pd.DataFrame, pd.Series)):
                X_instance = instance.to_frame().T if isinstance(instance, pd.Series) else instance
            else:
                X_instance = pd.DataFrame([instance], columns=self.feature_names_in_)
                
            # Get prediction
            prediction = self.predict(X_instance)[0]
            prediction_proba = self.predict_proba(X_instance)[0]
            
            explanations['prediction'] = prediction
            explanations['prediction_proba'] = prediction_proba
        except Exception as e:
            self._log(f"Failed to get prediction: {e}")
        
        # Use explainer for detailed explanations if available
        if hasattr(self, 'explainer_') and self.explainer_ is not None:
            # Get feature names for processed data if available
            if feature_names is None and hasattr(self, 'get_feature_names_out'):
                try:
                    feature_names = self.get_feature_names_out()
                except:
                    pass
                    
            # Get local explanations
            try:
                local_explanations = self.explainer_.explain_instance(instance_array, feature_names)
                explanations.update(local_explanations)
            except Exception as e:
                self._log(f"Failed to get explanations from explainer: {e}")
            
        return explanations
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after preprocessing."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            return self.preprocessor_.get_feature_names_out()
        except:
            # Fallback
            dummy_df = pd.DataFrame({col: [0] for col in self.feature_names_in_})
            n_features = self.preprocessor_.transform(dummy_df).shape[1]
            return [f'feature_{i}' for i in range(n_features)]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the fitted model to file."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        save_model(self, filepath)
        self._log(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BinaryClassifier':
        """Load a fitted model from file."""
        return load_model(filepath)
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'BinaryClassifier':
        """Create classifier from YAML configuration file."""
        from .utils import load_config
        
        config = load_config(config_path)
        
        # Extract model configuration
        model_config = config.get('model', {})
        preprocessing_config = config.get('preprocessing', {})
        training_config = config.get('training', {})
        evaluation_config = config.get('evaluation', {})
        
        # Merge all configurations
        all_config = {}
        all_config.update(model_config)
        all_config.update(preprocessing_config)
        all_config.update(training_config)
        all_config.update(evaluation_config)
        
        return cls(**all_config)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {
            'model': self.model,
            'models_to_try': self.models_to_try,
            'ensemble_models': self.ensemble_models,
            'ensemble_method': self.ensemble_method,
            'meta_learner': self.meta_learner,
            'handle_missing': self.handle_missing,
            'handle_categorical': self.handle_categorical,
            'handle_scaling': self.handle_scaling,
            'handle_imbalanced': self.handle_imbalanced,
            'optimize_threshold': self.optimize_threshold,
            'calibrate': self.calibrate,
            'feature_selection': self.feature_selection,
            'feature_engineering': self.feature_engineering,
            'explain': self.explain_enabled,
            'tune_hyperparameters': self.tune_hyperparameters,
            'tuning_budget': self.tuning_budget,
            'cv_folds': self.cv_folds,
            'scoring': self.scoring,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        return params
    
    def set_params(self, **params) -> 'BinaryClassifier':
        """Set the parameters of this estimator."""
        valid_params = self.get_params()
        
        for key, value in params.items():
            if key in valid_params:
                if key == 'explain':
                    # Handle explain parameter specially
                    self.explain_enabled = value
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        
        return self
    
    def __sklearn_tags__(self):
        """Return sklearn tags."""
        try:
            # Try to use new sklearn tags system
            from sklearn.utils._tags import Tags
            # Tags constructor requires estimator_type and target_tags
            tags = Tags(
                estimator_type="classifier",
                target_tags={"two_d_array": True}
            )
            return tags
        except Exception:
            # Fallback to mock tags for older sklearn versions
            class MockInputTags:
                def __init__(self):
                    self.two_d_array = True
                    self.sparse = False
                    self.categorical = False
                    self.string = False
                    self.dict = False
                    self.positive_only = False
                    
            class MockTags:
                def __init__(self):
                    self.estimator_type = "classifier"
                    self.requires_fit = True
                    self.requires_y = True
                    self.requires_positive_X = False
                    self.requires_X = True
                    self.allow_nan = False
                    self.poor_score = False
                    self.no_validation = False
                    self.multiclass_only = False
                    self.binary_only = False
                    self.multilabel = False
                    self.multioutput = False
                    self.multioutput_only = False
                    self.stateless = False
                    self.pairwise = False
                    self.input_tags = MockInputTags()
                    
            return MockTags()

    def __repr__(self) -> str:
        """String representation of the classifier."""
        params = self.get_params()
        param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:5]])
        return f"BinaryClassifier({param_str}, ...)"