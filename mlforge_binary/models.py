"""
Model wrappers and hyperparameter spaces for MLForge-Binary
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')
warnings.filterwarnings('ignore', message='.*but LGBMClassifier was fitted with feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import warnings

# Optional imports for gradient boosting
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    import os
    import logging
    # Suppress LightGBM warnings globally
    os.environ['LIGHTGBM_VERBOSITY'] = '-1'
    # Disable LightGBM logging through Python logging
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """Base wrapper for all models with common interface."""
    
    _estimator_type = "classifier"
    
    def _more_tags(self):
        return {
            'binary_only': True,
            'requires_y': True,
            'requires_fit': True,
            '_xfail_checks': {
                'check_sample_weights_invariance': 'not implemented',
                'check_estimators_dtypes': 'not implemented'
            }
        }
    
    def __sklearn_tags__(self):
        # For compatibility with newer sklearn versions
        try:
            from sklearn.utils._tags import Tags
            # Create a basic Tags object for classifier
            tags = Tags()
            tags.estimator_type = "classifier"
            tags.requires_fit = True
            return tags
        except Exception as e:
            # Import the basic classifier mixin tags
            try:
                from sklearn.base import ClassifierMixin
                # Use parent class method if available
                return ClassifierMixin.__sklearn_tags__(self)
            except Exception:
                # Last resort fallback - create a mock tags object
                class MockTags:
                    def __init__(self):
                        self.estimator_type = "classifier"
                        self.requires_fit = True
                        self.binary_only = True
                        # Add missing attributes that sklearn might expect
                        self.input_tags = MockInputTags()
                        self.target_tags = MockTargetTags()
                        self.array_api_support = False
                        self.no_validation = False
                        self.non_deterministic = False
                        self._skip_test = False
                        self.stateless = False

                class MockInputTags:
                    def __init__(self):
                        self.pairwise = False
                        self.allow_nan = False
                        self.requires_positive_X = False
                        self.X_types = ["2darray"]

                class MockTargetTags:
                    def __init__(self):
                        self.required = True
                        self.y_types = ["1dlabels"]
                        self.multiclass_only = False
                        self.multilabel = False
                        self.multioutput_only = False
                        self.multioutput = False

                return MockTags()
    
    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.model_params = kwargs
        self.model_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y, **fit_params):
        """Fit the model."""
        self.model_ = self._create_model()
        
        # Handle special fitting requirements
        if self.model_type in ['xgboost', 'lightgbm', 'catboost']:
            result = self._prepare_gradient_boosting_params(X, y, fit_params)
            # For LightGBM, the fitting is handled in _prepare_gradient_boosting_params
            if self.model_type == 'lightgbm' and hasattr(self, 'is_fitted_') and self.is_fitted_:
                return result
            else:
                fit_params = result
        
        self.model_.fit(X, y, **fit_params)
        self.is_fitted_ = True
        
        # Set classes_ attribute required by sklearn
        if hasattr(self.model_, 'classes_'):
            self.classes_ = self.model_.classes_
        else:
            # Fallback for models that don't have classes_ attribute
            import numpy as np
            self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        self._check_fitted()
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        self._check_fitted()
        return self.model_.predict_proba(X)
    
    def decision_function(self, X):
        """Get confidence scores for the samples."""
        self._check_fitted()
        if hasattr(self.model_, 'decision_function'):
            return self.model_.decision_function(X)
        elif hasattr(self.model_, 'predict_proba'):
            # For models without decision_function, use log-odds
            proba = self.model_.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1] - proba[:, 0]
            else:
                return proba
        else:
            raise AttributeError(f"Model {self.model_type} does not support decision_function")
    
    def _check_fitted(self):
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
    
    def _create_model(self):
        """Create the actual model instance."""
        if self.model_type == 'logistic':
            return LogisticRegression(**self.model_params)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == 'extra_trees':
            return ExtraTreesClassifier(**self.model_params)
        elif self.model_type == 'svm':
            return SVC(probability=True, **self.model_params)
        elif self.model_type == 'knn':
            return KNeighborsClassifier(**self.model_params)
        elif self.model_type == 'naive_bayes':
            return GaussianNB(**self.model_params)
        elif self.model_type == 'neural_network':
            return MLPClassifier(**self.model_params)
        elif self.model_type == 'sgd':
            return SGDClassifier(**self.model_params)
        elif self.model_type == 'ridge':
            return RidgeClassifier(**self.model_params)
        elif self.model_type == 'xgboost' and HAS_XGB:
            return xgb.XGBClassifier(**self.model_params)
        elif self.model_type == 'lightgbm' and HAS_LGB:
            # Disable warnings for this specific instance
            import logging
            lgb_logger = logging.getLogger('lightgbm')
            lgb_logger.setLevel(logging.CRITICAL)
            return lgb.LGBMClassifier(**self.model_params)
        elif self.model_type == 'catboost' and HAS_CB:
            return cb.CatBoostClassifier(**self.model_params)
        else:
            raise ValueError(f"Unknown or unavailable model type: {self.model_type}")
    
    def _prepare_gradient_boosting_params(self, X, y, fit_params):
        """Prepare parameters for gradient boosting models."""
        if self.model_type == 'xgboost':
            # Add eval_set for early stopping
            if 'eval_set' not in fit_params and hasattr(self.model_, 'early_stopping_rounds'):
                fit_params['eval_set'] = [(X, y)]
                fit_params['verbose'] = False
        elif self.model_type == 'lightgbm':
            # Redirect stdout and stderr to suppress all LightGBM output
            import sys
            import os
            from contextlib import redirect_stdout, redirect_stderr
            
            # Store original stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # Redirect to devnull during fit
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    # Add eval_set for early stopping and suppress warnings
                    if 'eval_set' not in fit_params and hasattr(self.model_, 'early_stopping_rounds'):
                        fit_params['eval_set'] = [(X, y)]
                        fit_params['callbacks'] = [lgb.log_evaluation(0)]  # Suppress output
                    # Always add callback to suppress all LightGBM output
                    if 'callbacks' not in fit_params:
                        fit_params['callbacks'] = [lgb.log_evaluation(0)]
                    else:
                        if lgb.log_evaluation(0) not in fit_params['callbacks']:
                            fit_params['callbacks'].append(lgb.log_evaluation(0))
                    # Ensure feature names are consistent
                    if hasattr(X, 'columns'):
                        # If X is a DataFrame, ensure column names are strings
                        X.columns = [str(col) for col in X.columns]
                    
                    # Perform the actual fit with suppressed output
                    return_val = self.model_.fit(X, y, **fit_params)
            
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Skip the normal fit process since we already did it
            self.is_fitted_ = True
            
            # Set classes_ attribute required by sklearn
            if hasattr(self.model_, 'classes_'):
                self.classes_ = self.model_.classes_
            else:
                # Fallback for models that don't have classes_ attribute
                import numpy as np
                self.classes_ = np.unique(y)
            
            return self
        elif self.model_type == 'catboost':
            # CatBoost is verbose by default, suppress output
            if 'verbose' not in fit_params:
                fit_params['verbose'] = False
        
        return fit_params


class AutoModelSelector:
    """Automatically select best model from a set of candidates."""
    
    def __init__(self, 
                 models_to_try: Optional[List[str]] = None,
                 cv_folds: int = 5,
                 scoring: str = 'roc_auc',
                 random_state: int = 42):
        
        self.models_to_try = models_to_try or self._get_default_models()
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_model_type_ = None
        self.model_scores_ = {}
        
    def _get_default_models(self) -> List[str]:
        """Get default list of models to try."""
        models = ['logistic', 'random_forest', 'extra_trees']
        
        # Add gradient boosting models if available
        if HAS_LGB:
            models.append('lightgbm')
        if HAS_XGB:
            models.append('xgboost')
        if HAS_CB:
            models.append('catboost')
            
        return models
    
    def select_best_model(self, X, y) -> str:
        """Select best model based on cross-validation performance."""
        best_score = -np.inf
        best_model = 'logistic'  # fallback
        
        for model_type in self.models_to_try:
            try:
                # Get default parameters for this model
                params = get_default_params(model_type, random_state=self.random_state)
                
                # Create model
                model = ModelWrapper(model_type, **params)
                
                # Cross-validate
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                           scoring=self.scoring, n_jobs=-1)
                
                mean_score = scores.mean()
                self.model_scores_[model_type] = {
                    'mean': mean_score,
                    'std': scores.std(),
                    'scores': scores
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model_type
                    
            except Exception as e:
                # Model failed, skip it
                print(f"Warning: Model {model_type} failed with error: {e}")
                continue
        
        self.best_model_type_ = best_model
        return best_model


def get_default_params(model_type: str, random_state: int = 42) -> Dict[str, Any]:
    """Get default parameters for each model type."""
    
    params = {
        'logistic': {
            'random_state': random_state,
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'random_forest': {
            'n_estimators': 100,
            'random_state': random_state,
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        'extra_trees': {
            'n_estimators': 100,
            'random_state': random_state,
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        'svm': {
            'random_state': random_state,
            'class_weight': 'balanced',
            'kernel': 'rbf'
        },
        'knn': {
            'n_neighbors': 5,
            'n_jobs': -1
        },
        'naive_bayes': {},
        'neural_network': {
            'hidden_layer_sizes': (100,),
            'random_state': random_state,
            'max_iter': 500
        },
        'sgd': {
            'random_state': random_state,
            'class_weight': 'balanced',
            'loss': 'log_loss'
        },
        'ridge': {
            'random_state': random_state,
            'class_weight': 'balanced'
        }
    }
    
    # Gradient boosting models
    if HAS_XGB:
        params['xgboost'] = {
            'random_state': random_state,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'verbosity': 0  # Suppress warnings
        }
    
    if HAS_LGB:
        params['lightgbm'] = {
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': -1,
            'class_weight': 'balanced',
            'min_child_samples': 1,   # Minimum samples in a leaf
            'min_split_gain': 0.0,    # No minimum gain required
            'min_child_weight': 1e-10, # Very small minimum child weight
            'reg_alpha': 0.0,         # No L1 regularization
            'reg_lambda': 0.0,        # No L2 regularization  
            'feature_pre_filter': False,  # Disable feature name validation
            'force_col_wise': True,   # Force column-wise histogram building
            'max_depth': 10,          # Allow deeper trees
            'num_leaves': 31,         # Default number of leaves
            'learning_rate': 0.1,     # Learning rate
            'n_estimators': 10,       # Very few estimators to prevent warnings
            'min_data_in_leaf': 1,    # Allow single data point leaves
            'feature_fraction': 1.0,  # Use all features
            'bagging_fraction': 1.0,  # Use all data
            'boost_from_average': True,  # Boost from average
            'objective': 'binary',    # Explicitly set binary objective
            'metric': 'binary_logloss'  # Explicitly set metric
        }
    
    if HAS_CB:
        params['catboost'] = {
            'random_state': random_state,
            'verbose': False,
            'auto_class_weights': 'Balanced'
        }
    
    return params.get(model_type, {})


def get_hyperparameter_space(model_type: str) -> Dict[str, Any]:
    """Get hyperparameter search space for each model type."""
    
    spaces = {
        'logistic': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'extra_trees': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'neural_network': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['tanh', 'relu'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    # Gradient boosting spaces
    if HAS_XGB:
        spaces['xgboost'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    if HAS_LGB:
        spaces['lightgbm'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 30],
            'min_split_gain': [0.0, 0.1, 0.2]
        }
    
    if HAS_CB:
        spaces['catboost'] = {
            'iterations': [100, 200, 300],
            'depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    return spaces.get(model_type, {})


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier that combines multiple models."""
    
    def __init__(self, 
                 models: List[str],
                 ensemble_method: str = 'voting',
                 meta_learner: str = 'logistic',
                 random_state: int = 42):
        
        self.models = models
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner
        self.random_state = random_state
        
        self.ensemble_ = None
        self.base_models_ = {}
        
    def fit(self, X, y):
        """Fit the ensemble."""
        # Set classes_ attribute required by sklearn
        import numpy as np
        self.classes_ = np.unique(y)
        
        if self.ensemble_method == 'voting':
            self._fit_voting_ensemble(X, y)
        elif self.ensemble_method == 'stacking':
            self._fit_stacking_ensemble(X, y)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return self
    
    def _fit_voting_ensemble(self, X, y):
        """Fit voting ensemble."""
        estimators = []
        
        for model_name in self.models:
            params = get_default_params(model_name, self.random_state)
            model = ModelWrapper(model_name, **params)
            estimators.append((model_name, model))
            
        self.ensemble_ = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        
        self.ensemble_.fit(X, y)
    
    def _fit_stacking_ensemble(self, X, y):
        """Fit stacking ensemble."""
        from sklearn.ensemble import StackingClassifier
        
        base_estimators = []
        
        for model_name in self.models:
            params = get_default_params(model_name, self.random_state)
            model = ModelWrapper(model_name, **params)
            base_estimators.append((model_name, model))
        
        # Create meta-learner
        meta_params = get_default_params(self.meta_learner, self.random_state)
        meta_model = ModelWrapper(self.meta_learner, **meta_params)
        
        self.ensemble_ = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        self.ensemble_.fit(X, y)
    
    def predict(self, X):
        """Make predictions."""
        return self.ensemble_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.ensemble_.predict_proba(X)
    
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