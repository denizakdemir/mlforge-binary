"""
Automatic preprocessing pipeline for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

from .utils import (
    detect_missing_pattern, detect_categorical_columns, 
    detect_high_cardinality_columns, detect_imbalance
)


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """Automatic preprocessing pipeline that adapts to data characteristics."""
    
    def __init__(self,
                 handle_missing: str = "auto",
                 handle_categorical: str = "auto", 
                 handle_scaling: str = "auto",
                 handle_imbalanced: str = "auto",
                 feature_selection: bool = False,
                 feature_engineering: bool = False,
                 missing_threshold: float = 0.95,
                 correlation_threshold: float = 0.95,
                 max_features: Optional[int] = None,
                 random_state: int = 42):
        
        self.handle_missing = handle_missing
        self.handle_categorical = handle_categorical
        self.handle_scaling = handle_scaling
        self.handle_imbalanced = handle_imbalanced
        self.feature_selection = feature_selection
        self.feature_engineering = feature_engineering
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.random_state = random_state
        
        self.preprocessor_ = None
        self.resampler_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.categorical_columns_ = None
        self.numeric_columns_ = None
        
    def _analyze_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze data characteristics to determine preprocessing strategy."""
        analysis = {}
        
        # Missing value analysis
        analysis['missing'] = detect_missing_pattern(X)
        
        # Column type analysis
        analysis['categorical_columns'] = detect_categorical_columns(X)
        analysis['numeric_columns'] = [col for col in X.columns 
                                     if col not in analysis['categorical_columns']]
        analysis['high_cardinality_columns'] = detect_high_cardinality_columns(X)
        
        # Data quality issues
        analysis['columns_to_drop'] = []
        
        # Drop columns with too many missing values
        for col in X.columns:
            missing_pct = X[col].isnull().sum() / len(X)
            if missing_pct > self.missing_threshold:
                analysis['columns_to_drop'].append(col)
        
        # Detect highly correlated features
        numeric_cols = analysis['numeric_columns']
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_pairs = [(col, row) for col in upper_tri.columns 
                              for row in upper_tri.index 
                              if upper_tri.loc[row, col] > self.correlation_threshold]
            
            # Keep first feature from each highly correlated pair
            for col1, col2 in high_corr_pairs:
                if col2 not in analysis['columns_to_drop']:
                    analysis['columns_to_drop'].append(col2)
        
        # Class imbalance analysis
        if y is not None:
            analysis['is_imbalanced'] = detect_imbalance(y)
            analysis['class_distribution'] = y.value_counts(normalize=True).to_dict()
        
        return analysis
    
    def _create_missing_imputer(self, strategy: str, columns: List[str]) -> Any:
        """Create missing value imputer based on strategy."""
        if strategy == "simple":
            return SimpleImputer(strategy='median')
        elif strategy == "iterative":
            if ITERATIVE_IMPUTER_AVAILABLE:
                return IterativeImputer(random_state=self.random_state, max_iter=10)
            else:
                # Fallback to simple imputer if iterative is not available
                return SimpleImputer(strategy="median")
        elif strategy == "none":
            return "passthrough"
        else:  # auto
            # Use iterative for numeric, simple for categorical
            if ITERATIVE_IMPUTER_AVAILABLE:
                return IterativeImputer(random_state=self.random_state, max_iter=10)
            else:
                # Fallback to simple imputer if iterative is not available
                return SimpleImputer(strategy="median")
    
    def _create_categorical_encoder(self, strategy: str, columns: List[str]) -> Any:
        """Create categorical encoder based on strategy."""
        if strategy == "onehot":
            return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        elif strategy == "target":
            return TargetEncoder(random_state=self.random_state, cv=5)
        elif strategy == "none":
            return LabelEncoder()
        else:  # auto
            # Use target encoding for high cardinality, one-hot for low cardinality
            high_cardinality_cols = detect_high_cardinality_columns(pd.DataFrame(columns={col: [] for col in columns}))
            if any(col in high_cardinality_cols for col in columns):
                return TargetEncoder(random_state=self.random_state, cv=5)
            else:
                return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    def _create_scaler(self, strategy: str) -> Any:
        """Create scaler based on strategy."""
        if strategy == "standard":
            return StandardScaler()
        elif strategy == "robust":
            return RobustScaler()
        elif strategy == "minmax":
            return MinMaxScaler()
        elif strategy == "none":
            return "passthrough"
        else:  # auto
            return RobustScaler()  # More robust to outliers
    
    def _create_feature_selector(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Create feature selector based on configuration."""
        if not self.feature_selection:
            return "passthrough"
        
        n_features = X.shape[1]
        k = min(self.max_features or n_features, n_features)
        
        # Use mutual information for feature selection
        return SelectKBest(score_func=mutual_info_classif, k=k)
    
    def _create_resampler(self, strategy: str) -> Optional[Any]:
        """Create resampler for handling imbalanced data."""
        if strategy == "none" or strategy == "auto":
            return None
        elif strategy == "smote":
            if IMBALANCED_LEARN_AVAILABLE:
                return SMOTE(random_state=self.random_state)
            else:
                return None
        elif strategy == "adasyn":
            if IMBALANCED_LEARN_AVAILABLE:
                return ADASYN(random_state=self.random_state)
            else:
                return None
        elif strategy == "undersample":
            if IMBALANCED_LEARN_AVAILABLE:
                return RandomUnderSampler(random_state=self.random_state)
            else:
                return None
        elif strategy == "smote_tomek":
            if IMBALANCED_LEARN_AVAILABLE:
                return SMOTETomek(random_state=self.random_state)
            else:
                return None
        else:
            return None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Optional[Union[np.ndarray, pd.Series]] = None):
        """Fit the preprocessing pipeline."""
        from .utils import validate_input_data
        
        X, y = validate_input_data(X, y)
        self.feature_names_in_ = X.columns.tolist()
        
        # Analyze data characteristics
        analysis = self._analyze_data(X, y)
        
        # Remove problematic columns
        X_clean = X.drop(columns=analysis['columns_to_drop'])
        
        # Update column lists after dropping
        self.categorical_columns_ = [col for col in analysis['categorical_columns'] 
                                   if col not in analysis['columns_to_drop']]
        self.numeric_columns_ = [col for col in analysis['numeric_columns']
                               if col not in analysis['columns_to_drop']]
        
        # Create preprocessing pipelines for different column types
        transformers = []
        
        # Numeric pipeline
        if self.numeric_columns_:
            numeric_pipeline = Pipeline([
                ('imputer', self._create_missing_imputer(self.handle_missing, self.numeric_columns_)),
                ('scaler', self._create_scaler(self.handle_scaling))
            ])
            transformers.append(('numeric', numeric_pipeline, self.numeric_columns_))
        
        # Categorical pipeline  
        if self.categorical_columns_:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', self._create_categorical_encoder(self.handle_categorical, self.categorical_columns_))
            ])
            transformers.append(('categorical', categorical_pipeline, self.categorical_columns_))
        
        # Create main preprocessor
        if transformers:
            self.preprocessor_ = ColumnTransformer(
                transformers=transformers,
                remainder='drop'  # Drop any remaining columns
            )
        else:
            # No valid columns found
            self.preprocessor_ = "passthrough"
        
        # Add feature engineering if requested
        pipeline_steps = []
        
        if self.preprocessor_ != "passthrough":
            pipeline_steps.append(('preprocessor', self.preprocessor_))
        
        # Feature engineering (polynomial features for linear models)
        if self.feature_engineering:
            pipeline_steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        
        # Feature selection
        if self.feature_selection and y is not None:
            # Fit preprocessor first to get transformed features
            if pipeline_steps:
                temp_pipeline = Pipeline(pipeline_steps)
                X_temp = temp_pipeline.fit_transform(X_clean, y)
            else:
                X_temp = X_clean
            
            feature_selector = self._create_feature_selector(pd.DataFrame(X_temp), y)
            if feature_selector != "passthrough":
                pipeline_steps.append(('feature_selection', feature_selector))
        
        # Create final pipeline
        if pipeline_steps:
            self.preprocessor_ = Pipeline(pipeline_steps)
            self.preprocessor_.fit(X_clean, y)
        
        # Create resampler for imbalanced data
        if y is not None and analysis.get('is_imbalanced', False):
            if self.handle_imbalanced == "auto":
                # Use SMOTE for imbalanced data
                self.resampler_ = SMOTE(random_state=self.random_state)
            else:
                self.resampler_ = self._create_resampler(self.handle_imbalanced)
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform the data using fitted preprocessor."""
        from .utils import validate_input_data
        
        X, _ = validate_input_data(X)
        
        if self.preprocessor_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Remove same columns that were dropped during fit
        if hasattr(self, 'columns_to_drop_'):
            X = X.drop(columns=self.columns_to_drop_, errors='ignore')
        
        if self.preprocessor_ == "passthrough":
            return X.values
        
        return self.preprocessor_.transform(X)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit the preprocessor and transform the data."""
        self.fit(X, y)
        X_transformed = self.transform(X)
        
        # Apply resampling if needed and y is provided
        if y is not None and self.resampler_ is not None:
            from .utils import validate_input_data
            _, y = validate_input_data(X, y)
            X_resampled, y_resampled = self.resampler_.fit_resample(X_transformed, y)
            return X_resampled, y_resampled.values
        
        return X_transformed if y is None else (X_transformed, y.values)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if self.preprocessor_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if self.preprocessor_ == "passthrough":
            return self.feature_names_in_
        
        try:
            return self.preprocessor_.get_feature_names_out().tolist()
        except AttributeError:
            # Fallback for older sklearn versions
            n_features = self.preprocessor_.transform(pd.DataFrame({col: [0] for col in self.feature_names_in_})).shape[1]
            return [f'feature_{i}' for i in range(n_features)]