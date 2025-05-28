"""
AutoML functionality for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from .classifier import BinaryClassifier
from .utils import validate_input_data, Timer, format_number


class AutoML:
    """Automated machine learning for binary classification."""
    
    def __init__(self,
                 time_budget: int = 3600,  # seconds
                 ensemble_size: int = 5,
                 metric: str = 'roc_auc',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True):
        
        self.time_budget = time_budget
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.leaderboard_ = None
        self.best_model_ = None
        self.ensemble_model_ = None
        self.experiment_results_ = []
        
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AutoML] {message}")
    
    def _get_model_configurations(self) -> List[Dict[str, Any]]:
        """Get list of model configurations to try."""
        configurations = [
            # Individual models with default settings
            {'model': 'lightgbm'},
            {'model': 'xgboost'},
            {'model': 'catboost'},
            {'model': 'random_forest'},
            {'model': 'extra_trees'},
            {'model': 'logistic'},
            
            # Models with different preprocessing
            {'model': 'lightgbm', 'handle_missing': 'simple'},
            {'model': 'xgboost', 'handle_categorical': 'target'},
            {'model': 'catboost', 'handle_scaling': 'standard'},
            
            # Models with feature engineering
            {'model': 'logistic', 'feature_engineering': True},
            {'model': 'random_forest', 'feature_selection': True},
            
            # Calibrated models
            {'model': 'lightgbm', 'calibrate': True},
            {'model': 'xgboost', 'calibrate': True},
            
            # Ensemble models
            {
                'model': 'ensemble',
                'ensemble_models': ['lightgbm', 'xgboost', 'logistic'],
                'ensemble_method': 'voting'
            },
            {
                'model': 'ensemble',
                'ensemble_models': ['catboost', 'random_forest', 'extra_trees'],
                'ensemble_method': 'stacking'
            },
            
            # Auto model selection
            {'model': 'auto'},
            {'model': 'auto', 'calibrate': True},
            {'model': 'auto', 'feature_selection': True},
        ]
        
        return configurations
    
    def _evaluate_configuration(self, 
                              config: Dict[str, Any], 
                              X_train: np.ndarray, 
                              y_train: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single model configuration."""
        
        try:
            with Timer() as timer:
                # Create classifier
                clf = BinaryClassifier(
                    random_state=self.random_state,
                    verbose=False,
                    **config
                )
                
                # Cross-validate
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        clf, X_train, y_train,
                        cv=self.cv_folds,
                        scoring=self.metric,
                        n_jobs=1  # Use single job for parallel execution
                    )
                
                result = {
                    'config': config,
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores,
                    'training_time': timer.elapsed_time,
                    'model_name': self._config_to_name(config)
                }
                
                self._log(f"✓ {result['model_name']}: {result['mean_score']:.4f} (±{result['std_score']:.4f})")
                return result
                
        except Exception as e:
            self._log(f"✗ {self._config_to_name(config)}: Failed - {str(e)}")
            return {
                'config': config,
                'mean_score': -np.inf,
                'std_score': np.inf,
                'scores': np.array([]),
                'training_time': 0,
                'model_name': self._config_to_name(config),
                'error': str(e)
            }
    
    def _config_to_name(self, config: Dict[str, Any]) -> str:
        """Convert configuration to readable name."""
        model = config.get('model', 'unknown')
        
        if model == 'ensemble':
            ensemble_models = config.get('ensemble_models', [])
            ensemble_method = config.get('ensemble_method', 'voting')
            return f"ensemble_{ensemble_method}_{'_'.join(ensemble_models[:2])}"
        
        name = model
        
        # Add preprocessing modifiers
        if config.get('calibrate', False):
            name += "_calibrated"
        if config.get('feature_engineering', False):
            name += "_feateng"
        if config.get('feature_selection', False):
            name += "_featsel"
        
        return name
    
    def fit(self, 
           X: Union[np.ndarray, pd.DataFrame],
           y: Union[np.ndarray, pd.Series]) -> 'AutoML':
        """Run automated machine learning."""
        
        self._log(f"Starting AutoML with {self.time_budget} second budget...")
        
        # Validate input
        X, y = validate_input_data(X, y)
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        self._log(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")
        
        # Get configurations to try
        configurations = self._get_model_configurations()
        self._log(f"Evaluating {len(configurations)} model configurations...")
        
        # Time-bounded evaluation
        start_time = time.time()
        results = []
        
        if self.n_jobs == 1:
            # Sequential execution
            for i, config in enumerate(configurations):
                if time.time() - start_time > self.time_budget:
                    self._log(f"Time budget exceeded, stopping after {i} configurations")
                    break
                
                result = self._evaluate_configuration(config, X_train.values, y_train.values)
                results.append(result)
                
        else:
            # Parallel execution
            max_workers = min(self.n_jobs if self.n_jobs > 0 else 4, len(configurations))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_config = {
                    executor.submit(self._evaluate_configuration, config, X_train.values, y_train.values): config
                    for config in configurations
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    if time.time() - start_time > self.time_budget:
                        self._log("Time budget exceeded, cancelling remaining jobs")
                        # Cancel remaining futures
                        for f in future_to_config:
                            if not f.done():
                                f.cancel()
                        break
                    
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        config = future_to_config[future]
                        self._log(f"Configuration failed: {config} - {e}")
        
        # Store all results
        self.experiment_results_ = results
        
        # Create leaderboard
        self._create_leaderboard(results)
        
        # Train best model
        self._train_best_model(X_train, y_train)
        
        # Train ensemble if requested
        if self.ensemble_size > 1:
            self._train_ensemble_model(X_train, y_train)
        
        # Final evaluation
        self._final_evaluation(X_test, y_test)
        
        elapsed_time = time.time() - start_time
        self._log(f"AutoML completed in {elapsed_time:.1f} seconds")
        
        return self
    
    def _create_leaderboard(self, results: List[Dict[str, Any]]) -> None:
        """Create leaderboard from results."""
        # Filter successful results
        valid_results = [r for r in results if r['mean_score'] > -np.inf]
        
        if not valid_results:
            raise ValueError("No valid models found")
        
        # Sort by score
        valid_results.sort(key=lambda x: x['mean_score'], reverse=True)
        
        # Create DataFrame
        leaderboard_data = []
        for result in valid_results:
            leaderboard_data.append({
                'rank': len(leaderboard_data) + 1,
                'model': result['model_name'],
                'score': result['mean_score'],
                'std': result['std_score'],
                'training_time': result['training_time']
            })
        
        self.leaderboard_ = pd.DataFrame(leaderboard_data)
        
        # Display top results
        self._log("\
Top 10 Models:")
        self._log(self.leaderboard_.head(10).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    def _train_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the best model on full training data."""
        if not self.experiment_results_:
            raise ValueError("No experiments run yet")
        
        # Get best configuration
        best_result = max(self.experiment_results_, key=lambda x: x['mean_score'])
        best_config = best_result['config']
        
        self._log(f"Training best model: {best_result['model_name']}")
        
        # Train final model
        self.best_model_ = BinaryClassifier(
            random_state=self.random_state,
            verbose=False,
            **best_config
        )
        
        self.best_model_.fit(X_train, y_train)
    
    def _train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train ensemble of top models."""
        # Get top configurations
        valid_results = [r for r in self.experiment_results_ if r['mean_score'] > -np.inf]
        valid_results.sort(key=lambda x: x['mean_score'], reverse=True)
        
        top_configs = [r['config'] for r in valid_results[:self.ensemble_size]]
        
        self._log(f"Training ensemble of top {len(top_configs)} models...")
        
        # Create ensemble using voting with unique model names
        model_names = []
        seen_models = {}
        for i, config in enumerate(top_configs):
            base_model = config.get('model', 'logistic')
            # Create unique name for sklearn voting ensemble
            if base_model in seen_models:
                seen_models[base_model] += 1
                unique_name = f"{base_model}_{seen_models[base_model]}"
            else:
                seen_models[base_model] = 0
                unique_name = base_model
            model_names.append(unique_name)
        
        self.ensemble_model_ = BinaryClassifier(
            model='ensemble',
            ensemble_models=model_names,
            ensemble_method='voting',
            random_state=self.random_state,
            verbose=False
        )
        
        self.ensemble_model_.fit(X_train, y_train)
    
    def _final_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate final models on test set."""
        self._log("\
Final Test Set Evaluation:")
        
        # Evaluate best model
        if self.best_model_:
            y_pred = self.best_model_.predict(X_test)
            y_proba = self.best_model_.predict_proba(X_test)[:, 1]
            
            if self.metric == 'roc_auc':
                score = roc_auc_score(y_test, y_proba)
            else:
                from sklearn.metrics import get_scorer
                scorer = get_scorer(self.metric)
                score = scorer(self.best_model_, X_test, y_test)
            
            self._log(f"Best Model Test {self.metric.upper()}: {score:.4f}")
        
        # Evaluate ensemble
        if self.ensemble_model_:
            y_pred = self.ensemble_model_.predict(X_test)
            y_proba = self.ensemble_model_.predict_proba(X_test)[:, 1]
            
            if self.metric == 'roc_auc':
                score = roc_auc_score(y_test, y_proba)
            else:
                from sklearn.metrics import get_scorer
                scorer = get_scorer(self.metric)
                score = scorer(self.ensemble_model_, X_test, y_test)
            
            self._log(f"Ensemble Test {self.metric.upper()}: {score:.4f}")
    
    def get_best_model(self) -> BinaryClassifier:
        """Get the best trained model."""
        if self.best_model_ is None:
            raise ValueError("No model trained yet. Call fit() first.")
        return self.best_model_
    
    def get_ensemble_model(self) -> Optional[BinaryClassifier]:
        """Get the ensemble model if available."""
        return self.ensemble_model_
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the best model."""
        return self.get_best_model().predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities using the best model."""
        return self.get_best_model().predict_proba(X)


def compare_models(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    models: List[str] = None,
    metrics: List[str] = None,
    cv: int = 5,
    include_preprocessing_variants: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """Compare multiple models and return leaderboard."""
    
    if models is None:
        models = ['auto', 'xgboost', 'lightgbm', 'catboost', 'logistic', 'random_forest']
    
    if metrics is None:
        metrics = ['roc_auc', 'f1', 'brier_score']
    
    # Validate input
    X_train, y_train = validate_input_data(X_train, y_train)
    X_test, y_test = validate_input_data(X_test, y_test)
    
    results = []
    
    for model_name in models:
        print(f"Evaluating {model_name}...")
        
        configurations = [{'model': model_name}]
        
        # Add preprocessing variants if requested
        if include_preprocessing_variants and model_name != 'auto':
            configurations.extend([
                {'model': model_name, 'handle_categorical': 'target'},
                {'model': model_name, 'calibrate': True},
                {'model': model_name, 'feature_selection': True}
            ])
        
        for config in configurations:
            try:
                # Create and train model
                clf = BinaryClassifier(
                    random_state=random_state,
                    verbose=False,
                    **config
                )
                
                with Timer() as timer:
                    clf.fit(X_train, y_train)
                
                # Calculate metrics
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)
                
                from .evaluation import MetricsCalculator
                calc = MetricsCalculator()
                metric_values = calc.calculate_all_metrics(y_test.values, y_pred, y_proba)
                
                # Create result row
                result = {
                    'model': model_name,
                    'config': str(config),
                    'training_time': timer.elapsed_time
                }
                
                # Add requested metrics
                for metric in metrics:
                    if metric == 'roc_auc':
                        result[metric] = metric_values.get('roc_auc', np.nan)
                    elif metric == 'f1':
                        result[metric] = metric_values.get('f1_score', np.nan)
                    elif metric == 'brier_score':
                        result[metric] = metric_values.get('brier_score', np.nan)
                    elif metric == 'inference_time':
                        # Measure inference time
                        with Timer() as inf_timer:
                            _ = clf.predict(X_test[:100] if len(X_test) > 100 else X_test)
                        result[metric] = inf_timer.elapsed_time * 1000  # Convert to ms
                
                results.append(result)
                
            except Exception as e:
                print(f"  ✗ {config}: Failed - {e}")
                continue
    
    # Create DataFrame and sort by first metric
    df = pd.DataFrame(results)
    if not df.empty and metrics:
        primary_metric = metrics[0]
        ascending = primary_metric in ['brier_score', 'inference_time']  # Lower is better
        df = df.sort_values(primary_metric, ascending=ascending)
    
    return df


def quick_experiment(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    time_budget: int = 300,  # 5 minutes
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Run a quick AutoML experiment."""
    
    print(f"Running quick experiment with {time_budget} second budget...")
    
    # Split data
    X, y = validate_input_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Run AutoML
    automl = AutoML(
        time_budget=time_budget,
        random_state=random_state,
        verbose=True
    )
    
    automl.fit(X_train, y_train)
    
    # Get best model and evaluate
    best_model = automl.get_best_model()
    
    # Final evaluation
    evaluation_results = best_model.evaluate(
        X_test, y_test, 
        generate_report=False
    )
    
    return {
        'automl': automl,
        'best_model': best_model,
        'leaderboard': automl.leaderboard_,
        'test_metrics': evaluation_results['metrics'],
        'experiment_results': automl.experiment_results_
    }