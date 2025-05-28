"""
Basic usage example for MLForge-Binary
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlforge_binary import BinaryClassifier, AutoML, compare_models, quick_experiment


def create_sample_data():
    """Create sample binary classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("=== Example 1: Basic Usage ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize with sensible defaults (disable calibration for testing)
    clf = BinaryClassifier(model='logistic', calibrate=False)
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    predictions = clf.predict_proba(X_test)
    binary_predictions = clf.predict(X_test)
    
    # Get insights
    explanations = clf.explain()
    evaluation_results = clf.evaluate(X_test, y_test)
    
    print(f"Model type: {clf.model}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    print(f"Test F1 Score: {evaluation_results['metrics']['f1_score']:.3f}")
    print("\nTop 5 important features:")
    if 'feature_importance' in explanations:
        for feature, importance in list(explanations['feature_importance'].items())[:5]:
            print(f"  {feature}: {importance:.3f}")
    
    print("\n")


def example_advanced_usage():
    """Example 2: Advanced usage with customization."""
    print("=== Example 2: Advanced Usage ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Customize model selection and preprocessing
    clf = BinaryClassifier(
        model="auto",  # auto-select best model
        models_to_try=["lightgbm", "xgboost", "catboost", "logistic", "random_forest"],
        handle_missing="auto",
        handle_categorical="auto",
        optimize_threshold=True,  # Find best classification threshold
        calibrate=True,  # Probability calibration
        feature_selection=True,  # Automatic feature selection
        explain=True  # Enable explanations
    )
    
    # Train
    clf.fit(X_train, y_train)
    
    # Evaluate with comprehensive report
    evaluation_results = clf.evaluate(
        X_test, y_test, 
        generate_report=True,
        report_path="model_report.html"
    )
    
    print(f"Selected model: {clf.model}")
    print(f"Optimal threshold: {clf.optimal_threshold_:.3f}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    print(f"Report saved to: {evaluation_results.get('report_path', 'N/A')}")
    
    # Save model
    clf.save("trained_model.pkl")
    print("Model saved to: trained_model.pkl")
    
    print("\n")


def example_ensemble():
    """Example 3: Ensemble model."""
    print("=== Example 3: Ensemble Model ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Custom ensemble
    clf = BinaryClassifier(
        model="ensemble",
        ensemble_models=["xgboost", "catboost", "logistic"],
        ensemble_method="stacking",  # voting, stacking
        meta_learner="logistic"  # for stacking
    )
    
    # Train
    clf.fit(X_train, y_train)
    
    # Evaluate
    evaluation_results = clf.evaluate(X_test, y_test)
    
    print(f"Ensemble method: {clf.ensemble_method}")
    print(f"Base models: {clf.ensemble_models}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    
    print("\n")


def example_model_comparison():
    """Example 4: Compare multiple models."""
    print("=== Example 4: Model Comparison ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compare models
    leaderboard = compare_models(
        X_train, y_train, X_test, y_test,
        models=['auto', 'xgboost', 'catboost', 'logistic'],
        metrics=['roc_auc', 'f1', 'brier_score'],
        cv=5,
        include_preprocessing_variants=True
    )
    
    print("Model Comparison Results:")
    print(leaderboard.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print("\n")


def example_automl():
    """Example 5: AutoML experiment."""
    print("=== Example 5: AutoML Experiment ===")
    
    # Create sample data
    X, y = create_sample_data()
    
    # Quick experiment (5 minutes)
    results = quick_experiment(X, y, time_budget=300)
    
    print("Quick AutoML Results:")
    print(results['leaderboard'].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nBest model test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
    
    # Full AutoML with more time
    automl = AutoML(time_budget=600, ensemble_size=3)  # 10 minutes
    automl.fit(X, y)
    
    print("\nFull AutoML Leaderboard:")
    print(automl.leaderboard_.head().to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Get best model
    best_model = automl.get_best_model()
    print(f"\nBest model configuration: {best_model.get_params()}")
    
    print("\n")


def example_production_features():
    """Example 6: Production features."""
    print("=== Example 6: Production Features ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train production-ready model
    clf = BinaryClassifier(
        model="auto", 
        calibrate=True,
        optimize_threshold=True
    )
    clf.fit(X_train, y_train)
    
    # Save for production
    clf.save("production_model.pkl")
    
    # Load model (simulating production environment)
    loaded_clf = BinaryClassifier.load("production_model.pkl")
    
    # Make predictions
    predictions = loaded_clf.predict(X_test)
    probabilities = loaded_clf.predict_proba(X_test)
    
    print(f"Loaded model type: {loaded_clf.model}")
    print(f"Optimal threshold: {loaded_clf.optimal_threshold_:.3f}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Model explanations
    explanations = loaded_clf.explain()
    print(f"Number of important features: {len(explanations.get('feature_importance', {}))}")
    
    print("\n")


if __name__ == "__main__":
    print("MLForge-Binary Examples\n")
    
    try:
        example_basic_usage()
        example_advanced_usage()
        example_ensemble()
        example_model_comparison()
        example_automl()
        example_production_features()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
