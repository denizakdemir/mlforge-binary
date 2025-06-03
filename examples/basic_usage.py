"""
Basic usage example for MLForge-Binary

This example demonstrates the core functionality of MLForge-Binary, including:
- Basic model training and prediction
- Model evaluation and metrics
- XAI (Explainable AI) features with global and instance-level explanations
- Advanced customization options
- Ensemble models
- Model comparison
- AutoML functionality
- Production deployment features

The example showcases both global model explanations (feature importance) and
instance-level explanations for individual predictions using:
- clf.explain() - For global model explanations
- clf.explain_instance() - For explaining individual predictions
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
    
    # Get global insights (feature importance)
    explanations = clf.explain()
    evaluation_results = clf.evaluate(X_test, y_test)
    
    print(f"Model type: {clf.model}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    print(f"Test F1 Score: {evaluation_results['metrics']['f1_score']:.3f}")
    print("\nTop 5 important features:")
    if 'feature_importance' in explanations:
        for feature, importance in list(explanations['feature_importance'].items())[:5]:
            print(f"  {feature}: {importance:.3f}")
    
    # Get instance-level explanations for a single prediction
    print("\nExplaining a single prediction:")
    instance_idx = 0
    instance = X_test.iloc[[instance_idx]]
    instance_prediction = clf.predict(instance)[0]
    instance_probability = clf.predict_proba(instance)[0, 1]
    
    # Get instance explanation
    instance_explanation = clf.explain_instance(instance)
    
    print(f"Prediction for instance {instance_idx}: {instance_prediction} (probability: {instance_probability:.3f})")
    
    if 'contribution_summary' in instance_explanation:
        print("\nTop contributing features:")
        contributions = instance_explanation['contribution_summary']
        for feature, contribution in list(contributions.items())[:3]:
            direction = "increases" if contribution > 0 else "decreases"
            print(f"  {feature}: {contribution:+.3f} ({direction} probability)")
    
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
    
    # Compare models - avoid using 'auto' which may cause issues in ensemble creation
    leaderboard = compare_models(
        X_train, y_train, X_test, y_test,
        models=['random_forest', 'xgboost', 'lightgbm', 'logistic'],  # Changed models to avoid auto
        metrics=['roc_auc', 'f1', 'brier_score'],
        cv=3,  # Reduced for speed
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
    
    # Quick experiment (30 seconds)
    results = quick_experiment(
        X, y, 
        time_budget=30  # Reduced to 30 seconds
    )
    
    print("Quick AutoML Results:")
    print(results['leaderboard'].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nBest model test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
    
    # Full AutoML with more time but minimal configuration
    automl = AutoML(
        time_budget=60,  # Reduced to 60 seconds for quicker execution
        ensemble_size=2  # Reduced ensemble size
    )
    
    try:
        automl.fit(X, y)
        
        print("\nFull AutoML Leaderboard:")
        print(automl.leaderboard_.head().to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        # Get best model
        best_model = automl.get_best_model()
        print(f"\nBest model configuration: {best_model.get_params()}")
    except Exception as e:
        print(f"\nAutoML encountered an error: {e}")
        print("This can happen with some model combinations in AutoML.")
    
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
        optimize_threshold=True,
        explain=True  # Enable XAI features
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
    
    # Global model explanations
    global_explanations = loaded_clf.explain()
    print(f"Number of important features: {len(global_explanations.get('feature_importance', {}))}")
    
    # Instance-level explanations (for deployment)
    print("\nInstance-level explanations (production example):")
    
    # Simulate incoming prediction request
    new_instance = X_test.iloc[[0]]  # First test instance for demonstration
    
    # Make prediction
    prediction = loaded_clf.predict(new_instance)[0]
    probability = loaded_clf.predict_proba(new_instance)[0, 1]
    print(f"Prediction: {prediction} (probability: {probability:.4f})")
    
    # Explain this specific prediction
    instance_explanation = loaded_clf.explain_instance(new_instance)
    
    # Use explanation for business decision logic
    print("Explanation details for business logic:")
    
    if 'contribution_summary' in instance_explanation:
        # Sort features by absolute contribution
        contributions = instance_explanation['contribution_summary']
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Show top influencing features
        print("Top influencing features:")
        for feature, contribution in sorted_contributions[:3]:
            direction = "increases" if contribution > 0 else "decreases"
            print(f"  {feature}: {contribution:+.4f} ({direction} probability)")
        
        # Example business logic based on explanation
        if probability > 0.7 and any(abs(contrib) > 0.1 for _, contrib in sorted_contributions[:3]):
            print("High confidence prediction with strong feature support")
        elif probability > 0.5 and probability < 0.7:
            print("Moderate confidence prediction - consider additional checks")
        else:
            print("Low confidence prediction - manual review recommended")
    
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
