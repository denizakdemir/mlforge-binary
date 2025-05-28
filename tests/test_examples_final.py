"""Final test of all working examples (excluding AutoML for now)"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlforge_binary import BinaryClassifier, compare_models

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
    
    # Evaluate
    evaluation_results = clf.evaluate(X_test, y_test)
    
    print(f"Model type: {clf.model}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    print(f"Test F1 Score: {evaluation_results['metrics']['f1_score']:.3f}")

def example_advanced_usage():
    """Example 2: Advanced usage with customization."""
    print("\n=== Example 2: Advanced Usage ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Customize model selection and preprocessing
    clf = BinaryClassifier(
        model="auto",  # auto-select best model
        models_to_try=["lightgbm", "xgboost", "catboost", "logistic", "random_forest"],
        calibrate=True,  # Probability calibration
        optimize_threshold=True,  # Find best classification threshold
        feature_selection=True,  # Automatic feature selection
        explain=True  # Enable explanations
    )
    
    # Train
    clf.fit(X_train, y_train)
    
    # Evaluate with comprehensive report
    evaluation_results = clf.evaluate(
        X_test, y_test, 
        generate_report=True,
        report_path="final_report.html"
    )
    
    print(f"Selected model: {clf.model}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    print(f"Report saved to: {evaluation_results.get('report_path', 'N/A')}")

def example_ensemble():
    """Example 3: Ensemble model."""
    print("\n=== Example 3: Ensemble Model ===")
    
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

def example_model_comparison():
    """Example 4: Compare multiple models."""
    print("\n=== Example 4: Model Comparison ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compare models (smaller set for speed)
    leaderboard = compare_models(
        X_train, y_train, X_test, y_test,
        models=['logistic', 'xgboost', 'catboost'],
        metrics=['roc_auc', 'f1'],
        cv=3,
        include_preprocessing_variants=False  # Faster
    )
    
    print("Model Comparison Results:")
    print(leaderboard.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

if __name__ == "__main__":
    print("MLForge-Binary Final Examples Test\n")
    
    try:
        example_basic_usage()
        example_advanced_usage()
        example_ensemble()
        example_model_comparison()
        
        print("\n🎉 All examples completed successfully!")
        print("✅ Basic Usage: Working")
        print("✅ Advanced Usage: Working (with HTML report)")
        print("✅ Ensemble Models: Working") 
        print("✅ Model Comparison: Working")
        print("✅ Report Generation: Working (with interactive plots)")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()