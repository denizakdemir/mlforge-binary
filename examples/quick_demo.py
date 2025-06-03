"""
Quick demo of MLForge-Binary core functionality

This quick demo showcases the main features of MLForge-Binary including:
- Basic classification with default settings
- Automatic model selection
- AutoML for hyperparameter tuning
- XAI (Explainable AI) features for model interpretability
  - Global explanations (feature importance)
  - Instance-level explanations for individual predictions
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlforge_binary import BinaryClassifier, AutoML


def main():
    """Run quick demonstration of key features."""
    print("MLForge-Binary Quick Demo\n")
    
    # Create sample data
    print("Creating sample dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Example 1: Basic Usage
    print("=== Example 1: Basic Classifier ===")
    clf = BinaryClassifier(model='logistic', calibrate=False)
    clf.fit(X_train, y_train)
    
    # Evaluate
    results = clf.evaluate(X_test, y_test)
    print(f"Model: {clf.model}")
    print(f"ROC AUC: {results['metrics']['roc_auc']:.3f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
    print()
    
    # Example 2: Auto Model Selection with XAI
    print("=== Example 2: Auto Model Selection with XAI ===")
    clf_auto = BinaryClassifier(
        model='auto',
        models_to_try=['logistic', 'random_forest', 'lightgbm'],
        calibrate=False,
        explain=True  # Enable XAI features
    )
    clf_auto.fit(X_train, y_train)
    
    results_auto = clf_auto.evaluate(X_test, y_test)
    print(f"Selected model: {clf_auto.model}")
    print(f"ROC AUC: {results_auto['metrics']['roc_auc']:.3f}")
    print(f"F1 Score: {results_auto['metrics']['f1_score']:.3f}")
    
    # Get global model explanations
    explanations = clf_auto.explain()
    
    print("\nModel Explanations:")
    if 'feature_importance' in explanations:
        importance_dict = explanations['feature_importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 5 Important Features:")
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.4f}")
    
    # Get instance-level explanation for a single prediction
    print("\nInstance-level Explanation:")
    instance = X_test.iloc[[0]]  # First test instance
    prediction = clf_auto.predict(instance)[0]
    probability = clf_auto.predict_proba(instance)[0, 1]
    
    # Get instance explanation
    instance_explanation = clf_auto.explain_instance(instance)
    
    print(f"Prediction: {prediction} (probability: {probability:.4f})")
    
    if 'contribution_summary' in instance_explanation:
        print("Feature Contributions:")
        contributions = instance_explanation['contribution_summary']
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        for feature, contribution in sorted_contributions[:3]:
            direction = "increases" if contribution > 0 else "decreases"
            print(f"  {feature}: {contribution:+.4f} ({direction} probability)")
    
    print()
    
    # Example 3: AutoML (quick version)
    print("=== Example 3: AutoML (30 seconds) ===")
    automl = AutoML(
        time_budget=30,  # 30 seconds
        random_state=42
    )
    automl.fit(X_train, y_train)
    
    # Get best model performance
    best_row = automl.leaderboard_.iloc[0]
    best_model = best_row['model']
    best_score = best_row['score']
    print(f"Best model: {best_model}")
    print(f"Best CV score: {best_score:.3f}")
    
    # Test performance
    test_preds = automl.predict_proba(X_test)
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, test_preds[:, 1])
    print(f"Test ROC AUC: {test_auc:.3f}")
    
    # Get the best model from AutoML and explain it
    best_clf = automl.get_best_model()
    if hasattr(best_clf, 'explain'):
        print("\nExplaining AutoML Best Model:")
        try:
            best_explanations = best_clf.explain()
            if 'feature_importance' in best_explanations:
                imp_dict = best_explanations['feature_importance']
                sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
                
                print("Top 3 Important Features:")
                for feature, importance in sorted_imp[:3]:
                    print(f"  {feature}: {importance:.4f}")
        except Exception as e:
            print(f"Could not get explanations: {e}")
    
    print()
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()