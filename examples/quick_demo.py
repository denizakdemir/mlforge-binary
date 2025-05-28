"""
Quick demo of MLForge-Binary core functionality
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
    
    # Example 2: Auto Model Selection
    print("=== Example 2: Auto Model Selection ===")
    clf_auto = BinaryClassifier(
        model='auto',
        models_to_try=['logistic', 'random_forest', 'lightgbm'],
        calibrate=False
    )
    clf_auto.fit(X_train, y_train)
    
    results_auto = clf_auto.evaluate(X_test, y_test)
    print(f"Selected model: {clf_auto.model}")
    print(f"ROC AUC: {results_auto['metrics']['roc_auc']:.3f}")
    print(f"F1 Score: {results_auto['metrics']['f1_score']:.3f}")
    print()
    
    # Example 3: AutoML (quick version)
    print("=== Example 3: AutoML (30 seconds) ===")
    automl = AutoML(
        time_budget=30,  # 30 seconds
        random_state=42
    )
    automl.fit(X_train, y_train)
    
    # Get best model performance
    best_score = automl.best_score_
    best_model = automl.best_model_type_
    print(f"Best model: {best_model}")
    print(f"Best CV score: {best_score:.3f}")
    
    # Test performance
    test_preds = automl.predict_proba(X_test)
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, test_preds[:, 1])
    print(f"Test ROC AUC: {test_auc:.3f}")
    print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()