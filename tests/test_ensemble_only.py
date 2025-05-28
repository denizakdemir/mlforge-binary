"""Test only the ensemble example"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlforge_binary import BinaryClassifier

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

def test_ensemble():
    """Test ensemble model."""
    print("=== Testing Ensemble Model ===")
    
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
    
    print("Ensemble test completed successfully!")

if __name__ == "__main__":
    test_ensemble()