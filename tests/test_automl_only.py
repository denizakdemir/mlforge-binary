"""Test only the AutoML functionality"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlforge_binary import AutoML, quick_experiment

def create_sample_data():
    """Create sample binary classification dataset."""
    X, y = make_classification(
        n_samples=500,  # Smaller dataset for faster testing
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

def test_automl():
    """Test AutoML functionality."""
    print("=== Testing AutoML ===")
    
    # Create sample data
    X, y = create_sample_data()
    
    # Test quick_experiment first (simpler)
    print("Testing quick_experiment...")
    try:
        results = quick_experiment(X, y, time_budget=30)  # 30 seconds
        print(f"Quick experiment succeeded! Best ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
    except Exception as e:
        print(f"Quick experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test full AutoML
    print("\nTesting full AutoML...")
    try:
        automl = AutoML(time_budget=60, ensemble_size=2)  # 1 minute, smaller ensemble
        automl.fit(X, y)
        
        print("AutoML training succeeded!")
        print(f"Number of models evaluated: {len(automl.leaderboard_)}")
        
        best_model = automl.get_best_model()
        print(f"Best model type: {best_model}")
        
    except Exception as e:
        print(f"AutoML failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_automl()