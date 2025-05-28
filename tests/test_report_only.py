"""Test only the report generation to see if plots are working"""

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

def test_report():
    """Test report generation with plots."""
    print("=== Testing Report Generation ===")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    clf = BinaryClassifier(
        model="catboost", 
        calibrate=True,
        optimize_threshold=True
    )
    
    # Train
    clf.fit(X_train, y_train)
    
    # Generate comprehensive report
    evaluation_results = clf.evaluate(
        X_test, y_test, 
        generate_report=True,
        report_path="test_report.html"
    )
    
    print(f"Model: {clf.model}")
    print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
    print(f"Report saved to: {evaluation_results.get('report_path', 'N/A')}")
    
    print("Report generation test completed successfully!")

if __name__ == "__main__":
    test_report()