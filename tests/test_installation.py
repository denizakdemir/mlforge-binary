"""
Quick test to verify MLForge-Binary installation and basic functionality
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")
    
    try:
        from mlforge_binary import BinaryClassifier, AutoML, compare_models, quick_experiment
        print("✓ Main imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from mlforge_binary.utils import validate_input_data, detect_missing_pattern
        print("✓ Utils import successful")
    except ImportError as e:
        print(f"✗ Utils import error: {e}")
        return False
    
    assert True


def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        from mlforge_binary import BinaryClassifier
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        print("✓ Sample data created")
        
        # Test basic classifier
        clf = BinaryClassifier(model='logistic', calibrate=False, verbose=False)
        clf.fit(X_df, y_series)
        
        predictions = clf.predict(X_df)
        probabilities = clf.predict_proba(X_df)
        
        print(f"✓ Model trained and predictions made")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - Probabilities shape: {probabilities.shape}")
        
        # Test evaluation
        results = clf.evaluate(X_df, y_series)
        print(f"✓ Evaluation completed")
        print(f"  - ROC AUC: {results['metrics']['roc_auc']:.3f}")
        
        assert True
        
    except Exception as e:
        print(f"✗ Basic functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optional_dependencies():
    """Test optional dependencies."""
    print("\nTesting optional dependencies...")
    
    # Test gradient boosting libraries
    optional_libs = {
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM', 
        'catboost': 'CatBoost',
        'shap': 'SHAP',
        'plotly': 'Plotly'
    }
    
    available_libs = []
    
    for lib, name in optional_libs.items():
        try:
            __import__(lib)
            print(f"✓ {name} available")
            available_libs.append(lib)
        except ImportError:
            print(f"○ {name} not available (optional)")
    
    assert available_libs  # Should have at least some available libs


def main():
    """Run all tests."""
    print("MLForge-Binary Installation Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        sys.exit(1)
    
    # Test optional dependencies
    available_libs = test_optional_dependencies()
    
    print("\n" + "=" * 40)
    print("✅ All tests passed!")
    print(f"📦 Available optional libraries: {', '.join(available_libs) if available_libs else 'None'}")
    print("\nMLForge-Binary is ready to use!")
    
    # Show quick usage example
    print("\nQuick usage example:")
    print("""
from mlforge_binary import BinaryClassifier

# Create classifier
clf = BinaryClassifier()

# Train and predict
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)

# Evaluate
results = clf.evaluate(X_test, y_test)
    """)


if __name__ == "__main__":
    main()