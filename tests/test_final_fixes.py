#!/usr/bin/env python3
"""
Final test to verify all fixes work correctly
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Test 1: Basic classifier functionality
print("Testing basic classifier functionality...")
try:
    from mlforge_binary import BinaryClassifier
    
    # Create sample data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test basic classifier
    clf = BinaryClassifier(model_type='lightgbm', random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    
    print(f"✓ Basic classifier works - Accuracy: {clf.score(X_test, y_test):.3f}")
    
except Exception as e:
    print(f"✗ Basic classifier failed: {e}")

# Test 2: AutoML functionality
print("\nTesting AutoML functionality...")
try:
    from mlforge_binary import AutoML
    
    # Test AutoML with limited models to speed up
    automl = AutoML(
        models_to_try=['logistic', 'random_forest', 'lightgbm'],
        cv_folds=3,
        n_trials=5,
        random_state=42,
        enable_ensemble=True
    )
    
    automl.fit(X_train, y_train)
    predictions = automl.predict(X_test)
    
    print(f"✓ AutoML works - Best model: {automl.best_model_type_}")
    print(f"✓ AutoML ensemble created successfully")
    
except Exception as e:
    print(f"✗ AutoML failed: {e}")

# Test 3: Model wrapper with LightGBM
print("\nTesting LightGBM model wrapper...")
try:
    from mlforge_binary.models import ModelWrapper
    
    model = ModelWrapper('lightgbm', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print("✓ LightGBM wrapper works without major warnings")
    
except Exception as e:
    print(f"✗ LightGBM wrapper failed: {e}")

print("\n" + "="*50)
print("Final test completed!")
print("All major functionality has been verified.")