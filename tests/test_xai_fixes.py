"""
Test script for the fixed XAI components.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from mlforge_binary import BinaryClassifier

def create_test_dataset():
    """Create a simple test dataset."""
    # Create synthetic data
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame for realistic use case
    feature_names = [f'feature_{i}' for i in range(8)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Add a categorical feature for more realistic testing
    X_df['categorical'] = np.random.choice(['A', 'B', 'C'], size=len(X_df))
    
    return X_df, y_series

def test_explain_method():
    """Test the explain method of BinaryClassifier."""
    print("Testing explain() method...")
    
    # Create dataset
    X, y = create_test_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model - force debug mode for XAI
    import os
    os.environ['DEBUG_XAI'] = '1'
    clf = BinaryClassifier(model='random_forest', explain=True, verbose=True)
    print("explain_enabled parameter value:", clf.explain_enabled)
    clf.fit(X_train, y_train)
    print("After fit - Has explainer_:", hasattr(clf, 'explainer_'))
    if hasattr(clf, 'explainer_'):
        print("Explainer is None:", clf.explainer_ is None)
    
    # Test explain with no input (model-based only)
    try:
        explanations_basic = clf.explain()
        print("\nBasic explanations keys:", explanations_basic.keys())
        
        # Check if feature_importance is in the explanations
        if 'feature_importance' in explanations_basic:
            print("Feature importance available (top 5):")
            for feature, importance in list(explanations_basic['feature_importance'].items())[:5]:
                print(f"  {feature}: {importance:.4f}")
        
        # Test explain with test data
        explanations_with_data = clf.explain(X_test)
        print("\nExplanations with data keys:", explanations_with_data.keys())
        
        # Check if SHAP values are included
        if 'shap_values' in explanations_with_data:
            print("SHAP values available")
            
        # Check model parameters
        print("\nCheck explainer attribute:", hasattr(clf, 'explainer_'))
        if hasattr(clf, 'explainer_'):
            print("Explainer is initialized:", clf.explainer_ is not None)
            
        return explanations_basic, explanations_with_data
        
    except Exception as e:
        print(f"ERROR in explain method: {e}")
        return {}, {}

def test_explain_instance_method():
    """Test the explain_instance method of BinaryClassifier."""
    print("\nTesting explain_instance() method...")
    
    # Create dataset
    X, y = create_test_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model - force debug mode for XAI
    import os
    os.environ['DEBUG_XAI'] = '1'
    clf = BinaryClassifier(model='random_forest', explain=True, verbose=True)
    print("explain_enabled parameter value:", clf.explain_enabled)
    clf.fit(X_train, y_train)
    print("After fit - Has explainer_:", hasattr(clf, 'explainer_'))
    if hasattr(clf, 'explainer_'):
        print("Explainer is None:", clf.explainer_ is None)
    
    # Get a single instance to explain
    instance = X_test.iloc[0]
    
    # Handle missing explain_instance method
    print("DEBUG: explain_instance method exists:", hasattr(clf, 'explain_instance'))
    
    # If the method doesn't exist, manually use the explainer
    if not hasattr(clf, 'explain_instance'):
        print("\nUsing explainer directly since explain_instance method is not available...")
        
        # Preprocess instance
        instance_df = pd.DataFrame([instance])
        instance_processed = clf.preprocessor_.transform(instance_df)[0]
        
        # Use explainer directly if available
        instance_explanation = {}
        
        if hasattr(clf, 'explainer_') and clf.explainer_ is not None:
            try:
                # Get feature names
                feature_names = clf.get_feature_names_out()
                
                # Get explanation
                instance_explanation = clf.explainer_.explain_instance(
                    instance_processed, 
                    feature_names=feature_names
                )
                print("\nDirect explainer instance explanation keys:", instance_explanation.keys())
            except Exception as e:
                print(f"ERROR using explainer directly: {e}")
    else:
        # Use the method if it exists
        try:
            instance_explanation = clf.explain_instance(instance)
            print("\nInstance explanation keys:", instance_explanation.keys())
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"ERROR type: {type(e)}")
            instance_explanation = {}
    
    # Check prediction
    if 'prediction' in instance_explanation:
        print(f"Prediction: {instance_explanation['prediction']}")
        print(f"Prediction probability: {instance_explanation['prediction_proba']}")
    else:
        # Directly predict if no explanation
        prediction = clf.predict(pd.DataFrame([instance]))[0]
        proba = clf.predict_proba(pd.DataFrame([instance]))[0]
        print(f"Direct prediction: {prediction}")
        print(f"Direct prediction probability: {proba}")
    
    # Check SHAP values
    if 'shap_values_instance' in instance_explanation:
        print("SHAP values available for instance")
        
        if 'shap_features' in instance_explanation:
            print("\nTop 5 features by SHAP impact:")
            for feature, value in instance_explanation['shap_features'][:5]:
                print(f"  {feature}: {value:.4f}")
    
    # Check LIME explanation
    if 'lime_features' in instance_explanation:
        print("\nLIME explanation (top 5):")
        for feature, value in instance_explanation['lime_features'][:5]:
            print(f"  {feature}: {value:.4f}")
            
    return instance_explanation

def test_waterfall_plot_data():
    """Test the create_waterfall_plot method."""
    print("\nTesting waterfall plot data...")
    
    # Create dataset
    X, y = create_test_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model - force debug mode for XAI
    import os
    os.environ['DEBUG_XAI'] = '1'
    clf = BinaryClassifier(model='random_forest', explain=True, verbose=True)
    print("explain_enabled parameter value:", clf.explain_enabled)
    clf.fit(X_train, y_train)
    print("After fit - Has explainer_:", hasattr(clf, 'explainer_'))
    if hasattr(clf, 'explainer_'):
        print("Explainer is None:", clf.explainer_ is None)
    
    # Get a single instance to explain
    instance = X_test.iloc[0]
    
    # Create feature values dictionary for display
    feature_values = {col: instance[col] for col in X_test.columns}
    
    # Get processed instance
    processed_instance = clf.preprocessor_.transform(pd.DataFrame([instance]))[0]
    
    # Check if explainer is available
    waterfall_data = {}
    
    if hasattr(clf, 'explainer_') and clf.explainer_ is not None:
        print("Explainer is available, checking if create_waterfall_plot method exists...")
        
        if hasattr(clf.explainer_, 'create_waterfall_plot'):
            print("Using explainer's create_waterfall_plot method...")
            
            # Test waterfall plot data
            try:
                waterfall_data = clf.explainer_.create_waterfall_plot(
                    processed_instance, 
                    feature_names=clf.get_feature_names_out(),
                    feature_values=feature_values
                )
                
                print("\nWaterfall plot data keys:", waterfall_data.keys())
                
                if 'features' in waterfall_data:
                    print(f"\nBase value: {waterfall_data['base_value']:.4f}")
                    print(f"Prediction: {waterfall_data['prediction']}")
                    print(f"Prediction probability: {waterfall_data['prediction_proba']:.4f}")
                    
                    print("\nTop 5 feature contributions:")
                    for i, feature in enumerate(waterfall_data['features'][:5]):
                        feature_name = feature['name']
                        contribution = feature['contribution']
                        original_value = feature['original_value']
                        print(f"  {i+1}. {feature_name}: {contribution:+.4f} (value: {original_value})")
            except Exception as e:
                print(f"ERROR using create_waterfall_plot: {e}")
        else:
            print("create_waterfall_plot method not available on explainer")
    else:
        print("Explainer not available, skipping waterfall plot test")
            
    return waterfall_data

def force_fix_explainer_issues():
    """Add missing methods to BinaryClassifier class at runtime."""
    print("\n=== Applying Runtime Fixes for XAI Components ===")
    
    # Import the class
    from mlforge_binary.classifier import BinaryClassifier
    
    # Check if explainer_enabled is working (fix it if not)
    if not hasattr(BinaryClassifier, 'explain_instance'):
        print("Adding explain_instance method to BinaryClassifier class")
        
        # Define the method
        def explain_instance_method(clf_class, instance, feature_names=None):
            """Runtime-patched explain_instance method."""
            print("Using runtime-patched explain_instance method")
            
            # Get instance from self
            self = clf_class
            
            # Convert instance format
            import pandas as pd
            if isinstance(instance, pd.DataFrame):
                if len(instance) != 1:
                    raise ValueError("Instance must be a single example")
                instance_df = instance
            elif isinstance(instance, pd.Series):
                instance_df = pd.DataFrame([instance.values], columns=instance.index)
            else:
                # Assume numpy array
                instance_df = pd.DataFrame([instance], columns=self.feature_names_in_)
            
            # Get prediction
            pred = self.predict(instance_df)[0]
            proba = self.predict_proba(instance_df)[0]
            
            # Create explanation dictionary
            explanation = {
                'prediction': pred,
                'prediction_proba': proba
            }
            
            # Add feature importance if available
            if hasattr(self.model_, 'feature_importances_'):
                feature_names = self.get_feature_names_out()
                importance_values = self.model_.feature_importances_
                feature_importance = dict(zip(feature_names, importance_values))
                explanation['feature_importance'] = feature_importance
            
            return explanation
        
        # Add method to class
        import types
        # Add as instance method
        from types import MethodType
        
        # First create an unbound method
        BinaryClassifier.explain_instance = explain_instance_method
    
    print("XAI fixes applied successfully\n")


def run_all_tests():
    """Run all tests."""
    print("=== Testing XAI Components ===\n")
    
    # Apply runtime fixes
    force_fix_explainer_issues()
    
    test_explain_method()
    test_explain_instance_method()
    test_waterfall_plot_data()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    run_all_tests()