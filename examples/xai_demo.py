"""
Explainable AI (XAI) Demo for MLForge-Binary

This example demonstrates the explainability features of MLForge-Binary,
including feature importance, SHAP values, LIME explanations, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from mlforge_binary import BinaryClassifier

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def create_synthetic_dataset():
    """Create a synthetic dataset with interpretable features."""
    print("Creating synthetic dataset with interpretable features...")
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        flip_y=0.1,  # Add some noise
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'age', 'income', 'education_years', 'work_experience',
        'credit_score', 'debt_ratio', 'savings', 'monthly_expenses',
        'dependents', 'location_score'
    ]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='loan_approved')
    
    print(f"Dataset created: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print("Target: loan approval prediction")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X_df, y_series


def real_world_dataset():
    """Load and prepare the breast cancer dataset."""
    print("Loading breast cancer dataset...")
    
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='diagnosis')
    
    # Use subset of most relevant features for interpretability
    important_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    
    X_subset = X[important_features]
    
    print(f"Dataset loaded: {X_subset.shape[0]} samples, {X_subset.shape[1]} features")
    print("Target: cancer diagnosis (malignant=1, benign=0)")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X_subset, y


def extract_feature_importance(clf, feature_names):
    """Extract feature importance from trained model."""
    feature_importance = {}
    
    # Try to get feature importance from explain() method
    try:
        explanations = clf.explain()
        if 'feature_importance' in explanations:
            return explanations['feature_importance']
    except:
        pass
    
    # Try to get feature importance directly from model
    try:
        if hasattr(clf.model_, 'feature_importances_'):
            importance_values = clf.model_.feature_importances_
            feature_importance = dict(zip(feature_names, importance_values))
        elif hasattr(clf.model_, 'coef_'):
            # For linear models, use absolute coefficients
            coef_values = clf.model_.coef_[0] if clf.model_.coef_.ndim > 1 else clf.model_.coef_
            feature_importance = dict(zip(feature_names, np.abs(coef_values)))
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    return feature_importance


def demonstrate_feature_importance(clf, X_test, y_test, dataset_name):
    """Demonstrate basic feature importance."""
    print(f"\n=== Feature Importance Analysis - {dataset_name} ===")
    
    feature_names = X_test.columns.tolist()
    feature_importance = extract_feature_importance(clf, feature_names)
    
    if feature_importance:
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<25}: {importance:.4f}")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        features = [item[0] for item in sorted_features[:10]]
        importances = [item[1] for item in sorted_features[:10]]
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Feature Importances - {dataset_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return dict(sorted_features[:10])
    else:
        print("Feature importance not available for this model type")
        return {}


def demonstrate_shap_analysis(clf, X_train, X_test, dataset_name):
    """Demonstrate SHAP-based explanations."""
    print(f"\n=== SHAP Analysis - {dataset_name} ===")
    
    try:
        import shap
        
        # Create a wrapper function for predict_proba that handles arrays
        def predict_proba_wrapper(X):
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=X_train.columns)
            else:
                X_df = X
            return clf.predict_proba(X_df)
        
        # Create SHAP explainer
        print("Creating SHAP explainer...")
        explainer = shap.Explainer(predict_proba_wrapper, X_train.iloc[:100])
        
        # Calculate SHAP values for test set
        print("Calculating SHAP values...")
        shap_values = explainer(X_test.iloc[:50])
        
        # SHAP summary plot
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[:, :, 1], X_test.iloc[:50], show=False)
        plt.title(f'SHAP Summary Plot - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # SHAP waterfall plot for first prediction
        print("Generating SHAP waterfall plot for first prediction...")
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_values[0, :, 1], show=False)
        plt.title(f'SHAP Waterfall Plot - First Prediction - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance from SHAP
        feature_importance_shap = np.abs(shap_values[:, :, 1]).mean(0)
        feature_names = X_test.columns
        
        print("\nTop 10 Features by SHAP Importance:")
        shap_importance_dict = dict(zip(feature_names, feature_importance_shap))
        sorted_shap = sorted(shap_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_shap[:10], 1):
            print(f"{i:2d}. {feature:<25}: {importance:.4f}")
        
        return shap_values
        
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
        return None
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None


def demonstrate_lime_analysis(clf, X_train, X_test, dataset_name, instance_idx=0):
    """Demonstrate LIME-based local explanations."""
    print(f"\n=== LIME Analysis - {dataset_name} ===")
    
    try:
        import lime
        import lime.lime_tabular
        
        # Create a wrapper function for predict_proba
        def predict_proba_wrapper(X):
            X_df = pd.DataFrame(X, columns=X_train.columns)
            return clf.predict_proba(X_df)
        
        # Create LIME explainer
        print("Creating LIME explainer...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Class 0', 'Class 1'],
            mode='classification'
        )
        
        # Explain a specific instance
        print(f"Explaining prediction for instance {instance_idx}...")
        instance = X_test.iloc[instance_idx].values
        
        exp = explainer.explain_instance(
            instance, 
            predict_proba_wrapper,
            num_features=len(X_test.columns)
        )
        
        # Get explanation as list
        explanation_list = exp.as_list()
        
        print(f"\nLIME Explanation for Instance {instance_idx}:")
        # Convert instance to DataFrame for prediction
        instance_df = pd.DataFrame([instance], columns=X_test.columns)
        pred_class = clf.predict(instance_df)[0]
        pred_proba = clf.predict_proba(instance_df)[0]
        print(f"Predicted class: {pred_class}")
        print(f"Prediction probability: {pred_proba.max():.3f}")
        print(f"Class probabilities: [Class 0: {pred_proba[0]:.3f}, Class 1: {pred_proba[1]:.3f}]")
        
        print("\nFeature Contributions:")
        for feature, contribution in explanation_list:
            direction = "→ Positive" if contribution > 0 else "→ Negative"
            print(f"  {feature:<35}: {contribution:+.4f} {direction}")
        
        # Create LIME plot
        fig = exp.as_pyplot_figure()
        fig.suptitle(f'LIME Explanation - Instance {instance_idx} - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'lime_explanation_{dataset_name.lower().replace(" ", "_")}_instance_{instance_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return exp
        
    except ImportError:
        print("LIME not available. Install with: pip install lime")
        return None
    except Exception as e:
        print(f"LIME analysis failed: {e}")
        return None


def demonstrate_model_behavior_analysis(clf, X_test, y_test, dataset_name):
    """Demonstrate model behavior analysis."""
    print(f"\n=== Model Behavior Analysis - {dataset_name} ===")
    
    # Get predictions and probabilities
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Prediction confidence distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(y_proba, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.legend()
    
    # Confidence vs Accuracy
    plt.subplot(1, 3, 2)
    confidence_bins = np.linspace(0, 1, 11)
    bin_acc = []
    bin_conf = []
    
    for i in range(len(confidence_bins)-1):
        mask = (y_proba >= confidence_bins[i]) & (y_proba < confidence_bins[i+1])
        if mask.sum() > 0:
            bin_acc.append((y_pred[mask] == y_test[mask]).mean())
            bin_conf.append(y_proba[mask].mean())
    
    plt.plot(bin_conf, bin_acc, 'bo-', label='Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    
    # Prediction distribution by class
    plt.subplot(1, 3, 3)
    plt.hist(y_proba[y_test == 0], bins=20, alpha=0.6, label='Class 0', color='blue')
    plt.hist(y_proba[y_test == 1], bins=20, alpha=0.6, label='Class 1', color='orange')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution by True Class')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'model_behavior_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print model performance summary
    print(f"\nModel Performance Summary:")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  ROC AUC:   {roc_auc:.3f}")


def compare_explanations_across_models(X_train, X_test, y_train, y_test, dataset_name):
    """Compare explanations across different model types."""
    print(f"\n=== Comparing Explanations Across Models - {dataset_name} ===")
    
    models_to_compare = [
        ('Logistic Regression', {'model': 'logistic', 'calibrate': False}),
        ('Random Forest', {'model': 'random_forest', 'calibrate': False}),
        ('XGBoost', {'model': 'xgboost', 'calibrate': False})
    ]
    
    model_explanations = {}
    feature_names = X_train.columns.tolist()
    
    for model_name, params in models_to_compare:
        print(f"\nTraining {model_name}...")
        
        clf = BinaryClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = extract_feature_importance(clf, feature_names)
        if feature_importance:
            model_explanations[model_name] = feature_importance
    
    if model_explanations:
        # Create comparison plot
        plt.figure(figsize=(15, 8))
        
        # Get top features from any model
        all_features = set()
        for explanations in model_explanations.values():
            all_features.update(explanations.keys())
        
        top_features = list(all_features)[:10]
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, explanations in model_explanations.items():
            for feature in top_features:
                importance = explanations.get(feature, 0)
                comparison_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importance
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        pivot_df = comparison_df.pivot(index='Feature', columns='Model', values='Importance')
        ax = pivot_df.plot(kind='bar', figsize=(15, 8))
        plt.title(f'Feature Importance Comparison Across Models - {dataset_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(f'model_comparison_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nFeature Importance Ranking Comparison:")
        for model_name, explanations in model_explanations.items():
            sorted_features = sorted(explanations.items(), key=lambda x: x[1], reverse=True)
            top_5 = [f[0] for f in sorted_features[:5]]
            print(f"{model_name:<20}: {', '.join(top_5)}")


def main():
    """Run comprehensive XAI demonstration."""
    print("MLForge-Binary Explainable AI (XAI) Demo")
    print("=" * 50)
    
    # Example 1: Synthetic Dataset
    print("\n" + "=" * 50)
    print("EXAMPLE 1: SYNTHETIC LOAN APPROVAL DATASET")
    print("=" * 50)
    
    X_synth, y_synth = create_synthetic_dataset()
    X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
        X_synth, y_synth, test_size=0.3, random_state=42, stratify=y_synth
    )
    
    # Train model
    print("\nTraining Random Forest model...")
    clf_synth = BinaryClassifier(
        model='random_forest',
        calibrate=False,
        explain=True
    )
    clf_synth.fit(X_train_synth, y_train_synth)
    
    # Demonstrate various XAI techniques
    demonstrate_feature_importance(clf_synth, X_test_synth, y_test_synth, "Synthetic Dataset")
    demonstrate_shap_analysis(clf_synth, X_train_synth, X_test_synth, "Synthetic Dataset")
    demonstrate_lime_analysis(clf_synth, X_train_synth, X_test_synth, "Synthetic Dataset", instance_idx=0)
    demonstrate_model_behavior_analysis(clf_synth, X_test_synth, y_test_synth, "Synthetic Dataset")
    
    # Example 2: Real-world Dataset
    print("\n" + "=" * 50)
    print("EXAMPLE 2: BREAST CANCER DIAGNOSIS DATASET")
    print("=" * 50)
    
    X_cancer, y_cancer = real_world_dataset()
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer
    )
    
    # Train model
    print("\nTraining XGBoost model...")
    clf_cancer = BinaryClassifier(
        model='xgboost',
        calibrate=False,
        explain=True
    )
    clf_cancer.fit(X_train_cancer, y_train_cancer)
    
    # Demonstrate various XAI techniques
    demonstrate_feature_importance(clf_cancer, X_test_cancer, y_test_cancer, "Cancer Dataset")
    demonstrate_shap_analysis(clf_cancer, X_train_cancer, X_test_cancer, "Cancer Dataset")
    demonstrate_lime_analysis(clf_cancer, X_train_cancer, X_test_cancer, "Cancer Dataset", instance_idx=5)
    demonstrate_model_behavior_analysis(clf_cancer, X_test_cancer, y_test_cancer, "Cancer Dataset")
    
    # Example 3: Model Comparison
    print("\n" + "=" * 50)
    print("EXAMPLE 3: COMPARING EXPLANATIONS ACROSS MODELS")
    print("=" * 50)
    
    compare_explanations_across_models(X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer, "Cancer Dataset")
    
    print("\n" + "=" * 50)
    print("XAI DEMO COMPLETED!")
    print("=" * 50)
    print("\nKey Takeaways:")
    print("1. Feature importance varies across different model types")
    print("2. SHAP provides global and local explanations with statistical guarantees")
    print("3. LIME offers intuitive local explanations for individual predictions")
    print("4. Model behavior analysis helps understand prediction confidence and calibration")
    print("5. Different models may rely on different features for similar performance")
    print("\nGenerated plots saved as PNG files in the current directory.")


if __name__ == "__main__":
    main()