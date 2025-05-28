"""
XAI Demo for Mixed Data Types - MLForge-Binary

This example demonstrates explainability features with both categorical and continuous variables,
showing how MLForge-Binary handles mixed data types and provides interpretable explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

from mlforge_binary import BinaryClassifier

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def create_mixed_dataset():
    """Create a dataset with both categorical and continuous variables."""
    print("Creating mixed dataset with categorical and continuous variables...")
    
    # Generate base continuous features
    X_continuous, y = make_classification(
        n_samples=1500,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=42
    )
    
    # Create meaningful continuous feature names
    continuous_features = ['age', 'income', 'credit_score', 'debt_ratio', 'savings_amount', 'employment_years']
    X_cont_df = pd.DataFrame(X_continuous, columns=continuous_features)
    
    # Scale continuous features to meaningful ranges
    X_cont_df['age'] = (X_cont_df['age'] * 15 + 45).clip(18, 80).round().astype(int)
    X_cont_df['income'] = (X_cont_df['income'] * 30000 + 50000).clip(20000, 200000).round().astype(int)
    X_cont_df['credit_score'] = (X_cont_df['credit_score'] * 150 + 650).clip(300, 850).round().astype(int)
    X_cont_df['debt_ratio'] = (X_cont_df['debt_ratio'] * 0.3 + 0.3).clip(0, 1).round(3)
    X_cont_df['savings_amount'] = (X_cont_df['savings_amount'] * 50000 + 10000).clip(0, 200000).round().astype(int)
    X_cont_df['employment_years'] = (X_cont_df['employment_years'] * 10 + 5).clip(0, 40).round(1)
    
    # Create categorical features
    np.random.seed(42)
    n_samples = len(X_cont_df)
    
    # Education level (ordinal)
    education_map = {0: 'High School', 1: 'Associate', 2: 'Bachelor', 3: 'Master', 4: 'PhD'}
    education_probs = [0.3, 0.2, 0.3, 0.15, 0.05]
    education = np.random.choice(list(education_map.keys()), n_samples, p=education_probs)
    X_cont_df['education'] = [education_map[x] for x in education]
    
    # Employment type (nominal)
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Contract', 'Unemployed']
    employment_probs = [0.6, 0.15, 0.15, 0.08, 0.02]
    X_cont_df['employment_type'] = np.random.choice(employment_types, n_samples, p=employment_probs)
    
    # Location type (nominal)
    locations = ['Urban', 'Suburban', 'Rural']
    location_probs = [0.4, 0.45, 0.15]
    X_cont_df['location'] = np.random.choice(locations, n_samples, p=location_probs)
    
    # Housing status (nominal)
    housing_types = ['Own', 'Rent', 'Mortgage', 'Family']
    housing_probs = [0.3, 0.35, 0.25, 0.1]
    X_cont_df['housing_status'] = np.random.choice(housing_types, n_samples, p=housing_probs)
    
    # Marital status (nominal)
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    marital_probs = [0.35, 0.45, 0.15, 0.05]
    X_cont_df['marital_status'] = np.random.choice(marital_statuses, n_samples, p=marital_probs)
    
    # Add some logical relationships to make it more realistic
    # Higher education tends to correlate with higher income
    mask_high_edu = X_cont_df['education'].isin(['Master', 'PhD'])
    X_cont_df.loc[mask_high_edu, 'income'] *= 1.3
    X_cont_df.loc[mask_high_edu, 'income'] = X_cont_df.loc[mask_high_edu, 'income'].clip(upper=200000)
    
    # Self-employed tend to have more variable income
    mask_self_emp = X_cont_df['employment_type'] == 'Self-employed'
    X_cont_df.loc[mask_self_emp, 'income'] *= np.random.uniform(0.7, 1.5, mask_self_emp.sum())
    
    # Create target variable with logical relationships
    y_series = pd.Series(y, name='loan_approved')
    
    # Adjust target based on categorical features (make it more realistic)
    adjustment = np.zeros(n_samples)
    
    # Education effect
    edu_effects = {'High School': -0.1, 'Associate': 0, 'Bachelor': 0.1, 'Master': 0.2, 'PhD': 0.15}
    for edu, effect in edu_effects.items():
        mask = X_cont_df['education'] == edu
        adjustment[mask] += effect
    
    # Employment effect
    emp_effects = {'Full-time': 0.2, 'Part-time': -0.1, 'Self-employed': 0, 'Contract': -0.05, 'Unemployed': -0.5}
    for emp, effect in emp_effects.items():
        mask = X_cont_df['employment_type'] == emp
        adjustment[mask] += effect
    
    # Housing effect
    housing_effects = {'Own': 0.15, 'Mortgage': 0.1, 'Rent': 0, 'Family': -0.05}
    for housing, effect in housing_effects.items():
        mask = X_cont_df['housing_status'] == housing
        adjustment[mask] += effect
    
    # Apply adjustments to target
    prob_adjustment = 1 / (1 + np.exp(-adjustment))
    random_threshold = np.random.random(n_samples)
    y_adjusted = ((y + prob_adjustment) > random_threshold).astype(int)
    y_series = pd.Series(y_adjusted, name='loan_approved')
    
    print(f"Dataset created: {X_cont_df.shape[0]} samples, {X_cont_df.shape[1]} features")
    print("Features:")
    print("  Continuous: age, income, credit_score, debt_ratio, savings_amount, employment_years")
    print("  Categorical: education, employment_type, location, housing_status, marital_status")
    print("Target: loan approval prediction")
    print(f"Class distribution: {np.bincount(y_series)}")
    
    return X_cont_df, y_series


def analyze_data_types(X_df):
    """Analyze and display information about data types."""
    print("\n=== Data Type Analysis ===")
    
    # Identify categorical and continuous columns
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    continuous_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Continuous variables ({len(continuous_cols)}):")
    for col in continuous_cols:
        print(f"  {col}: min={X_df[col].min():.2f}, max={X_df[col].max():.2f}, mean={X_df[col].mean():.2f}")
    
    print(f"\nCategorical variables ({len(categorical_cols)}):")
    for col in categorical_cols:
        unique_vals = X_df[col].unique()
        print(f"  {col}: {len(unique_vals)} categories -> {list(unique_vals)}")
    
    return categorical_cols, continuous_cols


def plot_data_distribution(X_df, y_series, categorical_cols, continuous_cols):
    """Plot distribution of variables by target class."""
    print("\n=== Data Distribution Visualization ===")
    
    # Plot continuous variables
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(continuous_cols):
        axes[i].hist(X_df[y_series == 0][col], alpha=0.6, label='Not Approved', bins=20, color='red')
        axes[i].hist(X_df[y_series == 1][col], alpha=0.6, label='Approved', bins=20, color='green')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('continuous_variables_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot categorical variables
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(categorical_cols):
        # Create crosstab
        ct = pd.crosstab(X_df[col], y_series, normalize='index')
        ct.plot(kind='bar', ax=axes[i], color=['red', 'green'], alpha=0.7)
        axes[i].set_title(f'{col} Approval Rate')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Proportion')
        axes[i].legend(['Not Approved', 'Approved'])
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('categorical_variables_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def extract_feature_importance_mixed(clf, feature_names):
    """Extract feature importance from trained model for mixed data."""
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
            # Get actual feature names after preprocessing
            try:
                processed_feature_names = clf.get_feature_names_out()
                feature_importance = dict(zip(processed_feature_names, importance_values))
            except:
                # Fallback to original feature names
                feature_importance = dict(zip(feature_names, importance_values))
        elif hasattr(clf.model_, 'coef_'):
            # For linear models
            coef_values = clf.model_.coef_[0] if clf.model_.coef_.ndim > 1 else clf.model_.coef_
            try:
                processed_feature_names = clf.get_feature_names_out()
                feature_importance = dict(zip(processed_feature_names, np.abs(coef_values)))
            except:
                feature_importance = dict(zip(feature_names, np.abs(coef_values)))
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
    
    return feature_importance


def analyze_categorical_vs_continuous_importance(feature_importance, original_features, categorical_cols, continuous_cols):
    """Analyze importance of categorical vs continuous features."""
    print("\n=== Categorical vs Continuous Feature Importance ===")
    
    categorical_importance = {}
    continuous_importance = {}
    
    for feature, importance in feature_importance.items():
        # Try to map back to original feature
        base_feature = feature.split('_')[0] if '_' in feature else feature
        
        # Check if it's a categorical feature (might have encoding suffixes)
        is_categorical = False
        for cat_col in categorical_cols:
            if cat_col in feature or feature.startswith(cat_col):
                if cat_col not in categorical_importance:
                    categorical_importance[cat_col] = 0
                categorical_importance[cat_col] += importance
                is_categorical = True
                break
        
        # If not categorical, check if it's continuous
        if not is_categorical:
            for cont_col in continuous_cols:
                if cont_col in feature or feature == cont_col:
                    continuous_importance[cont_col] = importance
                    break
    
    print("Top Categorical Features:")
    sorted_cat = sorted(categorical_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_cat[:5], 1):
        print(f"  {i}. {feature:<20}: {importance:.4f}")
    
    print("\nTop Continuous Features:")
    sorted_cont = sorted(continuous_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_cont[:5], 1):
        print(f"  {i}. {feature:<20}: {importance:.4f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if categorical_importance:
        cat_features = list(categorical_importance.keys())
        cat_importances = list(categorical_importance.values())
        plt.barh(range(len(cat_features)), cat_importances, color='skyblue')
        plt.yticks(range(len(cat_features)), cat_features)
        plt.xlabel('Total Importance')
        plt.title('Categorical Feature Importance')
        plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    if continuous_importance:
        cont_features = list(continuous_importance.keys())
        cont_importances = list(continuous_importance.values())
        plt.barh(range(len(cont_features)), cont_importances, color='lightcoral')
        plt.yticks(range(len(cont_features)), cont_features)
        plt.xlabel('Importance')
        plt.title('Continuous Feature Importance')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('categorical_vs_continuous_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return categorical_importance, continuous_importance


def demonstrate_mixed_data_shap(clf, X_train, X_test, categorical_cols, continuous_cols, dataset_name):
    """Demonstrate SHAP analysis for mixed data types."""
    print(f"\n=== SHAP Analysis for Mixed Data - {dataset_name} ===")
    
    try:
        import shap
        
        # Get preprocessed data for SHAP analysis
        print("Getting preprocessed data for SHAP analysis...")
        X_train_processed = clf.preprocessor_.transform(X_train)
        X_test_processed = clf.preprocessor_.transform(X_test)
        
        # Get feature names after preprocessing
        try:
            processed_feature_names = clf.get_feature_names_out()
        except:
            processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
        
        # Create wrapper function that works with preprocessed data
        def predict_proba_wrapper(X):
            # X is already preprocessed numerical data
            return clf.model_.predict_proba(X)
        
        # Create SHAP explainer with preprocessed data
        print("Creating SHAP explainer for preprocessed data...")
        explainer = shap.Explainer(predict_proba_wrapper, X_train_processed[:100])
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer(X_test_processed[:50])
        
        # SHAP summary plot
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        
        # Create DataFrame for better labels
        X_test_processed_df = pd.DataFrame(X_test_processed[:50], columns=processed_feature_names)
        
        shap.summary_plot(shap_values[:, :, 1], X_test_processed_df, show=False, max_display=15)
        plt.title(f'SHAP Summary Plot - {dataset_name} (Preprocessed Features)')
        plt.tight_layout()
        plt.savefig(f'shap_mixed_data_summary_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze SHAP values by original feature type
        feature_importance_shap = np.abs(shap_values[:, :, 1]).mean(0)
        
        # Map processed features back to original feature types
        cat_shap_importance = {}
        cont_shap_importance = {}
        
        for i, processed_feature in enumerate(processed_feature_names):
            importance = feature_importance_shap[i]
            
            # Try to map back to original feature
            mapped_to_original = False
            for cat_col in categorical_cols:
                if cat_col in processed_feature.lower():
                    if cat_col not in cat_shap_importance:
                        cat_shap_importance[cat_col] = 0
                    cat_shap_importance[cat_col] += importance
                    mapped_to_original = True
                    break
            
            if not mapped_to_original:
                for cont_col in continuous_cols:
                    if cont_col in processed_feature.lower():
                        cont_shap_importance[cont_col] = importance
                        break
        
        print("\nTop Features by SHAP Importance (grouped by original feature):")
        if cat_shap_importance:
            print("Categorical Features:")
            sorted_cat_shap = sorted(cat_shap_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_cat_shap[:5], 1):
                print(f"  {i}. {feature:<20}: {importance:.4f}")
        
        if cont_shap_importance:
            print("Continuous Features:")
            sorted_cont_shap = sorted(cont_shap_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_cont_shap[:5], 1):
                print(f"  {i}. {feature:<20}: {importance:.4f}")
        
        return shap_values
        
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
        return None
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None


def demonstrate_mixed_data_lime(clf, X_train, X_test, y_train, categorical_cols, continuous_cols, instance_idx=0):
    """Demonstrate LIME analysis for mixed data types."""
    print(f"\n=== LIME Analysis for Mixed Data ===")
    
    try:
        import lime
        import lime.lime_tabular
        
        print("Note: LIME analysis works better with continuous variables.")
        print("For mixed data, we'll focus on a simpler explanation...")
        
        # Show instance details first
        print(f"\nInstance {instance_idx} details:")
        for col in X_test.columns:
            value = X_test.iloc[instance_idx][col]
            data_type = "Categorical" if col in categorical_cols else "Continuous"
            print(f"  {col} ({data_type}): {value}")
        
        # Get prediction for this instance
        instance_df = X_test.iloc[[instance_idx]]
        pred_class = clf.predict(instance_df)[0]
        pred_proba = clf.predict_proba(instance_df)[0]
        print(f"\nPredicted class: {pred_class}")
        print(f"Prediction probability: {pred_proba.max():.3f}")
        print(f"Class probabilities: [Not Approved: {pred_proba[0]:.3f}, Approved: {pred_proba[1]:.3f}]")
        
        # Try LIME with just continuous features for demonstration
        continuous_data = X_train[continuous_cols].values
        continuous_test_data = X_test[continuous_cols].values
        
        def predict_proba_continuous_wrapper(X):
            # Create full DataFrame with categorical features from the original instance
            full_instances = []
            for x_cont in X:
                # Start with the original instance
                full_instance = X_test.iloc[instance_idx].copy()
                # Replace continuous features with the perturbed values
                for i, col in enumerate(continuous_cols):
                    full_instance[col] = x_cont[i]
                full_instances.append(full_instance)
            
            full_df = pd.DataFrame(full_instances)
            return clf.predict_proba(full_df)
        
        # Create LIME explainer for continuous features only
        print("Creating LIME explainer for continuous features...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            continuous_data,
            feature_names=continuous_cols,
            class_names=['Not Approved', 'Approved'],
            mode='classification'
        )
        
        # Explain the continuous features for this instance
        continuous_instance = X_test.iloc[instance_idx][continuous_cols].values
        
        exp = explainer.explain_instance(
            continuous_instance, 
            predict_proba_continuous_wrapper,
            num_features=len(continuous_cols)
        )
        
        # Get explanation as list
        explanation_list = exp.as_list()
        
        print(f"\nLIME Explanation for Continuous Features (Instance {instance_idx}):")
        print("Feature Contributions:")
        for feature, contribution in explanation_list:
            direction = "→ Positive" if contribution > 0 else "→ Negative"
            print(f"  {feature:<25}: {contribution:+.4f} {direction}")
        
        # Create LIME plot
        fig = exp.as_pyplot_figure()
        fig.suptitle(f'LIME Explanation - Continuous Features - Instance {instance_idx}')
        plt.tight_layout()
        plt.savefig(f'lime_mixed_data_continuous_instance_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show categorical feature impact by comparison
        print(f"\nCategorical Feature Values for Instance {instance_idx}:")
        for col in categorical_cols:
            value = X_test.iloc[instance_idx][col]
            
            # Count how many approved vs not approved have this categorical value
            same_cat_mask = X_train[col] == value
            if same_cat_mask.sum() > 0:
                approval_rate = y_train[same_cat_mask].mean()
                print(f"  {col} = '{value}': {approval_rate:.1%} approval rate in training data")
        
        return exp
        
    except ImportError:
        print("LIME not available. Install with: pip install lime")
        return None
    except Exception as e:
        print(f"LIME analysis failed: {e}")
        return None


def main():
    """Run comprehensive mixed data XAI demonstration."""
    print("MLForge-Binary XAI Demo: Mixed Data Types (Categorical + Continuous)")
    print("=" * 80)
    
    # Create mixed dataset
    X_mixed, y_mixed = create_mixed_dataset()
    
    # Analyze data types
    categorical_cols, continuous_cols = analyze_data_types(X_mixed)
    
    # Plot data distributions
    plot_data_distribution(X_mixed, y_mixed, categorical_cols, continuous_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_mixed, y_mixed, test_size=0.3, random_state=42, stratify=y_mixed
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models with different algorithms to show how they handle mixed data
    models_to_test = [
        ('Random Forest', {'model': 'random_forest', 'calibrate': False}),
        ('XGBoost', {'model': 'xgboost', 'calibrate': False}),
        ('Logistic Regression', {'model': 'logistic', 'calibrate': False})
    ]
    
    for model_name, params in models_to_test:
        print("\n" + "=" * 80)
        print(f"TESTING {model_name.upper()}")
        print("=" * 80)
        
        # Train model
        print(f"\nTraining {model_name}...")
        clf = BinaryClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"{model_name} Performance:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  ROC AUC:   {roc_auc:.3f}")
        
        # Feature importance analysis
        print(f"\n=== Feature Importance Analysis - {model_name} ===")
        feature_names = X_train.columns.tolist()
        feature_importance = extract_feature_importance_mixed(clf, feature_names)
        
        if feature_importance:
            print("\nTop 10 Most Important Features:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"{i:2d}. {feature:<30}: {importance:.4f}")
            
            # Analyze by data type
            analyze_categorical_vs_continuous_importance(
                feature_importance, feature_names, categorical_cols, continuous_cols
            )
        
        # SHAP analysis
        demonstrate_mixed_data_shap(clf, X_train, X_test, categorical_cols, continuous_cols, f"{model_name}")
        
        # LIME analysis for one instance
        demonstrate_mixed_data_lime(clf, X_train, X_test, y_train, categorical_cols, continuous_cols, instance_idx=0)
        
        print(f"\n{model_name} analysis completed!")
    
    print("\n" + "=" * 80)
    print("MIXED DATA XAI DEMO COMPLETED!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. MLForge-Binary automatically handles categorical variables through preprocessing")
    print("2. Feature importance can be analyzed separately for categorical vs continuous variables")
    print("3. SHAP and LIME work seamlessly with mixed data types")
    print("4. Different models may emphasize different types of features")
    print("5. Categorical features often get expanded into multiple binary features")
    print("\nGenerated plots and explanations saved as PNG files.")


if __name__ == "__main__":
    main()