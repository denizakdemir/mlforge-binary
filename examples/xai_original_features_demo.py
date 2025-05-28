"""
XAI Demo with Original Feature Mapping - MLForge-Binary

This example demonstrates how to map feature importance and SHAP values back to 
the original categorical variables and their specific categories, rather than 
showing dummy/encoded variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from mlforge_binary import BinaryClassifier

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def create_interpretable_dataset():
    """Create a dataset with both categorical and continuous variables for interpretation."""
    print("Creating interpretable dataset with mixed variable types...")
    
    # Generate base continuous features
    X_continuous, y = make_classification(
        n_samples=1200,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        flip_y=0.03,
        random_state=42
    )
    
    # Create meaningful continuous feature names and scale them
    continuous_features = ['age', 'annual_income', 'credit_score', 'debt_to_income', 'account_balance']
    X_cont_df = pd.DataFrame(X_continuous, columns=continuous_features)
    
    # Scale continuous features to realistic ranges
    X_cont_df['age'] = (X_cont_df['age'] * 12 + 35).clip(22, 70).round().astype(int)
    X_cont_df['annual_income'] = (X_cont_df['annual_income'] * 25000 + 45000).clip(25000, 150000).round().astype(int)
    X_cont_df['credit_score'] = (X_cont_df['credit_score'] * 120 + 650).clip(300, 850).round().astype(int)
    X_cont_df['debt_to_income'] = (X_cont_df['debt_to_income'] * 0.4 + 0.2).clip(0, 0.8).round(3)
    X_cont_df['account_balance'] = (X_cont_df['account_balance'] * 30000 + 5000).clip(0, 100000).round().astype(int)
    
    # Create categorical features with clear categories
    np.random.seed(42)
    n_samples = len(X_cont_df)
    
    # Education level (ordinal with clear impact)
    education_levels = ['High School', 'Some College', 'Bachelor Degree', 'Master Degree', 'PhD']
    education_probs = [0.25, 0.20, 0.35, 0.15, 0.05]
    X_cont_df['education'] = np.random.choice(education_levels, n_samples, p=education_probs)
    
    # Employment status (nominal with clear impact)
    employment_status = ['Full-time', 'Part-time', 'Self-employed', 'Student', 'Unemployed']
    employment_probs = [0.60, 0.15, 0.15, 0.05, 0.05]
    X_cont_df['employment_status'] = np.random.choice(employment_status, n_samples, p=employment_probs)
    
    # Home ownership (nominal)
    home_ownership = ['Own', 'Rent', 'Mortgage', 'Family']
    home_probs = [0.25, 0.40, 0.30, 0.05]
    X_cont_df['home_ownership'] = np.random.choice(home_ownership, n_samples, p=home_probs)
    
    # Geographic region (nominal)
    regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
    region_probs = [0.20, 0.20, 0.20, 0.20, 0.20]
    X_cont_df['region'] = np.random.choice(regions, n_samples, p=region_probs)
    
    # Create more realistic target with logical relationships
    y_series = pd.Series(y, name='loan_approved')
    
    # Adjust target based on logical relationships
    prob_adjustments = np.zeros(n_samples)
    
    # Education impact
    edu_impact = {'High School': -0.2, 'Some College': -0.1, 'Bachelor Degree': 0.1, 
                  'Master Degree': 0.2, 'PhD': 0.15}
    for edu, impact in edu_impact.items():
        mask = X_cont_df['education'] == edu
        prob_adjustments[mask] += impact
    
    # Employment impact
    emp_impact = {'Full-time': 0.3, 'Part-time': 0.0, 'Self-employed': 0.1, 
                  'Student': -0.1, 'Unemployed': -0.5}
    for emp, impact in emp_impact.items():
        mask = X_cont_df['employment_status'] == emp
        prob_adjustments[mask] += impact
    
    # Home ownership impact
    home_impact = {'Own': 0.2, 'Mortgage': 0.1, 'Rent': 0.0, 'Family': -0.1}
    for home, impact in home_impact.items():
        mask = X_cont_df['home_ownership'] == home
        prob_adjustments[mask] += impact
    
    # Apply adjustments
    adjusted_probs = 1 / (1 + np.exp(-prob_adjustments))
    random_adjustments = np.random.random(n_samples) * 0.3
    final_probs = (y + adjusted_probs + random_adjustments) > 1.2
    y_series = pd.Series(final_probs.astype(int), name='loan_approved')
    
    print(f"Dataset created: {X_cont_df.shape[0]} samples, {X_cont_df.shape[1]} features")
    print("Continuous features: age, annual_income, credit_score, debt_to_income, account_balance")
    print("Categorical features: education, employment_status, home_ownership, region")
    print(f"Target distribution: {np.bincount(y_series)} (Not Approved, Approved)")
    
    return X_cont_df, y_series


def create_feature_mapping(X_original, X_processed_feature_names):
    """Create mapping from processed features back to original features."""
    print("\n=== Creating Feature Mapping ===")
    
    # Identify original feature types
    categorical_cols = X_original.select_dtypes(include=['object']).columns.tolist()
    continuous_cols = X_original.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create mapping dictionary
    feature_mapping = {
        'categorical_to_processed': defaultdict(list),
        'continuous_to_processed': {},
        'processed_to_original': {},
        'categorical_categories': {}
    }
    
    # Store categorical categories
    for col in categorical_cols:
        feature_mapping['categorical_categories'][col] = sorted(X_original[col].unique())
    
    # Map processed features back to original
    for processed_feature in X_processed_feature_names:
        processed_lower = processed_feature.lower()
        
        # Check if it maps to a categorical feature
        mapped = False
        for cat_col in categorical_cols:
            if cat_col.lower() in processed_lower:
                feature_mapping['categorical_to_processed'][cat_col].append(processed_feature)
                feature_mapping['processed_to_original'][processed_feature] = cat_col
                mapped = True
                break
        
        # If not categorical, check continuous
        if not mapped:
            for cont_col in continuous_cols:
                if cont_col.lower() in processed_lower or processed_lower == cont_col.lower():
                    feature_mapping['continuous_to_processed'][cont_col] = processed_feature
                    feature_mapping['processed_to_original'][processed_feature] = cont_col
                    mapped = True
                    break
        
        if not mapped:
            print(f"Warning: Could not map processed feature '{processed_feature}' to original feature")
    
    print("Feature mapping created:")
    print(f"  Continuous features: {len(continuous_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    
    for cat_col in categorical_cols:
        n_processed = len(feature_mapping['categorical_to_processed'][cat_col])
        categories = feature_mapping['categorical_categories'][cat_col]
        print(f"    {cat_col}: {n_processed} processed features, categories: {categories}")
    
    return feature_mapping, categorical_cols, continuous_cols


def extract_original_feature_importance(clf, feature_mapping, categorical_cols, continuous_cols):
    """Extract feature importance and map back to original features."""
    print("\n=== Extracting Original Feature Importance ===")
    
    # Get processed feature importance - try direct access first
    processed_importance = {}
    try:
        print(f"Model type: {type(clf.model_)}")
        print(f"Has feature_importances_: {hasattr(clf.model_, 'feature_importances_')}")
        
        # Access the actual model - clf.model_ is the ModelWrapper, clf.model_.model_ is the sklearn model
        if hasattr(clf.model_, 'model_') and clf.model_.model_ is not None:
            actual_model = clf.model_.model_
            wrapper_type = "ModelWrapper"
            print(f"Found {wrapper_type}, actual model type: {type(actual_model)}")
            
            # Check if sklearn model is further wrapped (e.g., in CalibratedClassifierCV)
            if hasattr(actual_model, 'calibrated_classifiers_'):
                # CalibratedClassifierCV case
                actual_model = actual_model.calibrated_classifiers_[0].estimator
                wrapper_type = "ModelWrapper -> CalibratedClassifierCV"
                print(f"Found {wrapper_type}, final model type: {type(actual_model)}")
            # Don't unwrap Random Forest/Extra Trees to individual trees - we want the ensemble model
            elif hasattr(actual_model, 'estimator') and not hasattr(actual_model, 'feature_importances_'):
                actual_model = actual_model.estimator
                wrapper_type = "ModelWrapper -> estimator wrapper"
                print(f"Found {wrapper_type}, final model type: {type(actual_model)}")
            elif hasattr(actual_model, 'base_estimator') and not hasattr(actual_model, 'feature_importances_'):
                actual_model = actual_model.base_estimator
                wrapper_type = "ModelWrapper -> base_estimator wrapper"
                print(f"Found {wrapper_type}, final model type: {type(actual_model)}")
        else:
            actual_model = clf.model_
            wrapper_type = "direct"
            print(f"Direct model access, type: {type(actual_model)}")
        
        if hasattr(actual_model, 'feature_importances_'):
            importance_values = actual_model.feature_importances_
            print(f"Feature importances shape: {importance_values.shape}")
            
            # Try to get feature names
            try:
                processed_feature_names = clf.get_feature_names_out()
                print(f"Got feature names: {len(processed_feature_names)} names")
            except Exception as e:
                print(f"get_feature_names_out failed: {e}")
                # Fallback if get_feature_names_out doesn't work
                n_features = len(importance_values)
                processed_feature_names = [f'feature_{i}' for i in range(n_features)]
                print(f"Using fallback feature names: {len(processed_feature_names)} names")
            
            processed_importance = dict(zip(processed_feature_names, importance_values))
            print(f"Extracted {len(processed_importance)} feature importances from model")
            
            # Debug: print first few feature names and importances
            print("First 5 features and importances:")
            for i, (name, imp) in enumerate(list(processed_importance.items())[:5]):
                print(f"  {name}: {imp:.4f}")
            
        elif hasattr(actual_model, 'coef_'):
            try:
                processed_feature_names = clf.get_feature_names_out()
            except:
                coef_values = actual_model.coef_[0] if actual_model.coef_.ndim > 1 else actual_model.coef_
                processed_feature_names = [f'feature_{i}' for i in range(len(coef_values))]
            
            coef_values = actual_model.coef_[0] if actual_model.coef_.ndim > 1 else actual_model.coef_
            processed_importance = dict(zip(processed_feature_names, np.abs(coef_values)))
            print(f"Extracted {len(processed_importance)} coefficients from linear model")
        else:
            # Try explain() method as fallback
            try:
                explanations = clf.explain()
                if 'feature_importance' in explanations:
                    processed_importance = explanations['feature_importance']
                    print(f"Extracted {len(processed_importance)} feature importances from explain() method")
            except Exception as explain_error:
                print(f"explain() method failed: {explain_error}")
                
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return {}, {}, {}
    
    if not processed_importance:
        print("No feature importance available")
        return {}, {}, {}
    
    # Map back to original features
    original_categorical_importance = {}
    original_continuous_importance = {}
    categorical_detailed_importance = {}
    
    # Aggregate categorical feature importance
    for cat_col in categorical_cols:
        total_importance = 0
        detailed_importance = {}
        
        for processed_feature in feature_mapping['categorical_to_processed'][cat_col]:
            if processed_feature in processed_importance:
                importance = processed_importance[processed_feature]
                total_importance += importance
                
                # Try to identify which category this processed feature represents
                category_name = None
                for category in feature_mapping['categorical_categories'][cat_col]:
                    if category.lower().replace(' ', '_') in processed_feature.lower():
                        category_name = category
                        break
                
                if category_name:
                    detailed_importance[category_name] = importance
                else:
                    # If we can't identify the category, use the processed feature name
                    detailed_importance[processed_feature] = importance
        
        original_categorical_importance[cat_col] = total_importance
        categorical_detailed_importance[cat_col] = detailed_importance
    
    # Map continuous feature importance
    for cont_col in continuous_cols:
        if cont_col in feature_mapping['continuous_to_processed']:
            processed_feature = feature_mapping['continuous_to_processed'][cont_col]
            if processed_feature in processed_importance:
                original_continuous_importance[cont_col] = processed_importance[processed_feature]
    
    print("Original feature importance extracted:")
    print("\nCategorical Features (total importance):")
    sorted_cat = sorted(original_categorical_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_cat, 1):
        print(f"  {i}. {feature:<20}: {importance:.4f}")
    
    print("\nContinuous Features:")
    sorted_cont = sorted(original_continuous_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_cont, 1):
        print(f"  {i}. {feature:<20}: {importance:.4f}")
    
    return original_categorical_importance, original_continuous_importance, categorical_detailed_importance


def plot_original_feature_importance(categorical_importance, continuous_importance, categorical_detailed, dataset_name):
    """Plot feature importance for original features."""
    print("\n=== Plotting Original Feature Importance ===")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall categorical importance
    if categorical_importance:
        ax1 = axes[0, 0]
        cat_features = list(categorical_importance.keys())
        cat_importances = list(categorical_importance.values())
        
        bars = ax1.barh(range(len(cat_features)), cat_importances, color='skyblue')
        ax1.set_yticks(range(len(cat_features)))
        ax1.set_yticklabels(cat_features)
        ax1.set_xlabel('Total Importance')
        ax1.set_title('Categorical Features - Total Importance')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + max(cat_importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
    
    # Plot 2: Overall continuous importance
    if continuous_importance:
        ax2 = axes[0, 1]
        cont_features = list(continuous_importance.keys())
        cont_importances = list(continuous_importance.values())
        
        bars = ax2.barh(range(len(cont_features)), cont_importances, color='lightcoral')
        ax2.set_yticks(range(len(cont_features)))
        ax2.set_yticklabels(cont_features)
        ax2.set_xlabel('Importance')
        ax2.set_title('Continuous Features - Importance')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + max(cont_importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
    
    # Plot 3: Detailed categorical importance (category-level)
    if categorical_detailed:
        ax3 = axes[1, 0]
        
        # Flatten category-level importance
        all_categories = []
        all_category_importance = []
        all_colors = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(categorical_detailed)))
        
        for i, (cat_feature, category_dict) in enumerate(categorical_detailed.items()):
            for category, importance in category_dict.items():
                all_categories.append(f"{cat_feature}: {category}")
                all_category_importance.append(importance)
                all_colors.append(colors[i])
        
        # Sort by importance
        sorted_data = sorted(zip(all_categories, all_category_importance, all_colors), 
                           key=lambda x: x[1], reverse=True)
        
        if sorted_data:
            categories, importances, colors = zip(*sorted_data[:15])  # Top 15
            
            bars = ax3.barh(range(len(categories)), importances, color=colors)
            ax3.set_yticks(range(len(categories)))
            ax3.set_yticklabels(categories)
            ax3.set_xlabel('Importance')
            ax3.set_title('Categorical Feature Categories - Detailed Importance')
            ax3.invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # Plot 4: Combined comparison
    ax4 = axes[1, 1]
    
    # Combine all features for comparison
    all_features = []
    all_importance = []
    all_types = []
    
    for feature, importance in categorical_importance.items():
        all_features.append(feature)
        all_importance.append(importance)
        all_types.append('Categorical')
    
    for feature, importance in continuous_importance.items():
        all_features.append(feature)
        all_importance.append(importance)
        all_types.append('Continuous')
    
    if all_features:
        # Sort by importance
        sorted_combined = sorted(zip(all_features, all_importance, all_types), 
                               key=lambda x: x[1], reverse=True)
        features, importances, types = zip(*sorted_combined)
        
        colors = ['skyblue' if t == 'Categorical' else 'lightcoral' for t in types]
        bars = ax4.barh(range(len(features)), importances, color=colors)
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels(features)
        ax4.set_xlabel('Importance')
        ax4.set_title('All Features - Combined Importance')
        ax4.invert_yaxis()
        
        # Add legend
        categorical_patch = plt.Rectangle((0, 0), 1, 1, fc='skyblue', label='Categorical')
        continuous_patch = plt.Rectangle((0, 0), 1, 1, fc='lightcoral', label='Continuous')
        ax4.legend(handles=[categorical_patch, continuous_patch], loc='lower right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'original_feature_importance_{dataset_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def analyze_categorical_category_impact(X_train, y_train, categorical_cols):
    """Analyze the impact of each category within categorical variables."""
    print("\n=== Analyzing Categorical Category Impact ===")
    
    category_impact_analysis = {}
    
    for cat_col in categorical_cols:
        print(f"\n{cat_col.upper()}:")
        category_analysis = {}
        
        for category in sorted(X_train[cat_col].unique()):
            mask = X_train[cat_col] == category
            if mask.sum() > 0:
                approval_rate = y_train[mask].mean()
                count = mask.sum()
                category_analysis[category] = {
                    'approval_rate': approval_rate,
                    'count': count,
                    'percentage_of_data': count / len(X_train) * 100
                }
                
                print(f"  {category:<20}: {approval_rate:.1%} approval rate "
                      f"({count:4d} samples, {count/len(X_train)*100:.1f}% of data)")
        
        category_impact_analysis[cat_col] = category_analysis
    
    return category_impact_analysis


def create_original_waterfall_plot(clf, X_train, X_test, y_train, y_test, feature_mapping, categorical_cols, continuous_cols, dataset_name, instance_idx=0):
    """Create waterfall plot for a specific instance using original features."""
    print(f"\n=== SHAP Waterfall Plot - Original Features - {dataset_name} ===")
    
    try:
        import shap
        
        # Get preprocessed data
        X_train_processed = clf.preprocessor_.transform(X_train)
        X_test_processed = clf.preprocessor_.transform(X_test)
        
        # Create SHAP explainer
        def predict_proba_wrapper(X):
            return clf.model_.predict_proba(X)
        
        print("Creating SHAP explainer for waterfall plot...")
        explainer = shap.Explainer(predict_proba_wrapper, X_train_processed[:100])
        
        # Get SHAP values for the specific instance
        print(f"Calculating SHAP values for instance {instance_idx}...")
        instance_processed = X_test_processed[instance_idx:instance_idx+1]
        shap_values = explainer(instance_processed)
        
        # Get processed feature names
        try:
            processed_feature_names = clf.get_feature_names_out()
        except:
            processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
        
        # Get the actual SHAP values array
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = shap_values
        
        # Handle SHAP values format
        if len(shap_array.shape) == 3:
            shap_values_to_use = shap_array[0, :, 1]  # Use positive class for first instance
        elif len(shap_array.shape) == 2:
            shap_values_to_use = shap_array[0, :]  # First instance
        else:
            print(f"Unexpected SHAP values shape: {shap_array.shape}")
            return None
        
        print(f"SHAP values shape for instance: {shap_values_to_use.shape}")
        
        # Map SHAP values back to original features
        original_shap_contributions = {}
        original_feature_values = {}
        
        # Initialize
        for cat_col in categorical_cols:
            original_shap_contributions[cat_col] = 0
            original_feature_values[cat_col] = X_test.iloc[instance_idx][cat_col]
        
        for cont_col in continuous_cols:
            original_shap_contributions[cont_col] = 0
            original_feature_values[cont_col] = X_test.iloc[instance_idx][cont_col]
        
        # Aggregate SHAP values
        for i, processed_feature in enumerate(processed_feature_names):
            if i < len(shap_values_to_use):
                if processed_feature in feature_mapping['processed_to_original']:
                    original_feature = feature_mapping['processed_to_original'][processed_feature]
                    original_shap_contributions[original_feature] += shap_values_to_use[i]
        
        # Get model prediction for this instance
        prediction_proba = clf.predict_proba(X_test.iloc[instance_idx:instance_idx+1])[0, 1]
        prediction_binary = clf.predict(X_test.iloc[instance_idx:instance_idx+1])[0]
        
        print(f"\nInstance {instance_idx} Analysis:")
        print(f"Prediction: {prediction_binary} (probability: {prediction_proba:.3f})")
        print(f"Actual label: {y_test.iloc[instance_idx]}")
        
        print(f"\nOriginal Feature Values:")
        for feature, value in original_feature_values.items():
            print(f"  {feature}: {value}")
        
        print(f"\nSHAP Contributions (mapped to original features):")
        sorted_contributions = sorted(original_shap_contributions.items(), 
                                    key=lambda x: abs(x[1]), reverse=True)
        for feature, contribution in sorted_contributions:
            direction = "→ Increases" if contribution > 0 else "→ Decreases"
            print(f"  {feature:<20}: {contribution:+.4f} {direction} approval probability")
        
        # Create waterfall plot - make it wider to accommodate labels with values
        plt.figure(figsize=(16, 10))
        
        # Calculate baseline (expected value)
        baseline = y_train.mean()
        
        # Prepare data for waterfall plot
        features = list(original_shap_contributions.keys())
        contributions = list(original_shap_contributions.values())
        
        # Sort by absolute contribution for better visualization
        sorted_data = sorted(zip(features, contributions), key=lambda x: abs(x[1]), reverse=True)
        features_sorted, contributions_sorted = zip(*sorted_data)
        
        # Only show top 10 features to avoid clutter
        max_features = min(10, len(features_sorted))
        features_plot = features_sorted[:max_features]
        contributions_plot = contributions_sorted[:max_features]
        
        # Create waterfall chart
        x_pos = np.arange(len(features_plot) + 2)  # +2 for baseline and final
        y_values = [baseline]
        
        # Calculate cumulative values
        current_value = baseline
        for contrib in contributions_plot:
            current_value += contrib
            y_values.append(current_value)
        
        # Create labels with feature names and values
        feature_labels = []
        for feature in features_plot:
            value = original_feature_values[feature]
            
            # Format the value nicely
            if feature in categorical_cols:
                # For categorical variables, show the category value
                feature_label = f"{feature}:\n{value}"
            else:
                # For continuous variables, format based on the feature type
                if isinstance(value, (int, float)):
                    if feature in ['age']:
                        feature_label = f"{feature}:\n{int(value)}"
                    elif feature in ['annual_income', 'account_balance']:
                        feature_label = f"{feature}:\n${int(value):,}"
                    elif feature in ['debt_to_income']:
                        feature_label = f"{feature}:\n{value:.1%}"
                    elif feature in ['credit_score']:
                        feature_label = f"{feature}:\n{int(value)}"
                    else:
                        feature_label = f"{feature}:\n{value:.2f}"
                else:
                    feature_label = f"{feature}:\n{value}"
            
            feature_labels.append(feature_label)
        
        bar_labels = ['Expected\nValue'] + feature_labels + ['Final\nPrediction']
        
        # Baseline bar
        plt.bar(0, baseline, color='lightgray', alpha=0.7, label='Expected Value')
        
        # Contribution bars
        for i, (feature, contrib) in enumerate(zip(features_plot, contributions_plot)):
            x = i + 1
            bottom = y_values[i]
            
            if contrib > 0:
                color = 'lightgreen'
                plt.bar(x, contrib, bottom=bottom, color=color, alpha=0.8)
            else:
                color = 'lightcoral'
                plt.bar(x, abs(contrib), bottom=bottom + contrib, color=color, alpha=0.8)
            
            # Add contribution value on bar
            text_y = bottom + contrib/2
            plt.text(x, text_y, f'{contrib:+.3f}', ha='center', va='center', 
                    fontweight='bold', fontsize=9)
        
        # Final prediction bar
        final_x = len(features_plot) + 1
        plt.bar(final_x, y_values[-1], color='gold', alpha=0.8, label='Final Prediction')
        
        # Add connecting lines
        for i in range(len(y_values) - 1):
            if i < len(contributions_plot) and contributions_plot[i] != 0:
                x1, x2 = i + 0.4, i + 1.6
                y = y_values[i + 1]
                plt.plot([x1, x2], [y, y], 'k--', alpha=0.5, linewidth=1)
        
        # Customize plot
        plt.xticks(x_pos, bar_labels, rotation=0, ha='center', fontsize=9)
        plt.ylabel('Approval Probability', fontsize=12)
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx} - {dataset_name}\n'
                 f'Prediction: {prediction_binary} (prob: {prediction_proba:.3f})', fontsize=14, pad=20)
        
        # Add horizontal line at 0.5 (decision boundary)
        plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='Decision Boundary (0.5)')
        
        # Add value labels for expected value and final prediction
        plt.text(0, baseline + 0.02, f'{baseline:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(final_x, y_values[-1] + 0.02, f'{y_values[-1]:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Add more space at bottom for labels
        
        # Save plot
        filename = f'waterfall_original_features_instance_{instance_idx}_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nWaterfall plot saved as: {filename}")
        
        # Show detailed explanation for categorical variables
        print(f"\nDetailed Categorical Variable Analysis for Instance {instance_idx}:")
        for cat_col in categorical_cols:
            if cat_col in original_shap_contributions:
                value = original_feature_values[cat_col]
                contribution = original_shap_contributions[cat_col]
                direction = "increases" if contribution > 0 else "decreases"
                
                print(f"\n{cat_col.upper()}: '{value}'")
                print(f"  Impact: {contribution:+.4f} ({direction} approval probability)")
                
                # Show how this category compares to others
                if cat_col in feature_mapping['categorical_categories']:
                    categories = feature_mapping['categorical_categories'][cat_col]
                    category_mask = X_train[cat_col] == value
                    if category_mask.sum() > 0:
                        category_approval_rate = y_train[category_mask].mean()
                        overall_approval_rate = y_train.mean()
                        
                        print(f"  Category approval rate: {category_approval_rate:.1%}")
                        print(f"  Overall approval rate: {overall_approval_rate:.1%}")
                        print(f"  Difference: {category_approval_rate - overall_approval_rate:+.1%}")
        
        return {
            'features': features_plot,
            'contributions': contributions_plot,
            'prediction_proba': prediction_proba,
            'prediction_binary': prediction_binary,
            'baseline': baseline,
            'feature_values': original_feature_values
        }
        
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
        return None
    except Exception as e:
        print(f"Waterfall plot creation failed: {e}")
        return None


def demonstrate_original_shap_analysis(clf, X_train, X_test, feature_mapping, categorical_cols, continuous_cols, dataset_name):
    """Demonstrate SHAP analysis mapped back to original features."""
    print(f"\n=== SHAP Analysis - Original Features - {dataset_name} ===")
    
    try:
        import shap
        
        # Get preprocessed data
        X_train_processed = clf.preprocessor_.transform(X_train)
        X_test_processed = clf.preprocessor_.transform(X_test)
        
        # Create SHAP explainer
        def predict_proba_wrapper(X):
            return clf.model_.predict_proba(X)
        
        print("Creating SHAP explainer...")
        explainer = shap.Explainer(predict_proba_wrapper, X_train_processed[:100])
        
        print("Calculating SHAP values...")
        shap_values = explainer(X_test_processed[:30])
        
        # Get processed feature names
        try:
            processed_feature_names = clf.get_feature_names_out()
        except:
            processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
        
        # Map SHAP values back to original features
        print("Mapping SHAP values to original features...")
        
        # Aggregate SHAP values for categorical features
        original_shap_values = {}
        shap_importance_by_original = {}
        
        # Get number of samples from SHAP values
        if hasattr(shap_values, 'values'):
            # For newer SHAP versions that return Explanation objects
            shap_array = shap_values.values
        else:
            # For older versions or if already numpy array
            shap_array = shap_values
        
        n_samples = shap_array.shape[0]
        print(f"Number of samples for SHAP: {n_samples}")
        
        # Initialize with zeros
        for cat_col in categorical_cols:
            original_shap_values[cat_col] = np.zeros(n_samples)
            shap_importance_by_original[cat_col] = 0
        
        for cont_col in continuous_cols:
            original_shap_values[cont_col] = np.zeros(n_samples)
            shap_importance_by_original[cont_col] = 0
        
        # Aggregate SHAP values - check if we have the right dimensions
        print(f"SHAP array shape: {shap_array.shape}")
        print(f"Processed feature names: {len(processed_feature_names)}")
        
        # Handle different SHAP value formats
        if len(shap_array.shape) == 3:
            # Binary classification: [samples, features, classes]
            shap_values_to_use = shap_array[:, :, 1]  # Use positive class
        elif len(shap_array.shape) == 2:
            # [samples, features] - already for the positive class
            shap_values_to_use = shap_array
        else:
            print(f"Unexpected SHAP values shape: {shap_array.shape}")
            return None
        
        print(f"Using SHAP values shape: {shap_values_to_use.shape}")
        
        # Aggregate SHAP values
        for i, processed_feature in enumerate(processed_feature_names):
            if i < shap_values_to_use.shape[1]:  # Check bounds
                if processed_feature in feature_mapping['processed_to_original']:
                    original_feature = feature_mapping['processed_to_original'][processed_feature]
                    original_shap_values[original_feature] += shap_values_to_use[:, i]
                    shap_importance_by_original[original_feature] += np.abs(shap_values_to_use[:, i]).mean()
                else:
                    print(f"Warning: {processed_feature} not found in feature mapping")
        
        print("\nSHAP Feature Importance (mapped to original features):")
        print("Categorical Features:")
        cat_shap = {k: v for k, v in shap_importance_by_original.items() if k in categorical_cols}
        sorted_cat_shap = sorted(cat_shap.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_cat_shap, 1):
            print(f"  {i}. {feature:<20}: {importance:.4f}")
        
        print("Continuous Features:")
        cont_shap = {k: v for k, v in shap_importance_by_original.items() if k in continuous_cols}
        sorted_cont_shap = sorted(cont_shap.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_cont_shap, 1):
            print(f"  {i}. {feature:<20}: {importance:.4f}")
        
        # Create SHAP summary plot for original features
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        original_features = list(original_shap_values.keys())
        original_shap_matrix = np.column_stack([original_shap_values[feat] for feat in original_features])
        
        # Use only the test samples that we used for SHAP analysis (first 30)
        X_test_subset = X_test.iloc[:original_shap_matrix.shape[0]]
        original_data_matrix = np.column_stack([X_test_subset[feat].values if feat in X_test_subset.columns 
                                              else np.zeros(len(X_test_subset)) for feat in original_features])
        
        # Create DataFrame for SHAP plot - use only the test samples we used for SHAP
        X_test_shap = X_test.iloc[:len(original_shap_matrix)]  # Match the number of SHAP samples
        shap_df = pd.DataFrame(original_data_matrix, columns=original_features)
        
        # Manual summary plot since we aggregated features
        feature_importance = [shap_importance_by_original[feat] for feat in original_features]
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        # Plot top features
        top_n = min(10, len(original_features))
        top_indices = sorted_indices[:top_n]
        
        for i, idx in enumerate(top_indices):
            feature = original_features[idx]
            values = original_shap_matrix[:, idx]
            y_pos = top_n - i - 1
            
            # Color by value for continuous, by category for categorical
            if feature in categorical_cols:
                colors = plt.cm.tab10(i / top_n)
                plt.scatter(values, [y_pos] * len(values), alpha=0.6, c=[colors] * len(values), s=20)
            else:
                # Use the corresponding feature values from our data matrix
                feature_values = original_data_matrix[:, idx]
                plt.scatter(values, [y_pos] * len(values), alpha=0.6, 
                           c=feature_values, cmap='viridis', s=20)
        
        plt.yticks(range(top_n), [original_features[i] for i in top_indices])
        plt.xlabel('SHAP Value (impact on model output)')
        plt.title(f'SHAP Summary - Original Features - {dataset_name}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(f'shap_original_features_{dataset_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return original_shap_values
        
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
        return None
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None


def main():
    """Run comprehensive original feature mapping XAI demonstration."""
    print("MLForge-Binary XAI Demo: Original Feature Mapping")
    print("=" * 60)
    
    # Create interpretable dataset
    X_data, y_data = create_interpretable_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    clf = BinaryClassifier(
        model='random_forest',
        calibrate=False,  # Disable calibration to get direct access to Random Forest
        explain=True
    )
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.3f}")
    print(f"  ROC AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    
    # Get processed feature names and create mapping
    try:
        processed_feature_names = clf.get_feature_names_out()
    except:
        print("Warning: Could not get processed feature names")
        processed_feature_names = [f'feature_{i}' for i in range(clf.preprocessor_.transform(X_train).shape[1])]
    
    feature_mapping, categorical_cols, continuous_cols = create_feature_mapping(X_train, processed_feature_names)
    
    # Extract original feature importance
    cat_importance, cont_importance, cat_detailed = extract_original_feature_importance(
        clf, feature_mapping, categorical_cols, continuous_cols
    )
    
    # Plot original feature importance
    if cat_importance or cont_importance:
        plot_original_feature_importance(cat_importance, cont_importance, cat_detailed, "Random Forest")
    
    # Analyze categorical category impact
    category_impact = analyze_categorical_category_impact(X_train, y_train, categorical_cols)
    
    # SHAP analysis with original feature mapping
    demonstrate_original_shap_analysis(clf, X_train, X_test, feature_mapping, 
                                     categorical_cols, continuous_cols, "Random Forest")
    
    # Create waterfall plots for interesting instances
    print("\n" + "=" * 60)
    print("WATERFALL PLOT ANALYSIS")
    print("=" * 60)
    
    # Find interesting instances to explain
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    # Find instances with different characteristics
    interesting_instances = []
    
    # High confidence approval
    high_conf_approved = np.where((y_pred == 1) & (y_pred_proba > 0.8))[0]
    if len(high_conf_approved) > 0:
        interesting_instances.append(("High Confidence Approval", high_conf_approved[0]))
    
    # High confidence rejection  
    high_conf_rejected = np.where((y_pred == 0) & (y_pred_proba < 0.2))[0]
    if len(high_conf_rejected) > 0:
        interesting_instances.append(("High Confidence Rejection", high_conf_rejected[0]))
    
    # Close to decision boundary
    close_boundary = np.where((y_pred_proba > 0.45) & (y_pred_proba < 0.55))[0]
    if len(close_boundary) > 0:
        interesting_instances.append(("Close to Decision Boundary", close_boundary[0]))
    
    # If no interesting instances found, just use first few
    if not interesting_instances:
        interesting_instances = [("Instance", 0), ("Instance", 1), ("Instance", 2)]
    
    # Create waterfall plots for interesting instances
    for description, instance_idx in interesting_instances[:3]:  # Limit to 3 plots
        print(f"\nCreating waterfall plot for: {description} (Instance {instance_idx})")
        waterfall_result = create_original_waterfall_plot(
            clf, X_train, X_test, y_train, y_test, feature_mapping, categorical_cols, continuous_cols, 
            "Random Forest", instance_idx
        )
    
    print("\n" + "=" * 60)
    print("ORIGINAL FEATURE MAPPING DEMO COMPLETED!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. Feature importance is aggregated back to original categorical variables")
    print("2. Category-level importance shows which specific categories matter most")
    print("3. SHAP values are properly mapped to original features, not dummy variables")
    print("4. Categorical variables show their total impact across all encoded features")
    print("5. Waterfall plots explain individual predictions using original feature names")
    print("6. Categorical variables show specific category values and their impact")
    print("7. Analysis provides actionable insights about original business variables")
    print("\nGenerated plots and analysis saved as PNG files.")
    print("\nWaterfall plots show:")
    print("- How each original feature contributes to individual predictions")
    print("- Categorical variables displayed with their actual values (e.g., 'education: PhD')")
    print("- Step-by-step build-up from baseline to final prediction")
    print("- Comparison to decision boundary and category approval rates")


if __name__ == "__main__":
    main()