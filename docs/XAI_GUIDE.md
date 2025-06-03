# MLForge-Binary XAI (Explainable AI) Guide

## Overview

MLForge-Binary provides advanced explainable AI capabilities that solve the common problem of interpreting machine learning models with categorical variables. Traditional XAI tools show feature importance for encoded dummy variables (e.g., `education_bachelor`, `education_master`), but business stakeholders need explanations in terms of the original variables (e.g., `education` with category analysis).

## Key XAI Features

### 🎯 Original Feature Mapping
- Maps dummy variable importance back to original categorical variables
- Provides category-level impact analysis  
- Shows business-friendly variable names and values

### 🌊 Waterfall Plots with Variable Values
- Individual prediction explanations showing actual variable values
- Step-by-step progression from baseline to final prediction
- Categorical variables displayed with their specific values

### 📊 Comprehensive SHAP Integration
- Automatic aggregation of SHAP values for dummy variables
- Category approval rate comparisons
- Multiple visualization types (summary, waterfall, feature importance)

### 🔍 LIME Explanations
- Local interpretable model-agnostic explanations
- Understand the factors influencing individual predictions
- Visualize feature contributions for specific instances

## Quick Start

### Basic XAI Usage

The XAI functionality is built directly into the BinaryClassifier class. There are two main methods for getting explanations:

```python
from mlforge_binary import BinaryClassifier

# Train model with XAI enabled (default is True)
clf = BinaryClassifier(model='random_forest', explain=True)
clf.fit(X_train, y_train)

# 1. Get global explanations (feature importance)
explanations = clf.explain()
print("Feature Importance:", explanations['feature_importance'])

# 2. Get individual instance explanations
instance = X_test.iloc[0]  # Single instance
instance_explanation = clf.explain_instance(instance)
print("Prediction:", instance_explanation['prediction'])
print("Prediction Probability:", instance_explanation['prediction_proba'])
```

### Advanced XAI Examples

Run the comprehensive XAI examples to see all features:

```bash
# Basic XAI with SHAP and LIME
python examples/xai_demo.py

# XAI with mixed data types (categorical + continuous)
python examples/xai_mixed_data_demo.py

# Advanced: Original feature mapping and waterfall plots
python examples/xai_original_features_demo.py
```

## XAI Examples Breakdown

### 1. Basic XAI Demo (`xai_demo.py`)

**Features:**
- SHAP summary plots
- LIME explanations for individual instances
- Feature importance analysis
- Model behavior visualization

**Usage with Updated API:**
```python
from mlforge_binary import BinaryClassifier

# Train model with XAI enabled
clf = BinaryClassifier(model='random_forest', explain=True)
clf.fit(X_train, y_train)

# Get global explanations (feature importance)
explanations = clf.explain(X_test)

# Get individual instance explanation
instance = X_test.iloc[0]
instance_explanation = clf.explain_instance(instance)
```

**Datasets:**
- Synthetic classification data
- Real-world cancer dataset
- Multiple model types (Random Forest, XGBoost, Logistic Regression)

**Generated Plots:**
- `shap_summary_*.png` - SHAP feature importance
- `shap_waterfall_*.png` - Individual prediction explanations
- `lime_explanation_*.png` - LIME local explanations
- `model_behavior_*.png` - Decision boundary visualizations

### 2. Mixed Data Demo (`xai_mixed_data_demo.py`)

**Features:**
- Handles categorical and continuous variables together
- Proper encoding and explanation mapping
- Multiple model comparisons

**Usage with Updated API:**
```python
# Train model with mixed data
clf = BinaryClassifier(model='random_forest', explain=True)
clf.fit(X_train, y_train)

# Get explanations for mixed data
explanations = clf.explain(X_test)

# Handle categorical features
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
continuous_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Analyze by data type
analyze_categorical_vs_continuous_importance(
    explanations['feature_importance'], feature_names, categorical_cols, continuous_cols
)
```

**Key Insights:**
- Shows how categorical variables contribute to predictions
- Compares different model types on same data
- Demonstrates preprocessing effects on explanations

**Generated Plots:**
- `shap_mixed_data_summary_*.png` - Mixed data SHAP analysis
- `*_variables_distribution.png` - Data distribution analysis

### 3. Original Features Demo (`xai_original_features_demo.py`)

**Features:**
- **Feature Mapping**: Maps processed features back to original variables
- **Waterfall Plots**: Shows actual variable values in explanations
- **Category Analysis**: Detailed breakdown of categorical variable impact
- **Business-Friendly Output**: Uses original variable names throughout

**Usage with Updated API:**
```python
# Train model
clf = BinaryClassifier(model='random_forest', explain=True)
clf.fit(X_train, y_train)

# Get processed feature names
processed_feature_names = clf.get_feature_names_out()

# Create mapping from processed to original features
feature_mapping, categorical_cols, continuous_cols = create_feature_mapping(
    X_train, processed_feature_names
)

# Extract feature importance and map to original features
cat_importance, cont_importance, cat_detailed = extract_original_feature_importance(
    clf, feature_mapping, categorical_cols, continuous_cols
)

# Explain individual instance with original feature values
instance = X_test.iloc[0]
instance_explanation = clf.explain_instance(instance)
```

**Key Functions:**
- `create_feature_mapping()` - Maps processed to original features
- `extract_original_feature_importance()` - Aggregates dummy variable importance
- `create_original_waterfall_plot()` - Individual prediction explanations
- `analyze_categorical_category_impact()` - Category-level analysis

## Feature Mapping Methodology

### Problem with Traditional XAI

```python
# Traditional XAI output (not business-friendly):
Feature Importance:
- education_bachelor_degree: 0.12
- education_master_degree: 0.08  
- education_phd: 0.15
- education_high_school: 0.05
```

### MLForge-Binary Solution

```python
# MLForge-Binary XAI output (business-friendly):
Feature Importance:
- education: 0.40 (total)
  - PhD: 0.15 (55% approval rate vs 50% overall)
  - Bachelor: 0.12 (52% approval rate)
  - Master: 0.08 (48% approval rate)
  - High School: 0.05 (45% approval rate)
```

## Waterfall Plot Features

### Variable Value Display

Waterfall plots show actual variable values for each instance:

```
Expected Value (50.0%)
  → education: PhD (+0.045)
  → annual_income: $85,000 (+0.123)
  → debt_to_income: 25.0% (-0.067)
  → employment_status: Full-time (+0.032)
  → Final Prediction: 78.3% (Approved)
```

### Formatting by Variable Type

- **Categorical**: Shows the actual category value
  - `education: PhD`
  - `employment_status: Full-time`

- **Currency**: Properly formatted monetary values
  - `annual_income: $65,000`
  - `account_balance: $12,500`

- **Percentages**: Ratio values as percentages
  - `debt_to_income: 35.0%`

- **Integers**: Clean integer display
  - `age: 29`
  - `credit_score: 720`

## Category-Level Analysis

### Approval Rate Comparisons

For each categorical variable, the system shows:

```
EDUCATION: 'PhD'
  Impact: +0.045 (increases approval probability)
  Category approval rate: 55.3%
  Overall approval rate: 50.0%
  Difference: +5.3%
```

This tells you:
1. How this specific category affects the prediction
2. The historical approval rate for this category
3. How it compares to the overall population

### Business Value

- **Loan Officers**: Understand why specific applications were approved/rejected
- **Compliance Teams**: Audit decisions with clear reasoning
- **Product Managers**: Identify which customer segments are most/least likely to be approved
- **Risk Analysts**: Understand risk factors in business terms

## Generated Visualizations

### Feature Importance Plots
- **4-panel layout**: Categorical total, continuous, category-detailed, combined
- **Original variable names**: Business-friendly labels
- **Sorted by importance**: Most impactful features first

### SHAP Summary Plots
- **Original feature aggregation**: Dummy variables combined
- **Color-coded impact**: Positive/negative contributions
- **Multiple instance analysis**: Population-level insights

### Waterfall Plots
- **Individual explanations**: Why this specific person was approved/rejected
- **Actual values displayed**: Real data values, not encoded numbers
- **Step-by-step progression**: From baseline expectation to final decision
- **Decision boundary reference**: Visual threshold line

## Using the Built-in XAI API

MLForge-Binary now provides a unified API for explanations through the BinaryClassifier class:

### Global Model Explanations

Use the `explain()` method to get global model explanations:

```python
# Get basic explanations without data
explanations = clf.explain()

# Get explanations with SHAP values using test data
explanations_with_shap = clf.explain(X_test)
```

Returns a dictionary with:
- `model_type`: The type of model used
- `optimal_threshold`: Classification threshold 
- `training_performance`: Performance metrics
- `feature_importance`: Feature importance values
- `shap_values`: SHAP values (if SHAP is installed and data provided)

### Instance-Level Explanations

Use the `explain_instance()` method to explain individual predictions:

```python
# Get explanation for a single instance
instance = X_test.iloc[0]
instance_explanation = clf.explain_instance(instance)
```

Returns a dictionary with:
- `prediction`: The predicted class
- `prediction_proba`: Prediction probability
- `shap_values_instance`: SHAP values for this instance
- `shap_features`: Feature-level contributions
- `lime_explanation`: LIME explanation object
- `lime_features`: Feature-level LIME contributions

## Technical Implementation

### Feature Mapping Process

1. **Identify Original Variables**: Separate categorical and continuous features
2. **Map Processed Features**: Connect dummy variables back to source categories
3. **Aggregate Importance**: Sum dummy variable importance by original feature
4. **Category Analysis**: Calculate approval rates and comparisons

### SHAP & LIME Integration

1. **Calculate SHAP Values**: Use processed features (required for model compatibility)
2. **Aggregate by Original Feature**: Sum SHAP values for related dummy variables
3. **Apply LIME**: Generate local explanations for individual instances
4. **Create Visualizations**: Use original feature names and values

### Waterfall Plot Construction

1. **Extract Instance Values**: Get original variable values for the specific instance
2. **Format by Type**: Apply appropriate formatting (currency, percentage, etc.)
3. **Calculate Contributions**: Map SHAP values to original features
4. **Build Progressive Chart**: Show step-by-step impact accumulation

## Best Practices

### When to Use Each XAI Method

- **Feature Importance** (`explain()`): Overall model understanding, feature selection
- **SHAP Summary** (`explain(X_test)`): Population-level insights, feature interaction analysis  
- **LIME** (`explain_instance()`): Individual decision explanations, local behavior
- **Waterfall Plots** (custom): Customer communications, decision explanation
- **Category Analysis** (custom): Business intelligence, risk factor identification

### Model Selection for XAI

- **Tree-based models** (Random Forest, XGBoost): Direct feature importance available
- **Linear models**: Coefficient-based importance
- **Complex models**: SHAP and LIME provide model-agnostic explanations

### Interpretation Guidelines

1. **Focus on Business Impact**: Use original variable names and values
2. **Consider Category Context**: Compare individual categories to overall rates
3. **Understand Limitations**: XAI shows correlation, not necessarily causation
4. **Validate Insights**: Cross-reference with domain expertise

## Troubleshooting

### Common Issues

1. **Missing XAI Libraries**: Install with `pip install shap lime`
2. **Explainer Not Initialized**: Set `explain=True` when creating BinaryClassifier
3. **Plot Generation Errors**: Ensure matplotlib backend is available
4. **Feature Mapping Failures**: Check that categorical variables are properly identified

### Performance Considerations

- **SHAP Calculation**: Can be slow for large datasets (use sampling)
- **Explainer Creation**: The first call may be slow as it initializes the explainer
- **Memory Usage**: Waterfall plots store individual instance data

## Future Enhancements

- More visualization options built directly into the API
- Automated report generation for compliance
- Interactive visualizations with Plotly
- Model comparison XAI analysis
- Time-series explanation support

---

For more examples and advanced usage, see the `examples/` directory and generated plots in `plots/`.