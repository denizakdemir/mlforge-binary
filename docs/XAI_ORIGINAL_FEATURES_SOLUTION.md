# XAI Original Features Mapping Solution

## Problem Solved

The user requested: **"I need a way to map the importances and the dependence to the original variables and the categories of categorical variables. Not the dummy variables"**

## Solution

Created `examples/xai_original_features_demo.py` that demonstrates how to:

1. **Map Feature Importance Back to Original Variables**
   - Automatically identifies which processed features (e.g., `education_bachelor_degree`) map to original categorical variables (e.g., `education`)
   - Aggregates dummy variable importances back to their parent categorical variables
   - Preserves continuous variable importance as-is

2. **Extract Category-Level Insights**
   - Shows which specific categories within categorical variables are most important
   - Provides approval rates for each category to understand business impact
   - Maps importance from encoded features back to interpretable categories

3. **SHAP Analysis with Original Feature Mapping**
   - Calculates SHAP values on processed features (required for model compatibility)
   - Aggregates SHAP values back to original categorical variables
   - Creates meaningful visualizations using original feature names

## Key Features

### Feature Mapping Function
```python
def create_feature_mapping(X_original, X_processed_feature_names):
    """Create mapping from processed features back to original features."""
```
- Maps `education_bachelor_degree` → `education`
- Maps `employment_status_full_time` → `employment_status`
- Preserves `age`, `income`, etc. as continuous variables

### Original Feature Importance Extraction
```python
def extract_original_feature_importance(clf, feature_mapping, categorical_cols, continuous_cols):
    """Extract feature importance and map back to original features."""
```
- Handles ModelWrapper → RandomForestClassifier unwrapping
- Aggregates categorical dummy variable importances
- Provides category-level detailed breakdown

### SHAP with Original Feature Aggregation
```python
def demonstrate_original_shap_analysis(clf, X_train, X_test, feature_mapping, categorical_cols, continuous_cols, dataset_name):
    """Demonstrate SHAP analysis mapped back to original features."""
```
- Calculates SHAP on processed features for model compatibility
- Aggregates SHAP values back to original categorical variables
- Creates proper visualizations with original feature names

## Results Example

### Feature Importance (Original Variables)
```
Categorical Features (total importance):
  1. region              : 0.0469
  2. education           : 0.0379
  3. employment_status   : 0.0356
  4. home_ownership      : 0.0315

Continuous Features:
  1. debt_to_income      : 0.2792
  2. annual_income       : 0.1783
  3. account_balance     : 0.1697
  4. credit_score        : 0.1423
  5. age                 : 0.0785
```

### Category-Level Analysis
```
EDUCATION:
  Bachelor Degree     : 51.4% approval rate ( 292 samples, 34.8% of data)
  High School         : 48.2% approval rate ( 222 samples, 26.4% of data)
  Master Degree       : 47.0% approval rate ( 134 samples, 16.0% of data)
  PhD                 : 55.3% approval rate (  38 samples, 4.5% of data)
  Some College        : 51.3% approval rate ( 154 samples, 18.3% of data)
```

### SHAP Analysis (Original Features)
```
SHAP Feature Importance (mapped to original features):
Categorical Features:
  1. region              : 0.0273
  2. home_ownership      : 0.0271
  3. education           : 0.0146
  4. employment_status   : 0.0108
Continuous Features:
  1. debt_to_income      : 0.2032
  2. account_balance     : 0.0921
  3. annual_income       : 0.0609
  4. age                 : 0.0472
  5. credit_score        : 0.0318
```

## Technical Implementation

### Model Access Fix
The key breakthrough was properly accessing the underlying sklearn model:
```python
# Access: clf.model_ (ModelWrapper) → clf.model_.model_ (RandomForestClassifier)
if hasattr(clf.model_, 'model_') and clf.model_.model_ is not None:
    actual_model = clf.model_.model_
    # Now can access actual_model.feature_importances_
```

### Feature Mapping Logic
```python
# Map processed feature names back to original
for processed_feature in X_processed_feature_names:
    for cat_col in categorical_cols:
        if cat_col.lower() in processed_feature.lower():
            feature_mapping['categorical_to_processed'][cat_col].append(processed_feature)
            break
```

### Importance Aggregation
```python
# Aggregate categorical feature importance
for cat_col in categorical_cols:
    total_importance = 0
    for processed_feature in feature_mapping['categorical_to_processed'][cat_col]:
        if processed_feature in processed_importance:
            total_importance += processed_importance[processed_feature]
    original_categorical_importance[cat_col] = total_importance
```

## Generated Visualizations

1. **original_feature_importance_random_forest.png** - 4-panel plot showing:
   - Categorical features total importance
   - Continuous features importance
   - Category-level detailed importance
   - Combined comparison

2. **shap_original_features_random_forest.png** - SHAP summary plot with original feature names

3. **waterfall_original_features_instance_X_random_forest.png** - Individual prediction explanations showing:
   - Step-by-step contribution of each original feature
   - Categorical variables with their actual values (e.g., "education:\nPhD")
   - Continuous variables with formatted values (e.g., "annual_income:\n$45,000", "debt_to_income:\n35.0%")
   - Build-up from baseline expectation to final prediction
   - Visual decision boundary reference
   - Color-coded positive/negative contributions

## Business Value

- **Interpretable Results**: Users see `education` importance, not `education_bachelor_degree`
- **Actionable Insights**: Can understand which education levels drive approvals
- **Compliance Ready**: Explanations use original business terminology
- **Category Analysis**: Shows impact of specific categories within variables

## Usage

```python
from mlforge_binary import BinaryClassifier

# Train model (automatic preprocessing handles categorical encoding)
clf = BinaryClassifier(model='random_forest', calibrate=False, explain=True)
clf.fit(X_train, y_train)

# Extract original feature mapping and importance
feature_mapping, cat_cols, cont_cols = create_feature_mapping(X_train, clf.get_feature_names_out())
cat_importance, cont_importance, cat_detailed = extract_original_feature_importance(
    clf, feature_mapping, cat_cols, cont_cols
)

# SHAP analysis with original feature mapping
demonstrate_original_shap_analysis(clf, X_train, X_test, feature_mapping, cat_cols, cont_cols, "Model")
```

## Waterfall Plot Examples

The demo generates waterfall plots for three types of instances:

### High Confidence Approval (Instance 4)
```
Prediction: 1 (probability: 0.990)
Waterfall Plot Labels Show:
  education: Some College      (+0.0056 impact)
  employment_status: Full-time (+0.0041 impact)  
  home_ownership: Family       (+0.0017 impact)
  debt_to_income: 80.0%        (+0.2550 impact)
  annual_income: $25,000       (+0.0599 impact)
  region: Midwest              (+0.0092 impact)
  age: 29                      (+0.0581 impact)
  credit_score: 385            (+0.0096 impact)
  account_balance: $0          (+0.0946 impact)
```

### High Confidence Rejection (Instance 8)  
```
Prediction: 0 (probability: 0.070)
Waterfall Plot Labels Show:
  education: Bachelor Degree   (-0.0025 impact)
  employment_status: Unemployed (-0.0125 impact)
  home_ownership: Mortgage     (+0.0332 impact)
  debt_to_income: 0.0%         (-0.2571 impact)
  annual_income: $25,000       (-0.0122 impact)
  account_balance: $80,557     (-0.1356 impact)
  age: 41                      (-0.0117 impact)
  credit_score: 448            (-0.0265 impact)
  region: West                 (+0.0028 impact)
```

### Close to Decision Boundary (Instance 61)
```
Prediction: 1 (probability: 0.500)
Waterfall Plot Labels Show:
  education: Some College      (-0.0092 impact)
  employment_status: Part-time (-0.0134 impact)
  home_ownership: Own          (-0.0152 impact)
  debt_to_income: 0.0%         (-0.0835 impact)
  annual_income: $30,746       (-0.0223 impact)
  account_balance: $0          (+0.1216 impact)
  age: 22                      (-0.0227 impact)
  credit_score: 648            (+0.0489 impact)
  region: West                 (+0.0038 impact)
```

Each waterfall plot shows how the model builds up from the baseline expectation (50% approval rate) to the final prediction, with each original feature contributing positively or negatively.

## Complete Solution

This solution completely addresses the user's need for mapping XAI results back to original business variables instead of encoded dummy variables, now including waterfall plots that show categorical variables with their actual category values and detailed impact analysis.