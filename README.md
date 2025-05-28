# MLForge-Binary

A simple, production-ready binary classification library that handles common ML challenges automatically.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mlforge-binary.git
cd mlforge-binary

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from mlforge_binary import BinaryClassifier

# Initialize classifier
clf = BinaryClassifier(model='auto')

# Train
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Evaluate
score = clf.score(X_test, y_test)
```

## ✨ Features

- **Automatic preprocessing**: Handles missing values, categorical encoding, scaling
- **Multiple algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, etc.
- **AutoML**: Automatic model selection and hyperparameter tuning
- **Ensemble methods**: Voting and stacking classifiers
- **Model evaluation**: Comprehensive metrics and visualizations
- **🎯 Advanced XAI (Explainable AI)**: 
  - Feature importance mapped to original variables
  - SHAP analysis with categorical variable support
  - Waterfall plots showing actual variable values
  - Category-level impact analysis
  - Business-friendly explanations
- **Production ready**: Model persistence, monitoring, and deployment utilities

## 📖 Examples

### AutoML Usage

```python
from mlforge_binary import AutoML

# Initialize AutoML
automl = AutoML(random_state=42)

# Fit with automatic model selection
automl.fit(X_train, y_train)

# Get predictions
predictions = automl.predict(X_test)
probabilities = automl.predict_proba(X_test)

# View best model
print(f"Best model: {automl.best_model_type_}")
```

### Model Comparison

```python
from mlforge_binary import compare_models

# Compare multiple models
results = compare_models(
    X_train, y_train, X_val, y_val,
    models=['logistic', 'random_forest', 'xgboost', 'lightgbm']
)
print(results)
```

### Quick Experiment

```python
from mlforge_binary import quick_experiment

# Run a quick ML experiment
results = quick_experiment(X, y, test_size=0.2, random_state=42)
print(results)
```

### 🎯 Explainable AI (XAI) Examples

#### Feature Importance with Original Variables

```python
from mlforge_binary import BinaryClassifier

# Train model with mixed data types
clf = BinaryClassifier(model='random_forest', explain=True)
clf.fit(X_train, y_train)

# Get feature importance mapped to original variables
# (not dummy/encoded variables)
explanations = clf.explain()
print("Feature Importance (Original Variables):")
for feature, importance in explanations['feature_importance'].items():
    print(f"  {feature}: {importance:.4f}")
```

#### SHAP Analysis with Categorical Variables

```python
# Advanced XAI analysis with SHAP
# Shows categorical variables with their actual values
# Example output:
# - education: Bachelor Degree → +0.052 impact
# - employment_status: Full-time → +0.023 impact  
# - annual_income: $65,000 → +0.087 impact

# Run comprehensive XAI examples
python examples/xai_demo.py                    # Basic XAI
python examples/xai_mixed_data_demo.py         # Mixed data types
python examples/xai_original_features_demo.py  # Original feature mapping
```

#### Waterfall Plots for Individual Predictions

```python
# Waterfall plots show step-by-step prediction explanation
# with actual variable values displayed:
# 
# Expected Value (50.0%) 
#   → education: PhD (+0.045)
#   → annual_income: $85,000 (+0.123) 
#   → debt_to_income: 25.0% (-0.067)
#   → Final Prediction: 78.3% (Approved)
```

## 🛠️ Development

### Run Examples

```bash
# Basic usage examples
python examples/basic_usage.py
python examples/quick_demo.py

# XAI (Explainable AI) examples
python examples/xai_demo.py                    # Basic XAI with SHAP and LIME
python examples/xai_mixed_data_demo.py         # XAI with mixed data types  
python examples/xai_original_features_demo.py  # Advanced: Original feature mapping
```

### Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python tests/test_installation.py
```

## 📋 Requirements

See `requirements.txt` for full dependencies.

**Core requirements:**
- Python 3.8+
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.20.0

**Optional dependencies:**
- xgboost (for XGBoost models)
- lightgbm (for LightGBM models)  
- catboost (for CatBoost models)
- optuna (for hyperparameter tuning)
- shap (for SHAP explanations and waterfall plots)
- lime (for LIME explanations)
- plotly (for visualizations)
- matplotlib (for plots)
- seaborn (for enhanced visualizations)

## 📊 Project Structure

```
mlforge_binary/
├── __init__.py         # Main exports
├── classifier.py       # BinaryClassifier class  
├── automl.py          # AutoML functionality
├── models.py          # Model wrappers
├── preprocessing.py    # Data preprocessing
├── evaluation.py      # Model evaluation
├── explainer.py       # Model explanations
├── utils.py           # Utility functions
├── fairness.py        # Fairness analysis
├── monitor.py         # Model monitoring
└── cli.py             # Command line interface

examples/
├── basic_usage.py                  # Basic usage examples
├── quick_demo.py                   # Quick demonstration
├── xai_demo.py                     # Basic XAI examples
├── xai_mixed_data_demo.py          # XAI with mixed data types
└── xai_original_features_demo.py   # Advanced XAI with feature mapping

plots/                              # Generated visualization outputs
tests/                              # Test suite
docs/                               # Documentation
```

## 🎯 Key XAI Features

### Original Feature Mapping
- **Problem**: Traditional XAI shows importance for dummy variables (`education_bachelor`, `education_master`)
- **Solution**: MLForge-Binary maps back to original variables (`education` with category breakdown)
- **Benefit**: Business stakeholders see interpretable variable names

### Waterfall Plots with Variable Values
- Shows actual values: `annual_income: $65,000`, `education: PhD`
- Step-by-step prediction explanation from baseline to final decision
- Category-level analysis for categorical variables

### Comprehensive SHAP Integration
- Automatic aggregation of dummy variable SHAP values
- Category impact analysis with approval rate comparisons
- Business-friendly explanations and visualizations

## 📄 License

This project is licensed under the MIT License.