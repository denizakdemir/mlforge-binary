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
- **Model explainability**: Feature importance and SHAP values
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

## 🛠️ Development

### Run Examples

```bash
# Basic usage examples
python examples/basic_usage.py
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
- shap (for model explanations)
- plotly (for visualizations)

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
└── ...
```

## 📄 License

This project is licensed under the MIT License.