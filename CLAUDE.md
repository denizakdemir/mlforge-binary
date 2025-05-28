# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_installation.py
python tests/simple_test.py

# Run syntax validation
python tests/simple_test.py
```

### Code Quality
```bash
# Format code (if black is installed)
black mlforge_binary/

# Lint code (if flake8 is installed)
flake8 mlforge_binary/

# Type checking (if mypy is installed)
mypy mlforge_binary/
```

### Running Examples
```bash
# Basic usage example
python examples/basic_usage.py

# Quick demo
python examples/quick_demo.py
```

## Architecture Overview

MLForge-Binary is a scikit-learn compatible binary classification library with automatic preprocessing and model selection capabilities.

### Core Components

**Main API Classes:**
- `BinaryClassifier`: Main scikit-learn compatible classifier with automatic preprocessing
- `AutoML`: Automated machine learning with model comparison and ensemble creation

**Internal Modules:**
- `preprocessing.py`: `AutoPreprocessor` - handles missing values, categorical encoding, scaling, imbalanced data
- `models.py`: Model wrappers, `AutoModelSelector`, `EnsembleClassifier` - unified interface for ML algorithms
- `evaluation.py`: `ComprehensiveEvaluator`, `MetricsCalculator`, `VisualizationGenerator` - model assessment
- `explainer.py`: Model interpretability using SHAP and LIME
- `fairness.py`: Bias detection and fairness metrics
- `utils.py`: Validation, serialization, timing utilities

### Key Design Patterns

**Pipeline Architecture**: BinaryClassifier coordinates preprocessing → model selection → evaluation as one cohesive pipeline that can be tuned end-to-end.

**Auto-Everything**: The library follows an "auto" pattern where string parameters like `"auto"` trigger intelligent defaults (e.g., `handle_missing="auto"` chooses the best missing value strategy).

**Scikit-learn Compatibility**: All main classes follow sklearn patterns with `fit()`, `predict()`, `predict_proba()` methods and can be used in sklearn pipelines.

**Model Abstraction**: `ModelWrapper` provides a unified interface across different ML libraries (sklearn, xgboost, lightgbm, catboost) with consistent hyperparameter handling.

### Data Flow

1. **Input Validation**: `validate_input_data()` ensures data quality
2. **Preprocessing**: `AutoPreprocessor` handles missing values, encoding, scaling based on data characteristics  
3. **Model Selection**: `AutoModelSelector` or explicit model choice with optional hyperparameter tuning
4. **Training**: Model fitting with optional calibration and threshold optimization
5. **Evaluation**: Comprehensive metrics, visualizations, and explanations

### Extension Points

- Add new models by implementing the `ModelWrapper` interface in `models.py`
- Extend preprocessing by adding strategies to `AutoPreprocessor`
- Add evaluation metrics in `evaluation.py`
- New fairness metrics can be added to `fairness.py`

## Dependencies

Core ML stack: scikit-learn, xgboost, lightgbm, catboost, pandas, numpy
AutoML: optuna for hyperparameter tuning
Explanations: shap, lime
Imbalanced data: imbalanced-learn
Visualization: plotly, matplotlib, seaborn