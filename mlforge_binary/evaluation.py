"""
Comprehensive evaluation and reporting for MLForge-Binary
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, validation_curve
import warnings
from pathlib import Path
import json
from datetime import datetime

from .utils import format_number, Timer


class MetricsCalculator:
    """Calculate comprehensive classification metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix derived metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Probability-based metrics
        if y_proba is not None:
            y_proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            metrics['average_precision'] = average_precision_score(y_true, y_proba_pos)
            metrics['brier_score'] = brier_score_loss(y_true, y_proba_pos)
            
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except ValueError:
                metrics['log_loss'] = np.nan
            
            # Calibration metrics
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba_pos, n_bins=10
            )
            metrics['expected_calibration_error'] = np.abs(
                fraction_of_positives - mean_predicted_value
            ).mean()
        
        return metrics
    
    @staticmethod
    def calculate_threshold_metrics(y_true: np.ndarray, 
                                  y_proba: np.ndarray, 
                                  thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Calculate metrics across different probability thresholds."""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 17)
        
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred_thresh, None)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)


class VisualizationGenerator:
    """Generate comprehensive visualizations for model evaluation."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      title: str = "ROC Curve") -> go.Figure:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=1)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600, height=500
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> go.Figure:
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {ap_score:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline (random classifier)
        baseline = y_true.mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color="red",
                     annotation_text=f"Random Classifier ({baseline:.3f})")
        
        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600, height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: List[str] = None, title: str = "Confusion Matrix") -> go.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['Negative', 'Positive']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500, height=500
        )
        
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              title: str = "Calibration Curve") -> go.Figure:
        """Plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        fig = go.Figure()
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value, y=fraction_of_positives,
            mode='lines+markers',
            name='Model',
            line=dict(color='blue', width=2)
        ))
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash', width=1)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=600, height=500
        )
        
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_proba: np.ndarray,
                               title: str = "Threshold Analysis") -> go.Figure:
        """Plot metrics vs threshold."""
        threshold_df = MetricsCalculator.calculate_threshold_metrics(y_true, y_proba)
        
        fig = go.Figure()
        
        metrics_to_plot = ['precision', 'recall', 'f1_score']
        colors = ['blue', 'red', 'green']
        
        for metric, color in zip(metrics_to_plot, colors):
            fig.add_trace(go.Scatter(
                x=threshold_df['threshold'],
                y=threshold_df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Threshold',
            yaxis_title='Metric Value',
            width=800, height=500
        )
        
        return fig
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   title: str = "Prediction Distribution") -> go.Figure:
        """Plot distribution of predicted probabilities by true class."""
        fig = go.Figure()
        
        # Positive class
        pos_proba = y_proba[y_true == 1]
        fig.add_trace(go.Histogram(
            x=pos_proba,
            name='Positive Class',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability density'
        ))
        
        # Negative class
        neg_proba = y_proba[y_true == 0]
        fig.add_trace(go.Histogram(
            x=neg_proba,
            name='Negative Class',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability density'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted Probability',
            yaxis_title='Density',
            barmode='overlay',
            width=700, height=500
        )
        
        return fig
    
    def create_evaluation_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_proba: np.ndarray) -> go.Figure:
        """Create comprehensive evaluation dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'ROC Curve', 'Precision-Recall Curve', 'Confusion Matrix',
                'Calibration Curve', 'Threshold Analysis', 'Prediction Distribution'
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc_score:.3f})', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='red')), row=1, col=1)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        fig.add_trace(go.Scatter(x=recall, y=precision, name=f'PR (AP={ap_score:.3f})', line=dict(color='green')), row=1, col=2)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig.add_trace(go.Heatmap(z=cm, colorscale='Blues', showscale=False), row=1, col=3)
        
        # Calibration Curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
        fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, name='Calibration', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Perfect', line=dict(dash='dash', color='red')), row=2, col=1)
        
        # Threshold Analysis
        threshold_df = MetricsCalculator.calculate_threshold_metrics(y_true, y_proba)
        fig.add_trace(go.Scatter(x=threshold_df['threshold'], y=threshold_df['f1_score'], name='F1', line=dict(color='green')), row=2, col=2)
        
        # Prediction Distribution
        pos_proba = y_proba[y_true == 1]
        neg_proba = y_proba[y_true == 0]
        fig.add_trace(go.Histogram(x=pos_proba, name='Positive', opacity=0.7, nbinsx=20), row=2, col=3)
        fig.add_trace(go.Histogram(x=neg_proba, name='Negative', opacity=0.7, nbinsx=20), row=2, col=3)
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Evaluation Dashboard")
        
        return fig


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self):
        self.viz_generator = VisualizationGenerator()
    
    def generate_metrics_table(self, metrics: Dict[str, float]) -> str:
        """Generate HTML table of metrics."""
        html = "<table border='1' style='border-collapse: collapse; margin: 20px 0;'>\
"
        html += "<tr><th style='padding: 10px; background-color: #f2f2f2;'>Metric</th>"
        html += "<th style='padding: 10px; background-color: #f2f2f2;'>Value</th></tr>\
"
        
        # Define metric order and display names
        metric_display = {
            'roc_auc': 'ROC AUC',
            'average_precision': 'Average Precision',
            'f1_score': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall',
            'accuracy': 'Accuracy',
            'specificity': 'Specificity',
            'brier_score': 'Brier Score',
            'log_loss': 'Log Loss',
            'expected_calibration_error': 'Expected Calibration Error'
        }
        
        for metric_key, display_name in metric_display.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                if not np.isnan(value):
                    html += f"<tr><td style='padding: 10px;'>{display_name}</td>"
                    html += f"<td style='padding: 10px;'>{format_number(value)}</td></tr>\
"
        
        html += "</table>\
"
        return html
    
    def generate_html_report(self, 
                           model_name: str,
                           metrics: Dict[str, float],
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: np.ndarray,
                           feature_importance: Optional[Dict[str, float]] = None,
                           model_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive HTML report."""
        
        # Generate plots
        roc_fig = self.viz_generator.plot_roc_curve(y_true, y_proba)
        pr_fig = self.viz_generator.plot_precision_recall_curve(y_true, y_proba)
        cm_fig = self.viz_generator.plot_confusion_matrix(y_true, y_pred)
        cal_fig = self.viz_generator.plot_calibration_curve(y_true, y_proba)
        thresh_fig = self.viz_generator.plot_threshold_analysis(y_true, y_proba)
        dist_fig = self.viz_generator.plot_prediction_distribution(y_true, y_proba)
        
        # Convert plots to HTML
        roc_html = pyo.plot(roc_fig, output_type='div', include_plotlyjs=False)
        pr_html = pyo.plot(pr_fig, output_type='div', include_plotlyjs=False)
        cm_html = pyo.plot(cm_fig, output_type='div', include_plotlyjs=False)
        cal_html = pyo.plot(cal_fig, output_type='div', include_plotlyjs=False)
        thresh_html = pyo.plot(thresh_fig, output_type='div', include_plotlyjs=False)
        dist_html = pyo.plot(dist_fig, output_type='div', include_plotlyjs=False)
        
        # Generate metrics table
        metrics_table = self.generate_metrics_table(metrics)
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLForge Binary Classification Report - {model_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; }}
                .plot-container {{ display: inline-block; margin: 10px; }}
                .metrics-section {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MLForge Binary Classification Report</h1>
                <h2>Model: {model_name}</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section metrics-section">
                <h2>Performance Metrics</h2>
                {metrics_table}
            </div>
            
            <div class="section">
                <h2>Performance Curves</h2>
                <div class="plot-container">{roc_html}</div>
                <div class="plot-container">{pr_html}</div>
            </div>
            
            <div class="section">
                <h2>Confusion Matrix & Calibration</h2>
                <div class="plot-container">{cm_html}</div>
                <div class="plot-container">{cal_html}</div>
            </div>
            
            <div class="section">
                <h2>Threshold Analysis & Prediction Distribution</h2>
                <div class="plot-container">{thresh_html}</div>
                <div class="plot-container">{dist_html}</div>
            </div>
        """
        
        # Add feature importance if available
        if feature_importance:
            html += """
            <div class="section">
                <h2>Feature Importance</h2>
                <table border='1' style='border-collapse: collapse;'>
                    <tr><th style='padding: 10px; background-color: #f2f2f2;'>Feature</th>
                    <th style='padding: 10px; background-color: #f2f2f2;'>Importance</th></tr>
            """
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:20]:  # Top 20 features
                html += f"<tr><td style='padding: 10px;'>{feature}</td><td style='padding: 10px;'>{format_number(importance)}</td></tr>"
            
            html += "</table></div>"
        
        # Add model parameters if available
        if model_params:
            html += """
            <div class="section">
                <h2>Model Parameters</h2>
                <pre style='background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>
            """
            html += json.dumps(model_params, indent=2, default=str)
            html += "</pre></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, html_content: str, filepath: Union[str, Path]) -> None:
        """Save HTML report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report saved to: {filepath}")


class ComprehensiveEvaluator:
    """Main evaluator class that orchestrates all evaluation components."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.viz_generator = VisualizationGenerator()
        self.report_generator = ReportGenerator()
    
    def evaluate(self, 
                model,
                X_test: Union[np.ndarray, pd.DataFrame],
                y_test: np.ndarray,
                model_name: str = "Binary Classifier",
                generate_report: bool = False,
                report_path: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        
        with Timer() as timer:
            # Get predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_pred, y_proba)
            
            # Add evaluation metadata
            evaluation_results = {
                'model_name': model_name,
                'metrics': metrics,
                'evaluation_time': timer.elapsed_time,
                'n_samples': len(y_test),
                'n_positive': int(y_test.sum()),
                'n_negative': int(len(y_test) - y_test.sum())
            }
            
            # Generate report if requested
            if generate_report:
                if report_path is None:
                    report_path = f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                # Get model parameters if available
                model_params = None
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()
                
                # Generate HTML report
                html_content = self.report_generator.generate_html_report(
                    model_name=model_name,
                    metrics=metrics,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    model_params=model_params
                )
                
                self.report_generator.save_report(html_content, report_path)
                evaluation_results['report_path'] = str(report_path)
        
        return evaluation_results