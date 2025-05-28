"""
Command-line interface for MLForge-Binary
"""

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False
    
import pandas as pd
import numpy as np
from pathlib import Path

from .classifier import BinaryClassifier
from .automl import AutoML


def cli():
    """Main CLI entry point."""
    if not HAS_CLICK:
        print("CLI requires click package. Install with: pip install click")
        return
    
    @click.group()
    def main():
        """MLForge-Binary CLI for binary classification tasks."""
        pass
    
    @main.command()
    @click.argument('data_file')
    @click.option('--target', required=True, help='Target column name')
    @click.option('--model', default='auto', help='Model type to use')
    @click.option('--output', default='model.pkl', help='Output model file')
    @click.option('--report', is_flag=True, help='Generate evaluation report')
    def train(data_file, target, model, output, report):
        """Train a binary classifier."""
        # Load data
        data = pd.read_csv(data_file)
        X = data.drop(columns=[target])
        y = data[target]
        
        # Create and train model
        clf = BinaryClassifier(model=model)
        clf.fit(X, y)
        
        # Save model
        clf.save(output)
        click.echo(f"Model saved to {output}")
        
        # Generate report if requested
        if report:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf.fit(X_train, y_train)
            
            report_path = output.replace('.pkl', '_report.html')
            clf.evaluate(X_test, y_test, generate_report=True, report_path=report_path)
            click.echo(f"Report saved to {report_path}")
    
    @main.command()
    @click.argument('data_file')
    @click.option('--model', required=True, help='Model file to use')
    @click.option('--output', default='predictions.csv', help='Output predictions file')
    def predict(data_file, model, output):
        """Make predictions using a trained model."""
        # Load data and model
        X = pd.read_csv(data_file)
        clf = BinaryClassifier.load(model)
        
        # Make predictions
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)[:, 1]
        
        # Save results
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
        results.to_csv(output, index=False)
        click.echo(f"Predictions saved to {output}")
    
    @main.command()
    @click.argument('data_file')
    @click.option('--target', required=True, help='Target column name')
    @click.option('--time-budget', default=300, help='Time budget in seconds')
    def automl(data_file, target, time_budget):
        """Run AutoML experiment."""
        # Load data
        data = pd.read_csv(data_file)
        X = data.drop(columns=[target])
        y = data[target]
        
        # Run AutoML
        automl_model = AutoML(time_budget=time_budget)
        automl_model.fit(X, y)
        
        # Save best model
        best_model = automl_model.get_best_model()
        best_model.save('automl_best_model.pkl')
        
        # Show results
        click.echo("AutoML completed!")
        click.echo("Leaderboard:")
        click.echo(automl_model.leaderboard_.to_string(index=False))
    
    main()


if __name__ == '__main__':
    cli()
