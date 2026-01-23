"""Model evaluation utilities for comprehensive performance assessment."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class for classification and regression tasks."""

    def __init__(self, task: str = 'classification'):
        """Initialize the model evaluator.

        Args:
            task: Type of ML task ('classification' or 'regression')
        """
        self.task = task
        self.metrics = {}

    def evaluate_classification(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Evaluate classification model performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)

        Returns:
            Dictionary of evaluation metrics
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        if y_proba is not None and isinstance(y_proba, pd.Series):
            y_proba = y_proba.values

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }

        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                else:
                    # Multi-class classification
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")

        self.metrics = metrics
        logger.info(f"Classification metrics: {metrics}")
        return metrics

    def evaluate_regression(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate regression model performance.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of evaluation metrics
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred),
        }

        # Additional metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100

        self.metrics = metrics
        logger.info(f"Regression metrics: {metrics}")
        return metrics

    def get_classification_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> str:
        """Get detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as string
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        return classification_report(y_true, y_pred)

    def get_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Get confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix as numpy array
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
        """
        cm = self.get_confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()

    def plot_roc_curve(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_proba: Union[pd.Series, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve for binary classification.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_proba, pd.Series):
            y_proba = y_proba.values

        if len(np.unique(y_true)) != 2:
            logger.warning("ROC curve is only for binary classification")
            return

        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        else:
            plt.show()

    def plot_precision_recall_curve(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_proba: Union[pd.Series, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision-recall curve for binary classification.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_proba, pd.Series):
            y_proba = y_proba.values

        if len(np.unique(y_true)) != 2:
            logger.warning("Precision-Recall curve is only for binary classification")
            return

        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve plot saved to {save_path}")
        else:
            plt.show()

    def plot_residuals(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Plot residuals for regression tasks.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        residuals = y_true - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True)

        # Residuals distribution
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        else:
            plt.show()

    def cross_validate_and_evaluate(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5
    ) -> Dict[str, Any]:
        """Perform cross-validation and return comprehensive evaluation.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results and evaluation metrics
        """
        from sklearn.model_selection import cross_val_predict

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        logger.info(f"Performing {cv}-fold cross-validation evaluation")

        # Get cross-validation predictions
        if self.task == 'classification':
            y_pred_cv = cross_val_predict(model, X, y, cv=cv, method='predict')
            try:
                y_proba_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
            except:
                y_proba_cv = None
        else:
            y_pred_cv = cross_val_predict(model, X, y, cv=cv)
            y_proba_cv = None

        # Evaluate performance
        if self.task == 'classification':
            metrics = self.evaluate_classification(y, y_pred_cv, y_proba_cv)
        else:
            metrics = self.evaluate_regression(y, y_pred_cv)

        # Add cross-validation info
        result = {
            'metrics': metrics,
            'cv_folds': cv,
            'task': self.task
        }

        logger.info(f"Cross-validation evaluation completed: {result}")
        return result