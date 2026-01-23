"""Classification models for the machine learning study project."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class Classifier:
    """A comprehensive classification model class."""

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """Initialize the classifier.

        Args:
            model_type: Type of classifier to use
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.is_fitted = False

    def _create_model(self, **kwargs) -> Any:
        """Create the specified model.

        Args:
            **kwargs: Model parameters

        Returns:
            Initialized model
        """
        models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC,
            'decision_tree': DecisionTreeClassifier,
            'knn': KNeighborsClassifier,
            'naive_bayes': GaussianNB,
        }

        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return models[self.model_type](**kwargs)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'Classifier':
        """Fit the classifier to the training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.model_type} classifier")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info("Model fitted successfully")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_type} does not support probability predictions")

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, float]:
        """Perform cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        logger.info(f"Performing {cv}-fold cross-validation with {scoring} scoring")

        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)

        results = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores.tolist()
        }

        logger.info(f"Cross-validation results: {results}")
        return results

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available.

        Returns:
            Feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            logger.warning(f"{self.model_type} does not provide feature importance")
            return None


class EnsembleClassifier:
    """An ensemble classifier combining multiple models."""

    def __init__(self, models: List[Tuple[str, Classifier]], voting: str = 'hard'):
        """Initialize the ensemble classifier.

        Args:
            models: List of (name, model) tuples
            voting: Voting strategy ('hard' or 'soft')
        """
        self.models = models
        self.voting = voting
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model.model) for name, model in models],
            voting=voting
        )
        self.is_fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'EnsembleClassifier':
        """Fit the ensemble classifier.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ensemble classifier with {len(self.models)} models")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.ensemble_model.fit(X, y)
        self.is_fitted = True

        logger.info("Ensemble model fitted successfully")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with the ensemble.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before making predictions")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.ensemble_model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities with the ensemble.

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before making predictions")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.ensemble_model.predict_proba(X)