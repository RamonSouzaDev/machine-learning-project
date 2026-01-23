"""Base pipeline class for machine learning workflows."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for machine learning pipelines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_loader = DataLoader(self.config.get('data_dir', 'data'))
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.is_trained = False

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        pass

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded data."""
        pass

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and engineer features."""
        pass

    @abstractmethod
    def build_model(self) -> Any:
        """Build and configure the model."""
        pass

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary with train/test splits
        """
        logger.info(f"Splitting data with test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.config.get('stratify', True) else None
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        logger.info("Training model")

        if self.model is None:
            self.model = self.build_model()

        self.model.fit(X_train, y_train)
        self.is_trained = True

        logger.info("Model training completed")
        return self.model

    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        logger.info(f"Saving model to {filepath}")

        model_data = {
            'model': self.model,
            'config': self.config,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info("Model saved successfully")

    def load_model(self, filepath: str) -> None:
        """Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        logger.info(f"Loading model from {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']

        logger.info("Model loaded successfully")

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline.

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting pipeline execution")

        # Load data
        df = self.load_data()

        # Preprocess data
        df_processed = self.preprocess_data(df)

        # Create features
        df_features = self.create_features(df_processed)

        # Prepare for modeling
        target_col = self.config.get('target_column')
        if target_col not in df_features.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        X = df_features.drop(columns=[target_col])
        y = df_features[target_col]

        # Split data
        splits = self.split_data(X, y)

        # Train model
        model = self.train(splits['X_train'], splits['y_train'])

        # Make predictions on test set
        predictions = self.predict(splits['X_test'])

        results = {
            'model': model,
            'predictions': predictions,
            'splits': splits,
            'config': self.config
        }

        logger.info("Pipeline execution completed")
        return results