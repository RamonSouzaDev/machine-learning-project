"""Experiment pipeline for running machine learning experiments."""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from ..models.classifier import Classifier
from ..models.evaluator import ModelEvaluator
from ..models.regressor import Regressor
from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class ExperimentPipeline(BasePipeline):
    """Pipeline for running machine learning experiments."""

    def __init__(self, config_path: str):
        """Initialize the experiment pipeline.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        super().__init__(config)
        self.task = config.get('task', 'classification')
        self.evaluator = ModelEvaluator(task=self.task)

    def load_data(self) -> pd.DataFrame:
        """Load data based on configuration."""
        data_config = self.config.get('data', {})

        if 'sklearn_dataset' in data_config:
            # Load from sklearn
            dataset_name = data_config['sklearn_dataset']
            dataset = self.data_loader.load_sklearn_dataset(dataset_name)
            df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
            df[self.config['target_column']] = dataset['target']

        elif 'csv_file' in data_config:
            # Load from CSV
            df = self.data_loader.load_csv(data_config['csv_file'])

        elif 'url' in data_config:
            # Download and load from URL
            filename = data_config.get('filename', 'downloaded_data.csv')
            file_path = self.data_loader.download_file(data_config['url'], filename)
            df = self.data_loader.load_csv(file_path)

        else:
            raise ValueError("No valid data source specified in configuration")

        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data based on configuration."""
        preprocess_config = self.config.get('preprocessing', {})

        # Handle missing values
        if 'missing_values' in preprocess_config:
            strategy = preprocess_config['missing_values'].get('strategy', 'mean')
            columns = preprocess_config['missing_values'].get('columns')
            df = self.preprocessor.handle_missing_values(df, strategy=strategy, columns=columns)

        # Encode categorical variables
        if 'categorical_encoding' in preprocess_config:
            method = preprocess_config['categorical_encoding'].get('method', 'onehot')
            columns = preprocess_config['categorical_encoding'].get('columns')
            df = self.preprocessor.encode_categorical(df, columns=columns, method=method)

        # Scale features
        if 'scaling' in preprocess_config:
            method = preprocess_config['scaling'].get('method', 'standard')
            columns = preprocess_config['scaling'].get('columns')
            df = self.preprocessor.scale_features(df, columns=columns, method=method)

        # Remove outliers
        if 'outliers' in preprocess_config:
            method = preprocess_config['outliers'].get('method', 'iqr')
            threshold = preprocess_config['outliers'].get('threshold', 1.5)
            columns = preprocess_config['outliers'].get('columns')
            df = self.preprocessor.remove_outliers(df, columns=columns, method=method, threshold=threshold)

        logger.info(f"Data preprocessing completed. Shape: {df.shape}")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on configuration."""
        feature_config = self.config.get('features', {})

        # Polynomial features
        if 'polynomial' in feature_config:
            columns = feature_config['polynomial'].get('columns', [])
            degree = feature_config['polynomial'].get('degree', 2)
            if columns:
                df = self.feature_engineer.create_polynomial_features(df, columns, degree=degree)

        # Interaction features
        if 'interactions' in feature_config:
            feature_pairs = feature_config['interactions'].get('pairs', [])
            if feature_pairs:
                df = self.feature_engineer.create_interaction_features(df, feature_pairs)

        # Binning features
        if 'binning' in feature_config:
            columns = feature_config['binning'].get('columns', [])
            bins = feature_config['binning'].get('bins', 5)
            if columns:
                df = self.feature_engineer.create_binning_features(df, columns, bins=bins)

        # Statistical features
        if 'statistics' in feature_config:
            groupby_col = feature_config['statistics'].get('groupby_column')
            agg_cols = feature_config['statistics'].get('aggregate_columns', [])
            operations = feature_config['statistics'].get('operations', ['mean', 'std'])
            if groupby_col and agg_cols:
                df = self.feature_engineer.create_statistical_features(
                    df, groupby_col, agg_cols, operations
                )

        # Feature selection
        if 'selection' in feature_config:
            k = feature_config['selection'].get('k', 10)
            method = feature_config['selection'].get('method', 'mutual_info')
            target_col = self.config['target_column']

            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_selected = self.feature_engineer.select_features(X, y, method=method, k=k, task=self.task)
            df = pd.concat([X_selected, y], axis=1)

        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df

    def build_model(self) -> Union[Classifier, Regressor]:
        """Build the model based on configuration."""
        model_config = self.config.get('model', {})

        if self.task == 'classification':
            model_type = model_config.get('type', 'random_forest')
            params = model_config.get('parameters', {})
            model = Classifier(model_type=model_type, **params)
        else:
            model_type = model_config.get('type', 'random_forest')
            params = model_config.get('parameters', {})
            model = Regressor(model_type=model_type, **params)

        logger.info(f"Built {self.task} model: {model_type}")
        return model

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment and return results."""
        logger.info("Starting experiment execution")

        # Run the base pipeline
        pipeline_results = self.run_pipeline()

        # Evaluate the model
        X_test = pipeline_results['splits']['X_test']
        y_test = pipeline_results['splits']['y_test']
        predictions = pipeline_results['predictions']

        if self.task == 'classification':
            # Get probabilities if available
            try:
                probabilities = self.model.predict_proba(X_test)
                evaluation_results = self.evaluator.evaluate_classification(
                    y_test, predictions, probabilities
                )
            except:
                evaluation_results = self.evaluator.evaluate_classification(
                    y_test, predictions
                )
        else:
            evaluation_results = self.evaluator.evaluate_regression(y_test, predictions)

        # Cross-validation
        X = pd.concat([pipeline_results['splits']['X_train'], X_test])
        y = pd.concat([pipeline_results['splits']['y_train'], y_test])

        cv_results = self.evaluator.cross_validate_and_evaluate(
            self.model.model, X, y, cv=self.config.get('cross_validation', {}).get('folds', 5)
        )

        results = {
            'pipeline_results': pipeline_results,
            'evaluation_results': evaluation_results,
            'cross_validation_results': cv_results,
            'config': self.config
        }

        logger.info("Experiment execution completed")
        return results