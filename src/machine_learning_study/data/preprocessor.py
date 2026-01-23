"""Data preprocessing utilities for cleaning and transforming data."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """A comprehensive data preprocessor for cleaning and transforming data."""

    def __init__(self):
        """Initialize the data preprocessor."""
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            columns: Specific columns to impute. If None, impute all numeric columns.

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        if columns is None:
            # Get numeric columns with missing values
            columns = df.select_dtypes(include=[np.number]).columns[
                df.select_dtypes(include=[np.number]).isnull().any()
            ].tolist()

        logger.info(f"Handling missing values for columns: {columns} using strategy: {strategy}")

        for col in columns:
            if col not in self.imputers:
                if strategy == 'constant':
                    self.imputers[col] = SimpleImputer(strategy=strategy, fill_value=0)
                else:
                    self.imputers[col] = SimpleImputer(strategy=strategy)

            # Reshape for single feature
            values = df[col].values.reshape(-1, 1)
            df[col] = self.imputers[col].fit_transform(values).ravel()

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: Columns to encode. If None, encode all object columns.
            method: Encoding method ('onehot', 'ordinal', 'label')

        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.info(f"Encoding categorical columns: {columns} using method: {method}")

        for col in columns:
            if method == 'onehot':
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                    encoded = self.encoders[col].fit_transform(df[[col]])
                    feature_names = self.encoders[col].get_feature_names_out([col])
                else:
                    encoded = self.encoders[col].transform(df[[col]])
                    feature_names = self.encoders[col].get_feature_names_out([col])

                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)

            elif method == 'ordinal':
                if col not in self.encoders:
                    self.encoders[col] = OrdinalEncoder()
                    df[col] = self.encoders[col].fit_transform(df[[col]])
                else:
                    df[col] = self.encoders[col].transform(df[[col]])

            elif method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.encoders[col].transform(df[col])

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """Scale numerical features.

        Args:
            df: Input DataFrame
            columns: Columns to scale. If None, scale all numeric columns.
            method: Scaling method ('standard', 'minmax', 'robust')

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Scaling columns: {columns} using method: {method}")

        for col in columns:
            if col not in self.scalers:
                if method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                elif method == 'robust':
                    self.scalers[col] = RobustScaler()

            # Reshape for single feature
            values = df[col].values.reshape(-1, 1)
            df[col] = self.scalers[col].fit_transform(values).ravel()

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """Remove outliers from numerical columns.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers. If None, check all numeric columns.
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Removing outliers from columns: {columns} using method: {method}")

        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]

        return df

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a data quality report.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary containing data quality metrics
        """
        report = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_summary': df.describe().to_dict()
        }

        # Add categorical summary if any categorical columns exist
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report['categorical_summary'] = df[categorical_cols].describe().to_dict()

        return report