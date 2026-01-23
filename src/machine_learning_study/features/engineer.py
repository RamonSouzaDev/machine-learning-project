"""Feature engineering utilities for creating and transforming features."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """A comprehensive feature engineering class."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_selectors = {}

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
        interaction_only: bool = False
    ) -> pd.DataFrame:
        """Create polynomial features from existing features.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Degree of polynomial features
            interaction_only: If True, only interaction features are produced

        Returns:
            DataFrame with polynomial features added
        """
        from sklearn.preprocessing import PolynomialFeatures

        df = df.copy()

        logger.info(f"Creating polynomial features of degree {degree} for columns: {columns}")

        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )

        poly_features = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)

        # Remove original column names from feature names if they exist
        new_feature_names = []
        for name in feature_names:
            if name not in columns:
                new_feature_names.append(f"poly_{name}")

        if new_feature_names:
            poly_df = pd.DataFrame(
                poly_features[:, len(columns):],
                columns=new_feature_names,
                index=df.index
            )
            df = pd.concat([df, poly_df], axis=1)

        return df

    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """Create interaction features between pairs of features.

        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing feature pairs

        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()

        logger.info(f"Creating interaction features for pairs: {feature_pairs}")

        for feat1, feat2 in feature_pairs:
            interaction_name = f"{feat1}_{feat2}_interaction"
            df[interaction_name] = df[feat1] * df[feat2]

        return df

    def create_binning_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        bins: Union[int, List[float]] = 5,
        strategy: str = 'quantile'
    ) -> pd.DataFrame:
        """Create binned features from continuous variables.

        Args:
            df: Input DataFrame
            columns: Columns to bin
            bins: Number of bins or bin edges
            strategy: Binning strategy ('quantile', 'uniform')

        Returns:
            DataFrame with binned features added
        """
        from sklearn.preprocessing import KBinsDiscretizer

        df = df.copy()

        logger.info(f"Creating binned features for columns: {columns}")

        for col in columns:
            if strategy == 'quantile':
                discretizer = KBinsDiscretizer(
                    n_bins=bins if isinstance(bins, int) else len(bins) - 1,
                    encode='ordinal',
                    strategy='quantile'
                )
            elif strategy == 'uniform':
                discretizer = KBinsDiscretizer(
                    n_bins=bins if isinstance(bins, int) else len(bins) - 1,
                    encode='ordinal',
                    strategy='uniform'
                )

            binned = discretizer.fit_transform(df[[col]])
            df[f"{col}_binned"] = binned.ravel()

        return df

    def create_statistical_features(
        self,
        df: pd.DataFrame,
        groupby_col: str,
        agg_cols: List[str],
        operations: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Create statistical features based on grouping.

        Args:
            df: Input DataFrame
            groupby_col: Column to group by
            agg_cols: Columns to aggregate
            operations: Statistical operations to perform

        Returns:
            DataFrame with statistical features added
        """
        df = df.copy()

        logger.info(f"Creating statistical features grouped by {groupby_col}")

        for op in operations:
            if op == 'mean':
                stats = df.groupby(groupby_col)[agg_cols].transform('mean')
                df = df.join(stats.add_suffix(f'_{groupby_col}_mean'))
            elif op == 'std':
                stats = df.groupby(groupby_col)[agg_cols].transform('std')
                df = df.join(stats.add_suffix(f'_{groupby_col}_std'))
            elif op == 'min':
                stats = df.groupby(groupby_col)[agg_cols].transform('min')
                df = df.join(stats.add_suffix(f'_{groupby_col}_min'))
            elif op == 'max':
                stats = df.groupby(groupby_col)[agg_cols].transform('max')
                df = df.join(stats.add_suffix(f'_{groupby_col}_max'))

        return df

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info',
        k: int = 10,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """Select the most important features.

        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            k: Number of features to select
            task: Type of ML task ('classification' or 'regression')

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting top {k} features using {method}")

        if method == 'mutual_info':
            if task == 'classification':
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == 'f_statistic':
            if task == 'classification':
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'chi2':
            if task == 'classification':
                # Chi-squared works only for non-negative features
                selector = SelectKBest(score_func=chi2, k=k)
            else:
                raise ValueError("Chi-squared test is only for classification tasks")

        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected features: {selected_features}")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def create_time_features(
        self,
        df: pd.DataFrame,
        datetime_col: str
    ) -> pd.DataFrame:
        """Create time-based features from datetime column.

        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column

        Returns:
            DataFrame with time features added
        """
        df = df.copy()

        logger.info(f"Creating time features from column: {datetime_col}")

        # Ensure datetime column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Extract time components
        df[f"{datetime_col}_year"] = df[datetime_col].dt.year
        df[f"{datetime_col}_month"] = df[datetime_col].dt.month
        df[f"{datetime_col}_day"] = df[datetime_col].dt.day
        df[f"{datetime_col}_hour"] = df[datetime_col].dt.hour
        df[f"{datetime_col}_dayofweek"] = df[datetime_col].dt.dayofweek
        df[f"{datetime_col}_quarter"] = df[datetime_col].dt.quarter
        df[f"{datetime_col}_is_weekend"] = df[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)

        return df

    def create_text_features(
        self,
        df: pd.DataFrame,
        text_cols: List[str]
    ) -> pd.DataFrame:
        """Create text-based features from text columns.

        Args:
            df: Input DataFrame
            text_cols: List of text columns

        Returns:
            DataFrame with text features added
        """
        df = df.copy()

        logger.info(f"Creating text features for columns: {text_cols}")

        for col in text_cols:
            # Basic text features
            df[f"{col}_length"] = df[col].astype(str).str.len()
            df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
            df[f"{col}_uppercase_ratio"] = df[col].astype(str).str.findall(r'[A-Z]').str.len() / df[f"{col}_length"]
            df[f"{col}_lowercase_ratio"] = df[col].astype(str).str.findall(r'[a-z]').str.len() / df[f"{col}_length"]
            df[f"{col}_digit_ratio"] = df[col].astype(str).str.findall(r'\d').str.len() / df[f"{col}_length"]

        return df