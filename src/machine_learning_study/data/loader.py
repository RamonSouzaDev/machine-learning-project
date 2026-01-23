"""Data loading utilities for the machine learning study project."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests
from sklearn.datasets import load_iris, load_breast_cancer, load_digits

logger = logging.getLogger(__name__)


class DataLoader:
    """A comprehensive data loader for various data sources."""

    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the data loader.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the loaded data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading CSV data from {file_path}")
        return pd.read_csv(file_path, **kwargs)

    def load_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from an Excel file.

        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            DataFrame containing the loaded data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading Excel data from {file_path}")
        return pd.read_excel(file_path, **kwargs)

    def download_file(self, url: str, filename: str) -> Path:
        """Download a file from a URL.

        Args:
            url: URL to download from
            filename: Name to save the file as

        Returns:
            Path to the downloaded file
        """
        file_path = self.data_dir / filename

        logger.info(f"Downloading {url} to {file_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded {filename} successfully")
        return file_path

    def load_sklearn_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset from scikit-learn.

        Args:
            dataset_name: Name of the sklearn dataset ('iris', 'breast_cancer', 'digits')

        Returns:
            Dictionary containing the dataset
        """
        logger.info(f"Loading sklearn dataset: {dataset_name}")

        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        elif dataset_name == 'digits':
            data = load_digits()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return {
            'data': data.data,
            'target': data.target,
            'feature_names': data.feature_names,
            'target_names': data.target_names,
            'DESCR': data.DESCR
        }

    def load_from_kaggle(self, dataset: str, filename: Optional[str] = None) -> pd.DataFrame:
        """Load a dataset from Kaggle (requires kaggle API).

        Args:
            dataset: Kaggle dataset identifier (user/dataset-name)
            filename: Specific file to load from the dataset

        Returns:
            DataFrame containing the loaded data
        """
        try:
            import kaggle
        except ImportError:
            raise ImportError("kaggle package required for Kaggle datasets. Install with: pip install kaggle")

        logger.info(f"Downloading Kaggle dataset: {dataset}")

        # Download dataset
        kaggle.api.competition_download_files(dataset, path=str(self.data_dir))

        # Load the data
        if filename:
            file_path = self.data_dir / filename
        else:
            # Try to find CSV files in the downloaded dataset
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            file_path = csv_files[0]

        return self.load_csv(file_path)