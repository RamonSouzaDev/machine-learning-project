"""Experiment tracking utilities using MLflow."""

import logging
from typing import Any, Dict, Optional, Union

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-based experiment tracking for machine learning projects."""

    def __init__(
        self,
        experiment_name: str = "machine-learning-study",
        tracking_uri: Optional[str] = None
    ):
        """Initialize the experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")

        self.client = MlflowClient()

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Optional name for the run

        Returns:
            MLflow active run object
        """
        mlflow.set_experiment(self.experiment_name)

        if run_name:
            run = mlflow.start_run(run_name=run_name)
        else:
            run = mlflow.start_run()

        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time-series metrics
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics")

    def log_model(self, model: Any, model_name: str = "model") -> None:
        """Log a model to the current run.

        Args:
            model: Trained model object
            model_name: Name for the logged model
        """
        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged model: {model_name}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file or directory.

        Args:
            local_path: Local path to the artifact
            artifact_path: Optional artifact path within the run
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact
        """
        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"Logged dictionary to: {artifact_file}")

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags for the current run.

        Args:
            tags: Dictionary of tags to set
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        logger.info(f"Set {len(tags)} tags")

    def log_experiment_results(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Union[int, float]],
        model: Any,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log complete experiment results.

        Args:
            config: Experiment configuration
            metrics: Evaluation metrics
            model: Trained model
            additional_info: Additional information to log

        Returns:
            Run ID of the logged experiment
        """
        with self.start_run() as run:
            # Log parameters
            self.log_parameters(config)

            # Log metrics
            self.log_metrics(metrics)

            # Log model
            self.log_model(model)

            # Log additional information
            if additional_info:
                self.log_dict(additional_info, "experiment_info.json")

            # Set tags
            tags = {
                "task": config.get("task", "unknown"),
                "model_type": config.get("model", {}).get("type", "unknown"),
                "dataset": config.get("data", {}).get("sklearn_dataset", "unknown")
            }
            self.set_tags(tags)

            run_id = run.info.run_id
            logger.info(f"Completed logging experiment results for run: {run_id}")

            return run_id

    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Get information about a specific run.

        Args:
            run_id: Run ID to query

        Returns:
            Dictionary with run information
        """
        run = self.client.get_run(run_id)
        return {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags
        }

    def compare_runs(self, run_ids: list) -> pd.DataFrame:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with comparison results
        """
        import pandas as pd

        runs_data = []
        for run_id in run_ids:
            run_info = self.get_run_info(run_id)
            runs_data.append({
                "run_id": run_id,
                **run_info["params"],
                **run_info["metrics"]
            })

        return pd.DataFrame(runs_data)

    def get_best_run(
        self,
        metric: str,
        mode: str = "max",
        experiment_ids: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric.

        Args:
            metric: Metric to optimize
            mode: "max" or "min" for optimization direction
            experiment_ids: Optional list of experiment IDs to search

        Returns:
            Best run information or None if no runs found
        """
        if experiment_ids is None:
            experiment_ids = [self.experiment_id]

        best_run = None
        best_value = float('-inf') if mode == "max" else float('inf')

        for exp_id in experiment_ids:
            runs = self.client.search_runs(
                experiment_ids=[exp_id],
                filter_string=f"metrics.{metric} IS NOT NULL",
                order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"]
            )

            if runs:
                run = runs[0]
                value = run.data.metrics.get(metric)
                if (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
                    best_value = value
                    best_run = self.get_run_info(run.info.run_id)

        return best_run