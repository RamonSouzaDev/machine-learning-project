"""ML Pipelines module."""

from .experiment_pipeline import ExperimentPipeline

__all__ = ["ExperimentPipeline"]


def run_experiment(experiment_type: str, config_path: str) -> dict:
    """Run a machine learning experiment.

    Args:
        experiment_type: Type of experiment to run
        config_path: Path to configuration file

    Returns:
        Experiment results
    """
    pipeline = ExperimentPipeline(config_path)
    return pipeline.run_experiment()