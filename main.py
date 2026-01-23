#!/usr/bin/env python3
"""Main entry point for the Machine Learning Study Project.

This script demonstrates the usage of the machine learning study project
and serves as an example of how to structure a modern ML application.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from machine_learning_study.pipelines import run_experiment
from machine_learning_study.utils.logging_config import setup_logging


def main():
    """Main function to run the ML study project."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Study Project - A comprehensive ML study guide"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="classification_example",
        choices=["classification_example", "regression_example", "clustering_example"],
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting Machine Learning Study Project")
    logger.info(f"Running experiment: {args.experiment}")
    logger.info(f"Configuration file: {args.config}")

    try:
        # Run the selected experiment
        results = run_experiment(args.experiment, args.config)
        logger.info("Experiment completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()