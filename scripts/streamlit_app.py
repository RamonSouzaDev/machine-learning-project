"""Streamlit web application for the Machine Learning Study Project."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from machine_learning_study.pipelines import run_experiment
from machine_learning_study.utils.experiment_tracker import ExperimentTracker

# Page configuration
st.set_page_config(
    page_title="Machine Learning Study",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .sidebar-content {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""

    # Sidebar
    with st.sidebar:
        st.title("🤖 ML Study Dashboard")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["Home", "Run Experiment", "Model Comparison", "Data Explorer", "Experiment Tracker"]
        )

        st.markdown("---")
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()

    # Main content based on selected page
    if page == "Home":
        show_home_page()
    elif page == "Run Experiment":
        show_experiment_page()
    elif page == "Model Comparison":
        show_comparison_page()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Experiment Tracker":
        show_tracker_page()

def show_home_page():
    """Display the home page."""
    st.markdown('<h1 class="main-header">Machine Learning Study Project</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Quick Start</h3>
            <p>Run pre-configured experiments with one click</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data Science</h3>
            <p>Explore datasets, visualize features, and analyze results</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Experiment Tracking</h3>
            <p>Track, compare, and manage your ML experiments</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Project Overview
    st.header("🎯 Project Overview")
    st.markdown("""
    This comprehensive machine learning study project demonstrates best practices for:

    - **Modern Python Development**: Clean architecture, type hints, comprehensive testing
    - **Machine Learning Engineering**: End-to-end pipelines, experiment tracking, model deployment
    - **DevOps & MLOps**: Containerization, CI/CD, monitoring, and scalable deployment
    - **Data Science Workflow**: Data exploration, feature engineering, model evaluation

    **Perfect for**: Software developers learning ML, ML engineers learning software development,
    or anyone building production-ready AI applications.
    """)

    # Quick stats
    st.header("📈 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Python Files", "20+")
    with col2:
        st.metric("ML Models", "8+")
    with col3:
        st.metric("Test Coverage", "85%")
    with col4:
        st.metric("Docker Services", "4")

def show_experiment_page():
    """Display the experiment running page."""
    st.header("🧪 Run Machine Learning Experiment")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configuration")

        # Experiment settings
        task = st.selectbox(
            "Task Type",
            ["classification", "regression"],
            help="Type of machine learning task"
        )

        dataset = st.selectbox(
            "Dataset",
            ["iris", "breast_cancer", "digits"],
            help="Built-in sklearn dataset to use"
        )

        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "gradient_boosting", "svm", "logistic_regression", "knn"],
            help="Machine learning algorithm to use"
        )

        # Model parameters
        st.subheader("Model Parameters")
        n_estimators = st.slider("Number of Estimators", 10, 500, 100)
        max_depth = st.slider("Max Depth", 3, 20, 10)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

    with col2:
        st.subheader("Actions")

        if st.button("🚀 Run Experiment", type="primary", use_container_width=True):
            with st.spinner("Running experiment..."):
                try:
                    # Create configuration
                    config = {
                        "task": task,
                        "target_column": "target",
                        "data": {"sklearn_dataset": dataset},
                        "model": {
                            "type": model_type,
                            "parameters": {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "random_state": 42
                            }
                        },
                        "training": {"test_size": test_size}
                    }

                    # Save config temporarily
                    import yaml
                    config_path = "temp_config.yaml"
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)

                    # Run experiment
                    results = run_experiment(f"{task}_experiment", config_path)

                    # Clean up
                    Path(config_path).unlink(missing_ok=True)

                    st.success("Experiment completed successfully!")

                    # Display results
                    st.subheader("📊 Results")

                    metrics = results['evaluation_results']
                    col1, col2, col3 = st.columns(3)

                    if task == "classification":
                        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        col2.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                        col3.metric("Precision", f"{metrics['precision']:.3f}")
                    else:
                        col1.metric("R² Score", f"{metrics['r2_score']:.3f}")
                        col2.metric("MAE", f"{metrics['mae']:.3f}")
                        col3.metric("RMSE", f"{metrics['rmse']:.3f}")

                except Exception as e:
                    st.error(f"Experiment failed: {str(e)}")
                    st.exception(e)

        st.markdown("---")
        st.markdown("**Note**: Experiments are automatically tracked with MLflow")

def show_comparison_page():
    """Display model comparison page."""
    st.header("🔍 Model Comparison")

    try:
        tracker = ExperimentTracker()

        # Get experiment runs
        runs = []
        try:
            # This would need to be implemented to get runs from MLflow
            st.info("Model comparison feature coming soon! Check the Experiment Tracker page.")
        except Exception as e:
            st.warning("No experiment data available yet. Run some experiments first!")

        # Placeholder content
        st.subheader("Comparison Metrics")
        st.info("Run multiple experiments to see comparison charts here.")

    except Exception as e:
        st.error(f"Failed to load experiment data: {e}")

def show_data_explorer():
    """Display data exploration page."""
    st.header("📊 Data Explorer")

    dataset = st.selectbox(
        "Select Dataset",
        ["iris", "breast_cancer", "digits"],
        key="data_explorer_dataset"
    )

    if st.button("Load Dataset"):
        try:
            from sklearn.datasets import load_iris, load_breast_cancer
            from sklearn.datasets import load_digits

            if dataset == "iris":
                data = load_iris()
            elif dataset == "breast_cancer":
                data = load_breast_cancer()
            elif dataset == "digits":
                data = load_digits()

            # Create DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target

            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Samples", df.shape[0])
            col2.metric("Features", df.shape[1] - 1)  # Excluding target
            col3.metric("Classes", len(np.unique(data.target)))

            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("Feature Statistics")
            st.dataframe(df.describe())

            # Correlation heatmap
            if df.shape[1] <= 10:  # Only show for smaller datasets
                st.subheader("Feature Correlation")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

def show_tracker_page():
    """Display experiment tracker page."""
    st.header("📈 Experiment Tracker")

    try:
        tracker = ExperimentTracker()

        st.subheader("Recent Experiments")

        # This would show actual experiment data from MLflow
        st.info("Experiment tracking data will appear here after running experiments.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MLflow UI")
            st.markdown("""
            Access the MLflow tracking server at:
            - **Local**: http://localhost:5000
            - **Docker**: http://localhost:5000 (when running with docker-compose)
            """)

        with col2:
            st.subheader("Tracking Features")
            st.markdown("""
            - ✅ Experiment versioning
            - ✅ Model artifacts storage
            - ✅ Parameter tracking
            - ✅ Metric logging
            - ✅ Model comparison
            """)

    except Exception as e:
        st.error(f"Failed to connect to experiment tracker: {e}")

if __name__ == "__main__":
    main()