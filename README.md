# Machine Learning Study Project 🚀

A comprehensive machine learning study project demonstrating best practices for software development with AI technologies in 2025-2026.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.8%2B-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)

## 🌟 Overview

This project serves as a complete guide for studying software development practices combined with modern machine learning engineering principles. It's designed to showcase how to build production-ready ML systems following industry best practices.

### 🎯 Key Features

- **Modern Python Project Structure**: Organized with best practices for scalability
- **Comprehensive ML Pipeline**: End-to-end workflow from data to deployment
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Containerization**: Docker support for consistent environments
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Model Evaluation**: Comprehensive metrics and validation techniques
- **Feature Engineering**: Advanced feature creation and selection methods
- **Multiple ML Algorithms**: Classification and regression models ready to use

## 📁 Project Structure

```
machine-learning-study/
├── src/machine_learning_study/     # Main package
│   ├── data/                       # Data loading and preprocessing
│   ├── models/                     # ML models and algorithms
│   ├── features/                   # Feature engineering
│   ├── pipelines/                  # ML pipelines and workflows
│   └── utils/                      # Utilities and helpers
├── tests/                          # Unit and integration tests
├── notebooks/                      # Jupyter notebooks and examples
├── config/                         # Configuration files
├── models/                         # Model checkpoints and artifacts
├── data/                          # Datasets (raw and processed)
├── scripts/                        # Utility scripts
├── docs/                          # Documentation
├── .github/workflows/             # CI/CD pipelines
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RamonSouzaDev/machine-learning-project.git
   cd machine-learning-study
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Running Your First Experiment

1. **Basic usage with command line**
   ```bash
   python main.py --experiment classification_example
   ```

2. **Using Docker**
   ```bash
   docker-compose up ml-study
   ```

3. **Interactive Jupyter environment**
   ```bash
   docker-compose up jupyter
   # Access at http://localhost:8888
   ```

## 📊 Example Usage

### Classification Example

```python
from machine_learning_study.pipelines import run_experiment

# Run a classification experiment
results = run_experiment("classification_example", "config/default.yaml")

print("Model Performance:")
print(f"Accuracy: {results['evaluation_results']['accuracy']:.3f}")
print(f"F1-Score: {results['evaluation_results']['f1_score']:.3f}")
```

### Custom Experiment Configuration

```yaml
# config/custom_experiment.yaml
task: "classification"
target_column: "target"
data:
  sklearn_dataset: "breast_cancer"

model:
  type: "gradient_boosting"
  parameters:
    n_estimators: 200
    learning_rate: 0.1

features:
  selection:
    method: "mutual_info"
    k: 15
```

## 🏗️ Architecture & Best Practices

### Software Development Practices

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Type Hints**: Full type annotation for better code maintainability
- **Comprehensive Testing**: Unit and integration tests with pytest
- **Code Quality**: Linting with flake8, formatting with black and isort
- **Documentation**: Comprehensive docstrings and README files

### Machine Learning Engineering

- **Reproducible Experiments**: MLflow tracking for experiment management
- **Data Versioning**: DVC integration for dataset versioning
- **Model Validation**: Cross-validation and comprehensive metrics
- **Feature Engineering**: Automated feature creation and selection
- **Hyperparameter Tuning**: Integration with Optuna for optimization

### DevOps & Deployment

- **Containerization**: Docker for consistent development environments
- **CI/CD**: Automated testing and deployment pipelines
- **Environment Management**: Dependency management with pip-tools
- **Monitoring**: Logging and experiment tracking

## 🧪 Experiment Tracking

Track your experiments with MLflow:

```python
from machine_learning_study.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("my-experiment")
with tracker.start_run("experiment-1"):
    tracker.log_parameters({"learning_rate": 0.01, "epochs": 100})
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
    tracker.log_model(trained_model)
```

## 📈 Model Evaluation

Comprehensive evaluation metrics for both classification and regression:

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix
- Precision-Recall curves

### Regression Metrics
- MAE, MSE, RMSE, R² Score
- Residual plots and analysis

## 🔧 Configuration

Experiments are configured using YAML files. The configuration system supports:

- Data source specification
- Preprocessing pipeline configuration
- Feature engineering settings
- Model hyperparameters
- Evaluation parameters

## 🐳 Docker Deployment

### Development Environment
```bash
# Start development environment
docker-compose up

# Run specific services
docker-compose up ml-study jupyter mlflow
```

### Production Deployment
```bash
# Build production image
docker build -t ml-study:latest .

# Run production container
docker run -p 8000:8000 ml-study:latest
```

## 📚 Learning Path & Study Guide

This project demonstrates modern software development practices for AI/ML:

### 1. **Project Setup & Structure**
- Modern Python packaging with pyproject.toml
- Virtual environment management
- Dependency management best practices

### 2. **Data Engineering**
- ETL pipeline implementation
- Data validation and quality checks
- Scalable data processing

### 3. **Machine Learning Pipeline**
- Feature engineering techniques
- Model selection and training
- Hyperparameter optimization

### 4. **Model Deployment & Monitoring**
- Containerization with Docker
- API development with FastAPI
- Model monitoring and logging

### 5. **CI/CD & DevOps**
- Automated testing pipelines
- Continuous integration with GitHub Actions
- Deployment automation

### 6. **Experiment Management**
- MLflow for experiment tracking
- Model versioning and lineage
- Reproducible research practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python and ML libraries
- Inspired by industry best practices from leading tech companies
- Designed for educational purposes and practical ML engineering

## 📞 Contact

**Ramon Souza**
- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/ramon-mendes-b44456164)
- GitHub: [RamonSouzaDev](https://github.com/RamonSouzaDev)
- Email: dwmom@hotmail.com

---

## 🎓 Why This Project Matters

In 2025-2026, the intersection of software development and AI technologies is more critical than ever. This project demonstrates:

- **How to build scalable ML systems** following software engineering principles
- **Modern development practices** that work for both traditional software and AI projects
- **Production-ready ML pipelines** that can be deployed in real-world scenarios
- **Best practices** that bridge the gap between academic ML and industry applications

Whether you're a software developer learning ML or an ML engineer learning software development, this project provides a comprehensive guide to building AI-powered applications the right way.

⭐ **Star this repo if you find it helpful for your learning journey!**
