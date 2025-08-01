# MLFlow Demo Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A complete **MLOps pipeline** demonstrating production-ready machine learning workflows using MLflow, DagsHub, DVC, and Evidently for experiment tracking, model versioning, and drift detection.

## 🚀 Features

- **🔄 CI/CD Pipeline** - Automated model training and deployment
- **📊 Experiment Tracking** - MLflow integration with DagsHub
- **🗃️ Data Versioning** - DVC for dataset and artifact management  
- **📈 Drift Detection** - Evidently-based monitoring and alerts
- **🚀 Model Serving** - FastAPI REST API with interactive docs
- **🐳 Docker Ready** - Containerized deployment

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Project Structure](#-project-structure)
- [MLOps Pipeline](#-mlops-pipeline)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## ⚡ Quick Start

### Option 1: Model API (Recommended for Testing)
```bash
# Install and run
pip install -r requirements.txt
python simple_train.py    # Train model (~30 seconds)
python app.py             # Start API server

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**🌐 Access:** http://localhost:8000 | **📚 Docs:** http://localhost:8000/docs

### Option 2: Docker Deployment
```bash
docker build -t iris-model-api .
docker run -p 8000:8000 iris-model-api
```

### Option 3: Full MLOps Pipeline
```bash
# Complete pipeline with tracking
dvc repro                 # Run DVC pipeline
# OR
python -m src.pipeline    # Direct execution
```

## 🔌 API Usage

### Endpoints
- **`GET /`** - API status and welcome
- **`POST /predict`** - Make predictions (iris classification)

### Example Request
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())
# Output: {"prediction": 0, "probability": 0.95, "class": "setosa"}
```

## 📁 Project Structure

```
MLflow_demo/
├── app.py                    # FastAPI model serving
├── simple_train.py           # Quick model training
├── test_api.py              # API testing suite
├── Dockerfile               # Container configuration
│
├── data/
│   ├── raw/                    # Original datasets (DVC tracked)
│   ├── processed/              # Cleaned data
│   └── drift_baseline/         # Drift reports
│
├── src/
│   ├── pipeline.py             # End-to-end ML pipeline
│   ├── train.py               # Model training with MLflow
│   ├── data_preprocessing.py   # Data cleaning
│   └── drift_detection.py     # Evidently monitoring
│
├── scripts/                 # Utility scripts
├── dvc.yaml                 # Pipeline configuration
└── requirements.txt
```

## 🔄 MLOps Pipeline

### What It Does
1. **📥 Data Loading** - Iris dataset ingestion
2. **🧹 Data Preprocessing** - Cleaning and validation
3. **📊 Drift Detection** - Baseline creation and monitoring
4. **🤖 Model Training** - RandomForest with hyperparameter tracking
5. **📝 Model Registry** - MLflow model versioning
6. **📈 Evaluation** - Performance metrics logging

### Tracking & Monitoring
- **🌐 DagsHub MLflow**: [View Experiments](https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow)
- **💻 Local MLflow**: Run `mlflow ui` → http://localhost:5000
- **📊 Drift Reports**: Interactive HTML reports in `data/drift_baseline/`

### Logged Artifacts
- ✅ Model performance metrics (accuracy, precision, recall)
- ✅ Hyperparameters and training configuration  
- ✅ Data drift analysis reports
- ✅ Model artifacts and dependencies

## 🛠️ Setup & Configuration

### Prerequisites
- Python 3.11+
- Git
- Docker (optional)
- DagsHub account (for full pipeline)

### Environment Setup
```bash
# Clone repository
git clone https://github.com/yahiaehab10/MLFlow_demo.git
cd MLFlow_demo

# Install dependencies
pip install -r requirements.txt

# Configure DagsHub (optional)
dvc remote modify origin password <your-dagshub-token>
```

## 🧪 Testing

```bash
# Run API tests
python test_api.py

# Test pipeline
dvc repro --dry

# Manual testing
python -c "
import requests
r = requests.post('http://localhost:8000/predict', 
                 json={'features': [5.1, 3.5, 1.4, 0.2]})
print(r.json())
"
```

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| DVC authentication error | `dvc remote modify origin password <token>` |
| MLflow connection timeout | Check internet connection and DagsHub access |
| Missing dependencies | `pip install -r requirements.txt` |
| Docker build fails | Ensure Docker daemon is running |

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Essential commands for immediate use
- **[SERVING_GUIDE.md](SERVING_GUIDE.md)** - Comprehensive deployment guide
- **[MLflow Docs](https://mlflow.org/docs/latest/index.html)** - MLflow reference
- **[DagsHub Tutorial](https://dagshub.com/docs/tutorial/)** - MLOps collaboration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
