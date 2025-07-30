# MLFlow Demo Project

This repository demonstrates a complete MLOps pipeline using **MLflow**, **DagsHub**, **DVC**, and **Evidently** for comprehensive model lifecycle management, data versioning, and experiment tracking.

**Features:**

- **CI/CD Pipeline Active** - Automated deployment and model promotion
- **Model Serving API** - FastAPI REST API for real-time predictions
- **Docker Support** - Containerized deployment ready

## Features

- **MLflow Integration**: Experiment tracking, model registry, and artifact logging
- **DagsHub Pipeline**: Complete MLOps collaboration platform with visual pipelines
- **DVC Data Versioning**: Version-controlled datasets and model artifacts
- **Drift Detection**: Evidently-based data drift monitoring
- **Automated Pipeline**: End-to-end reproducible ML workflows
- **Model Serving**: FastAPI REST API with Docker deployment
- **Interactive Documentation**: Swagger UI at `/docs` endpoint

## Project Structure

```
MLflow_demo/
│
├── app.py                     # FastAPI model serving application
├── simple_train.py            # Quick model training script
├── test_api.py               # API testing suite
├── Dockerfile                # Docker configuration
├── setup.sh                  # Automated setup script
│
├── data/
│   ├── data_loader.py
│   ├── raw/                  # Original dataset (DVC tracked)
│   ├── processed/            # Cleaned data (pipeline output)
│   └── drift_baseline/       # Drift detection reports
│
├── src/
│   ├── data_preprocessing.py # Data cleaning pipeline
│   ├── drift_detection.py    # Evidently drift detection
│   ├── train.py             # Model training with MLflow
│   ├── evaluate.py          # Model evaluation
│   └── pipeline.py          # Complete end-to-end pipeline
│
├── scripts/
│   ├── monitor_model.py     # Model monitoring utilities
│   └── promote_model.py     # Model promotion utilities
│
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── dvc.yaml                 # DVC pipeline configuration
├── dvc.lock                 # Pipeline lock file
├── metrics.json             # Pipeline metrics output
└── requirements.txt
```

## Quick Start

### Option 1: Model Serving API (Recommended for Testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a simple model
python simple_train.py

# 3. Start the API server
python app.py
```

**API Available at:** http://localhost:8000
**Interactive Docs:** http://localhost:8000/docs

### Option 2: Docker Deployment

```bash
# Build and run with Docker
docker build -t iris-model-api .
docker run -p 8000:8000 iris-model-api
```

### Option 3: Complete MLOps Pipeline

```bash
# Run full pipeline with MLflow tracking
dvc repro
# OR
python -m src.pipeline
```

## Model Serving API

### API Endpoints

- **`GET /`** - Welcome message and API status
- **`POST /predict`** - Make predictions
  - Input format:
    ```json
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    ```
  - Returns prediction for the iris class (0=Setosa, 1=Versicolor, 2=Virginica)

### Example Usage

**cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Python:**

```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())  # {"prediction": 0, "features": [...]}
```

**Interactive Documentation:** Visit http://localhost:8000/docs

## Setup & Installation

### Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerized deployment)
- DagsHub account (for full MLOps pipeline)

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yahiaehab10/MLFlow_demo.git
   cd MLFlow_demo
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure DagsHub authentication (optional, for full pipeline):**

   ```bash
   # Get your token from: https://dagshub.com/user/settings/tokens
   dvc remote modify origin password <your-dagshub-token>
   ```

## Usage

### Quick Testing & Development

**Train and serve model locally:**

```bash
python simple_train.py  # Train model
python app.py           # Start API server
python test_api.py      # Run test suite
```

### Production Deployment

**Docker deployment:**

```bash
docker build -t iris-model-api .
docker run -d -p 8000:8000 iris-model-api
```

### MLOps Pipeline (Advanced)

**Complete pipeline with tracking:**

```bash
dvc repro               # DVC pipeline (recommended)
# OR
python -m src.pipeline  # Direct execution with MLflow logging
```

## MLflow & DagsHub Integration

### Experiment Tracking

All experiments are automatically tracked and logged to:

- **DagsHub MLflow**: [https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow](https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow)
- **Local MLflow UI**: Run `mlflow ui` and visit [http://localhost:5000](http://localhost:5000)

### Model Registry & Stage Management

- Models are automatically registered as `IrisRandomForest` in MLflow Model Registry
- Support for model stage transitions (`Staging`, `Production`)
- Model versioning and lineage tracking
- Artifact storage (drift reports, model files, metrics)

### Logged Metrics & Parameters

- **Parameters**: Data paths, model hyperparameters, random seeds
- **Metrics**: Accuracy, precision, recall
- **Artifacts**:
  - Drift baseline HTML reports
  - Model files (pickle format)
  - Analysis images (when available)

## DagsHub Features

### Data Pipeline Visualization

- Visual representation of your ML pipeline
- Dependency tracking between stages
- Real-time status monitoring

### Collaboration

- Version-controlled datasets with DVC
- Experiment comparison and analysis
- Team collaboration on ML projects
- Model performance monitoring

## Data Drift Detection

The pipeline includes comprehensive drift detection using **Evidently**:

- **Baseline Generation**: Creates reference profiles from training data
- **Drift Monitoring**: Compares new data against baseline
- **HTML Reports**: Interactive drift analysis reports
- **Automated Alerts**: Integration with MLflow for drift tracking

## Pipeline Workflow

1. **Data Loading**: Load raw iris dataset
2. **Data Cleaning**: Preprocess and clean data
3. **Drift Detection**: Generate baseline and detect drift
4. **Model Training**: Train RandomForest with MLflow tracking
5. **Model Registration**: Register model in MLflow registry
6. **Metrics Logging**: Save performance metrics
7. **Artifact Upload**: Store reports and models in DagsHub

## Data Versioning with DVC

- **Raw Data**: `data/raw/iris.csv` - Version controlled with DVC
- **Processed Data**: `data/processed/iris_clean.csv` - Pipeline output
- **Models**: Stored in MLflow model registry
- **Reports**: `data/drift_baseline/` - Drift analysis artifacts

## Configuration

### Environment Variables

The pipeline uses the following configuration:

- **DagsHub Repository**: `yahiaehab10/MLFlow_demo`
- **MLflow Tracking URI**: `https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow`
- **Model Name**: `IrisRandomForest`

### DVC Configuration

```yaml
# dvc.yaml
stages:
  full_pipeline:
    cmd: python -m src.pipeline
    deps:
      - data/raw/iris.csv
      - src/pipeline.py
      - src/data_preprocessing.py
      - src/train.py
      - src/drift_detection.py
    outs:
      - data/processed/iris_clean.csv
      - data/drift_baseline/iris_drift_baseline.html
    metrics:
      - metrics.json
```

## Troubleshooting

### Common Issues

1. **DVC Push Authentication Error**:

   ```bash
   dvc remote modify origin password <your-dagshub-token>
   ```

2. **MLflow Tracking URI Error**:

   - Ensure you have internet connectivity
   - Verify DagsHub repository access

3. **Pipeline Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Documentation

📚 **Additional Guides:**

- **[QUICK_START.md](QUICK_START.md)** - Essential commands for model serving
- **[SERVING_GUIDE.md](SERVING_GUIDE.md)** - Comprehensive deployment guide

📖 **External Resources:**

- **MLflow**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- **DagsHub**: [DagsHub Tutorial](https://dagshub.com/docs/tutorial/)
- **DVC**: [DVC Documentation](https://dvc.org/doc)
- **Evidently**: [Evidently Documentation](https://docs.evidentlyai.com/)

## License

This project is open source and available under the [MIT License](LICENSE).
