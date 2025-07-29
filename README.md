# MLFlow Demo Project

This repository demonstrates a complete MLOps pipeline using **MLflow**, **DagsHub**, **DVC**, and **Evidently** for comprehensive model lifecycle management, data versioning, and experiment tracking.

**CI/CD Pipeline Active** - Automated deployment and model promotion enabled!

## Features

- **MLflow Integration**: Experiment tracking, model registry, and artifact logging
- **DagsHub Pipeline**: Complete MLOps collaboration platform with visual pipelines
- **DVC Data Versioning**: Version-controlled datasets and model artifacts
- **Drift Detection**: Evidently-based data drift monitoring
- **Automated Pipeline**: End-to-end reproducible ML workflows

## Project Structure

```
MLflow_demo/
│
├── data/
│   ├── data_loader.py
│   ├── raw/                    # Original dataset (DVC tracked)
│   ├── processed/              # Cleaned data (pipeline output)
│   └── drift_baseline/         # Drift detection reports
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_drift.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   ├── drift_detection.py      # Evidently drift detection
│   ├── train.py               # Model training with MLflow
│   ├── evaluate.py            # Model evaluation
│   └── pipeline.py            # Complete end-to-end pipeline
│
├── dvc.yaml                   # DVC pipeline configuration
├── dvc.lock                   # Pipeline lock file
├── metrics.json               # Pipeline metrics output
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- Git
- DVC
- MLflow
- DagsHub account

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yahiaehab10/MLFlow_demo.git
   cd MLFlow_demo
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure DagsHub authentication (for data push):**

   ```bash
   # Get your token from: https://dagshub.com/user/settings/tokens
   dvc remote modify origin password <your-dagshub-token>
   ```

## Usage

### Run Complete Pipeline

**Option 1: DVC Pipeline (Recommended)**

```bash
dvc repro  # Runs the complete reproducible pipeline
```

**Option 2: Direct Python Execution**

```bash
python -m src.pipeline  # Runs with MLflow logging to DagsHub
```

### Individual Components

- **Data Processing**: `python -m src.data_preprocessing`
- **Model Training**: `python -m src.train`
- **Drift Detection**: `python -m src.drift_detection`

### DVC Commands

- **View Pipeline**: `dvc dag`
- **Check Pipeline Status**: `dvc status`
- **Push Data to DagsHub**: `dvc push`
- **Pull Data from DagsHub**: `dvc pull`

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

- **MLflow**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- **DagsHub**: [DagsHub Tutorial](https://dagshub.com/docs/tutorial/)
- **DVC**: [DVC Documentation](https://dvc.org/doc)
- **Evidently**: [Evidently Documentation](https://docs.evidentlyai.com/)

## License

This project is open source and available under the [MIT License](LICENSE).
