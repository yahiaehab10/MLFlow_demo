# MLflow_demo

This is a proof of concept project demonstrating the use of MLflow for model lifecycle management, DagsHub integration for collaboration, and Evidently for drift detection.

## Project Structure

```
Mlflow_demo/
│
├── data/
│   ├── raw/ # Original dataset
│   ├── processed/ # Cleaned, transformed data
│   └── drift_baseline/ # Baseline stats for drift detection
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_drift.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_quality.py
│   ├── drift_detection.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Data cleaning and preprocessing
2. Drift detection analysis
3. Model training and evaluation

## DagsHub Integration

See notebooks for MLflow tracking details.
