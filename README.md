# MLFlow Demo Project

This repository demonstrates a full ML model lifecycle using MLflow, Evidently for drift detection, and DagsHub for collaboration and tracking. It includes:

- **Model Lifecycle Management**: MLflow Tracking, Model Registry, Artifacts, Logging
- **Drift Detection**: Evidently
- **Collaboration**: DagsHub integration

## Project Structure

```
MLflow_demo/
│
├── data/
│   ├── data_loader.py
│   ├── raw/            # Original dataset
│   ├── processed/      # Cleaned, transformed data
│   └── drift_baseline/ # Baseline stats for drift detection
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_drift.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── drift_detection.py
│   ├── train.py
│   ├── evaluate.py
│   └── pipeline.py
├── requirements.txt
└── README.md
└── main.py
```

## Setup

- Python 3.11.11
- Install dependencies: `pip install -r requirements.txt`

## Usage

- Run the pipeline: `python main.py`
- Notebooks for step-by-step exploration in `notebooks/`

## Features

- MLflow Tracking and Model Registry
- Drift Detection with Evidently
- DagsHub integration for collaboration

## Data

- Uses datasets from `sklearn`

---

For more details, see the notebooks and source files.
