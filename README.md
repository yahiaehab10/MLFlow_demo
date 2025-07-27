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
- To train and register a model manually, run:
  ```bash
  python src/train.py
  ```
  You will be prompted to enter the model stage (`None`, `Staging`, or `Production`).

## Model Registry & Stage Transition

- After training, the model is automatically registered in the MLflow Model Registry as `IrisRandomForest`.
- You can choose to transition the model version to `Staging` or `Production` by providing the stage as input when running `train.py`.
- You can also manage model stages via the DagsHub Models tab or the MLflow UI.

### View Experiments and Models

- **DagsHub**: [MLflow_demo_MF on DagsHub](https://dagshub.com/yahiaehab10/MLflow_demo_MF)
  - Experiments, runs, parameters, metrics, and registered models are all visible here.
- **MLflow UI (local)**: Run `mlflow ui` and visit [http://localhost:5000](http://localhost:5000)

---

## Features

- MLflow Tracking and Model Registry
- Drift Detection with Evidently
- DagsHub integration for collaboration

## Data

- Uses datasets from `sklearn`

---

For more details, see the notebooks and source files.
