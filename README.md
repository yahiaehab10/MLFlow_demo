# MLflow Demo

This project demonstrates a minimal workflow using **MLflow** integrated with **DagsHub** and **Evidently**. It covers experiment tracking, model registry, and drift detection in a single, reproducible pipeline.

## Project Structure

```
mlflow-demo/
├── data/
├── notebooks/
│   └── exploratory.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   ├── drift_detection.py
│   ├── model_testing.py
│   └── pipeline.py
├── environment.yml
├── main.py
└── README.md
```

## Setup

Create the Conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate mlflow
```

## Running the Demo

Execute the full pipeline with:

```bash
python main.py
```

The script initializes MLflow tracking with DagsHub:

```python
import dagshub

dagshub.init(repo_owner="yahiaehab10", repo_name="MLFlow_demo", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow")
```

Metrics, parameters and artifacts (including an Evidently drift report) are logged to the remote MLflow server. Open the DagsHub interface to explore runs and the model registry.

## Reproducibility

All dependencies are captured in `environment.yml`. The dataset used is the built-in Iris dataset from scikit-learn for simplicity.

## Drift Detection

`src/drift_detection.py` uses Evidently to generate an HTML data drift report comparing the training data to the evaluation data. The report is logged as an MLflow artifact for easy inspection.

## Collaboration

Since the tracking URI points to DagsHub, everyone with access to the repository can view experiments remotely, facilitating collaboration and model lifecycle management.


