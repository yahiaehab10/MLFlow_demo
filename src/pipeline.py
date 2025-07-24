"""Simple ML pipeline demonstrating MLflow, Evidently, and DagsHub."""
import os

import mlflow

from .data_processing import load_data
from .model_training import train_model
from .model_testing import evaluate_model
from .drift_detection import generate_drift_report


def run_pipeline():
    """Execute the pipeline end-to-end."""
    # Initialize DagsHub integration
    import dagshub

    dagshub.init(repo_owner="yahiaehab10", repo_name="MLFlow_demo", mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow")

    X_train, X_test, y_train, y_test = load_data()

    model, val_acc = train_model(X_train, y_train, X_test, y_test)

    test_acc = evaluate_model(model, X_test, y_test)

    # Drift detection using Evidently
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "drift_report.html")
    generate_drift_report(X_train, X_test, report_path)
    mlflow.log_artifact(report_path, artifact_path="drift")

    mlflow.set_tag("val_accuracy", val_acc)
    mlflow.set_tag("test_accuracy", test_acc)



