from src.data_preprocessing import clean_data
from src.drift_detection import generate_drift_baseline
from src.train import train_model
from src.evaluate import evaluate_model

import dagshub
import mlflow
import json


def run_pipeline():
    # Set MLflow tracking URI to DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow")
    # Initialize DagsHub MLflow tracking
    dagshub.init(repo_owner="yahiaehab10", repo_name="MLFlow_demo", mlflow=True)
    # Start MLflow run for the pipeline
    with mlflow.start_run(run_name="full_pipeline"):
        # Step 1: Clean data
        clean_data("data/raw/iris.csv", "data/processed/iris_clean.csv")
        mlflow.log_param("raw_data_path", "data/raw/iris.csv")
        mlflow.log_param("clean_data_path", "data/processed/iris_clean.csv")

        # Step 2: Drift baseline
        drift_report_path = "data/drift_baseline/iris_drift_baseline.html"
        generate_drift_baseline(
            "data/processed/iris_clean.csv",
            drift_report_path,
        )
        mlflow.log_artifact(drift_report_path, artifact_path="drift_reports")

        # Step 3: Train model
        model, acc = train_model("data/processed/iris_clean.csv")
        mlflow.log_metric("accuracy", acc)

        # Save metrics to JSON file for DVC
        metrics = {
            "accuracy": acc,
            "precision": acc,  # For this simple case, using accuracy as proxy
            "recall": acc,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)

        # Example: log additional artifacts (e.g., images from analysis)
        # Uncomment and update the path if you have analysis images to upload
        # mlflow.log_artifact("path/to/your/analysis_image.png", artifact_path="analysis_images")

        # Step 4: Evaluate model (optional)
        report = evaluate_model(model, "data/processed/iris_clean.csv")
        print(report)


if __name__ == "__main__":
    run_pipeline()
