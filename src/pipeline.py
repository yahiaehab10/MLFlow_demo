from src.data_preprocessing import clean_data
from src.drift_detection import generate_drift_baseline
from src.train import train_model
from src.evaluate import evaluate_model

import os
import mlflow
import json


def setup_mlflow_tracking():
    """Setup MLflow tracking with proper authentication"""
    # Check if we have authentication credentials
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME", "yahiaehab10")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD", dagshub_token)

    if dagshub_token and mlflow_password:
        try:
            # Set up DagsHub MLflow with authentication
            mlflow.set_tracking_uri(
                "https://dagshub.com/yahiaehab10/MLFlow_demo.mlflow"
            )

            # Try to authenticate by creating a simple connection test
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments(max_results=1)
            print("✓ Successfully connected to DagsHub MLflow")
            return True

        except Exception as e:
            print(f"❌ DagsHub authentication failed: {e}")
            print("🔄 Falling back to local MLflow tracking")
            mlflow.set_tracking_uri("file:./mlruns")
            return False
    else:
        print("⚠️  No DagsHub credentials found, using local MLflow")
        mlflow.set_tracking_uri("file:./mlruns")
        return False


def run_pipeline():
    print("🚀 Starting ML Pipeline...")

    # Setup MLflow tracking
    dagshub_connected = setup_mlflow_tracking()

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
        
        # Only log artifacts if connected to DagsHub, skip for local MLflow
        if dagshub_connected:
            mlflow.log_artifact(drift_report_path, artifact_path="drift_reports")
        else:
            print(f"📁 Drift report saved locally at: {drift_report_path}")
            print("⚠️  Skipping artifact upload (local MLflow mode)")

        # Step 3: Train model
        model, acc = train_model("data/processed/iris_clean.csv", dagshub_connected=dagshub_connected)
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
