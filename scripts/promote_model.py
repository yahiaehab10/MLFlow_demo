#!/usr/bin/env python3
"""
Model promotion script for MLflow models.
Promotes models from staging to production based on performance criteria.
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient


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
            client = MlflowClient()
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


def get_latest_experiment_metrics():
    """Get metrics from the latest experiment run"""
    try:
        # Try to read metrics from local file first
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
            print(f"✓ Found metrics: {metrics}")
            return metrics
    except Exception as e:
        print(f"Could not read local metrics: {e}")

    # Fallback to MLflow tracking
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        if experiments:
            runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
            if runs:
                latest_run = runs[0]
                metrics = latest_run.data.metrics
                print(f"✓ Found MLflow metrics: {metrics}")
                return metrics
    except Exception as e:
        print(f"Could not read MLflow metrics: {e}")

    return {}


def promote_model():
    """Main model promotion logic"""
    print("🚀 Starting model promotion process...")

    # Setup connections
    dagshub_connected = setup_mlflow_tracking()

    # Get performance metrics
    metrics = get_latest_experiment_metrics()

    if not metrics:
        print("❌ No metrics found - cannot promote model")
        return False

    # Check if model meets promotion criteria
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    print(f"📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")

    # Define promotion thresholds
    min_accuracy = 0.85
    min_precision = 0.80
    min_recall = 0.80

    # Check promotion criteria
    meets_criteria = (
        accuracy >= min_accuracy and precision >= min_precision and recall >= min_recall
    )

    if meets_criteria:
        print("✅ Model meets promotion criteria!")
        print("🎯 Model is ready for production deployment")

        # Log promotion decision
        try:
            with mlflow.start_run(run_name="model_promotion"):
                mlflow.log_param("promotion_decision", "approved")
                mlflow.log_param("promotion_reason", "meets_all_criteria")
                mlflow.log_metric("final_accuracy", accuracy)
                mlflow.log_metric("final_precision", precision)
                mlflow.log_metric("final_recall", recall)
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

        return True
    else:
        print("❌ Model does not meet promotion criteria:")
        if accuracy < min_accuracy:
            print(f"   Accuracy {accuracy:.4f} < {min_accuracy}")
        if precision < min_precision:
            print(f"   Precision {precision:.4f} < {min_precision}")
        if recall < min_recall:
            print(f"   Recall {recall:.4f} < {min_recall}")

        # Log rejection
        try:
            with mlflow.start_run(run_name="model_promotion"):
                mlflow.log_param("promotion_decision", "rejected")
                mlflow.log_param("promotion_reason", "below_thresholds")
                mlflow.log_metric("final_accuracy", accuracy)
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

        return False


if __name__ == "__main__":
    success = promote_model()
    if success:
        print("🎉 Model promotion completed successfully!")
    else:
        print("⚠️  Model promotion rejected - improve model performance")
        exit(1)
