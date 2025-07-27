from src.data_preprocessing import clean_data
from src.drift_detection import generate_drift_baseline
from src.train import train_model

# from src.evaluate import evaluate_model

import dagshub
import mlflow


def run_pipeline():
    # Initialize DagsHub MLflow tracking
    dagshub.init(repo_owner="yahiaehab10", repo_name="MLflow_demo_MF", mlflow=True)
    # Start MLflow run for the pipeline
    with mlflow.start_run(run_name="full_pipeline"):
        # Step 1: Clean data
        clean_data("data/raw/iris.csv", "data/processed/iris_clean.csv")
        # Step 2: Drift baseline
        generate_drift_baseline(
            "data/processed/iris_clean.csv",
            "data/drift_baseline/iris_drift_baseline.html",
        )
        # Step 3: Train model
        model, acc = train_model("data/processed/iris_clean.csv")
        # Step 4: Evaluate model (optional)
        # report = evaluate_model(model, 'data/processed/iris_clean.csv')
        # print(report)


if __name__ == "__main__":
    run_pipeline()
