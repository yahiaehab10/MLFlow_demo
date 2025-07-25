import pandas as pd
import mlflow
from sklearn.metrics import classification_report

from data_quality import load_data


def evaluate_model(run_id: str):
    # Load test data
    df = load_data("data/processed/data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # Download model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Predict and evaluate
    preds = model.predict(X)
    report = classification_report(y, preds)
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Please provide run_id")
    else:
        evaluate_model(sys.argv[1])
