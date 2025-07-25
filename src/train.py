import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_quality import load_data


def train_model():
    df = load_data("data/processed/data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    with mlflow.start_run():
        mlflow.log_param("test_size", 0.2)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact("data/processed/data.csv", artifact_path="data")
        mlflow.sklearn.log_model(
            rf, artifact_path="model", registered_model_name="rf_model"
        )
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {acc}")


if __name__ == "__main__":
    train_model()
