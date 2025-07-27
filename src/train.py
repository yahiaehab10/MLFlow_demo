import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import dagshub


def train_model(data_path, stage=None):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Don't start a new run - use the current active run from pipeline
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc)
    model_info = mlflow.sklearn.log_model(clf, "model")
    # Register the model in the MLflow Model Registry
    model_uri = model_info.model_uri
    registered_model = mlflow.register_model(model_uri, "IrisRandomForest")
    print(f"Model trained with accuracy: {acc}")
    print(
        f"Model registered as 'IrisRandomForest', version: {registered_model.version}"
    )
    # Optionally transition the model to a specified stage
    if stage in ["Staging", "Production"]:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="IrisRandomForest", version=registered_model.version, stage=stage
        )
        print(
            f"Model version {registered_model.version} transitioned to stage: {stage}"
        )
    return clf, acc


if __name__ == "__main__":
    # When running standalone, initialize DagsHub and start an MLflow run
    dagshub.init(repo_owner="yahiaehab10", repo_name="MLFlow_demo", mlflow=True)
    stage = input("Enter model stage (None, Staging, Production): ") or None
    with mlflow.start_run():
        train_model("data/processed/iris_clean.csv", stage=stage)
