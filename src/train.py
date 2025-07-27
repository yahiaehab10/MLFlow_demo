import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import dagshub


def train_model(data_path):
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
    mlflow.sklearn.log_model(clf, "model")
    print(f"Model trained with accuracy: {acc}")
    return clf, acc


if __name__ == "__main__":
    # When running standalone, initialize DagsHub and start an MLflow run
    dagshub.init(repo_owner="yahiaehab10", repo_name="MLflow_demo_MF", mlflow=True)
    with mlflow.start_run():
        train_model("data/processed/iris_clean.csv")
