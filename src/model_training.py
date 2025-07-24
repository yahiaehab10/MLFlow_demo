"""Model training utilities."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow


def train_model(X_train, y_train, X_val, y_val):
    """Train a simple classifier and log with MLflow."""
    with mlflow.start_run(run_name="train_model"):
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
    return clf, acc
