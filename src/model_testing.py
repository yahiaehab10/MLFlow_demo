"""Model evaluation utilities."""

from sklearn.metrics import accuracy_score
import mlflow


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log metrics to MLflow."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("test_accuracy", acc)
    return acc
