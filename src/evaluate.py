import pandas as pd
from sklearn.metrics import classification_report
import mlflow


def evaluate_model(model, data_path):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1-score", report["weighted avg"]["f1-score"])
    return report
