#!/usr/bin/env python3
"""
Simple script to train and save an iris model for serving
"""
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_save_model():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save model with joblib for direct loading
    joblib.dump(model, "models/iris_model.pkl")
    print("Model saved to models/iris_model.pkl")

    # Also save with MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "iris_model", registered_model_name="IrisModel")
        mlflow.log_metric("accuracy", accuracy)
        print("Model logged to MLflow")

    return model


if __name__ == "__main__":
    train_and_save_model()
