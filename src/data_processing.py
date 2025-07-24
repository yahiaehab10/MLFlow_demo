"""Data loading and preprocessing utilities."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(test_size: float = 0.2, random_state: int = 42):
    """Load Iris dataset and split into train/test pandas DataFrames."""
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
