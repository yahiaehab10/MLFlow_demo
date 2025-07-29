"""
Basic tests for the ML pipeline
"""
import pytest
import os
import pandas as pd


def test_data_files_exist():
    """Test that required data files exist"""
    assert os.path.exists("data/raw/iris.csv"), "Raw data file should exist"
    

def test_data_loading():
    """Test that data can be loaded properly"""
    df = pd.read_csv("data/raw/iris.csv")
    assert not df.empty, "Data should not be empty"
    assert len(df.columns) >= 4, "Should have at least 4 feature columns"


def test_src_modules_importable():
    """Test that source modules can be imported"""
    try:
        import src.pipeline
        import src.train
        import src.data_preprocessing
        assert True
    except ImportError as e:
        pytest.fail(f"Could not import src modules: {e}")


def test_requirements_satisfied():
    """Test that key packages are available"""
    try:
        import mlflow
        import sklearn
        import pandas
        import numpy
        assert True
    except ImportError as e:
        pytest.fail(f"Required package missing: {e}")


def test_basic_math():
    """Simple test to ensure pytest works"""
    assert 1 + 1 == 2
    assert "hello" == "hello"
