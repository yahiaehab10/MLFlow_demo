import pytest
import pandas as pd
from src.data_preprocessing import clean_data, validate_data_schema

def test_validate_data_schema():
    """Test data schema validation"""
    # Create sample data
    data = {
        'sepal_length': [5.1, 4.9, 4.7],
        'sepal_width': [3.5, 3.0, 3.2],
        'petal_length': [1.4, 1.4, 1.3],
        'petal_width': [0.2, 0.2, 0.2],
        'target': [0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    # Test valid data
    assert validate_data_schema(df), "Valid data failed schema validation"
    
    # Test invalid data (missing column)
    invalid_df = df.drop('sepal_length', axis=1)
    assert not validate_data_schema(invalid_df), "Invalid data passed schema validation"
