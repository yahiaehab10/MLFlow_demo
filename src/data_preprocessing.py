import pandas as pd


def validate_data_schema(df):
    """Validate input data schema"""
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    
    # Check columns exist
    if not all(col in df.columns for col in expected_columns):
        return False
    
    # Check data types
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False
    
    # Check value ranges
    if df[numeric_columns].min().min() < 0:
        return False
    
    return True


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Example cleaning: drop NA, reset index
    df = df.dropna().reset_index(drop=True)
    
    # Validate data schema
    if not validate_data_schema(df):
        raise ValueError("Data validation failed")
    
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    clean_data("data/raw/iris.csv", "data/processed/iris_clean.csv")
