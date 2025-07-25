import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning: drop duplicates and fill missing values."""
    # Drop duplicates
    df = df.drop_duplicates()
    # Fill numeric missing values with median
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV without index."""
    df.to_csv(path, index=False)
