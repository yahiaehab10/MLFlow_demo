import pandas as pd


def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Example cleaning: drop NA, reset index
    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    clean_data("data/raw/iris.csv", "data/processed/iris_clean.csv")
