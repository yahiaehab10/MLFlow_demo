import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def generate_drift_baseline(processed_path, baseline_path):
    df = pd.read_csv(processed_path)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df, current_data=df)
    report.save_html(baseline_path)
    print(f"Drift baseline saved to {baseline_path}")


if __name__ == "__main__":
    generate_drift_baseline(
        "data/processed/iris_clean.csv", "data/drift_baseline/iris_drift_baseline.html"
    )
