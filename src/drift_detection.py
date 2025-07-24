"""Data drift detection utilities using Evidently."""

from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: str):
    """Generate a data drift report between reference and current datasets."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)
    return report
