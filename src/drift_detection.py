import shutil
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


def save_baseline(processed_path: str, baseline_path: str) -> None:
    """Copy processed data to baseline for drift detection."""
    shutil.copy(processed_path, baseline_path)


def generate_drift_report(
    baseline_path: str, current_path: str, report_path: str
) -> None:
    """Generate drift detection report comparing baseline and current data."""
    ref = pd.read_csv(baseline_path)
    curr = pd.read_csv(current_path)
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref, curr)
    dashboard.save(report_path)
