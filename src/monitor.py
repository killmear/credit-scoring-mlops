import argparse
import json

import numpy as np
import pandas as pd
import requests

from src.config import ROOT
from src.validation import FEATURES


def population_stability_index(reference, current):
    # Фиксированные интервалы вероятности не зависят от новых данных.
    edges = np.linspace(0, 1, 11)
    expected = np.histogram(reference, bins=edges)[0].astype(float) + 0.5
    actual = np.histogram(current, bins=edges)[0].astype(float) + 0.5
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def monitor(url):
    records = pd.read_csv(ROOT / "data/test.csv")[FEATURES].head(500)
    response = requests.post(
        url + "/predict", json={"records": records.to_dict("records")}, timeout=60
    )
    response.raise_for_status()
    current = [item["probability"] for item in response.json()["predictions"]]
    reference = pd.read_csv(ROOT / "reports/reference_predictions.csv").probability
    report = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "psi": population_stability_index(reference, current),
        "description": "Сравнение train и 500 test через API, имитация нового потока",
    }
    (ROOT / "reports/monitoring.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    monitor(parser.parse_args().url)
