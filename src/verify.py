import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config import ROOT, TARGET
from src.train import evaluate
from src.validation import FEATURES, validate_frame


def verify():
    test = pd.read_csv(ROOT / "data/test.csv")
    train = pd.read_csv(ROOT / "data/train.csv")
    validate_frame(train[FEATURES])
    model = joblib.load(ROOT / "models/model.joblib")
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    saved = pd.read_csv(ROOT / "reports/test_predictions.csv")
    np.testing.assert_array_equal(saved.ID, test.ID)
    np.testing.assert_allclose(probabilities, saved.probability, rtol=0, atol=1e-12)
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    for key, value in evaluate(test[TARGET], probabilities).items():
        np.testing.assert_allclose(value, metrics["test"][key], rtol=0, atol=1e-12)
    if set(train.ID) & set(test.ID):
        raise ValueError("Обнаружено пересечение разбиений")
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    client = mlflow.MlflowClient()
    for run_id in metrics["runs"]:
        run = client.get_run(run_id)
        if run.info.status != "FINISHED" or "cv_roc_auc" not in run.data.metrics:
            raise ValueError("Эксперимент не завершён")
    exported = mlflow.sklearn.load_model(
        "runs:/"
        + metrics["runs"][
            list(pd.read_csv(ROOT / "reports/experiments.csv").configuration).index(
                metrics["winner"]
            )
        ]
        + "/model"
    )
    np.testing.assert_allclose(
        exported.predict_proba(test[FEATURES].head(20)),
        model.predict_proba(test[FEATURES].head(20)),
    )
    report = {
        "prediction_rows_verified": len(test),
        "split_overlap": 0,
        "finished_experiments": len(metrics["runs"]),
        "mlflow_model_matches": True,
    }
    (ROOT / "reports/verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    verify()
