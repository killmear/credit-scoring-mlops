import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,  # noqa: E402
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import ROOT, SEED, TARGET  # noqa: E402
from src.features import make_pipeline  # noqa: E402
from src.validation import FEATURES  # noqa: E402


def evaluate(y, probabilities):
    predicted = probabilities >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def train():
    for name in ["reports", "models"]:
        (ROOT / name).mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "data/train.csv")
    x, y = data[FEATURES], data[TARGET]
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    experiment_name = "credit-scoring"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    configurations = [
        ("logistic_c01", {"model__C": [0.1]}),
        ("logistic_c1", {"model__C": [1.0]}),
    ]
    for leaves in [7, 15, 31]:
        configurations.append(
            (
                f"boosting_leaves{leaves}",
                {
                    "model": [
                        HistGradientBoostingClassifier(
                            max_iter=120,
                            learning_rate=0.08,
                            early_stopping=False,
                            random_state=SEED,
                        )
                    ],
                    "model__max_leaf_nodes": [leaves],
                },
            )
        )
    rows, candidates, run_ids = [], [], []
    for name, grid in tqdm(configurations, desc="Конфигурации", unit="модель"):
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                make_pipeline(),
                grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(x, y)
            score = float(search.best_score_)
            deviation = float(search.cv_results_["std_test_score"][0])
            mlflow.log_params(
                {
                    "seed": SEED,
                    "folds": 3,
                    "threshold": 0.5,
                    "configuration": name,
                    **{k: str(v[0]) for k, v in grid.items()},
                }
            )
            mlflow.log_metrics({"cv_roc_auc": score, "cv_std": deviation})
            mlflow.log_artifact(str(ROOT / "reports/data.json"))
            mlflow.sklearn.log_model(
                search.best_estimator_,
                name="model",
                input_example=x.head(2),
                code_paths=[str(ROOT / "src")],
                serialization_format="cloudpickle",
                pip_requirements=str(ROOT / "requirements-api.txt"),
            )
            rows.append(
                {
                    "configuration": name,
                    "cv_roc_auc": score,
                    "cv_std": deviation,
                    "run_id": run.info.run_id,
                }
            )
            candidates.append(search.best_estimator_)
            run_ids.append(run.info.run_id)
    winner = int(np.argmax([r["cv_roc_auc"] for r in rows]))
    model = candidates[winner]
    # Test открывается после завершения выбора по кросс-валидации.
    test = pd.read_csv(ROOT / "data/test.csv")
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    metrics = evaluate(test[TARGET], probabilities)
    model_path = ROOT / "models/model.joblib"
    temporary = model_path.with_suffix(".tmp")
    joblib.dump(model, temporary)
    temporary.replace(model_path)
    pd.DataFrame(rows).to_csv(ROOT / "reports/experiments.csv", index=False)
    pd.DataFrame({"ID": test.ID, "target": test[TARGET], "probability": probabilities}).to_csv(
        ROOT / "reports/test_predictions.csv", index=False
    )
    pd.DataFrame({"probability": model.predict_proba(x)[:, 1]}).to_csv(
        ROOT / "reports/reference_predictions.csv", index=False
    )
    fpr, tpr, _ = roc_curve(test[TARGET], probabilities)
    plt.figure(figsize=(12, 5))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {metrics["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Доля ложноположительных ответов")
    plt.ylabel("Полнота обнаружения дефолтов")
    plt.title("ROC-кривая выбранной модели на test")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "reports/roc.png")
    plt.close()
    report = {
        "winner": rows[winner]["configuration"],
        "threshold": 0.5,
        "cv_roc_auc": rows[winner]["cv_roc_auc"],
        "test": metrics,
        "runs": run_ids,
    }
    (ROOT / "reports/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with mlflow.start_run(run_id=run_ids[winner]):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.set_tag("selected", "true")
        mlflow.log_artifact(str(ROOT / "reports/roc.png"))
        mlflow.log_artifact(str(ROOT / "reports/metrics.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    train()
