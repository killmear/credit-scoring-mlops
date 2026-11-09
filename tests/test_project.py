import numpy as np
import pandas as pd
import pytest

from src.features import add_features, make_pipeline
from src.validation import FEATURES, validate_frame
from src.monitor import population_stability_index


@pytest.fixture
def sample():
    row = dict.fromkeys(FEATURES, 0)
    row.update(LIMIT_BAL=50000, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=35)
    return pd.DataFrame([row, row])


def test_valid_data(sample):
    assert validate_frame(sample)["success"]


@pytest.mark.parametrize(
    "column,value", [("AGE", -1), ("SEX", 3), ("PAY_0", 10), ("LIMIT_BAL", np.nan)]
)
def test_invalid_data(sample, column, value):
    sample.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_frame(sample)


def test_bad_schema(sample):
    with pytest.raises(ValueError):
        validate_frame(sample.drop(columns="AGE"))
    with pytest.raises(ValueError):
        validate_frame(sample.assign(unexpected=1))
    with pytest.raises(ValueError):
        validate_frame(sample.iloc[:0])


def test_text_and_fractional_category(sample):
    sample["AGE"] = "unknown"
    with pytest.raises(ValueError):
        validate_frame(sample)
    sample["AGE"] = 35
    sample["SEX"] = 1.5
    with pytest.raises(ValueError):
        validate_frame(sample)


def test_negative_bill_allowed(sample):
    sample.loc[0, "BILL_AMT1"] = -200
    assert validate_frame(sample)["success"]


def test_features_without_mutation(sample):
    original = sample.copy()
    transformed = add_features(sample)
    pd.testing.assert_frame_equal(sample, original)
    assert transformed["payment_mean"].eq(0).all()
    assert np.isfinite(transformed.select_dtypes(include="number").to_numpy()).all()


def test_pipeline_roundtrip(sample, tmp_path):
    import joblib

    model = make_pipeline()
    model.fit(sample, [0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict_proba(sample), joblib.load(path).predict_proba(sample))


def test_psi():
    values = np.linspace(0, 1, 100)
    assert population_stability_index(values, values) == pytest.approx(0)
    assert population_stability_index(values, np.ones(100)) > 0
    assert np.isfinite(population_stability_index(np.zeros(100), np.ones(100)))


def test_float_categories_match_integer(sample):
    pipeline = make_pipeline().fit(sample, [0, 1])
    np.testing.assert_allclose(
        pipeline.predict_proba(sample), pipeline.predict_proba(sample.astype(float))
    )


def test_metrics_on_perfect_predictions():
    from src.train import evaluate

    assert evaluate([0, 0, 1, 1], np.array([0.1, 0.2, 0.8, 0.9])) == {
        "roc_auc": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_saved_split():
    from src.config import ROOT, TARGET

    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    assert not set(train.ID) & set(test.ID)
    assert len(train) + len(test) == 30000
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.001
