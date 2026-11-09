import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.config import ROOT
from src.validation import FEATURES


def test_api_matches_model():
    records = pd.read_csv(ROOT / "data/test.csv")[FEATURES].head(3)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json={"records": records.to_dict("records")})
        assert response.status_code == 200
        actual = [row["probability"] for row in response.json()["predictions"]]
        model = joblib.load(ROOT / "models/model.joblib")
        np.testing.assert_allclose(actual, model.predict_proba(records)[:, 1])


def test_api_rejects_invalid_requests():
    row = pd.read_csv(ROOT / "data/test.csv")[FEATURES].iloc[0].to_dict()
    with TestClient(app) as client:
        assert client.post("/predict", json={"records": []}).status_code == 422
        assert client.post("/predict", json={"records": [{"AGE": 30}]}).status_code == 422
        row["AGE"] = -1
        assert client.post("/predict", json={"records": [row]}).status_code == 422
        row["AGE"] = "30"
        assert client.post("/predict", json={"records": [row]}).status_code == 422
