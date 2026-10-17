import joblib
import pytest
from fastapi.testclient import TestClient

from src.data import FEATURES, clean
from src.features import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


@pytest.fixture()
def client(tmp_path, raw_df, monkeypatch):
    df = clean(raw_df)
    model = Pipeline([("prep", build_preprocessor()),
                      ("clf", LogisticRegression(max_iter=500))])
    model.fit(df[FEATURES], df["target"])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    from app import main
    monkeypatch.setattr(main, "MODEL_PATH", path)
    monkeypatch.setattr(main, "_model", None)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_invalid_input(client):
    bad = {**SAMPLE, "age": -5}
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=missing).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "predictions_total" in text
