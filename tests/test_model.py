import json

import joblib

from src.data import FEATURES, clean
from src import train


def test_training_produces_artifacts(tmp_path, raw_df, monkeypatch):
    data_path = tmp_path / "heart.csv"
    clean(raw_df).to_csv(data_path, index=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    # shrink the grids so the test stays fast
    monkeypatch.setattr(train, "CANDIDATES", {
        "logistic_regression": (train.CANDIDATES["logistic_regression"][0], {"clf__C": [1]}),
        "random_forest": (train.CANDIDATES["random_forest"][0],
                          {"clf__n_estimators": [20], "clf__max_depth": [3]}),
    })
    summary = train.main(data_path, tmp_path / "models", experiment="pytest")
    assert summary["selected_model"] in {"logistic_regression", "random_forest"}
    model = joblib.load(tmp_path / "models" / "model.joblib")
    proba = model.predict_proba(clean(raw_df)[FEATURES].head(3))
    assert proba.shape == (3, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert json.loads((tmp_path / "models" / "metrics.json").read_text())["features"] == FEATURES
