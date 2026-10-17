"""Train, tune, evaluate and track models with MLflow.

Run:  python -m src.train            (uses data/heart.csv)
View: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data import FEATURES, TARGET, load_clean  # noqa: E402
from src.features import build_preprocessor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# model name -> (estimator, hyper-parameter grid tuned with GridSearchCV)
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300],
         "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3]},
    ),
}


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def test_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred),
        "test_recall": recall_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def log_plots(model, X, y, name: str, tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.set_title(f"ROC - {name}")
    fig.savefig(tmp / f"roc_{name}.png", dpi=120)
    mlflow.log_artifact(str(tmp / f"roc_{name}.png"), "plots")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax)
    ax.set_title(f"Confusion matrix - {name}")
    fig.savefig(tmp / f"cm_{name}.png", dpi=120)
    mlflow.log_artifact(str(tmp / f"cm_{name}.png"), "plots")
    plt.close(fig)


def main(data_path=None, model_dir=None, experiment="heart-disease") -> dict:
    model_dir = Path(model_dir or ROOT / "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(experiment)

    df = load_clean(data_path) if data_path else load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (estimator, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(make_pipeline(estimator), grid, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
            search.fit(X_tr, y_tr)
            best = search.best_estimator_

            # unbiased CV estimate of the tuned pipeline on the training split
            cvres = cross_validate(best, X_tr, y_tr, cv=cv, scoring=CV_SCORING)
            cv_metrics = {f"cv_{m}_mean": float(cvres[f"test_{m}"].mean()) for m in CV_SCORING}
            cv_metrics.update({f"cv_{m}_std": float(cvres[f"test_{m}"].std()) for m in CV_SCORING})
            t_metrics = test_metrics(best, X_te, y_te)

            mlflow.log_param("model", name)
            mlflow.log_param("n_train", len(X_tr))
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({**cv_metrics, **t_metrics})
            log_plots(best, X_te, y_te, name, ROOT / "reports" / "plots")
            mlflow.sklearn.log_model(
                best, name="model", serialization_format="cloudpickle",
                signature=infer_signature(X_te, best.predict(X_te)))

            results[name] = {**cv_metrics, **t_metrics, "best_params": search.best_params_}
            fitted[name] = best
            print(f"{name}: cv_roc_auc={cv_metrics['cv_roc_auc_mean']:.3f} "
                  f"test_roc_auc={t_metrics['test_roc_auc']:.3f}")

    # model selection: best cross-validated ROC-AUC
    winner = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    joblib.dump(fitted[winner], model_dir / "model.joblib")
    summary = {"selected_model": winner, "selection_metric": "cv_roc_auc_mean",
               "features": FEATURES, "results": results}
    (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Selected: {winner} -> {model_dir / 'model.joblib'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--model-dir", default=None)
    a = p.parse_args()
    main(a.data, a.model_dir)
