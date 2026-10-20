"""Offline inference: predict heart disease risk from a JSON or CSV file.

Uses the same saved pipeline (preprocessing + classifier) as the API, so the
results match what /predict returns.

Examples (run from the repository root):
    python -m src.predict --input samples/sample_patient.json
    python -m src.predict --input samples/sample_patients.csv --output out.csv
    python -m src.predict --json '{"age": 63, "sex": 1, ...}'
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # lets joblib import src.* when unpickling
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "models" / "model.joblib"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
LABELS = {0: "no disease", 1: "disease"}


def load_records(input_path=None, json_text=None):
    """Return a DataFrame from a .csv/.json file or an inline JSON string."""
    if json_text is not None:
        data = json.loads(json_text)
    elif input_path is not None:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        data = json.loads(path.read_text())
    else:
        raise ValueError("give --input FILE or --json TEXT")
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def predict(model, frame, threshold=0.5):
    """Return a DataFrame with prediction, label, confidence, probability."""
    cols = list(getattr(model, "feature_names_in_", FEATURES))
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    x = frame[cols]
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= threshold).astype(int)
    confidence = [p if y == 1 else 1 - p for p, y in zip(proba, pred)]
    return pd.DataFrame({
        "prediction": pred,
        "label": [LABELS[int(y)] for y in pred],
        "confidence": [round(float(c), 4) for c in confidence],
        "probability_disease": [round(float(p), 4) for p in proba],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", help="path to a .json or .csv file")
    parser.add_argument("--json", help="one patient (or a list) as JSON text")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", help="optional CSV file for the results")
    args = parser.parse_args(argv)
    try:
        frame = load_records(args.input, args.json)
        model = joblib.load(args.model)
        result = predict(model, frame)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(result.to_string(index=False))
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"saved results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
