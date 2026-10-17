"""Data loading and cleaning for the Heart Disease UCI dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "heart_raw.csv"
CLEAN_PATH = ROOT / "data" / "heart.csv"

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "target"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    - '?' placeholders -> NaN, everything numeric
    - multi-class target (0-4) -> binary (0 = no disease, 1 = disease)
    - exact duplicate rows dropped
    Imputation is NOT done here; it happens inside the sklearn pipeline so the
    medians/modes are learned from training folds only (no leakage).
    """
    out = df.copy()
    out = out.replace("?", np.nan)
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[TARGET] = (out[TARGET] > 0).astype(int)
    out = out.drop_duplicates().reset_index(drop=True)
    return out[COLUMNS]


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_and_save(raw_path: Path = RAW_PATH, out_path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = clean(pd.read_csv(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    d = prepare_and_save()
    print(f"Wrote {len(d)} cleaned rows to {CLEAN_PATH}")
    print(d.isna().sum()[d.isna().sum() > 0])
