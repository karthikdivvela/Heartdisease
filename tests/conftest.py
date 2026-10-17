import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Small synthetic frame with the UCI schema (incl. '?' placeholders)."""
    rng = np.random.default_rng(0)
    n = 200
    target = rng.integers(0, 5, n)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n) + (target > 0) * 3,
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(150, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n) - (target > 0) * 15,
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.random(n) * 4 + (target > 0), 1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(object),
        "thal": rng.choice([3, 6, 7], n).astype(object),
        "target": target,
    })[COLUMNS]
    df.loc[3, "ca"] = "?"
    df.loc[7, "thal"] = "?"
    return df
