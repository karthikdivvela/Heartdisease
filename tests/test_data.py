import numpy as np

from src.data import COLUMNS, FEATURES, clean
from src.features import build_preprocessor


def test_clean_handles_missing_and_binarises_target(raw_df):
    out = clean(raw_df)
    assert list(out.columns) == COLUMNS
    assert set(out["target"].unique()) <= {0, 1}
    assert out["ca"].isna().sum() >= 1          # '?' became NaN
    assert out["ca"].dtype.kind in "fi"


def test_clean_is_idempotent(raw_df):
    once = clean(raw_df)
    twice = clean(once)
    assert once.equals(twice)


def test_preprocessor_outputs_finite_matrix(raw_df):
    df = clean(raw_df)
    X = build_preprocessor().fit_transform(df[FEATURES])
    assert X.shape[0] == len(df)
    assert np.isfinite(np.asarray(X)).all()      # imputer removed every NaN
