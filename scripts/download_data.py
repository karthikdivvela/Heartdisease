"""Download the Heart Disease UCI (Cleveland) dataset to data/heart_raw.csv.

Usage:  python scripts/download_data.py
Tries the direct UCI file first, then falls back to the `ucimlrepo` package (id=45).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import COLUMNS, RAW_PATH  # noqa: E402

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")


def from_url() -> pd.DataFrame:
    return pd.read_csv(URL, header=None, names=COLUMNS)


def from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=45)
    df = ds.data.features.copy()
    df["target"] = ds.data.targets.iloc[:, 0].values
    return df[COLUMNS]


def main() -> None:
    try:
        df = from_url()
        print("Downloaded from UCI URL")
    except Exception as exc:  # network / format problem
        print(f"Direct download failed ({exc}); trying ucimlrepo...")
        df = from_ucimlrepo()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df)} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()
