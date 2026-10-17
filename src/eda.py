"""EDA: saves plots to reports/eda/. Run: python -m src.eda
Convert to a notebook with:  jupytext --to notebook src/eda.py  (or paste cells into Jupyter)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import NUMERIC, TARGET, load_clean  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "eda"


def run(df=None, out: Path = OUT) -> None:
    df = load_clean() if df is None else df
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.countplot(x=TARGET, data=df, ax=ax)
    ax.set_title("Class balance (0 = no disease, 1 = disease)")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)

    # 2. Histograms of numeric features split by target
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "histograms.png", dpi=150)
    plt.close(fig)

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Boxplots of numeric features vs target (outliers)
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(15, 3.5))
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "boxplots.png", dpi=150)
    plt.close(fig)

    print(f"EDA plots saved to {out}")


if __name__ == "__main__":
    run()
