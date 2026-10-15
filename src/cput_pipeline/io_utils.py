from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=index)
        return
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=index)
        except ImportError as exc:
            raise ImportError(
                "Saving parquet requires `pyarrow` or `fastparquet`. "
                "Install `pyarrow` before running the pipeline."
            ) from exc
        return
    raise ValueError(f"Unsupported dataframe output format: {path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires `pyarrow` or `fastparquet`. "
                "Install `pyarrow` before running the pipeline."
            ) from exc
    raise ValueError(f"Unsupported dataframe input format: {path}")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_joblib(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def load_joblib(path: Path) -> object:
    return joblib.load(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def close_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def histogram_with_cdf(
    series: pd.Series,
    title: str,
    xlabel: str,
    path: Path,
    bins: int = 40,
) -> None:
    values = pd.Series(series).dropna().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(values, bins=bins, color="tab:blue", alpha=0.85)
    axes[0].set_title(f"{title} Histogram")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")

    sorted_values = np.sort(values.to_numpy())
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values) if len(sorted_values) else np.array([])
    axes[1].plot(sorted_values, cdf, color="tab:orange")
    axes[1].set_title(f"{title} CDF")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Cumulative share")
    close_figure(fig, path)


def simple_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    path: Path,
    bins: int = 40,
) -> None:
    values = pd.Series(series).dropna().astype(float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=bins, color="tab:blue", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    close_figure(fig, path)


def bar_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    rotation: int = 45,
) -> None:
    values = series.copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(values.index.astype(str), values.values, color="tab:blue", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    close_figure(fig, path)


def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str | None,
    title: str,
    path: Path,
    alpha: float = 0.8,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if color_col and color_col in df.columns:
        categories = df[color_col].fillna("unknown").astype(str)
        for category in sorted(categories.unique()):
            mask = categories == category
            ax.scatter(df.loc[mask, x], df.loc[mask, y], s=20, alpha=alpha, label=category)
        ax.legend(title=color_col, fontsize=8)
    else:
        ax.scatter(df[x], df[y], s=20, alpha=alpha, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    close_figure(fig, path)


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    close_figure(fig, path)


def multi_histogram(
    grouped_values: Iterable[tuple[str, pd.Series]],
    title: str,
    xlabel: str,
    path: Path,
    bins: int = 30,
) -> None:
    grouped_values = list(grouped_values)
    n_panels = max(len(grouped_values), 1)
    cols = 2
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, series) in zip(axes, grouped_values):
        values = pd.Series(series).dropna().astype(float)
        ax.hist(values, bins=bins, alpha=0.85, color="tab:blue")
        ax.set_title(str(label))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
    for ax in axes[len(grouped_values) :]:
        ax.axis("off")
    fig.suptitle(title)
    close_figure(fig, path)
