"""Reproduce the published four-group association table from the derived release.

Evaluation labels are read only after each assignment is fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from reconstruct_dataset import TABLES, reconstruct  # noqa: E402

ANCHORS = {"H": 1, "T": 0}
TARGETS = ["B", "O", "N", "R"]
PC_COLS = ["PC1", "PC2", "PC3"]
KIN_COLS = [
    "mean_speed_kmh",
    "p95_speed_kmh",
    "harsh_accel_per_km",
    "harsh_brake_per_km",
]
EXPECTED = {
    "geography": 3,
    "time_of_day": 0,
    "speed_harsh_centroid": 3,
    "embedding_centroid": 4,
    "co_membership": 4,
    "spectral_runs_matching_all_four": 6,
    "spectral_runs_naming_none": 4,
}


def _majority(row: pd.Series):
    labeled = int(row.eval_n_driver_t) + int(row.eval_n_driver_h)
    if labeled == 0:
        return None
    return 1 if int(row.eval_n_driver_h) >= int(row.eval_n_driver_t) else 0


def _score(assigned: dict[str, int | None], groups: pd.DataFrame) -> int:
    correct = 0
    for code in TARGETS:
        truth = _majority(groups.loc[code])
        pred = assigned[code]
        if truth is not None and pred is not None and int(pred) == truth:
            correct += 1
    return correct


def _misses(assigned: dict[str, int | None], groups: pd.DataFrame) -> list[str]:
    missed = []
    for code in TARGETS:
        truth = _majority(groups.loc[code])
        pred = assigned[code]
        if not (truth is not None and pred is not None and int(pred) == truth):
            missed.append(code)
    return missed


def _nearer(groups: pd.DataFrame, column_h: str, column_t: str, smaller_wins: bool) -> dict[str, int | None]:
    assigned: dict[str, int | None] = dict(ANCHORS)
    for code, row in groups.iterrows():
        if code in ANCHORS:
            continue
        left = float(row[column_h])
        right = float(row[column_t])
        if np.isclose(left, right):
            assigned[str(code)] = None
        elif smaller_wins:
            assigned[str(code)] = 1 if left < right else 0
        else:
            assigned[str(code)] = 1 if left > right else 0
    return assigned


def _centroids(frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    return {
        str(code): group[columns].mean().to_numpy(dtype=float)
        for code, group in frame.groupby("group_code")
    }


def _nearest_centroid(centroids: dict[str, np.ndarray]) -> dict[str, int | None]:
    assigned: dict[str, int | None] = dict(ANCHORS)
    anchor_h = centroids["H"]
    anchor_t = centroids["T"]
    for code, vector in centroids.items():
        if code in ANCHORS:
            continue
        distance_h = float(np.linalg.norm(vector - anchor_h))
        distance_t = float(np.linalg.norm(vector - anchor_t))
        if np.isclose(distance_h, distance_t):
            assigned[code] = None
        else:
            assigned[code] = 1 if distance_h < distance_t else 0
    return assigned


def _speed(trips: pd.DataFrame) -> dict[str, int | None]:
    scaled = trips.copy()
    base = scaled[scaled["group_code"].isin(ANCHORS)]
    for column in KIN_COLS:
        mean = float(base[column].mean())
        deviation = float(base[column].std(ddof=0))
        scaled[column] = 0.0 if deviation == 0.0 else (scaled[column] - mean) / deviation
    return _nearest_centroid(_centroids(scaled, KIN_COLS))


def _propagate(labels: dict[str, int]) -> dict[str, int | None]:
    assigned: dict[str, int | None] = dict(ANCHORS)
    if labels["H"] == labels["T"]:
        for code in labels:
            if code not in ANCHORS:
                assigned[code] = None
        return assigned
    for code, label in labels.items():
        if code in ANCHORS:
            continue
        if label == labels["H"]:
            assigned[code] = 1
        elif label == labels["T"]:
            assigned[code] = 0
        else:
            assigned[code] = None
    return assigned


def reproduce() -> dict[str, int]:
    if not (TABLES / "groups.csv").exists():
        reconstruct()
    groups = pd.read_csv(TABLES / "groups.csv").set_index("group_code")
    segments = pd.read_csv(TABLES / "segments.csv")
    trips = pd.read_csv(TABLES / "trips_kinematics.csv")
    affinity = pd.read_csv(TABLES / "affinity.csv").set_index("group_code")
    runs = pd.read_csv(TABLES / "spectral_runs.csv")

    geography = _nearer(groups, "km_to_anchor_h", "km_to_anchor_t", smaller_wins=True)
    time_of_day = _nearer(groups, "hour_l1_to_anchor_h", "hour_l1_to_anchor_t", smaller_wins=True)
    speed = _speed(trips)
    embedding = _nearest_centroid(_centroids(segments, PC_COLS))
    membership = _nearer(affinity, "H", "T", smaller_wins=False)
    per_run = []
    for _, run in runs.groupby("run_index"):
        labels = dict(zip(run["group_code"], run["cluster_label"].astype(int)))
        per_run.append(_score(_propagate(labels), groups))
    results = {
        "geography": _score(geography, groups),
        "time_of_day": _score(time_of_day, groups),
        "speed_harsh_centroid": _score(speed, groups),
        "embedding_centroid": _score(embedding, groups),
        "co_membership": _score(membership, groups),
        "spectral_runs_matching_all_four": int(sum(count == 4 for count in per_run)),
        "spectral_runs_naming_none": int(sum(count == 0 for count in per_run)),
    }
    if results != EXPECTED:
        raise SystemExit(f"result mismatch: {results}")
    if _misses(geography, groups) != ["O"] or _misses(speed, groups) != ["O"]:
        raise SystemExit("the single geography or speed miss is not group O")
    if _misses(embedding, groups) or _misses(membership, groups):
        raise SystemExit("embedding or co-membership missed a target")
    if len(per_run) != 10:
        raise SystemExit(f"expected 10 spectral runs, found {len(per_run)}")
    return results


if __name__ == "__main__":
    print(reproduce())
