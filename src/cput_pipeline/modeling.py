from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from .config import PipelineConfig
from .dataset import (
    assign_hour_bucket,
    deduplicate_coordinates,
    round_or_unknown,
    split_dataframe,
    summarize_trip_driver,
)
from .io_utils import close_figure, save_dataframe, save_joblib, save_json, save_npz, heatmap

try:
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
except ImportError as exc:  # pragma: no cover - import guard
    MiniRocketMultivariate = None
    MINIROCKET_IMPORT_ERROR = exc
else:
    MINIROCKET_IMPORT_ERROR = None


FEATURE_COLUMNS = ["speed", "acceleration_est_1", "angular_acc"]
SCALING_MODE_NOTEBOOK = "notebook_bucket_zscore"
SCALING_MODE_SEGMENT = "segment_zscore"
VALID_SCALING_MODES = {SCALING_MODE_NOTEBOOK, SCALING_MODE_SEGMENT}


def _require_minirocket() -> None:
    if MiniRocketMultivariate is None:
        raise ImportError(
            "MiniRocketMultivariate is unavailable. Install `sktime` before running 04_cput_clustering.py."
        ) from MINIROCKET_IMPORT_ERROR


def _normalization_road_speed_key(value: object) -> float | str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "unknown"
    return float(numeric_value)


def resolve_scaling_mode(scaling_mode: str | None, config: PipelineConfig) -> str:
    mode = config.scaling_mode if scaling_mode is None else str(scaling_mode)
    if mode not in VALID_SCALING_MODES:
        raise ValueError(f"Unsupported scaling mode: {mode}")
    return mode


def scaling_mode_output_path(base_path: Path, scaling_mode: str) -> Path:
    if scaling_mode == SCALING_MODE_NOTEBOOK:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{scaling_mode}{base_path.suffix}")


def reconstruct_segment_series(
    points_grouped: pd.DataFrame,
    trips_grouped: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    trip_metadata = trips_grouped.set_index("drive_id")[
        ["pattern_group", "driver", "eligible_for_modeling", "pattern_group_size"]
    ].to_dict("index")
    segment_map: dict[str, pd.DataFrame] = {}
    for drive_id, trip in points_grouped.groupby("drive_id", sort=False):
        trip = trip.sort_values("orig_time").reset_index(drop=True)
        trip_dedup = deduplicate_coordinates(trip)
        segment_list = split_dataframe(trip_dedup, config.stop_eps)
        trip_info = trip_metadata.get(int(drive_id), {})
        for segment_index, segment in enumerate(segment_list, start=1):
            segment = segment.copy()
            segment = segment.dropna(subset=FEATURE_COLUMNS + ["orig_time"]).reset_index(drop=True)
            if segment.empty:
                continue
            segment_id = f"{int(drive_id)}_{segment_index}"
            segment["segment_id"] = segment_id
            segment["drive_id"] = int(drive_id)
            segment["pattern_group"] = trip_info.get("pattern_group", pd.NA)
            segment["driver"] = trip_info.get("driver", summarize_trip_driver(segment["driver"]))
            segment["eligible_for_modeling"] = bool(trip_info.get("eligible_for_modeling", False))
            segment["pattern_group_size"] = trip_info.get("pattern_group_size", pd.NA)
            segment["hour_bucket"] = assign_hour_bucket(int(segment["orig_time"].iloc[0].hour), config.hour_buckets)
            road_speed_value = segment["road_speed"].iloc[0]
            segment["normalization_road_speed"] = _normalization_road_speed_key(road_speed_value)
            segment["road_speed_bucket"] = round_or_unknown(
                pd.to_numeric(pd.Series([road_speed_value]), errors="coerce").iloc[0],
                decimals=0,
            )
            segment_map[segment_id] = segment
    save_joblib(segment_map, config.segment_timeseries_joblib)
    return segment_map


def normalize_time_series(
    df_list: list[pd.DataFrame],
    scaling_mode: str = SCALING_MODE_NOTEBOOK,
) -> list[pd.DataFrame]:
    if not df_list:
        return []
    normalized_input = [df.reset_index(drop=True).copy() for df in df_list]

    if scaling_mode == SCALING_MODE_NOTEBOOK:
        concatenated = pd.concat(normalized_input, axis=0)
        grouped = concatenated[FEATURE_COLUMNS].groupby(concatenated.index)
        mean_series = grouped.mean()
        std_series = grouped.std().replace(0, 1)

        normalized_list: list[pd.DataFrame] = []
        for df in normalized_input:
            normalized_df = df.copy()
            index_values = normalized_df.index
            for column in FEATURE_COLUMNS:
                normalized_df[column] = (
                    normalized_df[column] - mean_series.loc[index_values, column]
                ) / std_series.loc[index_values, column]
            normalized_list.append(normalized_df)
        return normalized_list

    if scaling_mode == SCALING_MODE_SEGMENT:
        normalized_list = []
        for df in normalized_input:
            normalized_df = df.copy()
            feature_means = normalized_df[FEATURE_COLUMNS].mean()
            feature_stds = normalized_df[FEATURE_COLUMNS].std().replace(0, 1).fillna(1)
            normalized_df[FEATURE_COLUMNS] = (
                normalized_df[FEATURE_COLUMNS] - feature_means
            ) / feature_stds
            normalized_list.append(normalized_df)
        return normalized_list

    raise ValueError(f"Unsupported scaling mode: {scaling_mode}")


def build_normalized_buckets(
    segment_map: dict[str, pd.DataFrame],
    config: PipelineConfig,
    scaling_mode: str | None = None,
    *,
    persist: bool = True,
) -> tuple[
    dict[int, list[pd.DataFrame]],
    dict[int, list[pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
]:
    scaling_mode = resolve_scaling_mode(scaling_mode, config)
    scaling_parameters: dict[tuple[int, float | str, int], list[pd.DataFrame]] = defaultdict(list)
    for segment in segment_map.values():
        if not bool(segment["eligible_for_modeling"].iloc[0]):
            continue
        pattern_group = segment["pattern_group"].iloc[0]
        if pd.isna(pattern_group):
            continue
        length = len(segment)
        if not (config.segment_length_min < length < config.segment_length_max):
            continue
        key = (
            length,
            segment["normalization_road_speed"].iloc[0],
            int(segment["hour_bucket"].iloc[0]),
        )
        scaling_parameters[key].append(segment)

    if config.min_bin_size_is_strict_lower_bound:
        filtered_parameters = {
            key: value for key, value in scaling_parameters.items() if len(value) > config.min_bin_size
        }
    else:
        filtered_parameters = {
            key: value for key, value in scaling_parameters.items() if len(value) >= config.min_bin_size
        }
    groups_splitted_normalized_ts: dict[int, list[pd.DataFrame]] = defaultdict(list)
    dict_length: dict[int, list[pd.DataFrame]] = defaultdict(list)
    bucket_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for key, series_list in filtered_parameters.items():
        normalized_list = normalize_time_series(series_list, scaling_mode=scaling_mode)
        length, road_speed_key, hour_bucket = key
        bucket_rows.append(
            {
                "scaling_mode": scaling_mode,
                "length": int(length),
                "normalization_road_speed": str(road_speed_key),
                "hour_bucket": int(hour_bucket),
                "segment_count": int(len(series_list)),
                "min_bin_size": int(config.min_bin_size),
                "min_bin_operator": ">" if config.min_bin_size_is_strict_lower_bound else ">=",
            }
        )
        concatenated_normalized = pd.concat(
            [df[FEATURE_COLUMNS].reset_index(drop=True) for df in normalized_list],
            axis=0,
            ignore_index=True,
        )
        for column in FEATURE_COLUMNS:
            values = concatenated_normalized[column]
            feature_rows.append(
                {
                    "scaling_mode": scaling_mode,
                    "length": int(length),
                    "normalization_road_speed": str(road_speed_key),
                    "hour_bucket": int(hour_bucket),
                    "feature": column,
                    "value_count": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        for normalized_df in normalized_list:
            pattern_group = int(normalized_df["pattern_group"].iloc[0])
            groups_splitted_normalized_ts[pattern_group].append(normalized_df)
            if len(normalized_df) >= 9:
                dict_length[len(normalized_df)].append(normalized_df)

    groups_splitted_normalized_ts = dict(sorted(groups_splitted_normalized_ts.items()))
    dict_length = dict(sorted(dict_length.items()))
    bucket_diagnostics_df = pd.DataFrame(
        bucket_rows,
        columns=[
            "scaling_mode",
            "length",
            "normalization_road_speed",
            "hour_bucket",
            "segment_count",
            "min_bin_size",
            "min_bin_operator",
        ],
    )
    if not bucket_diagnostics_df.empty:
        bucket_diagnostics_df = bucket_diagnostics_df.sort_values(
            ["length", "hour_bucket", "normalization_road_speed"],
            ignore_index=True,
        )
    feature_diagnostics_df = pd.DataFrame(
        feature_rows,
        columns=[
            "scaling_mode",
            "length",
            "normalization_road_speed",
            "hour_bucket",
            "feature",
            "value_count",
            "mean",
            "std",
            "min",
            "max",
        ],
    )
    if not feature_diagnostics_df.empty:
        feature_diagnostics_df = feature_diagnostics_df.sort_values(
            ["feature", "length", "hour_bucket", "normalization_road_speed"],
            ignore_index=True,
        )
    if not feature_diagnostics_df.empty:
        global_feature_rows = []
        for column in FEATURE_COLUMNS:
            mode_segments = [
                df[column]
                for group_segments in groups_splitted_normalized_ts.values()
                for df in group_segments
            ]
            if not mode_segments:
                continue
            values = pd.concat(mode_segments, ignore_index=True)
            global_feature_rows.append(
                {
                    "scaling_mode": scaling_mode,
                    "length": -1,
                    "normalization_road_speed": "ALL",
                    "hour_bucket": -1,
                    "feature": column,
                    "value_count": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        if global_feature_rows:
            feature_diagnostics_df = pd.concat(
                [feature_diagnostics_df, pd.DataFrame(global_feature_rows)],
                ignore_index=True,
            )
    if persist:
        save_joblib(groups_splitted_normalized_ts, config.normalized_segments_joblib)
        save_joblib(dict_length, config.length_buckets_joblib)
    save_dataframe(
        bucket_diagnostics_df,
        scaling_mode_output_path(
            config.cput_compare_root / "tables" / "normalization_bucket_diagnostics.csv",
            scaling_mode,
        ),
    )
    save_dataframe(
        feature_diagnostics_df,
        scaling_mode_output_path(
            config.cput_compare_root / "tables" / "normalization_feature_diagnostics.csv",
            scaling_mode,
        ),
    )
    summary_payload = {
        "scaling_mode": scaling_mode,
        "bucket_count": int(len(bucket_diagnostics_df)),
        "normalized_pattern_group_count": int(len(groups_splitted_normalized_ts)),
        "length_bucket_count": int(len(dict_length)),
        "segment_count": int(sum(len(value) for value in dict_length.values())),
        "min_bin_operator": ">" if config.min_bin_size_is_strict_lower_bound else ">=",
        "min_bin_size": int(config.min_bin_size),
    }
    save_json(
        summary_payload,
        scaling_mode_output_path(
            config.cput_compare_root / "tables" / "normalization_summary.json",
            scaling_mode,
        ),
    )
    return groups_splitted_normalized_ts, dict_length, bucket_diagnostics_df, feature_diagnostics_df


def convert_dfs_to_arrays(df_list: list[pd.DataFrame]) -> tuple[np.ndarray, pd.DataFrame]:
    arrays = []
    metadata_rows = []
    for df in df_list:
        arrays.append(df[FEATURE_COLUMNS].to_numpy().T)
        metadata_rows.append(
            {
                "segment_id": df["segment_id"].iloc[0],
                "drive_id": int(df["drive_id"].iloc[0]),
                "pattern_group": int(df["pattern_group"].iloc[0]),
                "driver": str(df["driver"].iloc[0]),
                "length": int(len(df)),
                "road_speed_bucket": str(df["road_speed_bucket"].iloc[0]),
                "hour_bucket": int(df["hour_bucket"].iloc[0]),
            }
        )
    return np.stack(arrays), pd.DataFrame(metadata_rows)


def compute_minirocket_features(
    dict_length: dict[int, list[pd.DataFrame]],
    config: PipelineConfig,
    random_state: int | None = None,
    persist: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    _require_minirocket()
    features_list: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []

    for length, df_list in dict_length.items():
        if not df_list:
            continue
        arrays, metadata = convert_dfs_to_arrays(df_list)
        rocket = MiniRocketMultivariate(
            num_kernels=config.minirocket_num_kernels,
            random_state=config.random_seed if random_state is None else int(random_state),
        )
        transformed = rocket.fit_transform(arrays)
        transformed = np.asarray(transformed)
        features_list.append(transformed)
        metadata["length_bucket"] = length
        metadata_frames.append(metadata)

    if not features_list:
        raise ValueError("No eligible normalized segments were available for MiniRocket feature generation.")

    X = np.vstack(features_list)
    metadata_df = pd.concat(metadata_frames, ignore_index=True)
    if persist:
        save_npz(
            config.minirocket_features_npz,
            X=X,
            segment_id=metadata_df["segment_id"].astype(str).to_numpy(),
            drive_id=metadata_df["drive_id"].to_numpy(),
            pattern_group=metadata_df["pattern_group"].to_numpy(),
            length_bucket=metadata_df["length_bucket"].to_numpy(),
        )
        save_dataframe(metadata_df, config.minirocket_metadata_csv)
    return X, metadata_df


def run_pca(
    X: np.ndarray,
    metadata_df: pd.DataFrame,
    config: PipelineConfig,
    n_components: int | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    component_count = config.pca_components if n_components is None else int(n_components)
    pca = PCA(n_components=component_count, random_state=config.random_seed)
    components = pca.fit_transform(X)
    component_names = [f"PC{index}" for index in range(1, component_count + 1)]
    df_pca = metadata_df.copy()
    for index, column_name in enumerate(component_names):
        df_pca[column_name] = components[:, index]

    if persist:
        variance_df = pd.DataFrame(
            {
                "component": component_names,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        save_dataframe(variance_df, config.pca_variance_csv)
    return df_pca


def plot_pca_scatter(df_pca: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = df_pca["pattern_group"].astype(str)
    for category in sorted(categories.unique(), key=lambda value: int(value)):
        mask = categories == category
        ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"], s=20, alpha=0.8, label=category)
    ax.set_title("MiniRocket PCA Scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="pattern_group", fontsize=8)
    close_figure(fig, path)


def plot_dbscan_group(group_df: pd.DataFrame, group_id: int, path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        group_df["PC1"],
        group_df["PC2"],
        group_df["PC3"],
        c=group_df["dbscan_label"],
        cmap="viridis",
        s=25,
    )
    ax.set_title(f"Pattern Group {group_id} DBSCAN Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(scatter, ax=ax, label="dbscan_label")
    close_figure(fig, path)


def select_feature_subset(
    X: np.ndarray,
    feature_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(random_state))
    subset_size = max(1, int(np.floor(X.shape[1] * feature_fraction)))
    selected_columns = np.sort(rng.choice(X.shape[1], size=subset_size, replace=False))
    return X[:, selected_columns], selected_columns


def _run_group_dbscan(
    group_df: pd.DataFrame,
    pca_columns: list[str],
    *,
    eps: float,
    min_samples: int,
) -> pd.Series:
    if len(group_df) < int(min_samples):
        return pd.Series(-1, index=group_df.index, dtype="int64")
    dbscan = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    return pd.Series(dbscan.fit_predict(group_df[pca_columns]), index=group_df.index)


def apply_group_outlier_removal(
    df_pca: pd.DataFrame,
    pattern_groups_df: pd.DataFrame,
    config: PipelineConfig,
    save_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    labeled_groups = []
    largest_cluster_rows = []
    prototype_vectors: dict[int, np.ndarray] = {}
    pca_columns = [column for column in df_pca.columns if column.startswith("PC")]
    neighborhood_lookup = (
        pattern_groups_df.set_index("pattern_group")["destination_neighborhood"].to_dict()
        if not pattern_groups_df.empty
        else {}
    )

    for group_id, group_df in df_pca.groupby("pattern_group", sort=True):
        group_df = group_df.copy()
        if len(group_df) < config.outlier_dbscan_min_samples:
            group_df["dbscan_label"] = -1
        else:
            dbscan = DBSCAN(eps=config.outlier_dbscan_eps, min_samples=config.outlier_dbscan_min_samples)
            group_df["dbscan_label"] = dbscan.fit_predict(group_df[pca_columns])

        non_noise = group_df[group_df["dbscan_label"] != -1]
        neighborhood_name = neighborhood_lookup.get(int(group_id), str(group_id))
        if non_noise.empty:
            largest_cluster_rows.append(
                {
                    "pattern_group": int(group_id),
                    "destination_neighborhood": neighborhood_name,
                    "noise_rate": 1.0,
                    "largest_cluster_label": -1,
                    "largest_cluster_size": 0,
                    "largest_cluster_share": 0.0,
                    "retained_for_bagging": False,
                }
            )
        else:
            cluster_sizes = non_noise["dbscan_label"].value_counts()
            largest_cluster_label = int(cluster_sizes.idxmax())
            largest_cluster = non_noise[non_noise["dbscan_label"] == largest_cluster_label].copy()
            retained_for_bagging = len(largest_cluster) >= config.min_pattern_group_size_for_modeling
            prototype_vectors[int(group_id)] = largest_cluster[pca_columns].mean().to_numpy()
            largest_cluster_rows.append(
                {
                    "pattern_group": int(group_id),
                    "destination_neighborhood": neighborhood_name,
                    "noise_rate": float((group_df["dbscan_label"] == -1).mean()),
                    "largest_cluster_label": largest_cluster_label,
                    "largest_cluster_size": int(len(largest_cluster)),
                    "largest_cluster_share": float(len(largest_cluster) / len(group_df)),
                    "retained_for_bagging": bool(retained_for_bagging),
                    **{
                        f"largest_cluster_{column.lower()}_mean": float(largest_cluster[column].mean())
                        for column in pca_columns
                    },
                    **{
                        f"largest_cluster_{column.lower()}_std": float(largest_cluster[column].std(ddof=0))
                        for column in pca_columns
                    },
                }
            )
        if save_plots:
            plot_dbscan_group(
                group_df,
                int(group_id),
                config.cput_root / "figures" / f"dbscan_3d_group_{int(group_id)}.png",
            )
        labeled_groups.append(group_df)

    if not labeled_groups:
        raise ValueError("No PCA rows were available for DBSCAN.")

    return pd.concat(labeled_groups, ignore_index=True), pd.DataFrame(largest_cluster_rows), prototype_vectors


def run_dbscan(
    df_pca: pd.DataFrame,
    pattern_groups_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_df, largest_cluster_df, _ = apply_group_outlier_removal(
        df_pca,
        pattern_groups_df,
        config,
        save_plots=True,
    )
    save_dataframe(labeled_df, config.segments_embedding_csv)
    save_dataframe(
        labeled_df[["segment_id", "pattern_group", "dbscan_label"]],
        config.dbscan_labels_csv,
    )
    save_dataframe(largest_cluster_df, config.group_largest_cluster_stats_csv)
    return labeled_df, largest_cluster_df


def _choose_spectral_k(X_means: np.ndarray, max_k: int, random_state: int | None) -> int:
    n_groups = len(X_means)
    if n_groups <= 2:
        return 2
    max_candidate = min(max_k, n_groups - 1)
    best_k = 2
    best_score = -np.inf
    for k in range(2, max_candidate + 1):
        if random_state is None:
            spectral = SpectralClustering(n_clusters=k)
        else:
            spectral = SpectralClustering(n_clusters=k, random_state=random_state)
        labels = spectral.fit_predict(X_means)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(X_means, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _compute_iteration_group_centroids(
    df_pca: pd.DataFrame,
    group_ids: list[int],
    config: PipelineConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    pca_columns = [column for column in df_pca.columns if column.startswith("PC")]
    X_means: list[np.ndarray] = []
    centroid_rows: list[dict[str, object]] = []

    for group_id in group_ids:
        group_df = df_pca[df_pca["pattern_group"] == group_id].copy()
        if group_df.empty:
            continue

        if len(group_df) < config.outlier_dbscan_min_samples:
            group_df["dbscan_label"] = -1
        else:
            dbscan = DBSCAN(
                eps=config.outlier_dbscan_eps,
                min_samples=config.outlier_dbscan_min_samples,
            )
            group_df["dbscan_label"] = dbscan.fit_predict(group_df[pca_columns])

        largest_cluster_label = int(group_df["dbscan_label"].value_counts().idxmax())
        largest_cluster = group_df[group_df["dbscan_label"] == largest_cluster_label].copy()
        X_means.append(largest_cluster[pca_columns].mean().to_numpy())
        centroid_rows.append(
            {
                "pattern_group": int(group_id),
                "largest_cluster_label": largest_cluster_label,
                "largest_cluster_size": int(len(largest_cluster)),
                "group_size": int(len(group_df)),
            }
        )

    if not X_means:
        return np.empty((0, 0)), centroid_rows
    return np.vstack(X_means), centroid_rows


def _compute_notebook_bagging_centroids(
    df_pca: pd.DataFrame,
    group_ids: list[int],
    config: PipelineConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    pca_columns = [column for column in df_pca.columns if column.startswith("PC")]
    X_means: list[np.ndarray] = []
    centroid_rows: list[dict[str, object]] = []

    for group_id in group_ids:
        group_df = df_pca[df_pca["pattern_group"] == group_id].copy()
        if group_df.empty:
            continue

        group_df["dbscan_label"] = _run_group_dbscan(
            group_df,
            pca_columns,
            eps=config.outlier_dbscan_eps,
            min_samples=config.outlier_dbscan_min_samples,
        )
        largest_cluster_label = int(group_df["dbscan_label"].value_counts().idxmax())
        largest_cluster = group_df[group_df["dbscan_label"] == largest_cluster_label].copy()
        X_means.append(largest_cluster[pca_columns].mean().to_numpy())
        centroid_rows.append(
            {
                "pattern_group": int(group_id),
                "largest_cluster_label": largest_cluster_label,
                "largest_cluster_size": int(len(largest_cluster)),
                "group_size": int(len(group_df)),
            }
        )

    if not X_means:
        return np.empty((0, 0)), centroid_rows
    return np.vstack(X_means), centroid_rows


def run_spectral_bagging(
    dict_length: dict[int, list[pd.DataFrame]],
    pattern_groups_df: pd.DataFrame,
    config: PipelineConfig,
    eligible_group_ids: set[int],
    *,
    persist: bool = True,
    output_suffix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_path = config.spectral_bagging_matrix_csv
    assignments_path = config.spectral_bagging_assignments_csv
    figure_path = config.cput_root / "figures" / "spectral_bagging_matrix.png"
    if output_suffix:
        matrix_path = matrix_path.with_name(f"{matrix_path.stem}_{output_suffix}{matrix_path.suffix}")
        assignments_path = assignments_path.with_name(
            f"{assignments_path.stem}_{output_suffix}{assignments_path.suffix}"
        )
        figure_path = figure_path.with_name(f"{figure_path.stem}_{output_suffix}{figure_path.suffix}")
    if not eligible_group_ids:
        empty_matrix = pd.DataFrame()
        empty_assignments = pd.DataFrame()
        if persist:
            save_dataframe(empty_matrix, matrix_path)
            save_dataframe(empty_assignments, assignments_path)
        return empty_matrix, empty_assignments

    group_name_map = pattern_groups_df.set_index("pattern_group")["destination_neighborhood"].to_dict()
    sorted_group_ids = sorted(eligible_group_ids)
    group_names = [group_name_map[group_id] for group_id in sorted_group_ids]
    matrix = pd.DataFrame(0.0, index=group_names, columns=group_names)
    assignment_rows: list[dict[str, object]] = []
    random_states = np.random.RandomState(config.random_seed).randint(
        0, 10000, size=config.bagging_num_runs
    )

    for run_index, seed in enumerate(random_states, start=1):
        X_run, metadata_run = compute_minirocket_features(
            dict_length,
            config,
            random_state=int(seed),
            persist=False,
        )
        df_pca_run = run_pca(
            X_run,
            metadata_run,
            config,
            n_components=config.bagging_pca_components,
            persist=False,
        )
        X_means, centroid_rows = _compute_notebook_bagging_centroids(
            df_pca_run,
            sorted_group_ids,
            config,
        )
        group_ids_in_run = [int(row["pattern_group"]) for row in centroid_rows]
        if len(group_ids_in_run) < 2:
            continue

        best_k = _choose_spectral_k(X_means, config.bagging_max_k, random_state=None)
        spectral = SpectralClustering(
            n_clusters=best_k,
            random_state=config.bagging_spectral_random_seed,
        )
        spectral_labels = spectral.fit_predict(X_means)

        centroid_row_lookup = {int(row["pattern_group"]): row for row in centroid_rows}
        for group_id, cluster_label in zip(group_ids_in_run, spectral_labels):
            centroid_row = centroid_row_lookup[group_id]
            assignment_rows.append(
                {
                    "run_index": run_index,
                    "seed": int(seed),
                    "pattern_group": group_id,
                    "destination_neighborhood": group_name_map[group_id],
                    "spectral_cluster_label": int(cluster_label),
                    "selected_feature_count": int(X_run.shape[1]),
                    "largest_cluster_label": int(centroid_row["largest_cluster_label"]),
                    "largest_cluster_size": int(centroid_row["largest_cluster_size"]),
                    "group_size": int(centroid_row["group_size"]),
                    "best_k": int(best_k),
                }
            )

        for left_group_id, left_label in zip(group_ids_in_run, spectral_labels):
            for right_group_id, right_label in zip(group_ids_in_run, spectral_labels):
                if int(left_label) == int(right_label):
                    matrix.loc[group_name_map[left_group_id], group_name_map[right_group_id]] += 1

    normalized_matrix = (matrix / max(config.bagging_num_runs, 1)).round(2)
    assignments_df = pd.DataFrame(assignment_rows)
    if persist:
        save_dataframe(
            normalized_matrix.reset_index().rename(columns={"index": "destination_neighborhood"}),
            matrix_path,
        )
        save_dataframe(assignments_df, assignments_path)
        heatmap(
            normalized_matrix,
            title="Spectral Bagging Co-occurrence Matrix",
            path=figure_path,
        )
    return normalized_matrix, assignments_df


def analyze_seeded_spectral_matrix(
    matrix: pd.DataFrame,
    seed_to_driver_label: dict[str, str],
    *,
    clique_threshold: float,
    forced_mixed_groups: set[str] | None = None,
    borderline_low: float | None = None,
) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame(
            columns=[
                "group_name",
                "cluster_id",
                "cluster_members",
                "decision_status",
                "predicted_driver_label",
                "auc_include",
                "exclusion_reason",
            ]
        )

    forced_mixed_groups = set() if forced_mixed_groups is None else {str(value) for value in forced_mixed_groups}
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    groups = list(matrix.index)
    seed_names = [str(name) for name in seed_to_driver_label]
    if borderline_low is None:
        borderline_low = max(0.0, 1.0 - float(clique_threshold))

    visited: set[str] = set()
    cluster_members: dict[int, list[str]] = {}
    cluster_lookup: dict[str, int] = {}
    exclusion_reason_lookup: dict[str, str] = {}
    cluster_id = 0

    for group in groups:
        if group in visited:
            continue
        high_sim_groups = [
            candidate
            for candidate in groups
            if candidate not in visited and float(matrix.loc[group, candidate]) >= float(clique_threshold)
        ]
        if len(high_sim_groups) == 1:
            borderline_groups = [
                candidate
                for candidate in groups
                if candidate != group and float(borderline_low) < float(matrix.loc[group, candidate]) < float(clique_threshold)
            ]
            visited.add(group)
            if borderline_groups:
                exclusion_reason_lookup[group] = "borderline_singleton"
                continue
            cluster_id += 1
            cluster_members[cluster_id] = [group]
            cluster_lookup[group] = cluster_id
            continue

        valid_cluster = True
        for left_group in high_sim_groups:
            for right_group in high_sim_groups:
                if float(matrix.loc[left_group, right_group]) < float(clique_threshold):
                    valid_cluster = False
                    break
            if not valid_cluster:
                break
        if not valid_cluster:
            visited.add(group)
            exclusion_reason_lookup[group] = "failed_clique_check"
            continue

        cluster_id += 1
        cluster_members[cluster_id] = list(high_sim_groups)
        for member in high_sim_groups:
            visited.add(member)
            cluster_lookup[member] = cluster_id

    decision_rows: list[dict[str, object]] = []
    for group in groups:
        current_cluster_id = cluster_lookup.get(group)
        members = cluster_members.get(current_cluster_id, [])
        seeded_driver_labels = sorted(
            {
                seed_to_driver_label[member]
                for member in members
                if member in seed_to_driver_label
            }
        )
        decision_status = "confident"
        predicted_driver_label: str | None = None
        auc_include = False
        exclusion_reason = exclusion_reason_lookup.get(group, "")

        if group in forced_mixed_groups:
            decision_status = "mixed_excluded"
            exclusion_reason = "forced_mixed_group"
        elif current_cluster_id is None:
            decision_status = "ambiguous_excluded"
            exclusion_reason = exclusion_reason or "not_in_confident_clique"
        elif len(seeded_driver_labels) == 1:
            predicted_driver_label = seeded_driver_labels[0]
            auc_include = True
        elif len(seeded_driver_labels) > 1:
            decision_status = "ambiguous_excluded"
            exclusion_reason = "multi_seed_cluster"
        else:
            decision_status = "unseeded_excluded"
            exclusion_reason = "cluster_has_no_seed_group"

        row = {
            "group_name": group,
            "cluster_id": current_cluster_id,
            "cluster_members": "|".join(members),
            "decision_status": decision_status,
            "predicted_driver_label": predicted_driver_label,
            "auc_include": bool(auc_include),
            "exclusion_reason": exclusion_reason,
            "seed_similarity_gap": np.nan,
        }
        seed_scores: dict[str, float] = {}
        for seed_name in seed_names:
            score = float(matrix.loc[group, seed_name]) if seed_name in matrix.columns else np.nan
            row[f"similarity_to_{seed_name}"] = score
            seed_scores[seed_name] = score
        valid_scores = [value for value in seed_scores.values() if not pd.isna(value)]
        if valid_scores:
            row["seed_similarity_gap"] = float(max(valid_scores) - min(valid_scores))
        decision_rows.append(row)

    return pd.DataFrame(decision_rows)


def train_group_classifiers(
    labeled_df: pd.DataFrame,
    X: np.ndarray,
    config: PipelineConfig,
    eligible_group_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if eligible_group_ids is not None:
        labeled_df = labeled_df[labeled_df["pattern_group"].isin(sorted(eligible_group_ids))].copy()
    feature_index = pd.Series(np.arange(len(labeled_df)), index=labeled_df["segment_id"])
    metrics_rows = []
    predicted_segments = []
    trip_rows = []

    for group_id, group_df in labeled_df.groupby("pattern_group", sort=True):
        group_df = group_df.copy()
        labeled_group = group_df[group_df["dbscan_label"] != -1].copy()
        if labeled_group.empty:
            metrics_rows.append(
                {
                    "pattern_group": int(group_id),
                    "status": "skipped_no_non_noise_labels",
                    "num_training_segments": 0,
                    "num_classes": 0,
                }
            )
            continue

        class_counts = labeled_group["dbscan_label"].value_counts()
        valid_labels = class_counts[class_counts >= config.classifier_min_class_size].index
        labeled_group = labeled_group[labeled_group["dbscan_label"].isin(valid_labels)].copy()
        if labeled_group.empty:
            metrics_rows.append(
                {
                    "pattern_group": int(group_id),
                    "status": "skipped_after_class_filter",
                    "num_training_segments": 0,
                    "num_classes": 0,
                }
            )
            continue

        if labeled_group["dbscan_label"].nunique() < 2:
            constant_label = int(labeled_group["dbscan_label"].iloc[0])
            majority_driver = (
                labeled_group.loc[labeled_group["driver"] != "not known", "driver"].mode().iloc[0]
                if not labeled_group.loc[labeled_group["driver"] != "not known", "driver"].mode().empty
                else "not known"
            )
            metrics_rows.append(
                {
                    "pattern_group": int(group_id),
                    "status": "constant_single_class",
                    "num_training_segments": int(len(labeled_group)),
                    "num_classes": int(labeled_group["dbscan_label"].nunique()),
                }
            )
            for _, row in group_df.iterrows():
                predicted_segments.append(
                    {
                        "segment_id": row["segment_id"],
                        "drive_id": int(row["drive_id"]),
                        "pattern_group": int(group_id),
                        "driver": row["driver"],
                        "dbscan_label": int(row["dbscan_label"]),
                        "final_cluster_label": constant_label,
                        "confidence": 1.0,
                        "label_source": "dbscan" if int(row["dbscan_label"]) != -1 else "constant_single_class",
                        "cluster_driver_proxy": majority_driver,
                        "duration_proxy": float(row.get("duration_sec", np.nan)),
                    }
                )
            continue

        labeled_indices = feature_index.loc[labeled_group["segment_id"]].to_numpy()
        X_group = X[labeled_indices]
        y_group = labeled_group["dbscan_label"].to_numpy()
        try:
            train_idx, test_idx = train_test_split(
                np.arange(len(y_group)),
                test_size=config.classifier_test_size,
                random_state=config.random_seed,
                stratify=y_group,
            )
        except ValueError:
            metrics_rows.append(
                {
                    "pattern_group": int(group_id),
                    "status": "skipped_split_failure",
                    "num_training_segments": int(len(labeled_group)),
                    "num_classes": int(labeled_group["dbscan_label"].nunique()),
                }
            )
            continue
        classifier = MLPClassifier(
            hidden_layer_sizes=config.classifier_hidden_layer_sizes,
            max_iter=config.classifier_max_iter,
            random_state=config.random_seed,
        )
        classifier.fit(X_group[train_idx], y_group[train_idx])
        y_pred = classifier.predict(X_group[test_idx])
        metrics_rows.append(
            {
                "pattern_group": int(group_id),
                "status": "trained",
                "num_training_segments": int(len(train_idx)),
                "num_validation_segments": int(len(test_idx)),
                "num_classes": int(len(np.unique(y_group))),
                "accuracy": float(accuracy_score(y_group[test_idx], y_pred)),
                "macro_f1": float(f1_score(y_group[test_idx], y_pred, average="macro")),
            }
        )

        cm = confusion_matrix(y_group[test_idx], y_pred, labels=np.unique(y_group))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Pattern Group {int(group_id)} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        labels = [str(label) for label in np.unique(y_group)]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, int(cm[row, col]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        close_figure(fig, config.cput_root / "figures" / f"confusion_group_{int(group_id)}.png")

        group_indices = feature_index.loc[group_df["segment_id"]].to_numpy()
        X_all_group = X[group_indices]
        predicted_labels = classifier.predict(X_all_group)
        predicted_proba = classifier.predict_proba(X_all_group).max(axis=1)

        non_noise_majority_driver = (
            labeled_group[labeled_group["driver"] != "not known"]
            .groupby("dbscan_label")["driver"]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "not known")
            .to_dict()
        )

        for (_, row), predicted_label, confidence in zip(group_df.iterrows(), predicted_labels, predicted_proba):
            label_source = "dbscan" if int(row["dbscan_label"]) != -1 else "classifier"
            predicted_segments.append(
                {
                    "segment_id": row["segment_id"],
                    "drive_id": int(row["drive_id"]),
                    "pattern_group": int(group_id),
                    "driver": row["driver"],
                    "dbscan_label": int(row["dbscan_label"]),
                    "final_cluster_label": int(predicted_label),
                    "confidence": float(confidence),
                    "label_source": label_source,
                    "cluster_driver_proxy": non_noise_majority_driver.get(int(predicted_label), "not known"),
                    "duration_proxy": float(row.get("duration_sec", np.nan)),
                }
            )

    predicted_segments_df = pd.DataFrame(predicted_segments)
    metrics_df = pd.DataFrame(metrics_rows)

    if not predicted_segments_df.empty:
        for drive_id, drive_df in predicted_segments_df.groupby("drive_id", sort=True):
            cluster_scores = drive_df.assign(confidence=drive_df["confidence"].fillna(1.0)).groupby(
                "final_cluster_label"
            )["confidence"].sum()
            final_cluster_label = int(cluster_scores.idxmax())
            confidence = float(cluster_scores.max() / cluster_scores.sum()) if cluster_scores.sum() else 0.0
            driver = drive_df["driver"].iloc[0]
            cluster_driver_proxy = (
                drive_df.loc[drive_df["final_cluster_label"] == final_cluster_label, "cluster_driver_proxy"].mode()
            )
            proxy_driver = cluster_driver_proxy.iloc[0] if not cluster_driver_proxy.empty else "not known"
            trip_rows.append(
                {
                    "drive_id": int(drive_id),
                    "pattern_group": int(drive_df["pattern_group"].iloc[0]),
                    "driver": driver,
                    "predicted_cluster_label": final_cluster_label,
                    "confidence": confidence,
                    "predicted_driver_proxy": proxy_driver,
                    "proxy_driver_agreement": bool(driver != "not known" and driver == proxy_driver),
                }
            )

    predicted_trips_df = pd.DataFrame(trip_rows)
    save_dataframe(metrics_df, config.classifier_metrics_csv)
    save_dataframe(predicted_segments_df, config.pseudolabeled_segments_csv)
    save_dataframe(predicted_trips_df, config.pseudolabeled_trips_csv)
    return metrics_df, predicted_segments_df, predicted_trips_df
