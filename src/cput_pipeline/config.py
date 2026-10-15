from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    input_paths: tuple[Path, Path, Path]
    output_root: Path
    random_seed: int = 500
    stop_eps: float = 0.1
    opencage_api_key: str = ""
    opencage_language: str = "en"
    destination_precision: int = 4
    geocode_precision: int = 3
    neighborhood_fallback_precision: int = 3
    destination_name_overrides: dict[str, str] = field(default_factory=dict)
    route_cell_precision: int = 4
    destination_bucket_method: str = "round"
    use_coarse_time_of_day: bool = True
    min_trips_per_destination: int = 7
    min_pattern_group_size_for_modeling: int = 60
    hour_buckets: dict[int, list[int]] = field(
        default_factory=lambda: {
            1: [7, 8, 9, 10, 19, 20, 21, 22, 23],
            2: [11, 12, 13, 14, 15, 16, 17, 18],
            3: [0, 1, 2, 3, 4, 5, 6],
        }
    )
    coarse_time_of_day_buckets: dict[str, list[int]] = field(
        default_factory=lambda: {
            "morning": [5, 6, 7, 8, 9, 10, 11],
            "afternoon": [12, 13, 14, 15, 16, 17, 18, 19, 20],
            "evening_night": [ 21, 22, 23, 0, 1, 2, 3, 4],
        }
    )
    segment_length_min: int = 5
    segment_length_max: int = 40
    min_bin_size: int = 30
    min_bin_size_is_strict_lower_bound: bool = True
    scaling_mode: str = "notebook_bucket_zscore"
    comparison_scaling_modes: tuple[str, ...] = (
        "notebook_bucket_zscore",
        "segment_zscore",
    )
    minirocket_num_kernels: int = 10_000
    pca_components: int = 3
    bagging_pca_components: int = 5
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 5
    outlier_dbscan_eps: float = 1.0
    outlier_dbscan_min_samples: int = 5
    bagging_num_runs: int = 10
    bagging_feature_fraction: float = 1.0
    bagging_max_k: int = 5
    bagging_spectral_random_seed: int = 42
    classifier_hidden_layer_sizes: tuple[int, ...] = (64, 32)
    classifier_max_iter: int = 400
    classifier_test_size: float = 0.25
    classifier_min_class_size: int = 2
    overlap_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3)

    @property
    def eda_root(self) -> Path:
        return self.output_root / "eda"

    @property
    def cput_root(self) -> Path:
        return self.output_root / "cput"

    @property
    def cput_compare_root(self) -> Path:
        return self.cput_root / "scaling_compare"

    @property
    def log_root(self) -> Path:
        return self.output_root / "logs"

    @property
    def points_clean_parquet(self) -> Path:
        return self.eda_root / "cache" / "points_clean.parquet"

    @property
    def trips_metadata_csv(self) -> Path:
        return self.eda_root / "tables" / "trips_metadata.csv"

    @property
    def trips_metadata_parquet(self) -> Path:
        return self.eda_root / "cache" / "trips_metadata.parquet"

    @property
    def points_grouped_parquet(self) -> Path:
        return self.eda_root / "cache" / "points_with_pattern_groups.parquet"

    @property
    def trips_grouped_parquet(self) -> Path:
        return self.eda_root / "cache" / "trips_with_pattern_groups.parquet"

    @property
    def pattern_groups_csv(self) -> Path:
        return self.eda_root / "tables" / "pattern_groups.csv"

    @property
    def patternability_summary_json(self) -> Path:
        return self.eda_root / "tables" / "patternability_summary.json"

    @property
    def geocode_cache_json(self) -> Path:
        return self.eda_root / "cache" / "opencage_reverse_geocode_cache.json"

    @property
    def segments_metadata_csv(self) -> Path:
        return self.eda_root / "tables" / "segments_metadata.csv"

    @property
    def segments_metadata_parquet(self) -> Path:
        return self.eda_root / "cache" / "segments_metadata.parquet"

    @property
    def segment_timeseries_joblib(self) -> Path:
        return self.eda_root / "cache" / "segment_timeseries.joblib"

    @property
    def trip_route_cells_parquet(self) -> Path:
        return self.eda_root / "cache" / "trip_route_cells.parquet"

    @property
    def route_overlap_summary_csv(self) -> Path:
        return self.eda_root / "tables" / "route_overlap_summary.csv"

    @property
    def normalized_segments_joblib(self) -> Path:
        return self.cput_root / "cache" / "normalized_segments.joblib"

    @property
    def length_buckets_joblib(self) -> Path:
        return self.cput_root / "cache" / "dict_length.joblib"

    @property
    def minirocket_features_npz(self) -> Path:
        return self.cput_root / "cache" / "minirocket_features.npz"

    @property
    def minirocket_metadata_csv(self) -> Path:
        return self.cput_root / "tables" / "minirocket_metadata.csv"

    @property
    def pca_variance_csv(self) -> Path:
        return self.cput_root / "tables" / "pca_variance.csv"

    @property
    def segments_embedding_csv(self) -> Path:
        return self.cput_root / "tables" / "segments_embedding.csv"

    @property
    def dbscan_labels_csv(self) -> Path:
        return self.cput_root / "tables" / "dbscan_labels.csv"

    @property
    def group_largest_cluster_stats_csv(self) -> Path:
        return self.cput_root / "tables" / "group_largest_cluster_stats.csv"

    @property
    def spectral_bagging_matrix_csv(self) -> Path:
        return self.cput_root / "tables" / "spectral_bagging_matrix.csv"

    @property
    def spectral_bagging_assignments_csv(self) -> Path:
        return self.cput_root / "tables" / "spectral_bagging_assignments.csv"

    @property
    def classifier_metrics_csv(self) -> Path:
        return self.cput_root / "tables" / "classifier_metrics_by_group.csv"

    @property
    def pseudolabeled_trips_csv(self) -> Path:
        return self.cput_root / "tables" / "pseudolabeled_trips.csv"

    @property
    def pseudolabeled_segments_csv(self) -> Path:
        return self.cput_root / "tables" / "pseudolabeled_segments.csv"

    def ensure_output_dirs(self) -> None:
        for path in (
            self.eda_root / "figures",
            self.eda_root / "tables",
            self.eda_root / "cache",
            self.cput_root / "figures",
            self.cput_root / "tables",
            self.cput_root / "cache",
            self.cput_compare_root / "tables",
            self.cput_compare_root / "figures",
            self.log_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> PipelineConfig:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    output_root = project_root / "outputs"
    return PipelineConfig(
        project_root=project_root,
        input_paths=(
            data_dir / "chunk_1.csv",
            data_dir / "chunk_2.csv",
            data_dir / "chunk_3.csv",
        ),
        output_root=output_root,
        opencage_api_key=os.environ.get("OPENCAGE_API_KEY", ""),
    )
