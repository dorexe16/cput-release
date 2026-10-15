from __future__ import annotations

import itertools
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io_utils import save_dataframe, save_json

try:
    from opencage.geocoder import OpenCageGeocode
except ImportError as exc:  # pragma: no cover - import guard
    OpenCageGeocode = None
    OPENCAGE_IMPORT_ERROR = exc
else:
    OPENCAGE_IMPORT_ERROR = None

CANONICAL_COLUMNS = [
    "orig_time",
    "drive_id",
    "longitude",
    "latitude",
    "speed",
    "acceleration_est_1",
    "angular_acc",
    "road_speed",
    "vehicle_state",
    "driver",
    "hour",
]
NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "speed",
    "acceleration_est_1",
    "angular_acc",
    "road_speed",
    "vehicle_state",
    "drive_id",
]
HELPER_COLUMNS = ["lat_change", "lon_change", "change", "keep", "date"]
NEIGHBORHOOD_COMPONENT_KEYS = [
    "neighbourhood",
    "suburb",
]


def round_or_unknown(value: float | int | str | None, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals == 0:
        return str(int(round(numeric)))
    return f"{round(numeric, decimals):.{decimals}f}"


def normalize_driver(value: object) -> str:
    if pd.isna(value):
        return "not known"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "not known", "unknown"}:
        return "not known"
    return text


def assign_hour_bucket(hour: int, hour_buckets: dict[int, list[int]]) -> int:
    for bucket, hours in hour_buckets.items():
        if hour in hours:
            return bucket
    raise ValueError(f"Hour {hour} does not fall into any configured bucket.")


def assign_coarse_time_of_day(hour: int, coarse_buckets: dict[str, list[int]]) -> str:
    for bucket_name, hours in coarse_buckets.items():
        if hour in hours:
            return bucket_name
    raise ValueError(f"Hour {hour} does not fall into any configured coarse bucket.")


def haversine_km(
    lon1: pd.Series | np.ndarray | float,
    lat1: pd.Series | np.ndarray | float,
    lon2: pd.Series | np.ndarray | float,
    lat2: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    lon1_arr = np.radians(lon1)
    lat1_arr = np.radians(lat1)
    lon2_arr = np.radians(lon2)
    lat2_arr = np.radians(lat2)
    dlon = lon2_arr - lon1_arr
    dlat = lat2_arr - lat1_arr
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_arr) * np.cos(lat2_arr) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def create_destination_key(lat: float, lon: float, precision: int) -> str:
    return f"{round(lat, precision):.{precision}f}_{round(lon, precision):.{precision}f}"


def resolve_destination_name(destination_key: str, raw_name: str, config: PipelineConfig) -> str:
    override = config.destination_name_overrides.get(str(destination_key))
    if override:
        return override
    return str(raw_name)


def create_cell_key(lat: float, lon: float, precision: int) -> str:
    return create_destination_key(lat, lon, precision)


def trip_distance_km(df_trip: pd.DataFrame) -> float:
    shifted = df_trip[["longitude", "latitude"]].shift()
    mask = shifted.notna().all(axis=1)
    if not mask.any():
        return 0.0
    distances = haversine_km(
        shifted.loc[mask, "longitude"],
        shifted.loc[mask, "latitude"],
        df_trip.loc[mask, "longitude"],
        df_trip.loc[mask, "latitude"],
    )
    return float(np.asarray(distances).sum())


def enforce_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            if column == "orig_time":
                df[column] = pd.NaT
            elif column in {"drive_id", "vehicle_state"}:
                df[column] = pd.Series(dtype="float64")
            elif column == "driver":
                df[column] = "not known"
            elif column == "hour":
                df[column] = pd.Series(dtype="int64")
            else:
                df[column] = np.nan
    return df


def load_and_clean_points(config: PipelineConfig) -> pd.DataFrame:
    config.ensure_output_dirs()
    frames: list[pd.DataFrame] = []
    cleaning_log: dict[str, int | list[str]] = {"files_loaded": [], "negative_speed_rows": 0}
    for input_path in config.input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        df = pd.read_csv(input_path, parse_dates=["orig_time"])
        cleaning_log["files_loaded"].append(str(input_path))
        error_columns = [column for column in df.columns if column.startswith("ERROR")]
        drop_columns = error_columns + [column for column in HELPER_COLUMNS if column in df.columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)
        frames.append(df)

    points = pd.concat(frames, ignore_index=True)
    points = enforce_columns(points)

    for column in NUMERIC_COLUMNS:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points["orig_time"] = pd.to_datetime(points["orig_time"], errors="coerce")
    points["driver"] = points["driver"].map(normalize_driver)
    points["hour"] = points["orig_time"].dt.hour.astype("Int64")

    if "vehicle_state" in points.columns:
        before = len(points)
        points = points[points["vehicle_state"].isin([1, 2]) | points["vehicle_state"].isna()].copy()
        cleaning_log["vehicle_state_filtered_rows"] = int(before - len(points))

    before = len(points)
    points = points.drop_duplicates(subset=["drive_id", "orig_time"], keep="first").copy()
    cleaning_log["duplicate_drive_time_rows"] = int(before - len(points))

    before = len(points)
    coord_mask = points["latitude"].between(-90, 90) & points["longitude"].between(-180, 180)
    points = points[coord_mask].copy()
    cleaning_log["invalid_coordinate_rows"] = int(before - len(points))

    negative_speed_mask = points["speed"] < 0
    cleaning_log["negative_speed_rows"] = int(negative_speed_mask.fillna(False).sum())
    points.loc[negative_speed_mask, "speed"] = np.nan

    before = len(points)
    points = points.dropna(subset=["orig_time", "drive_id", "longitude", "latitude", "speed"]).copy()
    cleaning_log["rows_dropped_missing_essentials"] = int(before - len(points))

    points["drive_id"] = points["drive_id"].astype("int64")
    points["hour"] = points["hour"].astype("int64")
    points = points.sort_values(["drive_id", "orig_time"]).reset_index(drop=True)

    def serialize_config_value(value: object) -> object:
        if isinstance(value, tuple):
            return [serialize_config_value(item) for item in value]
        if isinstance(value, list):
            return [serialize_config_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): serialize_config_value(item) for key, item in value.items()}
        if hasattr(value, "as_posix"):
            return str(value)
        return value

    config_payload = {key: serialize_config_value(value) for key, value in asdict(config).items()}

    save_dataframe(points, config.points_clean_parquet)
    save_json(cleaning_log, config.log_root / "cleaning_summary.json")
    save_json(config_payload, config.log_root / "pipeline_config.json")
    return points


def summarize_trip_driver(values: pd.Series) -> str:
    unique_values = sorted(set(values.dropna().map(normalize_driver)))
    if not unique_values:
        return "not known"
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def summarize_trip_road_speed(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    mode = clean.mode()
    if not mode.empty:
        return float(mode.iloc[0])
    return float(clean.iloc[0])


def build_trips_metadata(points: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    trip_rows: list[dict[str, object]] = []
    mixed_driver_count = 0
    grouped = points.groupby("drive_id", sort=False)
    for drive_id, trip in grouped:
        trip = trip.sort_values("orig_time").reset_index(drop=True)
        driver_label = summarize_trip_driver(trip["driver"])
        if driver_label == "mixed":
            mixed_driver_count += 1
        trip_rows.append(
            {
                "drive_id": int(drive_id),
                "start_time": trip["orig_time"].iloc[0],
                "end_time": trip["orig_time"].iloc[-1],
                "duration_sec": float((trip["orig_time"].iloc[-1] - trip["orig_time"].iloc[0]).total_seconds()),
                "start_latitude": float(trip["latitude"].iloc[0]),
                "start_longitude": float(trip["longitude"].iloc[0]),
                "end_latitude": float(trip["latitude"].iloc[-1]),
                "end_longitude": float(trip["longitude"].iloc[-1]),
                "num_points": int(len(trip)),
                "distance_km": trip_distance_km(trip),
                "mean_speed": float(trip["speed"].mean()),
                "median_speed": float(trip["speed"].median()),
                "max_speed": float(trip["speed"].max()),
                "missing_speed_pct": float(trip["speed"].isna().mean() * 100),
                "missing_acceleration_pct": float(trip["acceleration_est_1"].isna().mean() * 100),
                "missing_angular_acc_pct": float(trip["angular_acc"].isna().mean() * 100),
                "road_speed": summarize_trip_road_speed(trip["road_speed"]),
                "driver": driver_label,
                "start_hour": int(trip["hour"].iloc[0]),
                "end_hour": int(trip["hour"].iloc[-1]),
            }
        )
    trips_df = pd.DataFrame(trip_rows).sort_values("drive_id").reset_index(drop=True)
    save_dataframe(trips_df, config.trips_metadata_csv)
    save_dataframe(trips_df, config.trips_metadata_parquet)
    save_json({"mixed_driver_trip_count": int(mixed_driver_count)}, config.log_root / "trip_summary.json")
    return trips_df


def extract_neighborhood_name(result: dict[str, object], fallback: str) -> str:
    components = result.get("components", {}) if isinstance(result, dict) else {}
    if not isinstance(components, dict):
        components = {}
    for key in NEIGHBORHOOD_COMPONENT_KEYS:
        value = components.get(key)
        if value:
            return str(value)
    road = components.get("road")
    if road:
        return str(road)
    return fallback


def _format_opencage_coordinate(value: float) -> str:
    rounded = round(float(value), 3)
    text = f"{rounded:.3f}"
    return f"{text}01"


def _tag_destination_candidates(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drives_df = candidate_df.copy()
    drives_df["driver"] = 0
    driver_neighborhoods: set[str] = set()
    first_index = None
    for index, row in drives_df.iterrows():
        if first_index is None:
            first_index = index
            drives_df.at[index, "driver"] = 1
            driver_neighborhoods.add(str(row["destination_neighborhood"]))
            continue
        if str(row["destination_neighborhood"]) in driver_neighborhoods:
            drives_df.at[index, "driver"] = 1
    return (
        drives_df[drives_df["driver"] == 1].copy(),
        drives_df[drives_df["driver"] == 0].copy(),
    )


def _collect_notebook_style_drive_ids(
    trip_lookup_df: pd.DataFrame,
    drives_group: pd.DataFrame,
    *,
    with_returns: bool = True,
) -> list[int]:
    matching_drive_ids: list[int] = []
    seen_drive_ids: set[int] = set()
    for row_df in trip_lookup_df.itertuples(index=False):
        for row_group in drives_group.itertuples(index=False):
            if (
                round(float(row_df.end_latitude_round), 3) == float(row_group.end_latitude)
                and round(float(row_df.end_longitude_round), 3) == float(row_group.end_longitude)
            ):
                drive_id = int(row_df.drive_id)
                if drive_id not in seen_drive_ids:
                    matching_drive_ids.append(drive_id)
                    seen_drive_ids.add(drive_id)
                break

    if with_returns:
        matched_rows = trip_lookup_df[trip_lookup_df["drive_id"].isin(matching_drive_ids)]
        for matched_row in matched_rows.itertuples(index=False):
            for candidate_row in trip_lookup_df.itertuples(index=False):
                if (
                    round(float(matched_row.end_latitude_round), 3)
                    == round(float(candidate_row.start_latitude_round), 3)
                    and round(float(matched_row.end_longitude_round), 3)
                    == round(float(candidate_row.start_longitude_round), 3)
                    and matched_row.trip_date == candidate_row.trip_date
                ):
                    drive_id = int(candidate_row.drive_id)
                    if drive_id not in seen_drive_ids:
                        matching_drive_ids.append(drive_id)
                        seen_drive_ids.add(drive_id)
                    break
    return matching_drive_ids


def reverse_geocode_destination_keys(
    unique_destinations: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, dict[str, object]]:
    cache_path = config.geocode_cache_json
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    if OpenCageGeocode is None:
        raise ImportError(
            "OpenCageGeocode is unavailable. Install `opencage` before building pattern groups."
        ) from OPENCAGE_IMPORT_ERROR

    geocoder = OpenCageGeocode(config.opencage_api_key)
    cache_updated = False
    geocode_disabled_reason: str | None = None

    for row in unique_destinations.itertuples(index=False):
        destination_key = str(row.destination_key)
        coarse_fallback = (
            "approx_"
            + create_destination_key(
                float(row.end_latitude),
                float(row.end_longitude),
                config.neighborhood_fallback_precision,
            )
        )
        if (
            destination_key in cache
            and cache[destination_key].get("neighborhood")
            and (
                "geocode_error" not in cache[destination_key]
                or cache[destination_key].get("fallback_precision") == config.neighborhood_fallback_precision
            )
        ):
            continue
        if geocode_disabled_reason is not None:
            cache[destination_key] = {
                "destination_key": destination_key,
                "destination_latitude": float(row.end_latitude),
                "destination_longitude": float(row.end_longitude),
                "neighborhood": coarse_fallback,
                "formatted": coarse_fallback,
                "fallback_precision": config.neighborhood_fallback_precision,
                "geocode_error": geocode_disabled_reason,
            }
            cache_updated = True
            continue
        try:
            results = geocoder.reverse_geocode(
                _format_opencage_coordinate(float(row.end_latitude)),
                _format_opencage_coordinate(float(row.end_longitude)),
                no_annotations="1",
                language=config.opencage_language,
                limit=1,
            )
            first_result = results[0] if results else {}
            cache[destination_key] = {
                "destination_key": destination_key,
                "destination_latitude": float(row.end_latitude),
                "destination_longitude": float(row.end_longitude),
                "neighborhood": extract_neighborhood_name(first_result, destination_key),
                "formatted": first_result.get("formatted", destination_key) if first_result else destination_key,
            }
        except Exception as exc:
            geocode_error = type(exc).__name__
            if geocode_error in {"NotAuthorizedError", "ForbiddenError"}:
                geocode_disabled_reason = geocode_error
            # Preserve a deterministic fallback so the pipeline remains runnable when the API is unavailable.
            cache[destination_key] = {
                "destination_key": destination_key,
                "destination_latitude": float(row.end_latitude),
                "destination_longitude": float(row.end_longitude),
                "neighborhood": coarse_fallback,
                "formatted": coarse_fallback,
                "fallback_precision": config.neighborhood_fallback_precision,
                "geocode_error": geocode_error,
            }
        cache_updated = True

    if cache_updated or not cache_path.exists():
        save_json(cache, cache_path)
    return cache


def build_pattern_groups(
    trips_df: pd.DataFrame,
    points: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trips_df = trips_df.copy()
    trips_df["trip_date"] = pd.to_datetime(trips_df["start_time"]).dt.date
    grouping_precision = 3
    trips_df["start_latitude_round"] = trips_df["start_latitude"].round(grouping_precision)
    trips_df["start_longitude_round"] = trips_df["start_longitude"].round(grouping_precision)
    trips_df["end_latitude_round"] = trips_df["end_latitude"].round(grouping_precision)
    trips_df["end_longitude_round"] = trips_df["end_longitude"].round(grouping_precision)
    trips_df["start_key"] = trips_df.apply(
        lambda row: create_destination_key(
            row["start_latitude_round"],
            row["start_longitude_round"],
            grouping_precision,
        ),
        axis=1,
    )
    trips_df["destination_key"] = trips_df.apply(
        lambda row: create_destination_key(
            row["end_latitude_round"],
            row["end_longitude_round"],
            grouping_precision,
        ),
        axis=1,
    )
    trips_df["hour_bucket"] = trips_df["start_hour"].map(
        lambda value: assign_hour_bucket(int(value), config.hour_buckets)
    )
    trips_df["coarse_time_of_day"] = trips_df["start_hour"].map(
        lambda value: assign_coarse_time_of_day(int(value), config.coarse_time_of_day_buckets)
    )

    coordinate_groups_df = (
        trips_df.groupby(
            ["end_latitude_round", "end_longitude_round", "destination_key"], as_index=False
        )
        .size()
        .rename(columns={"size": "num_drives"})
        .sort_values("num_drives", ascending=False)
        .reset_index(drop=True)
    )
    home_row = coordinate_groups_df.iloc[0]
    home_destination = str(home_row["destination_key"])
    home_trip_count = int(home_row["num_drives"])
    home_trip_share = float(home_trip_count / max(len(trips_df), 1))

    frequent_coordinate_groups = coordinate_groups_df[
        coordinate_groups_df["num_drives"] > config.min_trips_per_destination
    ].copy()
    frequent_coordinate_groups = frequent_coordinate_groups.rename(
        columns={
            "end_latitude_round": "end_latitude",
            "end_longitude_round": "end_longitude",
        }
    )
    geocode_lookup = reverse_geocode_destination_keys(frequent_coordinate_groups, config)
    frequent_coordinate_groups["destination_neighborhood"] = frequent_coordinate_groups["destination_key"].map(
        lambda key: geocode_lookup.get(key, {}).get("neighborhood", key)
    )
    frequent_coordinate_groups["destination_neighborhood"] = frequent_coordinate_groups.apply(
        lambda row: resolve_destination_name(
            str(row["destination_key"]),
            str(row["destination_neighborhood"]),
            config,
        ),
        axis=1,
    )
    frequent_coordinate_groups["destination_formatted"] = frequent_coordinate_groups["destination_key"].map(
        lambda key: geocode_lookup.get(key, {}).get("formatted", key)
    )
    home_destination_neighborhood = geocode_lookup.get(home_destination, {}).get(
        "neighborhood",
        home_destination,
    )
    filtered_coordinate_groups = frequent_coordinate_groups[
        frequent_coordinate_groups["destination_neighborhood"] != home_destination_neighborhood
    ].copy()

    trips_df["destination_neighborhood"] = trips_df["destination_key"].map(
        lambda key: resolve_destination_name(
            str(key),
            str(geocode_lookup.get(key, {}).get("neighborhood", key)),
            config,
        )
    )
    trips_df["destination_formatted"] = trips_df["destination_key"].map(
        lambda key: geocode_lookup.get(key, {}).get("formatted")
    )
    trips_df["pattern_group_key"] = pd.NA
    trips_df["pattern_group"] = pd.Series(pd.array([pd.NA] * len(trips_df), dtype="Int64"))
    trips_df["pattern_group_size"] = pd.Series(pd.array([pd.NA] * len(trips_df), dtype="Int64"))
    trips_df["eligible_for_modeling"] = False
    trips_df["return_trip_drive_id"] = pd.Series(pd.array([pd.NA] * len(trips_df), dtype="Int64"))
    trips_df["has_return_trip"] = False

    ordered_trips = trips_df.sort_values(["trip_date", "start_time", "end_time", "drive_id"]).reset_index(drop=True)
    next_trip_lookup: dict[int, dict[str, object]] = {}
    for _, date_group in ordered_trips.groupby("trip_date", sort=False):
        future_start_lookup: dict[str, int] = {}
        for row in reversed(list(date_group.itertuples(index=False))):
            current_drive_id = int(row.drive_id)
            destination_key = str(row.destination_key)
            matched_drive_id = future_start_lookup.get(destination_key)
            if matched_drive_id is not None:
                next_trip_lookup[current_drive_id] = {
                    "drive_id": matched_drive_id,
                    "trip_date": row.trip_date,
                    "start_key": destination_key,
                }
            future_start_lookup[str(row.start_key)] = current_drive_id

    neighborhood_groups = (
        filtered_coordinate_groups.groupby("destination_neighborhood", sort=False)
        .agg(
            coordinate_group_count=("destination_key", "nunique"),
            coordinate_trip_count=("num_drives", "sum"),
            destination_formatted=("destination_formatted", "first"),
        )
        .reset_index()
    )

    pattern_group_rows: list[dict[str, object]] = []
    for neighborhood_group in neighborhood_groups.itertuples(index=False):
        coordinate_keys = filtered_coordinate_groups.loc[
            filtered_coordinate_groups["destination_neighborhood"] == neighborhood_group.destination_neighborhood,
            "destination_key",
        ].tolist()
        outbound_trips = trips_df[trips_df["destination_key"].isin(coordinate_keys)].copy()
        if outbound_trips.empty:
            continue

        group_drive_ids = set(outbound_trips["drive_id"].astype(int).tolist())
        return_trip_ids: set[int] = set()
        return_map: dict[int, int] = {}

        for outbound_trip in outbound_trips.itertuples(index=False):
            next_trip = next_trip_lookup.get(int(outbound_trip.drive_id))
            if next_trip is None:
                continue
            if (
                next_trip["trip_date"] == outbound_trip.trip_date
                and next_trip["start_key"] == outbound_trip.destination_key
            ):
                return_trip_id = int(next_trip["drive_id"])
                return_trip_ids.add(return_trip_id)
                return_map[int(outbound_trip.drive_id)] = return_trip_id

        group_drive_ids.update(return_trip_ids)
        group_size = len(group_drive_ids)
        if group_size < config.min_pattern_group_size_for_modeling:
            continue

        group_id = len(pattern_group_rows) + 1
        group_mask = trips_df["drive_id"].isin(group_drive_ids)

        trips_df.loc[group_mask, "pattern_group_key"] = neighborhood_group.destination_neighborhood
        trips_df.loc[group_mask, "pattern_group"] = group_id
        trips_df.loc[group_mask, "pattern_group_size"] = group_size
        trips_df.loc[group_mask, "eligible_for_modeling"] = True
        trips_df.loc[trips_df["drive_id"].isin(return_trip_ids), "has_return_trip"] = True
        for outbound_drive_id, return_drive_id in return_map.items():
            trips_df.loc[trips_df["drive_id"] == outbound_drive_id, "return_trip_drive_id"] = return_drive_id
            trips_df.loc[trips_df["drive_id"] == outbound_drive_id, "has_return_trip"] = True

        group_trips = trips_df[group_mask].copy()
        representative_lat = float(group_trips["end_latitude"].median())
        representative_lon = float(group_trips["end_longitude"].median())
        hour_bucket_counts = (
            group_trips["hour_bucket"].value_counts().sort_index().rename(index=lambda value: f"bucket_{value}")
        )
        coarse_counts = (
            group_trips["coarse_time_of_day"]
            .value_counts()
            .rename(index=lambda value: f"coarse_{value}")
            .to_dict()
        )
        distinct_driver_count = int(group_trips["driver"].nunique())
        pattern_group_rows.append(
            {
                "pattern_group": group_id,
                "pattern_group_key": neighborhood_group.destination_neighborhood,
                "destination_key": outbound_trips["destination_key"].mode().iloc[0],
                "destination_neighborhood": neighborhood_group.destination_neighborhood,
                "destination_formatted": neighborhood_group.destination_formatted,
                "coordinate_group_count": int(neighborhood_group.coordinate_group_count),
                "coordinate_trip_count": int(neighborhood_group.coordinate_trip_count),
                "trip_count": int(group_size),
                "outbound_trip_count": int(len(outbound_trips)),
                "return_trip_count": int(len(return_trip_ids)),
                "return_trip_share": float(len(return_trip_ids) / group_size if group_size else 0.0),
                "representative_end_latitude": representative_lat,
                "representative_end_longitude": representative_lon,
                "distinct_driver_count": distinct_driver_count,
                "eligible_for_modeling": True,
                **hour_bucket_counts.to_dict(),
                **coarse_counts,
            }
        )

    points_grouped = points.copy()
    pattern_mapping = trips_df.set_index("drive_id")["pattern_group"]
    points_grouped["pattern_group"] = points_grouped["drive_id"].map(pattern_mapping).astype("Int64")

    pattern_groups_df = pd.DataFrame(pattern_group_rows)
    if not pattern_groups_df.empty:
        pattern_groups_df = pattern_groups_df.sort_values("trip_count", ascending=False).reset_index(drop=True)
        old_group_ids = pattern_groups_df["pattern_group"].tolist()
        remap = {
            old_group_id: new_group_id
            for new_group_id, old_group_id in enumerate(old_group_ids, start=1)
        }
        if remap:
            trips_df["pattern_group"] = trips_df["pattern_group"].map(remap).astype("Int64")
            points_grouped["pattern_group"] = points_grouped["pattern_group"].map(remap).astype("Int64")
        pattern_groups_df["pattern_group"] = range(1, len(pattern_groups_df) + 1)
    save_dataframe(trips_df, config.trips_grouped_parquet)
    save_dataframe(points_grouped, config.points_grouped_parquet)
    save_dataframe(pattern_groups_df, config.pattern_groups_csv)

    groupable_mask = trips_df["pattern_group"].notna()
    summary = {
        "total_trips": int(len(trips_df)),
        "groupable_trip_count": int(groupable_mask.sum()),
        "groupable_trip_pct": float(groupable_mask.mean() * 100 if len(trips_df) else 0.0),
        "groupable_drive_pct": float(groupable_mask.mean() * 100 if len(trips_df) else 0.0),
        "num_pattern_groups": int(pattern_groups_df["pattern_group"].nunique() if not pattern_groups_df.empty else 0),
        "home_destination_key": home_destination,
        "home_destination_neighborhood": home_destination_neighborhood,
        "home_trip_count": home_trip_count,
        "home_trip_share": home_trip_share,
        "home_destination_rule": "most_frequent_destination_neighborhood_from_top_rounded_destination_coordinate",
        "destination_count_distribution": coordinate_groups_df.set_index("destination_key")["num_drives"].to_dict(),
        "pattern_group_count_distribution": {
            row["pattern_group_key"]: int(row["trip_count"]) for row in pattern_group_rows
        },
        "return_trip_count": int(trips_df["has_return_trip"].fillna(False).sum()),
        "return_trip_pct": float(trips_df["has_return_trip"].fillna(False).mean() * 100 if len(trips_df) else 0.0),
    }
    save_json(summary, config.patternability_summary_json)
    return trips_df, points_grouped, pattern_groups_df


def deduplicate_coordinates(df_trip: pd.DataFrame) -> pd.DataFrame:
    df_trip = df_trip.copy()
    lat_change = df_trip["latitude"].ne(df_trip["latitude"].shift())
    lon_change = df_trip["longitude"].ne(df_trip["longitude"].shift())
    change = lat_change | lon_change
    keep = change.shift(-1, fill_value=True)
    return df_trip.loc[keep].reset_index(drop=True)


def split_dataframe(df_trip: pd.DataFrame, stop_eps: float) -> list[pd.DataFrame]:
    df_trip = df_trip.reset_index(drop=True)
    if df_trip.empty:
        return []
    segment_boundaries = [0]
    speeds = df_trip["speed"].to_numpy()
    for index in range(len(df_trip) - 1):
        current_is_stop = np.isfinite(speeds[index]) and speeds[index] <= stop_eps
        next_is_move = np.isfinite(speeds[index + 1]) and speeds[index + 1] > stop_eps
        if current_is_stop and next_is_move:
            segment_boundaries.append(index + 1)
    segment_boundaries.append(len(df_trip))

    segments: list[pd.DataFrame] = []
    for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
        segment = df_trip.iloc[start:end].copy().reset_index(drop=True)
        if not segment.empty:
            segments.append(segment)
    return segments


def build_segments_metadata(points_grouped: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    segment_rows: list[dict[str, object]] = []
    trip_cell_rows: list[dict[str, object]] = []

    grouped = points_grouped.groupby("drive_id", sort=False)
    for drive_id, trip in grouped:
        trip = trip.sort_values("orig_time").reset_index(drop=True)
        trip_dedup = deduplicate_coordinates(trip)
        pattern_group = trip["pattern_group"].dropna().iloc[0] if trip["pattern_group"].notna().any() else pd.NA
        driver = summarize_trip_driver(trip["driver"])
        cell_keys = sorted(
            {
                create_cell_key(lat, lon, config.route_cell_precision)
                for lat, lon in zip(trip_dedup["latitude"], trip_dedup["longitude"])
            }
        )
        trip_cell_rows.append(
            {
                "drive_id": int(drive_id),
                "driver": driver,
                "pattern_group": pattern_group,
                "route_cells": "|".join(cell_keys),
                "route_cell_count": int(len(cell_keys)),
            }
        )

        segments = split_dataframe(trip_dedup, config.stop_eps)
        for segment_index, segment in enumerate(segments, start=1):
            road_speed = summarize_trip_road_speed(segment["road_speed"])
            start_hour = int(segment["orig_time"].iloc[0].hour)
            segment_rows.append(
                {
                    "segment_id": f"{int(drive_id)}_{segment_index}",
                    "drive_id": int(drive_id),
                    "pattern_group": pattern_group,
                    "driver": driver,
                    "length": int(len(segment)),
                    "duration_sec": float(
                        (segment["orig_time"].iloc[-1] - segment["orig_time"].iloc[0]).total_seconds()
                    ),
                    "distance_km": trip_distance_km(segment),
                    "road_speed": road_speed,
                    "road_speed_bucket": round_or_unknown(road_speed, decimals=0),
                    "start_latitude": float(segment["latitude"].iloc[0]),
                    "start_longitude": float(segment["longitude"].iloc[0]),
                    "end_latitude": float(segment["latitude"].iloc[-1]),
                    "end_longitude": float(segment["longitude"].iloc[-1]),
                    "start_hour": start_hour,
                    "hour_bucket": assign_hour_bucket(start_hour, config.hour_buckets),
                    "num_points": int(len(segment)),
                    "eligible_length": bool(config.segment_length_min < len(segment) < config.segment_length_max),
                }
            )

    segments_df = pd.DataFrame(segment_rows)
    if not segments_df.empty:
        segments_df = segments_df.sort_values(["drive_id", "segment_id"]).reset_index(drop=True)
    trip_route_cells_df = pd.DataFrame(trip_cell_rows).sort_values("drive_id").reset_index(drop=True)
    save_dataframe(segments_df, config.segments_metadata_csv)
    save_dataframe(segments_df, config.segments_metadata_parquet)
    save_dataframe(trip_route_cells_df, config.trip_route_cells_parquet)
    return segments_df


def jaccard_overlap(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def build_route_overlap_summary(
    trips_df: pd.DataFrame,
    trip_route_cells_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = trips_df.merge(trip_route_cells_df, on=["drive_id", "driver", "pattern_group"], how="left")
    merged = merged[merged["driver"] != "not known"].copy()
    merged["route_cell_set"] = merged["route_cells"].fillna("").map(
        lambda value: set(filter(None, str(value).split("|")))
    )

    pair_rows: list[dict[str, object]] = []
    for left_idx, right_idx in itertools.combinations(range(len(merged)), 2):
        left = merged.iloc[left_idx]
        right = merged.iloc[right_idx]
        if left["driver"] == right["driver"]:
            continue
        pair_rows.append(
            {
                "drive_id_left": int(left["drive_id"]),
                "drive_id_right": int(right["drive_id"]),
                "driver_left": left["driver"],
                "driver_right": right["driver"],
                "driver_pair": " vs ".join(sorted([left["driver"], right["driver"]])),
                "same_pattern_group": bool(
                    pd.notna(left["pattern_group"])
                    and pd.notna(right["pattern_group"])
                    and int(left["pattern_group"]) == int(right["pattern_group"])
                ),
                "jaccard_overlap": jaccard_overlap(left["route_cell_set"], right["route_cell_set"]),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    summary_rows: list[dict[str, object]] = []
    if pair_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "scope",
                "driver_pair",
                "pair_count",
                "mean_overlap",
                "median_overlap",
                "share_overlap_gt_0.1",
                "share_overlap_gt_0.2",
                "share_overlap_gt_0.3",
            ]
        )
        save_dataframe(summary_df, config.route_overlap_summary_csv)
        return pair_df, summary_df

    def append_summary(scope: str, df_scope: pd.DataFrame) -> None:
        if df_scope.empty:
            return
        summary_rows.append(
            {
                "scope": scope,
                "driver_pair": "all",
                "pair_count": int(len(df_scope)),
                "mean_overlap": float(df_scope["jaccard_overlap"].mean()),
                "median_overlap": float(df_scope["jaccard_overlap"].median()),
                **{
                    f"share_overlap_gt_{threshold:.1f}": float((df_scope["jaccard_overlap"] > threshold).mean())
                    for threshold in config.overlap_thresholds
                },
            }
        )
        for driver_pair, driver_df in df_scope.groupby("driver_pair"):
            summary_rows.append(
                {
                    "scope": scope,
                    "driver_pair": driver_pair,
                    "pair_count": int(len(driver_df)),
                    "mean_overlap": float(driver_df["jaccard_overlap"].mean()),
                    "median_overlap": float(driver_df["jaccard_overlap"].median()),
                    **{
                        f"share_overlap_gt_{threshold:.1f}": float(
                            (driver_df["jaccard_overlap"] > threshold).mean()
                        )
                        for threshold in config.overlap_thresholds
                    },
                }
            )

    append_summary("all_cross_driver_pairs", pair_df)
    append_summary("same_pattern_group_cross_driver_pairs", pair_df[pair_df["same_pattern_group"]].copy())
    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, config.route_overlap_summary_csv)
    save_dataframe(pair_df, config.eda_root / "tables" / "route_overlap_pairs.csv")
    return pair_df, summary_df
