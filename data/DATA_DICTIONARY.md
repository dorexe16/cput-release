# Data dictionary

The release contains five tables. They are derived from one shared-vehicle log. They do not contain coordinates, clock times, destination names, or the original trip identifiers.

`groups.csv` has one row for each of the seven destination groups.

| Column | Meaning |
|---|---|
| `group_code` | `H` and `T` are the two labeled anchors. `B`, `O`, `N`, and `R` are the labeled targets. `U` has no driver labels. |
| `role` | `anchor_h`, `anchor_t`, `target`, or `unlabeled`. |
| `km_to_anchor_h`, `km_to_anchor_t` | Distance in kilometres from the group-mean endpoint to each anchor mean, already rounded to 0.01 km. The geography rule assigns the nearer anchor. |
| `hour_l1_to_anchor_h`, `hour_l1_to_anchor_t` | L1 distance between the group's hour-bin distribution and each anchor. The time-of-day rule assigns the smaller distance. The histograms themselves are not released. |
| `eval_n_driver_t`, `eval_n_driver_h` | Labeled-trip counts. Driver T is coded 0 and driver H is coded 1, matching the manuscript. These counts are used only to score an assignment. They are not inputs to the assignment rules. Group `U` is zero on both. |

`segments.csv` has one row for each saved segment in the seven groups (3,879 rows).

| Column | Meaning |
|---|---|
| `segment_id`, `trip_id` | Replacement identifiers. They are not the identifiers in the source log. |
| `group_code` | Destination group of the parent trip. |
| `PC1`, `PC2`, `PC3` | Saved three-component embedding. The transform was fit on all groups. |
| `dbscan_label` | Saved density label. Noise is `-1`. The published centroid uses every segment, including noise. |

`trips_kinematics.csv` has one row for each trip in the seven groups (1,033 rows).

| Column | Meaning |
|---|---|
| `trip_id` | Same replacement identifier as in `segments.csv`. |
| `group_code` | Destination group. |
| `mean_speed_kmh`, `p95_speed_kmh`, `harsh_accel_per_km`, `harsh_brake_per_km` | Trip summaries used by the speed centroid. |

`affinity.csv` is the 7 by 7 co-membership matrix. An entry is the fraction of the ten saved spectral runs in which the two groups shared a cluster.

`spectral_runs.csv` has 70 rows: ten runs times seven groups. `cluster_label` is the saved spectral label. A run names groups only when anchors `H` and `T` receive different labels.

The embedding, the affinity, and the cluster labels are outputs of a transductive fit on this vehicle. Recomputing them does not create a held-out test.
