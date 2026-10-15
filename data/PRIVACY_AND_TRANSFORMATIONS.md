# Privacy and transformations

This archive is a derived release, not the original GPS log and not a claim that the original log can be reconstructed.

## Removed before packaging

- Coordinates and endpoint means. Only the two distances from each group to the two anchors are kept, at 0.01 km.
- Clock times and dates. Only the L1 distance between hour-bin distributions is kept.
- Destination names, street addresses, and geocoder responses.
- Original trip and segment identifiers, replaced by new integers assigned in this build.
- Per-trip driver labels. Group-level labeled counts remain, because the manuscript scores groups.
- The most frequent destination, which the study excluded and which is not one of the seven groups.
- Raw points, route cells, and the kinematic series used to fit the embedding.
- Third-party corpora.

## What still reproduces

From these tables, `experiments/reproduce_association.py` recomputes:

| Rule | Correct target groups |
|---|---|
| Geography | 3/4 |
| Time of day | 0/4 |
| Speed and harsh-event centroid | 3/4 |
| Embedding centroid | 4/4 |
| Co-membership | 4/4 |

Six of the ten saved spectral runs match all four targets. Four leave the anchors in one cluster and name none. The single geography miss and the single speed miss are both group `O`.

Those equalities are the published result. Co-membership does not add a correct group beyond the embedding centroid.

## What this archive cannot reproduce

- A refit of the segment embedding or of the contextual normalization.
- Any comparison that needs the raw points, including route overlap.
- The insurance-filter check, which needs fields that were removed.
- The external trajectory corpora summarized in the Supplement.
- A second vehicle, a later year, or a held-out trip prediction. Those experiments were not part of the published claim.

## Residual risk

The embedding and the speed summaries are behavioral measurements of one vehicle. Someone who already holds the same log could try to match them. The anchor distances are coarse constraints: a person who already knows both anchor locations could narrow the other destinations. The release does not contain those locations or their names.

Small coordinate noise, a constant map shift, and hashing were not used. Those transforms do not make a repeated trajectory anonymous, and they would also change the distances.

## Authorization boundary

Research use of the source log was authorized by the authors, who also authorized release of this derived archive. The original trajectories are not included. No institutional approval identifier for the study was found in the project records.
