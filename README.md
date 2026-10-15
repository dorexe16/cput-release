# Linking recurrent destinations from two labeled anchors

This repository reproduces the group-association table from the accompanying manuscript. It does not contain the original GPS.

The scientific question is how to associate unlabeled recurrent destinations with two labeled destinations on one shared vehicle, using sparse kinematic measurements. The manuscript's method builds destination groups, normalizes segment kinematics by operating context, embeds the segments, and associates the other groups with the two anchors. MiniROCKET, principal components, and spectral clustering are existing algorithms. The claim that is specific to this work is the destination-to-anchor task and the way those pieces are arranged. On this vehicle, co-membership of repeated clusterings matches the same four target groups as the embedding centroid and does not add a further correct group.

## Reproduce the published table

From this directory, with NumPy and pandas installed:

```text
python data/verify_dataset.py
python experiments/reproduce_association.py
```

The expected counts are geography 3/4, time of day 0/4, speed centroid 3/4, embedding centroid 4/4, and co-membership 4/4. Six of ten saved spectral runs match all four targets. Four name none. These counts are group decisions on the same vehicle used to build the embedding. They are not held-out trip accuracy.

The repository is [https://github.com/dorexe16/cput-release](https://github.com/dorexe16/cput-release).

## Research implementation

`src/` is the processing implementation: segmentation, destination groups, context-bin scaling, MiniROCKET, principal components, and spectral co-membership. A full refit reads GPS chunks from `data/chunk_1.csv`, `data/chunk_2.csv`, and `data/chunk_3.csv` relative to the project root. Those chunks are not in this repository. Neighborhood lookup reads `OPENCAGE_API_KEY` from the environment. Do not commit a key.

The verified result in the paper is the association script above. It does not refit the embedding.

## License

`LICENSE` is the MIT license for this release.

## What is not in this repository

- The original trajectories, coordinates, timestamps, and destination names.
- A refit of the embedding. The released script reads the saved components.
- Third-party trajectory corpora. See `docs/THIRD_PARTY_DATASETS.md`.
- The route-grid comparison, the insurance-filter check, and any second vehicle.
- A geocoding credential. Do not add one to these files.

The earlier project history that contains those files is not this repository. This history starts from the derived release.

## Layout

| Path | Role |
|---|---|
| `data/` | Three archive chunks, checksums, reconstruction, and the data dictionary. |
| `experiments/reproduce_association.py` | Canonical entry point for the published table. |
| `tests/test_release.py` | Runs verification and checks the published counts. |
| `docs/REPRODUCTION.md` | Commands and the results that are not supported. |

## License and citation

Cite the accompanying manuscript. An earlier preprint by the same authors, SSRN 10.2139/ssrn.5517261, is an older version of this article, not an independent method. The code is released under the MIT license in `LICENSE`.
