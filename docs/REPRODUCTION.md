# Reproduction

Install the packages in `requirements.txt`. From the repository root:

```text
python data/verify_dataset.py
python experiments/reproduce_association.py
python tests/test_release.py
```

`verify_dataset.py` rebuilds the zip from the three chunks, checks its checksum, extracts the tables, and checks each table checksum. `reproduce_association.py` then recomputes the associations. It exits with an error if the counts differ from the manuscript.

The embedding centroid is an internal summary of the saved representation. Co-membership is another summary of the same vehicle. Matching counts mean that co-membership did not add a correct group on this case. They do not mean that a previously published method already performed the destination-to-anchor task.

## Not supported by this run

- Held-out driver identification.
- A calibrated score or a precision-coverage curve.
- A later year scored from an earlier fit.
- A second shared vehicle.
- Preservation of an insurance comparison.
- Redistribution of GeoLife, T-Drive, or Porto traces.

## Random seeds

The ten spectral runs use the seeds stored in `spectral_runs.csv`. The released script does not draw new clusters. The embedding was fit once, in the original pipeline, and only its saved components are here.
