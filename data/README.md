# Derived dataset

The three files `dataset.part01`, `dataset.part02`, and `dataset.part03` are a split of one zip archive. Each part is about 35 KB. GitHub blocks files larger than 100 MB; these parts are far below that limit, so Git LFS is not required. The split is the distribution format for this release, not a response to a size limit.

The measured archive size is 107,409 bytes. The part sizes are 35,803, 35,803, and 35,803 bytes.

Reconstruct and check:

```text
python data/reconstruct_dataset.py
python data/verify_dataset.py
```

Run both commands from the repository root. Reconstruction writes `data/tables/` and must match `data/table_checksums.sha256` byte for byte. `data/checksums.sha256` covers the three parts and the assembled zip.

The tables are documented in `DATA_DICTIONARY.md`. What was removed, and which results survive, is documented in `PRIVACY_AND_TRANSFORMATIONS.md`.
