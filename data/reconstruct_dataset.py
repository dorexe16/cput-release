"""Concatenate the three release chunks and extract the derived tables."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent
PARTS = [DATA / "dataset.part01", DATA / "dataset.part02", DATA / "dataset.part03"]
ARCHIVE = DATA / "dataset.zip"
TABLES = DATA / "tables"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _expected() -> dict[str, str]:
    expected = {}
    for line in (DATA / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        expected[name] = digest
    return expected


def reconstruct() -> Path:
    expected = _expected()
    chunks = []
    for path in PARTS:
        payload = path.read_bytes()
        digest = _sha256(payload)
        if digest != expected[path.name]:
            raise SystemExit(f"checksum mismatch for {path.name}")
        chunks.append(payload)
    archive = b"".join(chunks)
    if _sha256(archive) != expected["dataset.zip"]:
        raise SystemExit("checksum mismatch for the reconstructed archive")
    ARCHIVE.write_bytes(archive)
    TABLES.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as handle:
        for info in handle.infolist():
            if not info.filename.startswith("tables/") or info.is_dir():
                raise SystemExit(f"unexpected archive member: {info.filename}")
            target = TABLES / Path(info.filename).name
            target.write_bytes(handle.read(info))
    return TABLES


if __name__ == "__main__":
    print(reconstruct())
