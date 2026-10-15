"""Check chunk checksums and exact equality of the extracted tables."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))

from reconstruct_dataset import reconstruct  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tables = reconstruct()
    expected = {}
    for line in (DATA / "table_checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        expected[name] = digest
    found = {path.name: _sha256(path) for path in sorted(tables.glob("*.csv"))}
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        changed = sorted(name for name in expected.keys() & found.keys() if expected[name] != found[name])
        raise SystemExit(f"table mismatch missing={missing} extra={extra} changed={changed}")
    print("verified", len(found), "tables")


if __name__ == "__main__":
    main()
