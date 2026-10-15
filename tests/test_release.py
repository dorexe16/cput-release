"""Confirm reconstruction and the published group counts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "experiments"))

from verify_dataset import main as verify_main  # noqa: E402
from reproduce_association import reproduce  # noqa: E402


def main() -> None:
    verify_main()
    results = reproduce()
    if results["embedding_centroid"] != results["co_membership"]:
        raise SystemExit("co-membership and the embedding centroid diverged")
    print("release checks passed")


if __name__ == "__main__":
    main()
