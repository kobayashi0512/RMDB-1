#!/usr/bin/env python3
"""Verify the materialized historical snapshot without reading mutable source tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", default=Path(__file__).with_name("frozen_historical_evidence_v1"))
    args = parser.parse_args()
    root = Path(args.snapshot_dir).resolve()
    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, item in manifest["files"].items():
        path = root / item["filename"]
        if not path.is_file():
            failures.append(f"{name}: missing {path.name}")
        elif digest(path) != item["sha256"]:
            failures.append(f"{name}: SHA-256 differs")
    if failures:
        print("SNAPSHOT VERIFICATION FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    print(f"SNAPSHOT VERIFIED: {manifest['snapshot_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

