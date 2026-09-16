#!/usr/bin/env python3
"""Verify the one allowed source table set for pre-confirmation historical evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default=Path(__file__).with_name("historical_result_freeze.json"))
    args = parser.parse_args()
    freeze_path = Path(args.freeze).resolve()
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, source in payload["canonical_sources"].items():
        path = (freeze_path.parent / source["path"]).resolve()
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        found = digest(path)
        if found != source["sha256"]:
            failures.append(f"{name}: SHA-256 differs ({found})")
        if "data_rows" in source:
            with path.open(newline="", encoding="utf-8") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            if count != source["data_rows"]:
                failures.append(f"{name}: expected {source['data_rows']} data rows, found {count}")
    if failures:
        print("HISTORICAL FREEZE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    print(f"HISTORICAL FREEZE VERIFIED: {payload['freeze_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

