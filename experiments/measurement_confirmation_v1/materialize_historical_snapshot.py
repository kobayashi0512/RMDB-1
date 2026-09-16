#!/usr/bin/env python3
"""Copy the canonical historical tables into a hash-verified, non-overwriting snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default=Path(__file__).with_name("historical_result_freeze.json"))
    parser.add_argument("--out-dir", default=Path(__file__).with_name("frozen_historical_evidence_v1"))
    args = parser.parse_args()
    freeze_path = Path(args.freeze).resolve()
    destination = Path(args.out_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite historical snapshot: {destination}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True)
    copied: dict[str, dict[str, str | int]] = {}
    for source_name, source in freeze["canonical_sources"].items():
        origin = (freeze_path.parent / source["path"]).resolve()
        if not origin.is_file():
            raise FileNotFoundError(f"Missing canonical source: {origin}")
        if digest(origin) != source["sha256"]:
            raise ValueError(f"Canonical source changed before snapshot: {source_name}")
        filename = f"{source_name}{origin.suffix}"
        copied_path = destination / filename
        shutil.copy2(origin, copied_path)
        if digest(copied_path) != source["sha256"]:
            raise ValueError(f"Copy verification failed for {source_name}")
        copied[source_name] = {"filename": filename, "sha256": source["sha256"], "bytes": copied_path.stat().st_size}
    shutil.copy2(freeze_path, destination / "historical_result_freeze.json")
    snapshot = {
        "snapshot_id": "RMDB-historical-evidence-snapshot-20260909-v1",
        "freeze_file_sha256": digest(freeze_path),
        "files": copied,
        "interpretation": freeze["interpretation_boundary"],
        "overwrite_policy": "Never overwrite this directory. Create a separately named freeze only after an explicit protocol revision.",
    }
    (destination / "snapshot_manifest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"historical snapshot created: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

