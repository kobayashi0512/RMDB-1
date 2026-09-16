#!/usr/bin/env python3
"""Create a non-overwriting hash manifest for the audited pre-writing figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "artifacts/figures_prewrite_v1"
DESTINATION = FIGURES / "snapshot_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure freeze: {DESTINATION}")
    audit = json.loads((FIGURES / "audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("Cannot freeze a failed figure audit")
    records = {}
    for path in sorted(FIGURES.iterdir()):
        if path.is_file() and path.name != DESTINATION.name:
            records[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    result = {
        "snapshot_id": "RMDB-prewriting-figure-1-20260909-v1",
        "source_ledger_snapshot": audit["source_ledger_snapshot"],
        "overwrite_policy": "Never overwrite. Render a separately versioned figure when evidence changes.",
        "records": records,
    }
    DESTINATION.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
