#!/usr/bin/env python3
"""Materialize a non-overwriting, hash-verified snapshot of completed HRF experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "artifacts/hrf_iqa_readout_stress_test_v1"
RESOLUTION = ROOT / "artifacts/hrf_iqa_resolution_sensitivity_v1"
OUT = ROOT / "artifacts/frozen_external_experiments_20260909_v1"
FILES = {
    "quality": ["protocol.json", "manifest.csv", "per_image_features.csv", "feature_values_long.csv", "measurement_stress_gate.json", "run_complete.json", "audit.json", "REPORT_ZH.md"],
    "resolution": ["protocol.json", "manifest.csv", "per_image_features.csv", "feature_values_long.csv", "gate_native_vs_75pct.json", "gate_native_vs_50pct.json", "analysis_provenance.json", "run_complete.json", "audit.json", "REPORT_ZH.md"],
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen external snapshot: {OUT}")
    for directory in (QUALITY, RESOLUTION):
        audit = json.loads((directory / "audit.json").read_text())
        if not audit.get("passed"):
            raise ValueError(f"Refusing to freeze failed audit: {directory}")
    OUT.mkdir(parents=True)
    records = {}
    for label, names in FILES.items():
        source_dir = QUALITY if label == "quality" else RESOLUTION
        for name in names:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = OUT / f"{label}__{name}"
            shutil.copy2(source, target)
            if digest(source) != digest(target):
                raise ValueError(f"Copy mismatch: {source}")
            records[target.name] = {"source": str(source.relative_to(ROOT)), "sha256": digest(target), "bytes": target.stat().st_size}
    manifest = {
        "snapshot_id": "RMDB-external-HRF-experiments-20260909-v1",
        "records": records,
        "boundaries": [
            "HRF IQA quality experiment is a paired high-resolution acquisition-quality stress test, not standard test-retest or cross-camera evidence.",
            "HRF IQA resolution experiment is controlled digital downsampling, not an optical or cross-camera experiment.",
            "Neither experiment supports a clinical interchangeability claim or replaces VascX.",
        ],
        "overwrite_policy": "Never overwrite. A new experimental run requires a separately named freeze.",
    }
    (OUT / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

