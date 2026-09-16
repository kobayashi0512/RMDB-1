#!/usr/bin/env python3
"""Integrity audit for the locked HRF IQA digital-resolution sensitivity outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hrf_iqa_resolution_sensitivity_v1"
FEATURES = {"vessel_mass", "artery_mass", "vein_mass", "av_ratio", "vessel_density", "skeleton_density", "branch_density", "endpoint_density", "component_count", "fractal_dimension", "occupancy_entropy", "identity_entropy"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    with (OUT / "manifest.csv").open(newline="", encoding="utf-8") as handle: manifest = list(csv.DictReader(handle))
    with (OUT / "per_image_features.csv").open(newline="", encoding="utf-8") as handle: values = list(csv.DictReader(handle))
    with (OUT / "feature_values_long.csv").open(newline="", encoding="utf-8") as handle: long_values = list(csv.DictReader(handle))
    checks = {
        "exactly_54_inputs": len(manifest) == 54 and len({row["image_id"] for row in manifest}) == 54,
        "exactly_18_source_eyes": len({row["patient_id"] for row in manifest}) == 18,
        "three_locked_levels": {row["resolution_level"] for row in manifest} == {"native", "75pct", "50pct"},
        "native_high_resolution": all(max(int(row["native_width"]), int(row["native_height"])) >= 3000 for row in manifest if row["resolution_level"] == "native"),
        "all_manifest_hashes_match": all(digest(ROOT / row["original_path"]) == row["sha256"] for row in manifest),
        "five_arms_per_input": len(values) == 270 and all(sum(row["image_id"] == image for row in values) == 5 for image in {row["image_id"] for row in manifest}),
        "all_values_finite": all(set(row).issuperset(FEATURES) and all(math.isfinite(float(row[f])) for f in FEATURES) for row in values),
        "long_values_complete": len(long_values) == 3240 and all(math.isfinite(float(row["value"])) for row in long_values),
        "runner_hash_matches_protocol": digest(ROOT / "src/run_hrf_iqa_resolution_sensitivity.py") == protocol["source_hashes"]["runner"],
        "mass_hash_matches_protocol": digest(ROOT / "models/baseline/artifacts/udct_av_mass_calibrated/best.pt") == protocol["source_hashes"]["mass_weight"],
        "occupancy_hash_matches_protocol": digest(ROOT / "models/baseline/artifacts/fives_unet_resnet34/best.pt") == protocol["source_hashes"]["occupancy_weight"],
    }
    for target in ("75pct", "50pct"):
        gate = json.loads((OUT / f"gate_native_vs_{target}.json").read_text())
        scope = gate.get("input_scope", {})
        checks[f"{target}_gate_scope"] = len(gate.get("within_camera", [])) == 3 and not gate.get("cross_camera") and scope.get("images") == 36 and scope.get("feature_rows") == 2160
        checks[f"{target}_synthetic_boundary"] = gate.get("analysis_role", "").startswith("controlled digital downsampling sensitivity")
    failures = [name for name, passed in checks.items() if not passed]
    report = {"audit_id": "RMDB-HRF-IQA-digital-resolution-sensitivity-audit-v1", "passed": not failures, "checks": checks, "failures": failures, "scope": "integrity only; a passing audit does not turn digital downsampling into cross-camera evidence"}
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

