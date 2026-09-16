#!/usr/bin/env python3
"""Independent integrity audit for the locked HRF IQA readout stress test."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hrf_iqa_readout_stress_test_v1"
SOURCE = ROOT / "data/raw/HRF_IQA_20260909"
EXPECTED_ARMS = {"raw_vascx", "mass", "joint", "smooth", "cleanup"}
EXPECTED_FEATURES = {
    "vessel_mass", "artery_mass", "vein_mass", "av_ratio", "vessel_density", "skeleton_density",
    "branch_density", "endpoint_density", "component_count", "fractal_dimension", "occupancy_entropy", "identity_entropy",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    with (OUT / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    with (OUT / "per_image_features.csv").open(newline="", encoding="utf-8") as handle:
        features = list(csv.DictReader(handle))
    with (OUT / "feature_values_long.csv").open(newline="", encoding="utf-8") as handle:
        long_values = list(csv.DictReader(handle))
    gate = json.loads((OUT / "measurement_stress_gate.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: dict[str, bool] = {}
    checks["archive_hash_matches_protocol"] = digest(SOURCE / "allQuality.zip") == protocol["source"]["archive_sha256"]
    checks["exactly_36_manifest_images"] = len(manifest) == 36 and len({row["image_id"] for row in manifest}) == 36
    checks["exactly_18_pair_keys"] = len({row["patient_id"] for row in manifest}) == 18 and all(sum(item["patient_id"] == row["patient_id"] for item in manifest) == 2 for row in manifest)
    checks["source_roles_complete"] = {(row["patient_id"], row["pair_role"]) for row in manifest} == {(f"HRF_IQA_PAIR_{number:02d}", role) for number in range(1, 19) for role in ("bad", "good")}
    checks["native_images_high_resolution"] = all(max(int(row["native_width"]), int(row["native_height"])) >= 3000 for row in manifest)
    checks["no_prior_project_use"] = {row["prior_project_use"] for row in manifest} == {"false"}
    checks["five_arms_per_image"] = len(features) == 180 and all({row["arm"] for row in features if row["image_id"] == image_id} == EXPECTED_ARMS for image_id in {row["image_id"] for row in manifest})
    checks["all_features_finite"] = all(set(row).issuperset(EXPECTED_FEATURES) and all(math.isfinite(float(row[feature])) for feature in EXPECTED_FEATURES) for row in features)
    checks["long_format_cardinality"] = len(long_values) == 36 * 5 * 12 and not any(not math.isfinite(float(row["value"])) for row in long_values)
    checks["three_primary_gate_rows"] = len(gate.get("within_camera", [])) == 3 and not gate.get("cross_camera")
    checks["stress_test_boundary_recorded"] = gate.get("analysis_role") == "paired acquisition-quality stress test, not same-condition retest reliability"
    for name, expected_hash in protocol["source_hashes"].items():
        locations = {
            "VascX_weight": ROOT / "models/third_party/vascx_weights/av_july24.pt",
            "mass_weight": ROOT / "models/baseline/artifacts/udct_av_mass_calibrated/best.pt",
            "occupancy_weight": ROOT / "models/baseline/artifacts/fives_unet_resnet34/best.pt",
            "runner": ROOT / "src/run_hrf_iqa_readout_stress_test.py",
            "feature_operator": ROOT / "src/extract_mbrset_repeatability_features.py",
        }
        checks[f"source_hash_{name}"] = digest(locations[name]) == expected_hash
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    result = {
        "audit_id": "RMDB-HRF-IQA-readout-stress-test-audit-v1",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "scope": "integrity audit only; does not convert the paired quality perturbation into a standard test-retest or cross-camera study",
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

