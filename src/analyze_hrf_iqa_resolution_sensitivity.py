#!/usr/bin/env python3
"""Create locked per-contrast inputs, then run the generic measurement Gate."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hrf_iqa_resolution_sensitivity_v1"
GATE = ROOT / "experiments/measurement_confirmation_v1/analyze_measurement_gate.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    master = OUT / "feature_values_long.csv"
    with master.open(newline="", encoding="utf-8") as handle:
        all_values = list(csv.DictReader(handle))
    provenance = {"analysis_script_sha256": digest(Path(__file__)), "master_feature_values_sha256": digest(master), "contrasts": {}}
    for target in ("75pct", "50pct"):
        manifest = OUT / f"manifest_native_vs_{target}.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            cases = list(csv.DictReader(handle))
        selected_ids = {row["image_id"] for row in cases}
        values = [row for row in all_values if row["image_id"] in selected_ids]
        expected = len(selected_ids) * 5 * 12
        if len(selected_ids) != 36 or len(values) != expected:
            raise ValueError(f"{target}: expected 36 images / {expected} feature rows, found {len(selected_ids)} / {len(values)}")
        features = OUT / f"feature_values_native_vs_{target}.csv"
        with features.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "arm", "feature", "value"])
            writer.writeheader(); writer.writerows(values)
        output = OUT / f"gate_native_vs_{target}.json"
        subprocess.run([sys.executable, str(GATE), "--manifest", str(manifest), "--feature-values", str(features), "--out", str(output), "--bootstrap", "10000", "--seed", "20260909"], check=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        report["analysis_role"] = "controlled digital downsampling sensitivity; not an optical, cross-camera, or clinical transportability experiment"
        report["input_scope"] = {"manifest_sha256": digest(manifest), "feature_values_sha256": digest(features), "images": len(selected_ids), "feature_rows": len(values)}
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        provenance["contrasts"][target] = {"manifest_sha256": digest(manifest), "feature_values_sha256": digest(features), "gate_sha256": digest(output)}
    (OUT / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()

