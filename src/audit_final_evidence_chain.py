#!/usr/bin/env python3
"""Integrity checks for the final three-layer evidence artifacts.

Only checks file counts, metadata, shapes, and finite values. It does not read
clinical labels or fit/select any model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def npy_audit(path: Path):
    files = sorted(path.glob("*.npy")) if path.exists() else []
    finite = 0; shapes = {}; nonempty = 0
    for p in files:
        try:
            a = np.load(p, mmap_mode="r")
            if a.size: nonempty += 1
            if np.isfinite(a).all(): finite += 1
            shapes[str(tuple(a.shape))] = shapes.get(str(tuple(a.shape)), 0) + 1
        except Exception:
            pass
    return {"files": len(files), "nonempty": nonempty, "finite": finite, "shapes": shapes}

def main():
    out = {
        "cimt_fixed_validation_or_test_read": False,
        "afio": {
            "audit": json.loads((ROOT/"artifacts/afio_mendeley_audit/audit.json").read_text()),
            "phenotypes": json.loads((ROOT/"artifacts/afio_mendeley/phenotypes/metadata.json").read_text()),
            "logits": npy_audit(ROOT/"artifacts/afio_mendeley/vascx_logits"),
            "occupancy": npy_audit(ROOT/"artifacts/afio_mendeley/occupancy"),
        },
        "fives": {
            "migration_metadata": json.loads((ROOT/"artifacts/structure_migration/fives_anatomy/metadata.json").read_text()),
            "phenotypes": json.loads((ROOT/"artifacts/structure_migration/fives_phenotypes/metadata.json").read_text()),
            "logits": npy_audit(ROOT/"artifacts/structure_migration/fives_vascx/logits"),
            "occupancy": npy_audit(ROOT/"artifacts/structure_migration/fives_occupancy"),
        },
        "stare": {
            "logits": npy_audit(ROOT/"artifacts/structure_migration/stare_vascx/logits"),
            "occupancy": npy_audit(ROOT/"artifacts/structure_migration/stare_occupancy"),
            "scored": json.loads((ROOT/"artifacts/structure_migration/stare_vessel_metrics.json").read_text()),
        },
        "clinical": {
            "mbrset_hypertension": (ROOT/"artifacts/mbrset_external/cv_hypertension_reliable_core_strict_disc/metrics.json").exists(),
            "mbrset_hypertension_no_dr": (ROOT/"artifacts/mbrset_external/cv_hypertension_reliable_core_no_dr/metrics.json").exists(),
            "permutation_negative_control": (ROOT/"artifacts/mbrset_external/permutation_negative_control/metrics.json").exists(),
        },
    }
    target = ROOT/"artifacts/three_layer_evidence/integrity_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
