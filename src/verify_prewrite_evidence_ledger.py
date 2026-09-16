#!/usr/bin/env python3
"""Independently verify the immutable pre-writing evidence ledger."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts/prewriting_evidence_ledger_20260909_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (LEDGER / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    snapshot = json.loads((LEDGER / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert snapshot["snapshot_id"] == "RMDB-prewriting-evidence-ledger-20260909-v1"
    for name, record in snapshot["records"].items():
        path = LEDGER / name
        assert path.is_file(), f"Missing frozen ledger file: {name}"
        assert path.stat().st_size == record["bytes"], f"Size mismatch: {name}"
        assert sha256(path) == record["sha256"], f"Hash mismatch: {name}"

    audit = json.loads((LEDGER / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    results = read_csv("MASTER_RESULTS.csv")
    splits = read_csv("DATA_SPLIT_REGISTRY.csv")
    claims = read_csv("CLAIM_EVIDENCE_MATRIX.csv")
    assert len(results) == 53
    assert len(splits) == 5
    assert len(claims) == 5
    assert {row["experiment_id"] for row in results} == {
        "historical_mbrset_test_repeatability", "historical_fundus_avseg_reference",
        "hrf_quality_stress", "hrf_resolution_native_vs_75pct", "hrf_resolution_native_vs_50pct",
    }
    assert all(row["eligible_for_cross_camera_claim"].startswith("no") for row in splits)
    assert any("not independent" in row["independence"] for row in splits if row["data_role_id"] == "hrf_iqa_digital_resolution_sensitivity")
    boundary = (LEDGER / "VASCX_INCREMENT_BOUNDARY.md").read_text(encoding="utf-8")
    assert "不得使用" in boundary and "Resolves" in boundary, "Boundary must explicitly prohibit overclaim language"
    assert all("resolves" not in row["allowed_wording"].lower() for row in claims)
    assert all("replaces vascx" not in row["allowed_wording"].lower() for row in claims)
    print("LEDGER VERIFIED")
    print(f"snapshot_id={snapshot['snapshot_id']}")
    print(f"records={len(snapshot['records'])} results={len(results)} splits={len(splits)} claims={len(claims)}")


if __name__ == "__main__":
    main()
