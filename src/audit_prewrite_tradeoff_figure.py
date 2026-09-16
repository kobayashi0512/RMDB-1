#!/usr/bin/env python3
"""Audit the frozen-ledger trade-off figure without re-rendering it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts/prewriting_evidence_ledger_20260909_v1"
FIGURES = ROOT / "artifacts/figures_prewrite_v1"
BASE = FIGURES / "Figure_1_VascX_readout_tradeoffs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure_id"] == "RMDB-prewriting-figure-1-v1"
    assert manifest["source_ledger_manifest_sha256"] == sha256(LEDGER / "snapshot_manifest.json")
    assert manifest["source_results_sha256"] == sha256(LEDGER / "MASTER_RESULTS.csv")
    assert manifest["numeric_source_only"] is True
    assert "No cross-camera" in manifest["interpretation_boundary"]
    png, pdf, svg = BASE.with_suffix(".png"), BASE.with_suffix(".pdf"), BASE.with_suffix(".svg")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert "<svg" in svg.read_text(encoding="utf-8")[:1000]
    assert min(path.stat().st_size for path in (png, pdf, svg)) > 10_000
    readme = (FIGURES / "README.md").read_text(encoding="utf-8")
    assert "does not imply cross-camera validation" in readme
    result = {
        "passed": True,
        "source_ledger_snapshot": manifest["source_ledger_snapshot"],
        "source_results_sha256": manifest["source_results_sha256"],
        "formats_checked": ["png", "pdf", "svg"],
        "visual_inspection_required_before_submission": True,
        "interpretation_boundary_preserved": True,
    }
    (FIGURES / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
