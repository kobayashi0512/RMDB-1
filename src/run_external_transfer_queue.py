#!/usr/bin/env python3
"""Restartable queue for label-free high-resolution anatomy transfer audits.

The queue deliberately stops before any clinical claim: RAV and Fundus-AVSeg do
not provide systemic-disease labels in this workspace.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY_VASCX = ROOT / ".venv-vascx/bin/python"
PY_OCC = ROOT / ".venv/bin/python"
LOG = ROOT / "artifacts/external_transfer/queue.log"


def write(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(label: str, command: list[str], marker: Path, log_path: Path) -> None:
    if marker.is_file():
        write(f"SKIP {label}: marker exists {marker}")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write(f"START {label}: {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        write(f"FAIL {label}: exit={result.returncode}; inspect {log_path}")
        raise SystemExit(result.returncode)
    if not marker.is_file():
        write(f"FAIL {label}: command succeeded but marker missing {marker}")
        raise SystemExit(2)
    write(f"DONE {label}")


def run_array_step(label: str, command: list[str], output: Path, expected: int, log_path: Path) -> None:
    existing = list(output.glob("*.npy")) if output.is_dir() else []
    if len(existing) == expected:
        write(f"SKIP {label}: {len(existing)}/{expected} npy already present")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write(f"START {label}: {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        write(f"FAIL {label}: exit={result.returncode}; inspect {log_path}")
        raise SystemExit(result.returncode)
    if len(list(output.glob("*.npy"))) != expected:
        write(f"FAIL {label}: expected {expected} npy, found {len(list(output.glob('*.npy')))}")
        raise SystemExit(2)
    (output / ".complete").write_text("complete\n", encoding="utf-8")
    write(f"DONE {label}")


def main() -> None:
    base = ROOT / "artifacts/external_transfer"
    datasets = {
        "RAV": base / "RAV",
        "FundusAVSeg": base / "FundusAVSeg",
    }
    for name, directory in datasets.items():
        manifest = directory / "manifest.csv"
        anatomy = directory / "anatomy"
        logits = directory / "vascx_logits"
        occupancy = directory / "occupancy"
        phenotypes = directory / "phenotypes"
        run(
            f"{name}:anatomy",
            [str(PY_VASCX), "scripts/extract_vascx_anatomy.py", "--images-manifest", str(manifest),
             "--output", str(anatomy), "--batch-size", "4", "--inference-chunk-size", "128"],
            anatomy / "metadata.json", directory / "anatomy.log",
        )
        expected = len(pd.read_csv(manifest))
        run_array_step(
            f"{name}:logits",
            [str(PY_VASCX), "scripts/export_vascx_logits_manifest.py", "--eye-manifest", str(manifest),
             "--anatomy", str(anatomy), "--output", str(logits), "--batch-size", "4", "--chunk-size", "128"],
            logits, expected, directory / "logits.log",
        )
        run_array_step(
            f"{name}:occupancy",
            [str(PY_OCC), "scripts/export_occupancy_probabilities.py", "--images", str(anatomy / "preprocessed_rgb"),
             "--checkpoint", str(ROOT / "artifacts/fives_unet_resnet34/best.pt"), "--output", str(occupancy),
             "--batch-size", "8"],
            occupancy, expected, directory / "occupancy.log",
        )
        run(
            f"{name}:phenotypes",
            [str(PY_VASCX), "scripts/extract_dual_readout_phenotypes.py", "--eye-manifest", str(manifest),
             "--split", "external_anatomy_transfer", "--rgb", str(anatomy / "preprocessed_rgb"),
             "--disc", str(anatomy / "disc"), "--logits", str(logits), "--occupancy", str(occupancy),
             "--output", str(phenotypes), "--require-complete"],
            phenotypes / "metadata.json", directory / "phenotypes.log",
        )

    summary = {"status": "complete", "clinical_external_confirmation": "pending_cohort_access", "datasets": {}}
    for name, directory in datasets.items():
        anatomy = json.loads((directory / "anatomy/metadata.json").read_text())
        phenotypes = json.loads((directory / "phenotypes/metadata.json").read_text())
        summary["datasets"][name] = {
            "anatomy": anatomy,
            "phenotypes": phenotypes,
            "logits_npy": len(list((directory / "vascx_logits").glob("*.npy"))),
            "occupancy_npy": len(list((directory / "occupancy").glob("*.npy"))),
        }
    (base / "queue_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write(f"COMPLETE queue_summary={base / 'queue_summary.json'}")


if __name__ == "__main__":
    main()
