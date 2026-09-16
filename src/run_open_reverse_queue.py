#!/usr/bin/env python3
"""Restartable outcome-blind queue for the forward and reverse open-data audit.

The queue restores only AFIO, RAV, Fundus-AVSeg, and the frozen checkpoints.
It never restores or reads China Fundus CIMT.  Pixel annotations are opened
only after the frozen inference products have been written.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-rmdb/bin/python"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(log: Path, message: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def marker_complete(marker: Path) -> bool:
    if not marker.is_file():
        return False
    if marker.suffix.lower() == ".csv":
        try:
            return len(pd.read_csv(marker)) > 0
        except (pd.errors.EmptyDataError, OSError):
            return False
    if marker.suffix.lower() == ".json":
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        for key in ("complete", "complete_for_measurement"):
            if key in payload:
                return bool(payload[key])
    return True


def run(log: Path, label: str, command: list[str], marker: Path) -> None:
    if marker_complete(marker):
        write(log, f"SKIP {label}: {marker}")
        return
    step_log = log.parent / f"{label.replace(':', '_')}.log"
    write(log, f"START {label}")
    with step_log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        write(log, f"FAIL {label}: exit={result.returncode}; log={step_log}")
        raise SystemExit(result.returncode)
    if not marker.is_file():
        write(log, f"FAIL {label}: missing marker {marker}")
        raise SystemExit(2)
    write(log, f"DONE {label}")


def run_array(
    log: Path,
    label: str,
    command: list[str],
    directory: Path,
    expected: int,
) -> dict[str, object]:
    marker = directory / ".complete"
    if marker.is_file():
        audit = audit_arrays(directory, expected)
        if audit["complete"]:
            write(log, f"SKIP {label}: {expected}/{expected} verified")
            return audit
    step_log = log.parent / f"{label.replace(':', '_')}.log"
    write(log, f"START {label}")
    with step_log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        write(log, f"FAIL {label}: exit={result.returncode}; log={step_log}")
        raise SystemExit(result.returncode)
    audit = audit_arrays(directory, expected)
    if not audit["complete"]:
        write(log, f"FAIL {label}: {audit}")
        raise ValueError(f"Incomplete {label}: {audit}")
    marker.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write(log, f"DONE {label}: {expected}/{expected} verified")
    return audit


def find_parent(root: Path, required: tuple[str, ...]) -> Path:
    candidates = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for candidate in candidates:
        if all((candidate / item).exists() for item in required):
            return candidate
    raise FileNotFoundError(f"No directory under {root} contains {required}")


def expand_nested_zips(root: Path) -> dict[str, object]:
    marker = root / ".nested_archives_complete.json"
    if marker_complete(marker):
        return json.loads(marker.read_text(encoding="utf-8"))
    archives = sorted(path for path in root.rglob("*.zip") if path.is_file())
    extracted = []
    for archive in archives:
        destination = archive.parent / archive.stem
        destination.mkdir(parents=True, exist_ok=True)
        resolved_destination = destination.resolve()
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if target != resolved_destination and resolved_destination not in target.parents:
                    raise ValueError(f"Unsafe nested ZIP member: {member.filename}")
            handle.extractall(destination)
            extracted.append({"archive": str(archive), "members": len(handle.infolist())})
    report = {"complete": True, "archives": extracted, "completed_at": stamp()}
    marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_manifest(images: Path, dataset: str, output: Path) -> pd.DataFrame:
    files = sorted(
        path for path in images.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    rows = [
        {
            "id": path.stem,
            "patient_id": path.stem,
            "eye": "left",
            "path": str(path.resolve()),
            "split": f"external_{dataset.lower()}",
            "dataset": dataset,
        }
        for path in files
    ]
    frame = pd.DataFrame(rows)
    if frame.empty or frame.id.duplicated().any():
        raise ValueError(f"Invalid {dataset} image manifest: n={len(frame)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def link_compatibility_tree(root: Path, anatomy: Path, logits: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, source in (("preprocessed_rgb", anatomy / "preprocessed_rgb"), ("logits", logits)):
        destination = root / name
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace compatibility path {destination}")
        os.symlink(source.resolve(), destination, target_is_directory=True)


def audit_arrays(directory: Path, expected: int) -> dict[str, object]:
    files = sorted(directory.glob("*.npy"))
    invalid = []
    for path in files:
        value = np.load(path, mmap_mode="r")
        if value.size == 0 or not np.isfinite(value).all():
            invalid.append(path.name)
    return {
        "expected": expected,
        "found": len(files),
        "invalid": invalid,
        "complete": len(files) == expected and not invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=ROOT / "release_assets")
    parser.add_argument("--restored", type=Path, default=ROOT / "restored_assets")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reverse_open")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    queue_log = args.output / "queue.log"

    run(
        queue_log,
        "restore_assets",
        [str(PYTHON), "src/reassemble_reverse_assets.py", "--assets", str(args.assets), "--output", str(args.restored)],
        args.restored / "asset_restore_metadata.json",
    )
    rav_root = find_parent(args.restored / "RAV", ("dataset/rgb", "dataset/av_masks"))
    fundus_root = find_parent(args.restored / "Fundus_AVSeg", ("images", "annotation", "testing.txt"))
    fundus_source_metadata = (
        fundus_root / "metadata.csv"
        if (fundus_root / "metadata.csv").is_file()
        else fundus_root / "metadata.xlsx"
    )
    if not fundus_source_metadata.is_file():
        raise FileNotFoundError("Fundus-AVSeg metadata.csv or metadata.xlsx is required")
    afio_root = args.restored / "AFIO"
    expand_nested_zips(afio_root)
    model_root = args.restored / "models"
    av_weights = next(model_root.rglob("third_party/vascx_weights/av_july24.pt"))
    occupancy_weights = next(model_root.rglob("baseline/artifacts/fives_unet_resnet34/best.pt"))
    udct_weights = next(model_root.rglob("baseline/artifacts/udct_av_mass_calibrated/best.pt"))

    afio_dir = args.output / "AFIO"
    run(
        queue_log, "AFIO:audit",
        [str(PYTHON), "src/audit_afio_mendeley.py", "--root", str(afio_root), "--output", str(afio_dir / "dataset_audit")],
        afio_dir / "dataset_audit/audit.json",
    )
    afio_manifest = afio_dir / "manifest.csv"
    if not marker_complete(afio_manifest):
        run(
            queue_log, "AFIO:manifest",
            [str(PYTHON), "src/prepare_afio_manifest.py", "--root", str(afio_root), "--output", str(afio_manifest)],
            afio_manifest,
        )

    manifests: dict[str, tuple[pd.DataFrame, Path, Path]] = {}
    manifests["AFIO"] = (pd.read_csv(afio_manifest), afio_manifest, afio_root)
    for name, images, dataset_root in (
        ("RAV", rav_root / "dataset/rgb", rav_root),
        ("FundusAVSeg", fundus_root / "images", fundus_root),
    ):
        directory = args.output / name
        manifest = directory / "manifest.csv"
        frame = pd.read_csv(manifest) if manifest.is_file() else write_manifest(images, name, manifest)
        manifests[name] = (frame, manifest, dataset_root)

    fundus_metadata = args.output / "FundusAVSeg/metadata.csv"
    if not fundus_metadata.is_file():
        source_frame = (
            pd.read_excel(fundus_source_metadata)
            if fundus_source_metadata.suffix.lower() == ".xlsx"
            else pd.read_csv(fundus_source_metadata)
        )
        fundus_metadata.parent.mkdir(parents=True, exist_ok=True)
        source_frame.to_csv(fundus_metadata, index=False)

    for name, (frame, manifest, _) in manifests.items():
        directory = args.output / name
        anatomy = directory / "anatomy"
        logits = directory / "vascx_logits"
        occupancy = directory / "occupancy"
        phenotypes = directory / "phenotypes"
        run(
            queue_log, f"{name}:anatomy",
            [str(PYTHON), "src/extract_vascx_anatomy.py", "--images-manifest", str(manifest), "--output", str(anatomy),
             "--av-weights", str(av_weights), "--batch-size", "4", "--inference-chunk-size", "128"],
            anatomy / "metadata.json",
        )
        logits_audit = run_array(
            queue_log, f"{name}:logits",
            [str(PYTHON), "src/export_vascx_logits_manifest.py", "--eye-manifest", str(manifest), "--anatomy", str(anatomy),
             "--weights", str(av_weights), "--output", str(logits), "--batch-size", "4", "--chunk-size", "128"],
            logits, len(frame),
        )
        occupancy_audit = run_array(
            queue_log, f"{name}:occupancy",
            [str(PYTHON), "src/export_occupancy_probabilities.py", "--images", str(anatomy / "preprocessed_rgb"),
             "--checkpoint", str(occupancy_weights), "--output", str(occupancy), "--batch-size", "8"],
            occupancy, len(frame),
        )
        run(
            queue_log, f"{name}:phenotypes",
            [str(PYTHON), "src/extract_dual_readout_phenotypes.py", "--eye-manifest", str(manifest), "--split", f"external_{name.lower()}",
             "--rgb", str(anatomy / "preprocessed_rgb"), "--disc", str(anatomy / "disc"), "--logits", str(logits),
             "--occupancy", str(occupancy), "--udct", str(udct_weights), "--output", str(phenotypes), "--require-complete"],
            phenotypes / "metadata.json",
        )

    rav_dir = args.output / "RAV"
    run(
        queue_log, "RAV:hard_metrics",
        [str(PYTHON), "src/evaluate_av_segmentation.py", "--predictions", str(rav_dir / "anatomy/artery_vein"),
         "--ground-truth", str(rav_root / "dataset/av_masks"), "--output", str(rav_dir / "hard_metrics")],
        rav_dir / "hard_metrics/summary.json",
    )
    run(
        queue_log, "RAV:uncertainty",
        [str(PYTHON), "src/analyze_av_uncertainty.py", "--logits", str(rav_dir / "vascx_logits"),
         "--ground-truth", str(rav_root / "dataset/av_masks"), "--hard-metrics", str(rav_dir / "hard_metrics/per_image_metrics.csv"),
         "--output", str(rav_dir / "uncertainty")],
        rav_dir / "uncertainty/summary.json",
    )
    rav_metadata = next((path for path in rav_root.rglob("metadata.csv") if path.is_file()), None)
    if rav_metadata is not None:
        run(
            queue_log, "RAV:domain_gaps",
            [str(PYTHON), "src/analyze_av_domain_gaps.py", "--metrics", str(rav_dir / "hard_metrics/per_image_metrics.csv"),
             "--metadata", str(rav_metadata), "--output", str(rav_dir / "domain_gaps")],
            rav_dir / "domain_gaps/summary.json",
        )

    fundus_dir = args.output / "FundusAVSeg"
    compatibility = fundus_dir / "evaluation_vascx"
    link_compatibility_tree(compatibility, fundus_dir / "anatomy", fundus_dir / "vascx_logits")
    run(
        queue_log, "FundusAVSeg:locked_evaluation",
        [str(PYTHON), "src/evaluate_fundus_avseg_external.py", "--dataset", str(fundus_root), "--vascx", str(compatibility),
         "--occupancy", str(fundus_dir / "occupancy"), "--udct", str(udct_weights), "--metadata", str(fundus_metadata),
         "--output", str(fundus_dir / "locked_evaluation"), "--bootstrap-repetitions", str(args.bootstrap_repetitions)],
        fundus_dir / "locked_evaluation/summary.json",
    )

    reverse_output = args.output / "failure_backtracking"
    run(
        queue_log, "reverse:failure_backtracking",
        [
            str(PYTHON), "src/analyze_reverse_failure_backtracking.py",
            "--per-eye", f"AFIO={args.output / 'AFIO/phenotypes/per_eye.csv'}",
            "--per-eye", f"RAV={args.output / 'RAV/phenotypes/per_eye.csv'}",
            "--per-eye", f"FundusAVSeg={args.output / 'FundusAVSeg/phenotypes/per_eye.csv'}",
            "--pixel-metrics", f"RAV={args.output / 'RAV/hard_metrics/per_image_metrics.csv'}",
            "--pixel-metrics", f"FundusAVSeg={args.output / 'FundusAVSeg/locked_evaluation/per_image_metrics.csv'}",
            "--config", str(ROOT / "configs/rmdb/reverse_audit_v1.json"),
            "--output", str(reverse_output),
            "--bootstrap-repetitions", str(args.bootstrap_repetitions),
        ],
        reverse_output / "summary.json",
    )

    report = {
        "status": "complete",
        "completed_at": stamp(),
        "cimt_fixed_validation_or_test_used": False,
        "clinical_outcomes_used": False,
        "failure_backtracking": json.loads((reverse_output / "summary.json").read_text(encoding="utf-8")),
        "datasets": {},
    }
    for name, (frame, _, _) in manifests.items():
        directory = args.output / name
        report["datasets"][name] = {
            "images": len(frame),
            "anatomy": json.loads((directory / "anatomy/metadata.json").read_text(encoding="utf-8")),
            "phenotypes": json.loads((directory / "phenotypes/metadata.json").read_text(encoding="utf-8")),
            "logits": audit_arrays(directory / "vascx_logits", len(frame)),
            "occupancy": audit_arrays(directory / "occupancy", len(frame)),
        }
    (args.output / "queue_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write(queue_log, "COMPLETE open reverse queue")


if __name__ == "__main__":
    main()
