#!/usr/bin/env python3
"""Restartable VascX extraction of A/V, optic-disc, and fovea anatomy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_fundusprep.cli import _run_preprocessing
from rtnls_inference.ensembles.ensemble_heatmap_regression import HeatmapRegressionEnsemble
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_rows(patients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in patients.itertuples():
        rows.extend([
            {"id": f"{row.patient_id}_L", "patient_id": str(row.patient_id), "eye": "left", "path": row.left_path},
            {"id": f"{row.patient_id}_R", "patient_id": str(row.patient_id), "eye": "right", "path": row.right_path},
        ])
    return pd.DataFrame(rows)


def load_rows(
    patients_path: Path, images_manifest: Path | None
) -> tuple[pd.DataFrame, int]:
    if images_manifest is not None:
        rows = pd.read_csv(
            images_manifest,
            usecols=["id", "patient_id", "eye", "path"],
            dtype={"id": str, "patient_id": str},
        )
        if rows["id"].duplicated().any():
            raise ValueError("The image manifest contains duplicate IDs.")
        return rows, rows["patient_id"].nunique()
    patients = pd.read_csv(patients_path, dtype={"patient_id": str})
    return make_rows(patients), len(patients)


def entries(rows: pd.DataFrame, rgb_dir: Path, ce_dir: Path) -> list[dict]:
    return [
        {
            "id": row.id,
            "image": str(rgb_dir / f"{row.id}.png"),
            "contrast_enhanced": str(ce_dir / f"{row.id}.png"),
        }
        for row in rows.itertuples()
    ]


def chunks(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        yield frame.iloc[start:start + size]


def release_accelerator_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, default=Path("artifacts/bimfo_v1_1_cache/patients.csv"))
    parser.add_argument(
        "--images-manifest", type=Path,
        help="Optional generic patient-eye-view manifest with id/patient_id/eye/path.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/vascx_anatomy"))
    parser.add_argument("--av-weights", type=Path, default=Path("../third_party/vascx_weights/av_july24.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inference-chunk-size", type=int, default=256)
    parser.add_argument("--preprocess-jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Patient limit for smoke tests only.")
    args = parser.parse_args()

    rows, patient_count = load_rows(args.patients, args.images_manifest)
    if args.limit is not None:
        selected = rows["patient_id"].drop_duplicates().head(args.limit)
        rows = rows.loc[rows["patient_id"].isin(set(selected))].copy()
        patient_count = int(rows["patient_id"].nunique())
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, ce_dir = args.output / "preprocessed_rgb", args.output / "preprocessed_ce"
    av_dir, disc_dir = args.output / "artery_vein", args.output / "disc"
    for directory in (rgb_dir, ce_dir, av_dir, disc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    missing_preprocess = rows[
        ~rows.id.map(lambda identifier: (rgb_dir / f"{identifier}.png").is_file() and (ce_dir / f"{identifier}.png").is_file())
    ]
    if len(missing_preprocess):
        _run_preprocessing(
            files=[Path(path) for path in missing_preprocess.path],
            ids=missing_preprocess.id.tolist(),
            rgb_path=rgb_dir, ce_path=ce_dir,
            bounds_path=args.output / "bounds_latest_batch.csv",
            n_jobs=args.preprocess_jobs,
        )

    device = run_device()
    missing_av = rows[~rows.id.map(lambda identifier: (av_dir / f"{identifier}.png").is_file())]
    if len(missing_av):
        av_model = SegmentationEnsemble.from_torchscript(args.av_weights).to(device).eval()
        for chunk in chunks(missing_av, args.inference_chunk_size):
            av_model.predict_preprocessed(
                entries(chunk, rgb_dir, ce_dir), dest_path=av_dir,
                num_workers=0, batch_size=args.batch_size,
            )
            release_accelerator_cache(device)
        del av_model

    missing_disc = rows[~rows.id.map(lambda identifier: (disc_dir / f"{identifier}.png").is_file())]
    if len(missing_disc):
        disc_model = SegmentationEnsemble.from_huggingface(
            "Eyened/vascx:disc/disc_july24.pt"
        ).to(device).eval()
        for chunk in chunks(missing_disc, args.inference_chunk_size):
            disc_model.predict_preprocessed(
                entries(chunk, rgb_dir, ce_dir), dest_path=disc_dir,
                num_workers=0, batch_size=args.batch_size,
            )
            release_accelerator_cache(device)
        del disc_model

    fovea_path = args.output / "fovea.csv"
    existing_fovea = pd.read_csv(fovea_path, dtype={"id": str}) if fovea_path.exists() else pd.DataFrame(columns=["id", "x_fovea", "y_fovea"])
    missing_fovea = rows[~rows.id.isin(set(existing_fovea.id))]
    if len(missing_fovea):
        fovea_model = HeatmapRegressionEnsemble.from_huggingface(
            "Eyened/vascx:fovea/fovea_july24.pt"
        ).to(device).eval()
        predictions = []
        for chunk in chunks(missing_fovea, args.inference_chunk_size):
            predictions.append(fovea_model.predict_preprocessed(
                entries(chunk, rgb_dir, ce_dir),
                num_workers=0, batch_size=args.batch_size,
            ))
            release_accelerator_cache(device)
        predicted = pd.concat(predictions)
        predicted = predicted.rename(columns={"x0": "x_fovea", "y0": "y_fovea"})
        predicted.index.name = "id"
        existing_fovea = pd.concat([existing_fovea, predicted.reset_index()], ignore_index=True)
        existing_fovea.drop_duplicates("id", keep="last").to_csv(fovea_path, index=False)

    audit = []
    fovea = pd.read_csv(fovea_path, dtype={"id": str}).set_index("id")
    for row in rows.itertuples():
        av = np.asarray(Image.open(av_dir / f"{row.id}.png"))
        disc = np.asarray(Image.open(disc_dir / f"{row.id}.png"))
        yy, xx = np.where(disc > 0)
        counts = np.bincount(av.ravel(), minlength=4)
        audit.append({
            "id": row.id, "patient_id": row.patient_id, "eye": row.eye,
            "artery_fraction": float(counts[1] / av.size),
            "vein_fraction": float(counts[2] / av.size),
            "crossing_fraction": float(counts[3] / av.size),
            "disc_fraction": float(len(xx) / disc.size),
            "disc_detected": bool(len(xx)),
            "disc_x": float(xx.mean()) if len(xx) else np.nan,
            "disc_y": float(yy.mean()) if len(yy) else np.nan,
            "fovea_x": float(fovea.loc[row.id, "x_fovea"]),
            "fovea_y": float(fovea.loc[row.id, "y_fovea"]),
        })
    audit_frame = pd.DataFrame(audit)
    # A small number of macula-centred photographs do not contain enough optic
    # disc pixels for the segmentation threshold to yield a mask. Preserve that
    # failure explicitly, while supplying a reproducible anatomical fallback
    # centre from the robust eye-specific disc-to-fovea displacement. Downstream
    # disc-mask phenotypes must still exclude rows where disc_detected is false.
    detected = audit_frame[audit_frame["disc_detected"]].copy()
    detected["disc_dx"] = detected["disc_x"] - detected["fovea_x"]
    detected["disc_dy"] = detected["disc_y"] - detected["fovea_y"]
    displacement = detected.groupby("eye")[["disc_dx", "disc_dy"]].median()
    missing_disc = ~audit_frame["disc_detected"]
    for eye, indices in audit_frame[missing_disc].groupby("eye").groups.items():
        audit_frame.loc[indices, "disc_x"] = (
            audit_frame.loc[indices, "fovea_x"] + displacement.loc[eye, "disc_dx"]
        ).clip(0, 1023)
        audit_frame.loc[indices, "disc_y"] = (
            audit_frame.loc[indices, "fovea_y"] + displacement.loc[eye, "disc_dy"]
        ).clip(0, 1023)
    audit_frame["disc_center_imputed"] = ~audit_frame["disc_detected"]
    audit_frame.to_csv(args.output / "anatomy_audit.csv", index=False)
    metadata = {
        "patients": patient_count, "eyes": len(rows), "device": str(device),
        "av_weights": str(args.av_weights), "av_sha256": sha256(args.av_weights),
        "disc_model": "Eyened/vascx:disc/disc_july24.pt",
        "fovea_model": "Eyened/vascx:fovea/fovea_july24.pt",
        "preprocessed_resolution": 1024, "smoke_test_limit": args.limit,
        "disc_segmentation_failures": int(missing_disc.sum()),
        "disc_center_fallback": "eye-specific median disc-to-fovea displacement",
        "complete": bool(
            len(audit_frame) == len(rows)
            and audit_frame.drop(columns=["id", "patient_id", "eye"]).notna().all().all()
        ),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
