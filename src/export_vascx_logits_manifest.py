#!/usr/bin/env python3
"""Restartable VascX logit export restricted to an explicit patient split."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release(runtime_device: torch.device) -> None:
    if runtime_device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif runtime_device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument(
        "--eye-manifest", type=Path,
        help="Optional generic manifest with id/patient_id/path columns.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--anatomy", type=Path, default=Path("artifacts/vascx_anatomy"))
    parser.add_argument(
        "--weights", type=Path,
        default=Path("../third_party/vascx_weights/av_july24.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/cimt_dual_readout_development/vascx_logits"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--limit-eyes", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.eye_manifest is not None:
        manifest = pd.read_csv(
            args.eye_manifest,
            usecols=["id", "patient_id", "path"],
            dtype={"id": str, "patient_id": str},
        )
        rows = [
            (row.id, row.path) for row in manifest.itertuples(index=False)
        ]
        patient_count = manifest["patient_id"].nunique()
    else:
        manifest = pd.read_csv(
            args.manifest,
            usecols=["patient_id", "split", "left_path", "right_path"],
            dtype={"patient_id": str},
        )
        patients = manifest.loc[manifest.split == args.split].copy()
        rows = []
        for row in patients.itertuples():
            rows.extend([
                (f"{row.patient_id}_L", row.left_path),
                (f"{row.patient_id}_R", row.right_path),
            ])
        patient_count = len(patients)
    if args.limit_eyes is not None:
        rows = rows[:args.limit_eyes]
    missing = [
        (identifier, path) for identifier, path in rows
        if not (args.output / f"{identifier}.npy").is_file()
    ]
    print(
        f"split={args.split} patients={patient_count} eyes={len(rows)} "
        f"existing={len(rows)-len(missing)} missing={len(missing)}",
        flush=True,
    )
    if not missing:
        return
    runtime_device = device()
    model = SegmentationEnsemble.from_torchscript(args.weights).to(runtime_device).eval()
    rgb_dir, ce_dir = args.anatomy / "preprocessed_rgb", args.anatomy / "preprocessed_ce"
    completed = len(rows) - len(missing)
    for start in range(0, len(missing), args.chunk_size):
        group = missing[start:start + args.chunk_size]
        entries = [
            {
                "id": identifier,
                "image": str(rgb_dir / f"{identifier}.png"),
                "contrast_enhanced": str(ce_dir / f"{identifier}.png"),
            }
            for identifier, _ in group
        ]
        model.predict_logits_preprocessed(
            entries,
            dest_path=args.output,
            num_workers=0,
            batch_size=args.batch_size,
        )
        completed += len(group)
        release(runtime_device)
        print(f"completed={completed}/{len(rows)}", flush=True)
    print(
        f"complete split={args.split} eyes={len(rows)} "
        f"logits={len(list(args.output.glob('*.npy')))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
