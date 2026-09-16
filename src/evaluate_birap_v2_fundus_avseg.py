#!/usr/bin/env python3
"""Frozen evaluation of a BiRAP V2 checkpoint on Fundus-AVSeg.

The checkpoint is selected only with FIVES and RAV development data.  This
script never fits thresholds or parameters to the Fundus-AVSeg labels; it
reports the prespecified argmax and probability metrics once per image.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation

from train_birap_v2 import BiRAPV2, RetinalDevelopmentDataset, case_metrics, runtime_device
from train_udct_av_pilot import softmax_np


def bootstrap_delta(frame: pd.DataFrame, metric: str, challenger: str, baseline: str,
                    repetitions: int, seed: int) -> dict:
    pivot = frame.pivot(index="id", columns="model", values=metric).dropna()
    delta = pivot[challenger].to_numpy() - pivot[baseline].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(repetitions, len(delta)))
    means = delta[idx].mean(1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
        "images_improved": int(((delta < 0) if metric in ('brier', 'av_ratio_log_error') else (delta > 0)).sum()),
        "images_total": int(len(delta)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--vascx", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--occupancy-checkpoint", type=Path, required=True)
    p.add_argument("--mass-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--route-center", type=float, default=.35,
                   help="Fixed value selected on FIVES/RAV development only.")
    p.add_argument("--route-temperature", type=float, default=.15)
    p.add_argument("--bootstrap-repetitions", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--legacy-unregistered", action="store_true",
                   help="Forensic reproduction ONLY: omit the saved crop transform. These scores are invalid for scientific comparison.")
    args = p.parse_args()
    root = args.dataset
    images = sorted((root / "images").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No Fundus-AVSeg images below {root}")
    mask_root = root / "masks"
    if not mask_root.is_dir():
        # The CC BY 4.0 release uses the repository's original directory name.
        mask_root = root / "annotation"
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.legacy_unregistered:
        bounds_frame = pd.read_csv(args.vascx / 'bounds_latest_batch.csv').set_index('id')
        aligned_root = args.output / 'aligned_reference'
        aligned_root.mkdir(exist_ok=True)
    for image in images:
        mask = mask_root / image.name
        logits = args.vascx / "logits" / f"{image.stem}.npy"
        if not mask.is_file() or not logits.is_file():
            raise FileNotFoundError(f"Missing paired mask/logits for {image.stem}")
        input_image = image
        if not args.legacy_unregistered:
            raw_rgb = np.asarray(Image.open(image).convert('RGB'))
            saved_rgb_path = args.vascx / 'preprocessed_rgb' / image.name
            saved_rgb = np.asarray(Image.open(saved_rgb_path).convert('RGB'))
            if saved_rgb.shape[0] != saved_rgb.shape[1]:
                raise ValueError('Expected canonical square preprocessing geometry')
            bounds_dict = ast.literal_eval(re.sub(
                r'np\.float64\(([^()]*)\)', r'\1', bounds_frame.loc[image.stem, 'bounds']))
            transform, _ = CFIBounds.from_dict(raw_rgb, bounds_dict).crop(saved_rgb.shape[0])
            if not np.array_equal(transform.warp(raw_rgb), saved_rgb):
                raise ValueError(f'Cannot exactly reproduce cached crop for {image.stem}')
            annotation = np.asarray(Image.open(mask).convert('RGB'))
            if annotation.shape[:2] != raw_rgb.shape[:2]:
                raise ValueError(f'Raw image and annotation differ in geometry: {image.stem}')
            aligned_mask = aligned_root / image.name
            Image.fromarray(transform.warp(annotation, mode=Interpolation.NEAREST)).save(aligned_mask)
            input_image, mask = saved_rgb_path, aligned_mask
        records.append({"id": image.stem, "source": "rav", "image": input_image,
                        "mask": mask, "logits": logits, "label_convention": "fundus_avseg"})
    loader = DataLoader(RetinalDevelopmentDataset(records, False, args.size, (384, 512, 640, 768)),
                        batch_size=1, num_workers=0)
    device = runtime_device()
    model = BiRAPV2(args.occupancy_checkpoint, args.mass_checkpoint,
                    route_center=args.route_center, route_temperature=args.route_temperature,
                    residual_logit=True, residual_scale=.1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            identifier = batch.pop("id")[0]
            batch.pop("source")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, route_enabled=True, identity_enabled=True)
            target = batch["target"][0].cpu().numpy()
            vessel = batch["vessel"][0, 0].cpu().numpy()
            variants = {
                "vascx": out["base_probability"][0].cpu().numpy(),
                "birap_v2_joint": out["joint"][0].cpu().numpy(),
                "birap_v2_mass": out["mass"][0].cpu().numpy(),
                "birap_v2_final": out["final"][0].cpu().numpy(),
            }
            for name, probability in variants.items():
                rows.append({
                    "id": identifier, "model": name,
                    "route": float(out["route"][0].mean().cpu()),
                    **case_metrics(probability, target, vessel),
                })
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "per_image_metrics.csv", index=False)
    summary = {"images": int(frame.id.nunique()), "models": {},
               "label_convention": "Fundus-AVSeg: green crossing; white uncertain",
               "evaluation_version": "aligned_official_labels_20260905"}
    for name, group in frame.groupby("model"):
        summary["models"][name] = {k: float(group[k].mean()) for k in (
            "vessel_dice", "artery_dice", "vein_dice", "mean_av_dice",
            "type_accuracy", "av_ratio_log_error", "brier") if k in group}
    summary["paired_bootstrap"] = {}
    for challenger in ("birap_v2_joint", "birap_v2_mass", "birap_v2_final"):
        for metric in ("vessel_dice", "mean_av_dice", "brier", "av_ratio_log_error"):
            summary["paired_bootstrap"][f"{challenger}_minus_vascx_{metric}"] = bootstrap_delta(
                frame, metric, challenger, "vascx", args.bootstrap_repetitions, args.seed)
    summary["selection_boundary"] = {
        "fundus_avseg_labels_used_for_selection": False,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else None,
    }
    summary['coordinate_alignment'] = {
        'saved_image_crop_applied_to_reference': not args.legacy_unregistered,
        'canonical_input_used': not args.legacy_unregistered,
        'valid_scientific_comparison': not args.legacy_unregistered,
        'note': 'Legacy mode reproduces the historical coordinate error; do not use it as a model performance estimate.' if args.legacy_unregistered else 'All cached RGB crops reconstructed exactly; same image-only transform applied to annotations.'
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
