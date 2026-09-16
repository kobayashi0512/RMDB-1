#!/usr/bin/env python3
"""Audit frozen outputs and vessel-mask overlap on label-bearing transfer sets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "RAV": {"annotation": ROOT.parent / "data/external/RAV/dataset/av_masks", "raw_prefix": "RAV"},
    "FundusAVSeg": {"annotation": ROOT.parent / "data/external/Fundus-AVSeg/Fundus-AVSeg/annotation", "raw_prefix": "FundusAVSeg"},
}


def mask_overlap(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    inter = float(np.logical_and(pred, truth).sum())
    denom = float(pred.sum() + truth.sum())
    union = float(np.logical_or(pred, truth).sum())
    return (2.0 * inter / denom if denom else float("nan"), inter / union if union else float("nan"))


def main() -> None:
    root = ROOT / "artifacts/external_transfer"
    report: dict[str, object] = {"label_blind_features": True, "datasets": {}}
    for name, config in CONFIG.items():
        directory = root / name
        manifest = pd.read_csv(directory / "manifest.csv")
        dices, ious, logits_finite, occupancy_finite = [], [], [], []
        for row in manifest.itertuples(index=False):
            av = np.asarray(Image.open(directory / "anatomy/artery_vein" / f"{row.id}.png"))
            pred = av > 0
            logits = np.load(directory / "vascx_logits" / f"{row.id}.npy")
            occupancy = np.load(directory / "occupancy" / f"{row.id}.npy")
            logits_finite.append(bool(np.isfinite(logits).all() and logits.shape == (1024, 1024, 4)))
            occupancy_finite.append(bool(np.isfinite(occupancy).all()))
            annotation = config["annotation"] / f"{Path(row.path).stem}.png"
            truth = np.asarray(Image.open(annotation).resize((1024, 1024), Image.Resampling.NEAREST))
            truth = truth.any(axis=2) if truth.ndim == 3 else truth > 0
            dice, iou = mask_overlap(pred, truth)
            dices.append(dice)
            ious.append(iou)
        anatomy = json.loads((directory / "anatomy/metadata.json").read_text())
        phenotypes = json.loads((directory / "phenotypes/metadata.json").read_text())
        report["datasets"][name] = {
            "images": int(len(manifest)),
            "anatomy_complete": bool(anatomy["complete"]),
            "phenotypes_complete": bool(phenotypes["complete"]),
            "disc_failures": int(anatomy["disc_segmentation_failures"]),
            "logits_shape_finite": int(sum(logits_finite)),
            "occupancy_finite": int(sum(occupancy_finite)),
            "vessel_dice_mean": float(np.nanmean(dices)),
            "vessel_dice_median": float(np.nanmedian(dices)),
            "vessel_iou_mean": float(np.nanmean(ious)),
            "vessel_iou_median": float(np.nanmedian(ious)),
        }
    (root / "transfer_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
