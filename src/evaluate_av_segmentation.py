#!/usr/bin/env python3
"""Evaluate hard A/V labels against RGB-channel artery/vein ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum() + target.sum())
    return float(2 * np.logical_and(prediction, target).sum() / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for prediction_path in sorted(args.predictions.glob("*.png")):
        target_path = args.ground_truth / prediction_path.name
        if not target_path.exists():
            continue
        prediction = np.asarray(Image.open(prediction_path))
        target = np.asarray(Image.open(target_path).convert("RGB"))
        if prediction.shape != target.shape[:2]:
            target = np.asarray(Image.fromarray(target).resize(prediction.shape[::-1], resample=Image.Resampling.NEAREST))
        valid = target[..., 1] == 0  # Exclude green unknown-vessel pixels.
        target_a = (target[..., 0] > 0) & valid
        target_v = (target[..., 2] > 0) & valid
        pred_a = np.isin(prediction, [1, 3]) & valid
        pred_v = np.isin(prediction, [2, 3]) & valid
        typed = np.logical_xor(target_a, target_v)
        correct_type = ((prediction == 1) & target_a) | ((prediction == 2) & target_v)
        artery_dice = dice(pred_a, target_a)
        vein_dice = dice(pred_v, target_v)
        direct_mean = np.nanmean([artery_dice, vein_dice])
        swapped_mean = np.nanmean([dice(pred_v, target_a), dice(pred_a, target_v)])
        rows.append(
            {
                "image": prediction_path.name,
                "artery_dice": artery_dice,
                "vein_dice": vein_dice,
                "mean_av_dice": direct_mean,
                "swapped_mean_av_dice": swapped_mean,
                "swap_corrected_mean_av_dice": max(direct_mean, swapped_mean),
                "global_identity_swap": bool(swapped_mean > direct_mean),
                "swap_gain": float(max(0.0, swapped_mean - direct_mean)),
                "type_accuracy_on_gt_vessels": float(correct_type[typed].mean()) if typed.any() else float("nan"),
                "target_av_ratio": float(target_a.sum() / max(target_v.sum(), 1)),
                "predicted_av_ratio": float(pred_a.sum() / max(pred_v.sum(), 1)),
                "av_ratio_log_error": float(abs(np.log((pred_a.sum() + 1) / (pred_v.sum() + 1)) - np.log((target_a.sum() + 1) / (target_v.sum() + 1)))),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_image_metrics.csv", index=False)
    summary = {
        "images": len(frame),
        **{
            f"{column}_{stat}": float(getattr(frame[column], stat)())
            for column in ["artery_dice", "vein_dice", "mean_av_dice", "swap_corrected_mean_av_dice", "type_accuracy_on_gt_vessels", "av_ratio_log_error"]
            for stat in ["mean", "median", "std"]
        },
        "global_identity_swaps": int(frame.global_identity_swap.sum()),
        "global_identity_swap_fraction": float(frame.global_identity_swap.mean()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
