#!/usr/bin/env python3
"""Factorize vessel occupancy (FIVES) from conditional A/V identity (VascX).

The occupancy threshold is selected on a fixed non-holdout validation subset.
The analog camera domain remains untouched until the final evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

ANALOG = "TOPCON TRC-50VT - Analog Fundus Camera"


def target_labels(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[..., 0] > 0, rgb[..., 1] > 0, rgb[..., 2] > 0
    target = np.zeros(red.shape, np.int64)
    target[red & ~blue] = 1
    target[blue & ~red] = 2
    target[red & blue] = 3
    target[green] = -100
    return target


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denominator) if denominator else np.nan


def score(prediction: np.ndarray, target: np.ndarray, target_rgb: np.ndarray) -> dict:
    valid = target != -100
    pa, pv = np.isin(prediction, [1, 3]) & valid, np.isin(prediction, [2, 3]) & valid
    ta, tv = np.isin(target, [1, 3]) & valid, np.isin(target, [2, 3]) & valid
    typed = np.isin(target, [1, 2])
    target_vessel = target_rgb.max(-1) > 0
    predicted_vessel = prediction > 0
    return {
        "vessel_dice": dice(predicted_vessel, target_vessel),
        "artery_dice": dice(pa, ta),
        "vein_dice": dice(pv, tv),
        "mean_av_dice": np.nanmean([dice(pa, ta), dice(pv, tv)]),
        "type_accuracy": float((prediction[typed] == target[typed]).mean()),
        "av_ratio_log_error": float(abs(np.log((pa.sum() + 1) / (pv.sum() + 1)) - np.log((ta.sum() + 1) / (tv.sum() + 1)))),
    }


def load_resized(identifier: str, rav: Path, ground_truth: Path, size: int):
    image = Image.open(rav / "preprocessed_rgb" / f"RAV_{identifier}.png").convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    target_rgb = np.asarray(Image.open(ground_truth / f"RAV_{identifier}.png").convert("RGB").resize((size, size), Image.Resampling.NEAREST))
    logits = torch.from_numpy(np.load(rav / "logits" / f"RAV_{identifier}.npy").astype(np.float32)).permute(2, 0, 1)[None]
    logits = F.interpolate(logits, (size, size), mode="bilinear", align_corners=False)[0]
    return image, target_rgb, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rav", type=Path, default=Path("artifacts/rav_vascx_full"))
    parser.add_argument("--ground-truth", type=Path, default=Path("../data/external/RAV/dataset/av_masks"))
    parser.add_argument("--metadata", type=Path, default=Path("../data/external/RAV/metadata.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/fives_unet_resnet34/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/factorized_av"))
    parser.add_argument("--holdout-device", default=ANALOG)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, dtype={"id": str})
    holdout = metadata.loc[metadata.device == args.holdout_device, "id"].tolist()
    development = metadata.loc[metadata.device != args.holdout_device, "id"].tolist()
    random.Random(args.seed).shuffle(development)
    validation = development[:20]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint)
    model.eval()
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    probability_dir = args.output / "occupancy_probability"
    probability_dir.mkdir(parents=True, exist_ok=True)

    cases = {}
    with torch.inference_mode():
        for identifier in validation + holdout:
            image, target_rgb, vascx_logits = load_resized(identifier, args.rav, args.ground_truth, args.size)
            vessel_probability = model(normalize(image)[None].to(device)).sigmoid()[0, 0].cpu().numpy()
            np.save(probability_dir / f"RAV_{identifier}.npy", vessel_probability.astype(np.float16))
            vascx_probability = vascx_logits.softmax(0).numpy()
            conditional_label = vascx_probability[1:4].argmax(0) + 1
            target = target_labels(target_rgb)
            cases[identifier] = (vessel_probability, conditional_label, vascx_probability.argmax(0), target, target_rgb)

    thresholds = np.linspace(0.10, 0.90, 33)
    tuning_rows = []
    for threshold in thresholds:
        values = []
        for identifier in validation:
            vessel_probability, conditional_label, _, target, target_rgb = cases[identifier]
            prediction = np.where(vessel_probability >= threshold, conditional_label, 0)
            values.append(score(prediction, target, target_rgb)["mean_av_dice"])
        tuning_rows.append({"threshold": float(threshold), "validation_mean_av_dice": float(np.mean(values))})
    tuning = pd.DataFrame(tuning_rows)
    threshold = float(tuning.loc[tuning.validation_mean_av_dice.idxmax(), "threshold"])
    tuning.to_csv(args.output / "threshold_tuning.csv", index=False)

    rows = []
    for identifier in holdout:
        vessel_probability, conditional_label, baseline, target, target_rgb = cases[identifier]
        factorized = np.where(vessel_probability >= threshold, conditional_label, 0)
        rows.append({"id": identifier, "model": "vascx", **score(baseline, target, target_rgb)})
        rows.append({"id": identifier, "model": "factorized", **score(factorized, target, target_rgb)})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "holdout_metrics.csv", index=False)
    overall = frame.groupby("model").agg(
        n=("id", "size"), vessel_dice=("vessel_dice", "mean"), artery_dice=("artery_dice", "mean"),
        vein_dice=("vein_dice", "mean"), mean_av_dice=("mean_av_dice", "mean"),
        type_accuracy=("type_accuracy", "mean"), av_ratio_log_error=("av_ratio_log_error", "mean"),
    ).reset_index()
    overall.to_csv(args.output / "holdout_summary.csv", index=False)
    summary = {
        "selected_threshold": threshold,
        "validation_images": len(validation),
        "holdout_device": args.holdout_device,
        "holdout_images": len(holdout),
        "results": overall.to_dict("records"),
        "cimt_validation_used": False,
        "cimt_test_used": False,
        "fives_official_test_used_in_this_experiment": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
