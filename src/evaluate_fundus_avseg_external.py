#!/usr/bin/env python3
"""Locked, zero-tuning external validation on Fundus-AVSeg."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from train_udct_av_pilot import Corrector, hessian_field, softmax_np


def target_labels(rgb: np.ndarray) -> np.ndarray:
    red = (rgb[..., 0] == 255) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)
    blue = (rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 255)
    green = (rgb[..., 0] == 0) & (rgb[..., 1] == 255) & (rgb[..., 2] == 0)
    white = (rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] == 255)
    target = np.zeros(red.shape, np.int64)
    target[red] = 1
    target[blue] = 2
    target[green] = 3
    target[white] = -100
    return target


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denominator) if denominator else np.nan


def score(prediction: np.ndarray, target: np.ndarray, target_rgb: np.ndarray) -> dict:
    valid = target != -100
    pa = np.isin(prediction, [1, 3]) & valid
    pv = np.isin(prediction, [2, 3]) & valid
    ta = np.isin(target, [1, 3]) & valid
    tv = np.isin(target, [2, 3]) & valid
    typed = np.isin(target, [1, 2])
    target_vessel = target_rgb.max(-1) > 0
    artery_dice, vein_dice = dice(pa, ta), dice(pv, tv)
    return {
        "vessel_dice": dice(prediction > 0, target_vessel),
        "artery_dice": artery_dice,
        "vein_dice": vein_dice,
        "mean_av_dice": float(np.nanmean([artery_dice, vein_dice])),
        "type_accuracy": float((prediction[typed] == target[typed]).mean()),
        "av_ratio_log_error": float(
            abs(
                np.log((pa.sum() + 1) / (pv.sum() + 1))
                - np.log((ta.sum() + 1) / (tv.sum() + 1))
            )
        ),
    }


def prediction_to_rgb(prediction: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    rgb[prediction == 1] = (255, 0, 0)
    rgb[prediction == 2] = (0, 0, 255)
    rgb[prediction == 3] = (0, 255, 0)
    return rgb


def bootstrap_delta(
    frame: pd.DataFrame,
    metric: str,
    challenger: str,
    baseline: str,
    repetitions: int,
    seed: int,
) -> dict:
    pivot = frame.pivot(index="id", columns="model", values=metric).dropna()
    delta = pivot[challenger].to_numpy() - pivot[baseline].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(repetitions, len(delta)))
    means = delta[indices].mean(1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
        "images_improved": int((delta > 0).sum()),
        "images_total": int(len(delta)),
    }


def load_features(image_path: Path, logits_path: Path, size: int):
    rgb = np.asarray(
        Image.open(image_path).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        )
    )
    logits = np.load(logits_path).astype(np.float32)
    logits = cv2.resize(logits, (size, size), interpolation=cv2.INTER_LINEAR)
    probability = softmax_np(logits)
    entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(-1) / np.log(4)
    direction = hessian_field(rgb)
    image = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
    base = np.moveaxis(np.clip(logits, -12, 12) / 6.0, -1, 0)
    features = np.concatenate(
        [image, base, entropy[None].astype(np.float32), direction], axis=0
    )
    return (
        torch.from_numpy(features),
        torch.from_numpy(logits).permute(2, 0, 1),
        np.moveaxis(probability, -1, 0),
    )


def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return frame.groupby(group_columns, dropna=False).agg(
        n=("id", "size"),
        vessel_dice=("vessel_dice", "mean"),
        artery_dice=("artery_dice", "mean"),
        vein_dice=("vein_dice", "mean"),
        mean_av_dice=("mean_av_dice", "mean"),
        type_accuracy=("type_accuracy", "mean"),
        av_ratio_log_error=("av_ratio_log_error", "mean"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("../data/external/Fundus-AVSeg/Fundus-AVSeg"),
    )
    parser.add_argument(
        "--vascx", type=Path, default=Path("artifacts/fundus_avseg_vascx")
    )
    parser.add_argument(
        "--occupancy", type=Path,
        default=Path("artifacts/fundus_avseg_occupancy"),
    )
    parser.add_argument(
        "--udct", type=Path,
        default=Path("artifacts/udct_av_mass_calibrated/best.pt"),
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("../data/external/Fundus-AVSeg/Fundus-AVSeg/metadata.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/fundus_avseg_external_validation"),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--gate-center", type=float, default=.35)
    parser.add_argument("--gate-temperature", type=float, default=.20)
    parser.add_argument("--threshold", type=float, default=.525)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_directories = {
        name: args.output / "predictions" / name
        for name in ("vascx", "mass_udct", "factorized_mass_udct")
    }
    for directory in prediction_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    metadata["id"] = metadata["image name"].str.replace(".png", "", regex=False)
    official_test = {
        line.strip().removesuffix(".png")
        for line in (args.dataset / "testing.txt").read_text().splitlines()
        if line.strip()
    }
    metadata["official_split"] = np.where(
        metadata.id.isin(official_test), "testing", "training"
    )
    metadata_by_id = metadata.set_index("id")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = Corrector().to(device)
    checkpoint = torch.load(args.udct, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    with torch.inference_mode():
        for image_path in sorted((args.dataset / "images").glob("*.png")):
            identifier = image_path.stem
            features, base_logits, vascx_probability = load_features(
                args.vascx / "preprocessed_rgb" / image_path.name,
                args.vascx / "logits" / f"{identifier}.npy",
                args.size,
            )
            corrected_logits = model(
                features[None].to(device), base_logits[None].to(device)
            )[0]
            corrected_probability = corrected_logits.softmax(0).cpu().numpy()
            adapted_probability = np.load(
                args.occupancy / f"{identifier}.npy"
            ).astype(np.float32)
            entropy = -(
                vascx_probability
                * np.log(np.maximum(vascx_probability, 1e-8))
            ).sum(0) / np.log(4)
            confidence = 1 - entropy
            gate = 1 / (
                1 + np.exp(
                    -(confidence - args.gate_center) / args.gate_temperature
                )
            )
            fused_occupancy = (
                gate * (1 - vascx_probability[0])
                + (1 - gate) * adapted_probability
            )
            identity = corrected_probability[1:4].argmax(0) + 1
            predictions = {
                "vascx": vascx_probability.argmax(0),
                "mass_udct": corrected_probability.argmax(0),
                "factorized_mass_udct": np.where(
                    fused_occupancy >= args.threshold, identity, 0
                ),
            }
            target_rgb = np.asarray(
                Image.open(args.dataset / "annotation" / image_path.name)
                .convert("RGB")
                .resize((args.size, args.size), Image.Resampling.NEAREST)
            )
            target = target_labels(target_rgb)
            info = metadata_by_id.loc[identifier]
            for model_name, prediction in predictions.items():
                Image.fromarray(prediction_to_rgb(prediction)).save(
                    prediction_directories[model_name] / image_path.name
                )
                rows.append({
                    "id": identifier,
                    "model": model_name,
                    "disease": info["disease type"],
                    "quality": info["image quality"],
                    "official_split": info["official_split"],
                    **score(prediction, target, target_rgb),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_image_metrics.csv", index=False)
    overall = aggregate(frame, ["model"])
    by_disease = aggregate(frame, ["disease", "model"])
    by_quality = aggregate(frame, ["quality", "model"])
    official_test_frame = frame.query("official_split == 'testing'")
    official_test = aggregate(official_test_frame, ["model"])
    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    by_disease.to_csv(args.output / "by_disease.csv", index=False)
    by_quality.to_csv(args.output / "by_quality.csv", index=False)
    official_test.to_csv(args.output / "official_test_metrics.csv", index=False)

    metrics = (
        "vessel_dice", "mean_av_dice", "type_accuracy", "av_ratio_log_error"
    )
    factorized_deltas = {
        metric: bootstrap_delta(
            frame, metric, "factorized_mass_udct", "vascx",
            args.bootstrap_repetitions, args.seed,
        )
        for metric in metrics
    }
    mass_deltas = {
        metric: bootstrap_delta(
            frame, metric, "mass_udct", "vascx",
            args.bootstrap_repetitions, args.seed,
        )
        for metric in metrics
    }
    primary = factorized_deltas["mean_av_dice"]
    ratio = mass_deltas["av_ratio_log_error"]
    mass_dice = mass_deltas["mean_av_dice"]
    summary = {
        "locked_parameters": {
            "gate_center": args.gate_center,
            "gate_temperature": args.gate_temperature,
            "threshold": args.threshold,
            "input_size": args.size,
            "fundus_avseg_tuning_used": False,
        },
        "external_images": int(metadata.shape[0]),
        "official_test_images": int((metadata.official_split == "testing").sum()),
        "overall": overall.to_dict("records"),
        "factorized_mass_udct_minus_vascx": factorized_deltas,
        "mass_udct_minus_vascx": mass_deltas,
        "primary_replication_success": bool(
            primary["mean_delta"] > 0 and primary["ci95"][0] > 0
        ),
        "ratio_replication_success": bool(
            ratio["mean_delta"] < 0 and mass_dice["mean_delta"] >= 0
        ),
        "cimt_validation_used": False,
        "cimt_test_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
