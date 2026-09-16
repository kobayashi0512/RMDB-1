#!/usr/bin/env python3
"""Validation-selected fusion of vessel occupancy and conditional A/V identity.

The analog-camera domain is never used for model or hyper-parameter selection.
This script evaluates whether two independently useful corrections actually
compose: a domain-adapted vessel occupancy head and the UDCT A/V residual head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from train_udct_av_pilot import Corrector, load_case, softmax_np, target_labels


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
    predicted_vessel = prediction > 0
    artery_dice, vein_dice = dice(pa, ta), dice(pv, tv)
    return {
        "vessel_dice": dice(predicted_vessel, target_vessel),
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


def occupancy_candidates(
    vascx_probability: np.ndarray,
    adapted_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    vascx_vessel = 1.0 - vascx_probability[0]
    candidates = {
        "vascx": vascx_vessel,
        "adapted": adapted_probability,
    }
    for alpha in (0.25, 0.50, 0.75):
        candidates[f"blend_{alpha:.2f}"] = alpha * vascx_vessel + (1 - alpha) * adapted_probability

    entropy = -(vascx_probability * np.log(np.maximum(vascx_probability, 1e-8))).sum(0) / np.log(4)
    confidence = 1.0 - entropy
    for center in (0.35, 0.50, 0.65):
        for temperature in (0.10, 0.20):
            gate = 1.0 / (1.0 + np.exp(-(confidence - center) / temperature))
            candidates[f"entropy_c{center:.2f}_t{temperature:.2f}"] = (
                gate * vascx_vessel + (1 - gate) * adapted_probability
            )
    return candidates


def bootstrap_delta(frame: pd.DataFrame, metric: str, challenger: str, repetitions: int, seed: int) -> dict:
    pivot = frame.pivot(index="id", columns="model", values=metric).dropna()
    delta = pivot[challenger].to_numpy() - pivot["vascx"].to_numpy()
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repetitions)])
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "images_improved": int((delta > 0).sum()),
        "images_total": int(len(delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rav", type=Path, default=Path("artifacts/rav_vascx_full"))
    parser.add_argument("--ground-truth", type=Path, default=Path("../data/external/RAV/dataset/av_masks"))
    parser.add_argument("--udct", type=Path, default=Path("artifacts/udct_av_pilot/best.pt"))
    parser.add_argument("--split", type=Path, default=Path("artifacts/udct_av_pilot/split.json"))
    parser.add_argument(
        "--occupancy",
        type=Path,
        default=Path("artifacts/factorized_av_adapted/occupancy_probability"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/factorized_udct"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split.read_text())
    validation, holdout = split["validation"], split["holdout"]
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = Corrector().to(device)
    checkpoint = torch.load(args.udct, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cases = {}
    with torch.inference_mode():
        for identifier in validation + holdout:
            features, base_logits, target = load_case(identifier, args.rav, args.ground_truth)
            corrected_logits = model(features[None].to(device), base_logits[None].to(device))[0]
            corrected_probability = corrected_logits.softmax(0).cpu().numpy()
            vascx_probability = softmax_np(base_logits.permute(1, 2, 0).numpy())
            vascx_probability = np.moveaxis(vascx_probability, -1, 0)
            adapted_probability = np.load(args.occupancy / f"RAV_{identifier}.npy").astype(np.float32)
            target_rgb = np.asarray(
                Image.open(args.ground_truth / f"RAV_{identifier}.png")
                .convert("RGB")
                .resize((512, 512), Image.Resampling.NEAREST)
            )
            cases[identifier] = {
                "target": target.numpy(),
                "target_rgb": target_rgb,
                "vascx_probability": vascx_probability,
                "adapted_probability": adapted_probability,
                "identity": corrected_probability[1:4].argmax(0) + 1,
                "vascx_prediction": vascx_probability.argmax(0),
                "udct_prediction": corrected_probability.argmax(0),
            }

    thresholds = np.linspace(0.10, 0.90, 33)
    tuning_rows = []
    candidate_names = list(
        occupancy_candidates(
            cases[validation[0]]["vascx_probability"],
            cases[validation[0]]["adapted_probability"],
        )
    )
    for candidate_name in candidate_names:
        for threshold in thresholds:
            values = []
            for identifier in validation:
                case = cases[identifier]
                occupancy = occupancy_candidates(
                    case["vascx_probability"], case["adapted_probability"]
                )[candidate_name]
                prediction = np.where(occupancy >= threshold, case["identity"], 0)
                values.append(score(prediction, case["target"], case["target_rgb"])["mean_av_dice"])
            tuning_rows.append(
                {
                    "occupancy": candidate_name,
                    "threshold": float(threshold),
                    "validation_mean_av_dice": float(np.mean(values)),
                }
            )
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["validation_mean_av_dice", "occupancy"], ascending=[False, True]
    )
    tuning.to_csv(args.output / "validation_tuning.csv", index=False)
    selected = tuning.iloc[0]
    selected_name, selected_threshold = str(selected.occupancy), float(selected.threshold)

    rows = []
    for identifier in holdout:
        case = cases[identifier]
        occupancy = occupancy_candidates(
            case["vascx_probability"], case["adapted_probability"]
        )[selected_name]
        fused_prediction = np.where(occupancy >= selected_threshold, case["identity"], 0)
        for model_name, prediction in (
            ("vascx", case["vascx_prediction"]),
            ("udct", case["udct_prediction"]),
            ("factorized_udct", fused_prediction),
        ):
            rows.append(
                {
                    "id": identifier,
                    "model": model_name,
                    **score(prediction, case["target"], case["target_rgb"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "holdout_metrics.csv", index=False)
    overall = frame.groupby("model").agg(
        n=("id", "size"),
        vessel_dice=("vessel_dice", "mean"),
        artery_dice=("artery_dice", "mean"),
        vein_dice=("vein_dice", "mean"),
        mean_av_dice=("mean_av_dice", "mean"),
        type_accuracy=("type_accuracy", "mean"),
        av_ratio_log_error=("av_ratio_log_error", "mean"),
    ).reset_index()
    overall.to_csv(args.output / "holdout_summary.csv", index=False)
    deltas = {
        metric: bootstrap_delta(frame, metric, "factorized_udct", args.bootstrap_repetitions, args.seed)
        for metric in ("vessel_dice", "mean_av_dice", "type_accuracy")
    }
    summary = {
        "selection": {
            "occupancy": selected_name,
            "threshold": selected_threshold,
            "validation_mean_av_dice": float(selected.validation_mean_av_dice),
            "validation_images": len(validation),
        },
        "holdout_device": split["holdout_device"],
        "holdout_images": len(holdout),
        "results": overall.to_dict("records"),
        "factorized_udct_minus_vascx": deltas,
        "limitations": [
            "The device holdout contains only seven images.",
            "The fusion search is validation-selected and remains a pilot, not an external confirmation.",
            "Vessel occupancy and A/V identity were trained independently rather than end-to-end.",
        ],
        "cimt_validation_used": False,
        "cimt_test_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
