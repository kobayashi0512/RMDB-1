#!/usr/bin/env python3
"""Audit whether VascX soft confidence predicts A/V errors and improves AVR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, wilcoxon


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.5, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--hard-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hard = pd.read_csv(args.hard_metrics).set_index("image")
    rows, confidence_parts, correct_parts = [], [], []
    for logits_path in sorted(args.logits.glob("*.npy")):
        name = f"{logits_path.stem}.png"
        target_path = args.ground_truth / name
        if not target_path.exists():
            continue
        probability = softmax(np.load(logits_path))
        target = np.asarray(Image.open(target_path).convert("RGB"))
        if target.shape[:2] != probability.shape[:2]:
            target = np.asarray(
                Image.fromarray(target).resize(probability.shape[1::-1], Image.Resampling.NEAREST)
            )
        red, green, blue = target[..., 0] > 0, target[..., 1] > 0, target[..., 2] > 0
        typed = np.logical_xor(red, blue) & ~green
        av_probability = probability[..., 1:3]
        av_total = av_probability.sum(axis=-1)
        conditional = av_probability / np.maximum(av_total[..., None], 1e-6)
        prediction = conditional.argmax(axis=-1)
        truth = np.where(red, 0, 1)
        confidence = conditional.max(axis=-1)[typed]
        correct = (prediction == truth)[typed]
        # Bound memory while retaining a broad calibration sample.
        confidence_parts.append(confidence[::8])
        correct_parts.append(correct[::8])
        entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(axis=-1) / np.log(probability.shape[-1])
        soft_ratio = float((probability[..., 1] + probability[..., 3]).sum() / max((probability[..., 2] + probability[..., 3]).sum(), 1e-6))
        target_ratio = float((red & ~green).sum() / max((blue & ~green).sum(), 1))
        rows.append(
            {
                "image": name,
                "mean_entropy": float(entropy.mean()),
                "vessel_entropy": float(entropy[red | blue].mean()),
                "typed_confidence": float(confidence.mean()),
                "typed_accuracy": float(correct.mean()),
                "typed_ece": expected_calibration_error(confidence, correct),
                "soft_av_ratio": soft_ratio,
                "soft_av_ratio_log_error": float(abs(np.log((soft_ratio + 1e-6) / (target_ratio + 1e-6)))),
                "hard_av_ratio_log_error": float(hard.loc[name, "av_ratio_log_error"]),
                "mean_av_dice": float(hard.loc[name, "mean_av_dice"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_image_uncertainty.csv", index=False)
    confidence = np.concatenate(confidence_parts)
    correct = np.concatenate(correct_parts)
    retention = []
    for coverage in [1.0, 0.9, 0.75, 0.5, 0.25]:
        threshold = np.quantile(confidence, 1 - coverage)
        selected = confidence >= threshold
        retention.append(
            {"coverage": coverage, "confidence_threshold": float(threshold), "typed_accuracy": float(correct[selected].mean())}
        )
    pd.DataFrame(retention).to_csv(args.output / "selective_accuracy.csv", index=False)
    uncertainty_error = spearmanr(frame.vessel_entropy, 1 - frame.typed_accuracy)
    ratio_test = wilcoxon(frame.soft_av_ratio_log_error, frame.hard_av_ratio_log_error)
    summary = {
        "images": len(frame),
        "pixel_typed_ece": expected_calibration_error(confidence, correct),
        "vessel_entropy_vs_type_error_rho": float(uncertainty_error.statistic),
        "vessel_entropy_vs_type_error_p": float(uncertainty_error.pvalue),
        "hard_ratio_log_error_mean": float(frame.hard_av_ratio_log_error.mean()),
        "soft_ratio_log_error_mean": float(frame.soft_av_ratio_log_error.mean()),
        "soft_minus_hard_ratio_error": float((frame.soft_av_ratio_log_error - frame.hard_av_ratio_log_error).mean()),
        "paired_wilcoxon_p": float(ratio_test.pvalue),
        "selective_accuracy": retention,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
