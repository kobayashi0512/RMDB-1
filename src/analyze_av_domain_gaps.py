#!/usr/bin/env python3
"""Stratify A/V errors by device and age to identify domain-generalization gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.metrics)
    metadata = pd.read_csv(args.metadata, dtype={"id": str})
    metadata["image"] = "RAV_" + metadata.id + ".png"
    frame = metrics.merge(metadata, on="image", validate="one_to_one")
    frame["age_group"] = pd.cut(
        frame.patient_age, bins=[0, 49, 59, 69, 200],
        labels=["<50", "50-59", "60-69", ">=70"],
    )
    aggregations = {
        "n": ("image", "size"),
        "mean_av_dice": ("mean_av_dice", "mean"),
        "type_accuracy": ("type_accuracy_on_gt_vessels", "mean"),
        "av_ratio_log_error": ("av_ratio_log_error", "mean"),
        "failure_rate_dice_lt_0_70": ("mean_av_dice", lambda values: float((values < 0.70).mean())),
    }
    by_device = frame.groupby("device", observed=True).agg(**aggregations).sort_values("mean_av_dice")
    by_age = frame.groupby("age_group", observed=True).agg(**aggregations)
    by_device.to_csv(args.output / "by_device.csv")
    by_age.to_csv(args.output / "by_age.csv")
    frame.nsmallest(20, "mean_av_dice").to_csv(args.output / "worst_cases.csv", index=False)
    age_dice = spearmanr(frame.patient_age, frame.mean_av_dice)
    summary = {
        "images": len(frame),
        "age_dice_spearman_rho": float(age_dice.statistic),
        "age_dice_p": float(age_dice.pvalue),
        "worst_device": str(by_device.index[0]),
        "worst_device_mean_dice": float(by_device.iloc[0].mean_av_dice),
        "failures_dice_lt_0_70": int((frame.mean_av_dice < 0.70).sum()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(by_device.to_string())
    print(by_age.to_string())


if __name__ == "__main__":
    main()
