#!/usr/bin/env python3
"""Compute paired-view repeatability metrics from outcome-blind features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def icc_1_1(values: pd.DataFrame, feature: str) -> tuple[float, int]:
    data = values[["eye_key", feature]].dropna()
    groups = data.groupby("eye_key", sort=False)[feature]
    counts = groups.size().to_numpy(float)
    means = groups.mean().to_numpy(float)
    ss = groups.apply(lambda x: float(np.square(x.to_numpy(float) - x.mean()).sum())).to_numpy(float)
    keep = counts >= 2
    counts, means, ss = counts[keep], means[keep], ss[keep]
    if len(counts) < 2:
        return float("nan"), int(len(counts))
    grand = float(np.average(means, weights=counts))
    n, g = float(counts.sum()), float(len(counts))
    ms_between = float(np.sum(counts * (means - grand) ** 2) / max(g - 1, 1))
    ms_within = float(ss.sum() / max(n - g, 1))
    effective_k = (n - float(np.sum(counts**2)) / n) / max(g - 1, 1)
    return float((ms_between - ms_within) / (ms_between + (effective_k - 1) * ms_within + 1e-12)), int(g)


def bootstrap_icc(values: pd.DataFrame, feature: str, repetitions: int, seed: int) -> tuple[float, float, float, int]:
    data = values[["eye_key", feature]].dropna()
    groups = [group[feature].to_numpy(float) for _, group in data.groupby("eye_key", sort=False) if len(group) >= 2]
    point, n_groups = icc_1_1(values, feature)
    if n_groups < 2:
        return point, float("nan"), float("nan"), n_groups
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = rng.integers(0, n_groups, n_groups)
        rows = []
        for index, group in enumerate(selected):
            for view, value in enumerate(groups[group]):
                rows.append({"eye_key": f"g{index}", feature: value})
        sample = pd.DataFrame(rows)
        samples.append(icc_1_1(sample, feature)[0])
    return point, float(np.nanquantile(samples, .025)), float(np.nanquantile(samples, .975)), n_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"patient_id": str, "eye": str, "split": str})
    frame["eye_key"] = frame["patient_id"] + "__" + frame["eye"]
    features = [column for column in frame.columns if column.startswith(("raw_", "joint_", "adapted_"))]
    rows = []
    for split, subset in frame.groupby("split", sort=True):
        for feature in features:
            point, low, high, n_groups = bootstrap_icc(subset, feature, args.bootstrap_repetitions, args.seed)
            data = subset[["eye_key", feature]].dropna()
            grouped = data.groupby("eye_key")[feature]
            deviations = grouped.apply(lambda x: float(np.mean(np.abs(x.to_numpy(float) - x.mean()))))
            rows.append({
                "split": split,
                "feature": feature,
                "icc_1_1": point,
                "icc_ci95_low": low,
                "icc_ci95_high": high,
                "n_repeat_eyes": n_groups,
                "within_eye_mae": float(deviations.mean()) if len(deviations) else float("nan"),
                "within_eye_mae_median": float(deviations.median()) if len(deviations) else float("nan"),
            })
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "repeatability_baselines.csv", index=False)
    report = {
        "input_images": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "repeat_eyes": int(frame.groupby("eye_key").size().ge(2).sum()),
        "features": features,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "outcome_blind": True,
        "cimt_used": False,
        "outcomes_used": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.sort_values(["split", "icc_1_1"], ascending=[True, False]).head(30).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
