#!/usr/bin/env python3
"""Label-blind repeat-view reliability audit for mBRSET eye phenotypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260731


def icc_from_summaries(counts: np.ndarray, means: np.ndarray, within_ss: np.ndarray) -> float:
    """One-way random-effects ICC(1,1) from per-eye sufficient statistics."""
    if len(counts) < 2:
        return float("nan")
    grand = float(np.average(means, weights=counts))
    n, groups = float(counts.sum()), float(len(counts))
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(within_ss.sum())
    ms_between = ss_between / (groups - 1)
    ms_within = ss_within / (n - groups)
    effective_k = (n - float(np.sum(counts ** 2)) / n) / (groups - 1)
    return float((ms_between - ms_within) / (ms_between + (effective_k - 1) * ms_within))


def summaries(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = frame[["eye_key", column]].dropna()
    grouped = data.groupby("eye_key", sort=False)[column]
    count = grouped.size().to_numpy(dtype=float)
    mean = grouped.mean().to_numpy(dtype=float)
    sum_sq = grouped.apply(lambda x: float(np.sum(np.square(x.to_numpy(dtype=float))))).to_numpy(dtype=float)
    within = sum_sq - count * np.square(mean)
    keep = count >= 2
    return count[keep], mean[keep], within[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--feature", action="append", default=[],
                        help="Repeat to audit only pre-specified numeric features.")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"patient_id": str})
    frame["eye_key"] = frame["patient_id"] + "__" + frame["eye"]
    numeric = [c for c in frame.columns if c not in {"id", "patient_id", "eye", "split", "disc_detected", "eye_key"}]
    if args.feature:
        missing = sorted(set(args.feature) - set(numeric))
        if missing:
            raise KeyError(f"Unknown requested feature(s): {missing}")
        numeric = args.feature
    repeat_keys = frame.groupby("eye_key").size()
    eligible = frame.loc[frame["eye_key"].isin(repeat_keys[repeat_keys >= 2].index)].copy()
    rng = np.random.default_rng(SEED)
    records = []
    for column in numeric:
        counts, means, within = summaries(eligible, column)
        estimate = icc_from_summaries(counts, means, within)
        samples = np.empty(args.bootstrap_repetitions)
        for index in range(args.bootstrap_repetitions):
            selected = rng.integers(0, len(counts), len(counts))
            samples[index] = icc_from_summaries(counts[selected], means[selected], within[selected])
        records.append({"feature": column, "icc_1_1": estimate,
                        "ci95_low": float(np.nanquantile(samples, .025)),
                        "ci95_high": float(np.nanquantile(samples, .975)),
                        "n_repeat_eyes": int(len(counts))})
    result = pd.DataFrame(records).sort_values("icc_1_1", ascending=False)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "repeatability_icc.csv", index=False)
    (args.output / "repeatability_metadata.json").write_text(json.dumps({
        "label_blind": True, "images": int(len(frame)), "repeat_eyes": int(len(repeat_keys[repeat_keys >= 2])),
        "bootstrap_repetitions": args.bootstrap_repetitions, "method": "one_way_random_effects_icc_1_1",
    }, indent=2), encoding="utf-8")
    print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
