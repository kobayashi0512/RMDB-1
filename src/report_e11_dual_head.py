#!/usr/bin/env python3
"""Generate E11 dual-head and uncertainty figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["H0_mean", "H1_centile", "H2_nested_fusion"]
LABELS = ["Mean head", "Centile head", "Nested fusion"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.input / "summary_metrics.csv").set_index(
        "experiment"
    ).loc[ORDER]
    tuning = pd.read_csv(args.input / "outer_tuning.csv")
    coverage = pd.read_csv(args.input / "coverage_risk.csv")
    oof = pd.read_csv(args.input / "repeated_oof_predictions.csv")
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(3)
    axes[0].bar(
        x, summary.mae_mm_mean, yerr=summary.mae_mm_std, capsize=4,
        color=["#7F7F7F", "#2C7BB6", "#1A9850"],
    )
    axes[0].set(xticks=x, xticklabels=LABELS, ylabel="MAE (mm)")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].bar(
        x, summary.rmse_mm_mean, yerr=summary.rmse_mm_std, capsize=4,
        color=["#7F7F7F", "#2C7BB6", "#1A9850"],
    )
    axes[1].set(xticks=x, xticklabels=LABELS, ylabel="RMSE (mm)")
    axes[1].tick_params(axis="x", rotation=12)
    figure.suptitle("Dual-estimand trade-off")
    figure.tight_layout()
    figure.savefig(args.output / "dual_head_metrics.png", dpi=240)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    counts = tuning.selected_centile_weight.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#2C7BB6")
    ax.set(
        xlabel="Nested centile-head weight", ylabel="Outer folds",
        title="Weights selected without outer-fold access",
    )
    figure.tight_layout()
    figure.savefig(args.output / "selected_weights.png", dpi=240)
    plt.close(figure)

    coverage_summary = coverage.groupby("coverage").mae_mm.agg(["mean", "std"])
    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        coverage_summary.index, coverage_summary["mean"],
        yerr=coverage_summary["std"], marker="o", capsize=4,
    )
    ax.set(
        xlabel="Coverage retained (lowest disagreement first)",
        ylabel="Fusion MAE (mm)",
        title="Disagreement fails selective-risk validation",
    )
    ax.invert_xaxis()
    figure.tight_layout()
    figure.savefig(args.output / "coverage_risk_failure.png", dpi=240)
    plt.close(figure)

    first = oof.loc[oof.seed.eq(oof.seed.min())]
    error = np.abs(first.cimt_mm - first.fused_prediction_mm)
    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(first.disagreement_uncertainty, error, s=14, alpha=.45)
    rho = first[["disagreement_uncertainty"]].assign(
        error=error
    ).corr(method="spearman").iloc[0, 1]
    ax.set(
        xlabel="Normalized head disagreement",
        ylabel="Absolute fusion error (mm)",
        title=f"No useful uncertainty relation (seed rho={rho:.3f})",
    )
    figure.tight_layout()
    figure.savefig(args.output / "disagreement_vs_error.png", dpi=240)
    plt.close(figure)
    print(f"created 4 figures in {args.output}")


if __name__ == "__main__":
    main()
