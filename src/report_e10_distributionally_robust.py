#!/usr/bin/env python3
"""Generate descriptive figures for the completed E10 experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ORDER = [
    "E10_0_gaussian_e8b",
    "E10A_conditional_centile",
    "E10B_centile_group_dro",
]
LABEL = {
    "E10_0_gaussian_e8b": "Gaussian E8B",
    "E10A_conditional_centile": "Conditional centile",
    "E10B_centile_group_dro": "Centile + Group-DRO",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.input / "summary_metrics.csv").set_index(
        "experiment"
    ).loc[ORDER]
    oof = pd.read_csv(args.input / "repeated_oof_predictions.csv")
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(ORDER))
    axes[0].bar(
        x, summary.mae_mm_mean,
        yerr=summary.mae_mm_std, capsize=4, color=["#7F7F7F", "#2C7BB6", "#1A9850"]
    )
    axes[0].set(
        xticks=x, xticklabels=[LABEL[item] for item in ORDER],
        ylabel="Repeated-CV MAE (mm)", title="Average performance",
    )
    axes[0].tick_params(axis="x", rotation=15)
    width = .25
    for index, column in enumerate([
        "worst_age_group_mae_mm_mean",
        "worst_sex_mae_mm_mean",
        "worst_cimt_group_mae_mm_mean",
    ]):
        axes[1].bar(
            x + (index - 1) * width, summary[column], width,
            label=["Age quartile", "Sex", "CIMT tertile"][index],
        )
    axes[1].set(
        xticks=x, xticklabels=[LABEL[item] for item in ORDER],
        ylabel="Worst-group MAE (mm)", title="Prespecified stress groups",
    )
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output / "average_and_worst_group_mae.png", dpi=240)
    plt.close(figure)

    rows = []
    for experiment in ORDER:
        for seed, group in oof.loc[
            oof.experiment.eq(experiment)
        ].groupby("seed"):
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "mae": mean_absolute_error(
                    group.cimt_mm, group.prediction_mm
                ),
                "rmse": mean_squared_error(
                    group.cimt_mm, group.prediction_mm
                ) ** .5,
            })
    tradeoff = pd.DataFrame(rows)
    figure, ax = plt.subplots(figsize=(6.5, 5))
    for experiment, group in tradeoff.groupby("experiment"):
        ax.scatter(
            group.rmse, group.mae, s=45, label=LABEL[experiment], alpha=.8
        )
    ax.set(
        xlabel="RMSE (mm; lower is better)",
        ylabel="MAE (mm; lower is better)",
        title="Mean–median estimand trade-off",
    )
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.output / "mae_rmse_tradeoff.png", dpi=240)
    plt.close(figure)

    baseline = oof.loc[oof.experiment.eq(ORDER[0])]
    figure, ax = plt.subplots(figsize=(8, 4.8))
    positions, values, labels = [], [], []
    for position, experiment in enumerate(ORDER[1:]):
        challenger = oof.loc[oof.experiment.eq(experiment)]
        deltas = []
        for (seed, fold), group in challenger.groupby(["seed", "fold"]):
            reference = baseline.loc[
                baseline.seed.eq(seed) & baseline.fold.eq(fold)
            ]
            deltas.append(
                mean_absolute_error(group.cimt_mm, group.prediction_mm)
                - mean_absolute_error(
                    reference.cimt_mm, reference.prediction_mm
                )
            )
        positions.extend(np.arange(len(deltas)) + position * 27)
        values.extend(deltas)
        labels.append(LABEL[experiment])
    colors = np.where(np.asarray(values) <= 0, "#2C7BB6", "#D7191C")
    ax.bar(positions, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel="Outer folds (first 25: centile; second 25: Group-DRO)",
        ylabel="MAE delta versus Gaussian E8B (mm)",
        title="Direction consistency across 25 outer folds",
    )
    figure.tight_layout()
    figure.savefig(args.output / "fold_direction.png", dpi=240)
    plt.close(figure)
    print(f"created 3 figures in {args.output}")


if __name__ == "__main__":
    main()
