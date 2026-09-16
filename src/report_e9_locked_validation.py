#!/usr/bin/env python3
"""Create publication-style descriptive figures for the locked E9 evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.input / "predictions.csv")
    subgroups = pd.read_csv(args.input / "subgroups.csv")
    metrics = json.loads((args.input / "metrics.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    truth = predictions.cimt_mm.to_numpy()
    normative = predictions.normative_prediction_mm.to_numpy()
    e8b = predictions.e8b_prediction_mm.to_numpy()
    limits = [min(truth.min(), normative.min(), e8b.min()),
              max(truth.max(), normative.max(), e8b.max())]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, values, title in zip(
        axes, [normative, e8b], ["V0 normative", "V1 E8B"]
    ):
        ax.scatter(truth, values, s=18, alpha=.65)
        ax.plot(limits, limits, "--", color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Observed CIMT (mm)")
        ax.set_ylabel("Predicted CIMT (mm)")
    fig.tight_layout()
    fig.savefig(args.output / "observed_vs_predicted.png", dpi=240)
    plt.close(fig)

    improvement = np.abs(truth - normative) - np.abs(truth - e8b)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(improvement, bins=30, color="#2878B5", alpha=.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(improvement.mean(), color="#C82423", linewidth=2,
               label=f"Mean improvement={improvement.mean():.4f} mm")
    ax.set_xlabel("Absolute-error reduction (V0 − V1, mm)")
    ax.set_ylabel("Patients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "paired_error_improvement.png", dpi=240)
    plt.close(fig)

    label = predictions.accelerated.to_numpy()
    probability = predictions.e8d_probability.to_numpy()
    fpr, tpr, _ = roc_curve(label, probability)
    precision, recall, _ = precision_recall_curve(label, probability)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(fpr, tpr, linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axes[0].set(
        xlabel="False-positive rate", ylabel="True-positive rate",
        title=f"ROC (AUROC={metrics['e8d']['auroc']:.3f})",
    )
    axes[1].plot(recall, precision, linewidth=2)
    axes[1].axhline(label.mean(), linestyle="--", color="black", linewidth=1)
    axes[1].set(
        xlabel="Recall", ylabel="Precision",
        title=f"PR (AUPRC={metrics['e8d']['auprc']:.3f})",
    )
    fig.tight_layout()
    fig.savefig(args.output / "accelerated_roc_pr.png", dpi=240)
    plt.close(fig)

    labels = subgroups.variable.astype(str) + ": " + subgroups.level.astype(str)
    y = np.arange(len(subgroups))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    axes[0].barh(y, subgroups.mae_delta, color=np.where(
        subgroups.mae_delta <= 0, "#2C7BB6", "#D7191C"
    ))
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(xlabel="MAE delta (V1 − V0, mm)", yticks=y, yticklabels=labels)
    axes[1].barh(y, subgroups.e8d_auroc, color="#FDAE61")
    axes[1].axvline(.5, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="V2 AUROC")
    fig.tight_layout()
    fig.savefig(args.output / "subgroup_results.png", dpi=240)
    plt.close(fig)

    print(f"created 4 figures in {args.output}")


if __name__ == "__main__":
    main()
