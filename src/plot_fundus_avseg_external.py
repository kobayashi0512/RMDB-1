#!/usr/bin/env python3
"""Create locked external-validation diagnostic figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MODEL_LABELS = {
    "vascx": "VascX",
    "mass_udct": "Mass-UDCT",
    "factorized_mass_udct": "Gated Factorized",
}
MODEL_COLORS = {
    "vascx": "#64748B",
    "mass_udct": "#0F766E",
    "factorized_mass_udct": "#2563EB",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=Path("artifacts/fundus_avseg_external_validation"),
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("../data/external/Fundus-AVSeg/Fundus-AVSeg"),
    )
    args = parser.parse_args()
    figure_dir = args.results / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results / "per_image_metrics.csv")
    order = ["vascx", "mass_udct", "factorized_mass_udct"]

    overall = frame.groupby("model")[
        ["vessel_dice", "mean_av_dice", "type_accuracy", "av_ratio_log_error"]
    ].mean().loc[order]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for axis, metric, title in zip(
        axes,
        overall.columns,
        ["Vessel Dice", "Mean A/V Dice", "Type accuracy", "A/V ratio log error ↓"],
    ):
        values = overall[metric]
        axis.bar(
            range(3), values,
            color=[MODEL_COLORS[name] for name in order],
            width=.68,
        )
        axis.set_xticks(range(3), [MODEL_LABELS[name] for name in order], rotation=25, ha="right")
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=.2)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fundus-AVSeg locked zero-tuning external validation", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "overall_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pivot = frame.pivot(index="id", columns="model")
    metrics = ["vessel_dice", "mean_av_dice", "type_accuracy", "av_ratio_log_error"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    rng = np.random.default_rng(20260729)
    for axis, metric in zip(axes, metrics):
        delta = (
            pivot[metric]["mass_udct"] - pivot[metric]["vascx"]
        ).to_numpy()
        jitter = rng.normal(1, .035, size=len(delta))
        axis.scatter(jitter, delta, s=12, alpha=.55, color="#0F766E")
        axis.boxplot(delta, positions=[1], widths=.25, showfliers=False)
        axis.axhline(0, color="#991B1B", lw=1, ls="--")
        axis.set_xticks([1], ["Mass-UDCT − VascX"])
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Paired per-image changes (n=100)", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_deltas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    disease = frame.groupby(["disease", "model"]).mean_av_dice.mean().unstack()
    disease = disease.loc[["Normal", "DR", "AMD", "Glaucoma"], order]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(disease))
    width = .24
    for index, name in enumerate(order):
        axis.bar(
            x + (index - 1) * width, disease[name], width,
            label=MODEL_LABELS[name], color=MODEL_COLORS[name],
        )
    axis.set_xticks(x, disease.index)
    axis.set_ylabel("Mean A/V Dice")
    axis.set_title("Disease-stratified external performance")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "disease_stratified.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    delta = (
        pivot["mean_av_dice"]["mass_udct"] - pivot["mean_av_dice"]["vascx"]
    ).sort_values()
    selected = [delta.index[0], delta.index[len(delta) // 2], delta.index[-1]]
    image_rows = []
    for identifier in selected:
        information = frame.query("id == @identifier").iloc[0]
        images = [
            np.asarray(Image.open(args.dataset / "images" / f"{identifier}.png").convert("RGB").resize((512, 512))),
            np.asarray(Image.open(args.dataset / "annotation" / f"{identifier}.png").convert("RGB").resize((512, 512), Image.Resampling.NEAREST)),
            np.asarray(Image.open(args.results / "predictions" / "vascx" / f"{identifier}.png").convert("RGB")),
            np.asarray(Image.open(args.results / "predictions" / "mass_udct" / f"{identifier}.png").convert("RGB")),
            np.asarray(Image.open(args.results / "predictions" / "factorized_mass_udct" / f"{identifier}.png").convert("RGB")),
        ]
        image_rows.append((identifier, information, images))
    fig, axes = plt.subplots(3, 5, figsize=(14, 8.5))
    columns = ["Fundus", "Ground truth", "VascX", "Mass-UDCT", "Gated factorized"]
    for row, (identifier, information, images) in enumerate(image_rows):
        for column, image in enumerate(images):
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(columns[column])
        axes[row, 0].set_ylabel(
            f"{identifier}\n{information.disease}, {information.quality}\nΔDice={delta.loc[identifier]:+.4f}",
            fontsize=9,
        )
    fig.suptitle("External qualitative audit: worst, median and best paired change", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "qualitative_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"figures={len(list(figure_dir.glob('*.png')))} output={figure_dir}")


if __name__ == "__main__":
    main()
