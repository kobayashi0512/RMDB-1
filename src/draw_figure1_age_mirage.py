#!/usr/bin/env python3
"""Render RMDB Figure 1 from local cohort data and real retinal images.

This figure is deliberately a mixed conceptual/data figure: exact cohort values
and real inputs are drawn locally, while boxes/arrows only explain the task.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/manifest.csv"
OUT = ROOT / "figures/rmdb_age_mirage/output"

NAVY = "#16324F"
TEAL = "#2A9D8F"
TEAL_FILL = "#DDF3EF"
RED = "#D95D39"
RED_FILL = "#FCE8E3"
GOLD = "#E9C46A"
GRAY = "#68737D"
LIGHT = "#F5F7FA"
LINE = "#C9D2DB"


def rounded(ax, xy, wh, face, edge=NAVY, lw=1.7, radius=0.18):
    patch = FancyBboxPatch(
        xy, wh[0], wh[1],
        boxstyle=f"round,pad=0.016,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=NAVY, lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}", linewidth=lw,
        color=color, shrinkA=3, shrinkB=3,
    ))


def add_fundus(ax, image_path: str, xy, zoom=0.19):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (640, 640), method=Image.Resampling.LANCZOS)
    im = OffsetImage(np.asarray(img), zoom=zoom)
    ab = AnnotationBbox(im, xy, xycoords=ax.transAxes, frameon=True,
                        bboxprops={"edgecolor": NAVY, "linewidth": 1.4,
                                   "boxstyle": "round,pad=0.02", "facecolor": "white"})
    ax.add_artist(ab)


def load_rows():
    rows = []
    with MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                row["age"] = float(row["age"])
                row["sex"] = int(row["sex"])
                row["cimt_mm"] = float(row["cimt_mm"])
                rows.append(row)
    return rows


def fit_expected(rows):
    x = np.array([[1.0, r["age"], r["sex"]] for r in rows])
    y = np.array([r["cimt_mm"] for r in rows])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for r in rows:
        r["expected"] = float(beta @ np.array([1.0, r["age"], r["sex"]]))
        r["deviation"] = r["cimt_mm"] - r["expected"]
    return beta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    beta = fit_expected(rows)
    case_ids = ["502560002", "521238001"]
    cases = {r["patient_id"]: r for r in rows if r["patient_id"] in case_ids}
    if set(cases) != set(case_ids):
        raise RuntimeError("Expected Figure 1 cases are missing from the train manifest.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(18, 8), facecolor="white")
    fig.text(0.035, 0.982, "The age-proxy paradox in retinal cIMT modeling",
             fontsize=27, weight="bold", color="#102A43", va="top")
    fig.text(0.035, 0.937,
             "A high raw-cIMT score may identify chronological age before it identifies age-discordant vascular aging.",
             fontsize=15, color=GRAY, va="top")

    # Panel A
    a = fig.add_axes([0.035, 0.10, 0.275, 0.72])
    a.set_axis_off()
    rounded(a, (0.00, 0.00), (1.00, 1.00), "#F6F8FB", edge="#BAC6D2", lw=1.5)
    a.text(0.05, 0.94, "A  Conventional raw-cIMT task", transform=a.transAxes,
           fontsize=19, weight="bold", color=NAVY, va="top")
    a.text(0.05, 0.875, "What a model is often asked to do", transform=a.transAxes,
           fontsize=13.5, color=GRAY, va="top")

    rounded(a, (0.10, 0.62), (0.32, 0.16), "#E7EEF7", edge="#7894B5")
    a.text(0.26, 0.705, "Age + sex", transform=a.transAxes, ha="center", va="center",
           fontsize=16, weight="bold", color=NAVY)
    a.text(0.26, 0.65, "strong cIMT drivers", transform=a.transAxes, ha="center", va="center",
           fontsize=12.5, color=GRAY)
    rounded(a, (0.58, 0.62), (0.32, 0.16), "#F9EBD8", edge="#D2A660")
    a.text(0.74, 0.705, "Retinal image", transform=a.transAxes, ha="center", va="center",
           fontsize=16, weight="bold", color="#714D18")
    a.text(0.74, 0.65, "age-related appearance", transform=a.transAxes, ha="center", va="center",
           fontsize=12.5, color=GRAY)

    rounded(a, (0.24, 0.31), (0.52, 0.17), RED_FILL, edge=RED, lw=2.0)
    a.text(0.50, 0.415, "Raw cIMT label", transform=a.transAxes, ha="center", va="center",
           fontsize=18, weight="bold", color=RED)
    a.text(0.50, 0.355, "high vs low cIMT", transform=a.transAxes, ha="center", va="center",
           fontsize=14, color="#874333")
    arrow(a, (0.26, 0.62), (0.40, 0.48), NAVY)
    arrow(a, (0.74, 0.62), (0.60, 0.48), "#B47D28")
    rounded(a, (0.09, 0.09), (0.82, 0.12), "white", edge="#E3B7AE", lw=1.3)
    a.text(0.50, 0.15, "Risk: performance can be inflated by\nshared age information.",
           transform=a.transAxes, ha="center", va="center", fontsize=14.5, color=RED, weight="bold")

    # Panel B
    b = fig.add_axes([0.335, 0.10, 0.305, 0.68])
    b.set_facecolor("white")
    for spine in b.spines.values():
        spine.set_color("#BAC6D2")
        spine.set_linewidth(1.3)
    fig.text(0.335, 0.847, "B  Internal cohort: cIMT rises with age",
             fontsize=19, fontweight="bold", color=NAVY, va="top")
    fig.text(0.335, 0.817, "China-Fundus-CIMT train: 2,603 patients",
             fontsize=13.5, color=GRAY, va="top")
    for sex, name, color in [(0, "Female", "#97A7B8"), (1, "Male", "#BCC7D4")]:
        subset = [r for r in rows if r["sex"] == sex]
        b.scatter([r["age"] for r in subset], [r["cimt_mm"] for r in subset],
                  s=10, alpha=0.34, c=color, edgecolors="none", label=name, rasterized=True)
        ages = np.linspace(25, 80, 100)
        y = beta[0] + beta[1] * ages + beta[2] * sex
        b.plot(ages, y, color=NAVY if sex == 0 else "#5F7082", linewidth=2.2,
               linestyle="-" if sex == 0 else "--")
    for pid, color, label in [("502560002", TEAL, "age-concordant"), ("521238001", RED, "age-discordant")]:
        r = cases[pid]
        b.scatter(r["age"], r["cimt_mm"], s=140, c=color, edgecolors="white", linewidths=2.0, zorder=5)
        b.annotate(label, (r["age"], r["cimt_mm"]), xytext=(7, 10), textcoords="offset points",
                   fontsize=12.5, color=color, weight="bold")
    b.set_xlim(24, 82)
    b.set_ylim(0.42, 1.75)
    b.set_xlabel("Age (years)", fontsize=15)
    b.set_ylabel("Observed cIMT (mm)", fontsize=15)
    b.tick_params(labelsize=13)
    b.grid(axis="y", color="#E9EDF1", linewidth=0.9)
    b.legend(loc="upper left", frameon=False, ncol=2, fontsize=12)
    b.text(0.03, 0.05, "Lines: sex-specific linear expectations\nfor illustration of the conditioned target.",
           transform=b.transAxes, fontsize=12.5, color=GRAY,
           bbox={"facecolor": "white", "edgecolor": "#DCE3EA", "boxstyle": "round,pad=0.35"})

    # Panel C
    c = fig.add_axes([0.665, 0.10, 0.300, 0.72])
    c.set_axis_off()
    rounded(c, (0.00, 0.00), (1.00, 1.00), "#F7FBFA", edge="#9AC9C1", lw=1.5)
    c.text(0.05, 0.94, "C  Reframed clinical question", transform=c.transAxes,
           fontsize=19, weight="bold", color=TEAL, va="top")
    c.text(0.05, 0.875, "Who has cIMT higher than expected for age and sex?", transform=c.transAxes,
           fontsize=13.5, color=GRAY, va="top")

    left = cases["502560002"]
    right = cases["521238001"]
    add_fundus(c, left["left_path"], (0.27, 0.62), zoom=0.195)
    add_fundus(c, right["left_path"], (0.73, 0.62), zoom=0.195)
    c.text(0.27, 0.42, f"Female, {int(left['age'])} y", transform=c.transAxes,
           ha="center", fontsize=15, weight="bold", color=TEAL)
    c.text(0.73, 0.42, f"Female, {int(right['age'])} y", transform=c.transAxes,
           ha="center", fontsize=15, weight="bold", color=RED)
    rounded(c, (0.06, 0.18), (0.42, 0.18), TEAL_FILL, edge=TEAL, lw=1.6)
    rounded(c, (0.52, 0.18), (0.42, 0.18), RED_FILL, edge=RED, lw=1.6)
    c.text(0.27, 0.292, f"Observed: {left['cimt_mm']:.2f} mm\nExpected: {left['expected']:.2f} mm\nDeviation: {left['deviation']:+.2f} mm",
           transform=c.transAxes, ha="center", va="center", fontsize=13.5, color="#176F66", weight="bold")
    c.text(0.73, 0.292, f"Observed: {right['cimt_mm']:.2f} mm\nExpected: {right['expected']:.2f} mm\nDeviation: {right['deviation']:+.2f} mm",
           transform=c.transAxes, ha="center", va="center", fontsize=13.5, color="#9A3927", weight="bold")
    c.text(0.50, 0.065, "Real fundus images are measurement inputs -\nnot individual visual diagnoses.",
           transform=c.transAxes, ha="center", va="center", fontsize=13.3, color=GRAY)

    fig.text(0.035, 0.018,
             "RMDB reframes retinal vascular AI from predicting age-linked cIMT level to auditing age-discordant vascular aging at the patient level.",
             fontsize=15.5, color="#102A43", weight="bold")

    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"Figure1_age_mirage.{ext}", dpi=320 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
