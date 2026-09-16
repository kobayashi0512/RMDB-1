#!/usr/bin/env python3
"""Render RMDB Figure 3 with real local bilateral inputs and stored model outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/rmdb_age_mirage/output"
PID = "10323005"
RGB = ROOT / "artifacts/vascx_anatomy/preprocessed_rgb"
OCC = ROOT / "artifacts/cimt_dual_readout_development/occupancy"
LOGITS = ROOT / "artifacts/cimt_dual_readout_development/vascx_logits"
DISC = ROOT / "artifacts/vascx_anatomy/disc"
PATIENTS = ROOT / "artifacts/cimt_dual_readout_development/phenotypes_v2/per_patient.csv"

NAVY = "#16324F"
TEAL = "#168C83"
TEAL_FILL = "#DDF3EF"
RED = "#D85745"
BLUE = "#3C75B6"
AMBER = "#D9932D"
AMBER_FILL = "#FCF0DA"
GRAY = "#68737D"
GRID = "#C9D2DB"


def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def resize(a, size=380, mode="bilinear"):
    if a.ndim == 2:
        im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255))
        return np.asarray(im.resize((size, size), Image.Resampling.BILINEAR)) / 255.0
    im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255))
    return np.asarray(im.resize((size, size), Image.Resampling.BILINEAR)) / 255.0


def rgb_image(side):
    return np.asarray(Image.open(RGB / f"{PID}_{side}.png").convert("RGB")) / 255.0


def av_map(side):
    p = softmax(np.load(LOGITS / f"{PID}_{side}.npy").astype(np.float32))
    artery = p[..., 1] + p[..., 3]
    vein = p[..., 2] + p[..., 3]
    # The output is a probability visualization, not a hard segmentation.
    out = np.zeros((*artery.shape, 3), dtype=float)
    out[..., 0] = artery
    out[..., 1] = 0.12 * (artery + vein)
    out[..., 2] = vein
    return resize(np.clip(out * 1.55, 0, 1))


def occupancy_map(side):
    occ = np.load(OCC / f"{PID}_{side}.npy").astype(np.float32)
    return resize(occ)


def disc_circle(side, size=380):
    d = np.asarray(Image.open(DISC / f"{PID}_{side}.png").convert("L"))
    ys, xs = np.where(d > 100)
    if len(xs) < 8:
        return None
    sx, sy = size / d.shape[1], size / d.shape[0]
    cx, cy = xs.mean() * sx, ys.mean() * sy
    radius = np.sqrt(len(xs) / np.pi) * (sx + sy) / 2
    return cx, cy, radius


def panel(ax, title, subtitle, edge, face="#FAFCFD"):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                                boxstyle="round,pad=0.015,rounding_size=.03",
                                facecolor=face, edgecolor=edge, linewidth=1.6))
    ax.text(.045, .955, title, transform=ax.transAxes, va="top", fontsize=18.5,
            weight="bold", color=NAVY)
    ax.text(.045, .885, subtitle, transform=ax.transAxes, va="top", fontsize=13.1,
            color=GRAY)


def image_axis(fig, rect, image, title, cmap=None):
    ax = fig.add_axes(rect)
    ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(NAVY); s.set_linewidth(1.25)
    ax.set_title(title, fontsize=13.2, pad=5, color=NAVY, weight="bold")
    return ax


def box(ax, xy, w, h, title, body, face, edge):
    ax.add_patch(FancyBboxPatch(xy, w, h, transform=ax.transAxes,
                                boxstyle="round,pad=.014,rounding_size=.025",
                                facecolor=face, edgecolor=edge, linewidth=1.25))
    ax.text(xy[0]+w/2, xy[1]+h*.66, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=13.3, weight="bold", color=NAVY)
    ax.text(xy[0]+w/2, xy[1]+h*.30, body, transform=ax.transAxes, ha="center", va="center",
            fontsize=11.4, color=GRAY)


def fig_arrow(fig, start, end, color=NAVY):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=18, linewidth=2.0,
                                   color=color, shrinkA=4, shrinkB=4))


def patient_row():
    with PATIENTS.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["patient_id"] == PID:
                return r
    raise RuntimeError("Figure 3 patient missing from per_patient.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    row = patient_row()
    left, right = rgb_image("L"), rgb_image("R")
    left_av, right_av = av_map("L"), av_map("R")
    left_occ, right_occ = occupancy_map("L"), occupancy_map("R")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(21, 9.2), facecolor="white")
    fig.text(.03, .978, "RMDB: real bilateral retinal images to an auditable patient-level phenotype",
             fontsize=27, weight="bold", color="#102A43", va="top")
    fig.text(.03, .935,
             "The pipeline keeps complementary image evidence, anatomy, uncertainty and bilateral discordance instead of reducing each participant to one opaque score.",
             fontsize=15, color=GRAY, va="top")

    # four clear stage panels
    ax_a = fig.add_axes([.03, .18, .225, .66]); panel(ax_a, "A  Real bilateral inputs", "Illustrative train participant; distinct left/right eyes", "#9AAFC6")
    ax_b = fig.add_axes([.275, .18, .285, .66]); panel(ax_b, "B  Complementary readouts", "Real stored outputs; identity and occupancy are retained", "#A4CFC9", "#F8FCFB")
    ax_c = fig.add_axes([.580, .18, .190, .66]); panel(ax_c, "C  Anatomical reference", "Optic-disc reference anchors Zone B", "#E2C07A", "#FFFCF7")
    ax_d = fig.add_axes([.790, .18, .180, .66]); panel(ax_d, "D  Patient phenotype", "Explicit aggregation and provenance", "#BBA8D5", "#FBFAFD")

    # A images
    image_axis(fig, [.052, .422, .078, .262], left, "Left eye")
    image_axis(fig, [.155, .422, .078, .262], right, "Right eye")
    ax_a.text(.50, .332, "Two eyes are retained separately\nbefore patient-level aggregation.", transform=ax_a.transAxes,
              ha="center", va="center", fontsize=13.0, color=GRAY)
    ax_a.text(.50, .145, "Input quality and\noptic-disc status\nremain auditable", transform=ax_a.transAxes,
              ha="center", va="center", fontsize=14, color=NAVY, weight="bold",
              bbox={"boxstyle":"round,pad=.45", "facecolor":"#EAF0F6", "edgecolor":"#AABBD0"})

    # B actual maps (L top, R bottom)
    image_axis(fig, [.303, .475, .100, .235], left_av, "")
    image_axis(fig, [.425, .475, .100, .235], left_occ, "", cmap="Greens")
    image_axis(fig, [.303, .235, .100, .235], right_av, "")
    image_axis(fig, [.425, .235, .100, .235], right_occ, "", cmap="Greens")
    ax_b.text(.275, .765, "A/V identity", transform=ax_b.transAxes,
              ha="center", fontsize=13.4, color=NAVY, weight="bold")
    ax_b.text(.705, .765, "Vessel occupancy", transform=ax_b.transAxes,
              ha="center", fontsize=13.4, color=TEAL, weight="bold")
    ax_b.text(.03, .630, "LEFT", transform=ax_b.transAxes, ha="left", fontsize=11.6, color=GRAY, weight="bold")
    ax_b.text(.03, .270, "RIGHT", transform=ax_b.transAxes, ha="left", fontsize=11.6, color=GRAY, weight="bold")

    # C actual disc reference and measurements
    cimg_ax = image_axis(fig, [.604, .465, .135, .245], left, "Actual disc reference (left eye)")
    circ = disc_circle("L", size=left.shape[0])
    if circ:
        cx, cy, radius = circ
        cimg_ax.add_patch(Circle((cx, cy), radius * 2.0, edgecolor="#44B7B2", facecolor="none", linewidth=2.4))
        cimg_ax.text(.03, .05, "Zone B", transform=cimg_ax.transAxes, fontsize=12, color="#117C77", weight="bold",
                     bbox={"facecolor":"white", "edgecolor":"#7ECFC9", "boxstyle":"round,pad=.2"})
    box(ax_c, (.08, .250), .84, .105, "Vessel mass", "soft vessel / artery / vein fractions", TEAL_FILL, TEAL)
    box(ax_c, (.08, .125), .84, .105, "A/V + topology", "ratio, skeleton, endpoints, branches", "#EEF3F8", "#97ABC1")
    ax_c.text(.5, .078, "Zone B remains unavailable\nwhen disc localization fails.", transform=ax_c.transAxes,
              ha="center", va="center", fontsize=11.8, color=AMBER, weight="bold")

    # D retained phenotype schematic based on actual row values
    box(ax_d, (.09, .605), .82, .125, "Within-eye evidence", "mass + A/V + topology\n+ uncertainty", "#EEF3F8", "#97ABC1")
    box(ax_d, (.09, .415), .82, .125, "Bilateral aggregation", "mean state + |L − R|", "#EEE8F6", "#BBA8D5")
    box(ax_d, (.09, .220), .82, .135, "Locked E4 vector", "104 variables including\nage + sex", "#F6F0FB", VIOLET if False else "#BBA8D5")
    ax_d.text(.50, .115,
              f"Example retained\nmean occupancy entropy: {float(row['occupancy_entropy__bilateral_mean']):.3f}\n|L−R|: {float(row['occupancy_entropy__absolute_difference']):.3f}",
              transform=ax_d.transAxes, ha="center", va="center", fontsize=11.2, color=GRAY)
    for y0, y1 in [(.605, .540), (.415, .350)]:
        ax_d.add_patch(FancyArrowPatch((.50, y0), (.50, y1), transform=ax_d.transAxes,
                                       arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=NAVY))

    # stage arrows are horizontal and have their own whitespace lane
    fig_arrow(fig, (.257, .51), (.273, .51), NAVY)
    fig_arrow(fig, (.562, .51), (.578, .51), TEAL)
    fig_arrow(fig, (.772, .51), (.788, .51), AMBER)

    fig.text(.50, .098,
             "Grey labels specify what each output means; every displayed image and map comes from the same local participant and stored pipeline output.",
             ha="center", fontsize=13.5, color=GRAY)
    fig.text(.50, .055,
             "Illustrative processing display only — not an individual diagnostic explanation and not independent external clinical validation.",
             ha="center", fontsize=13.8, weight="bold", color="#4D3B6E")

    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"Figure3_rmdb_real_image_pipeline.{ext}", dpi=320 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
