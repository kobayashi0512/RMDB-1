#!/usr/bin/env python3
"""Render RMDB Figure 2 exclusively from locked local evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts/cimt_dual_readout_development/outer_cv/metrics.json"
COND = ROOT / "artifacts/cimt_dual_readout_development/normative_e7/metrics.json"
OUT = ROOT / "figures/rmdb_age_mirage/output"

NAVY = "#16324F"
SLATE = "#637487"
ORANGE = "#E38B32"
VIOLET = "#7B61A8"
VIOLET_LIGHT = "#EEE8F6"
GRAY = "#68737D"
GRID = "#E5EAF0"
RED = "#C74C3C"


def panel_box(fig, rect, title, subtitle, edge="#C7D1DC", face="#FBFCFD"):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    patch = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                           boxstyle="round,pad=0.014,rounding_size=0.035",
                           facecolor=face, edgecolor=edge, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(0.045, 0.955, title, va="top", transform=ax.transAxes,
            fontsize=19, weight="bold", color=NAVY)
    ax.text(0.045, 0.885, subtitle, va="top", transform=ax.transAxes,
            fontsize=13.2, color=GRAY)
    return ax


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(RAW.read_text())
    conditioned = json.loads(COND.read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(18, 8.5), facecolor="white")
    fig.text(0.035, 0.978, "Performance reversal after conditioning cIMT on age and sex",
             fontsize=27, weight="bold", color="#102A43", va="top")
    fig.text(0.035, 0.934,
             "The raw endpoint rewards age-linked information; the age-conditioned endpoint tests the remaining, clinically stricter signal.",
             fontsize=15, color=GRAY, va="top")

    # A. Raw endpoint AUROC
    panel_box(fig, [0.035, 0.19, 0.355, 0.66],
              "A  Conventional raw-cIMT endpoint",
              "OOF AUROC, 2,603 patients; locked five-fold outer CV")
    a = fig.add_axes([0.068, 0.29, 0.290, 0.44])
    labels = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    keys = ["E0_clinical", "E1_legacy_vascx", "E2_mass_udct", "E3_factorized", "E4_dual_readout", "E5_without_zone_b", "E6_without_uncertainty"]
    vals = [raw["overall"][k]["auroc"] for k in keys]
    colors = [SLATE, "#A8B4C0", "#A8B4C0", "#A8B4C0", ORANGE, "#A8B4C0", "#A8B4C0"]
    x = np.arange(len(vals))
    a.bar(x, vals, color=colors, width=0.68, edgecolor="white", linewidth=1)
    for xi, v, c in zip(x, vals, colors):
        a.text(xi, v + .0023, f"{v:.3f}", ha="center", va="bottom", fontsize=12.4,
               color=NAVY if c == SLATE else ("#8A4A11" if c == ORANGE else GRAY),
               weight="bold" if c in (SLATE, ORANGE) else "normal")
    a.axhline(0.5, color="#AAB5C1", linestyle="--", linewidth=1.1)
    a.set_ylim(.80, .87)
    a.set_ylabel("AUROC", fontsize=15)
    a.set_xticks(x, labels, fontsize=13)
    a.tick_params(axis="y", labelsize=12.5)
    a.grid(axis="y", color=GRID, linewidth=.9)
    for s in ["top", "right"]: a.spines[s].set_visible(False)
    a.spines["left"].set_color("#B7C2CD"); a.spines["bottom"].set_color("#B7C2CD")
    a.text(.50, -.14, "E0: clinical baseline     E4: full dual-readout model",
           transform=a.transAxes, ha="center", va="center", fontsize=12.7, color=GRAY)

    # B. paired deltas
    panel_box(fig, [0.415, 0.19, 0.265, 0.66],
              "B  Paired change vs E0",
              "Δ AUROC versus E0; paired bootstrap 95% CI")
    b = fig.add_axes([0.46, 0.35, 0.175, 0.355])
    pair_keys = ["E1_legacy_vascx__minus__E0_clinical", "E2_mass_udct__minus__E0_clinical",
                 "E3_factorized__minus__E0_clinical", "E4_dual_readout__minus__E0_clinical"]
    names = ["Legacy", "Mass", "Structure", "Dual"]
    y = np.arange(4)[::-1]
    for yi, key, name in zip(y, pair_keys, names):
        entry = raw["paired_bootstrap"][key]["auroc"]
        point = entry["mean_delta"]; lo, hi = entry["ci95"]
        color = ORANGE if name == "Dual" else NAVY
        b.errorbar(point, yi, xerr=[[point-lo], [hi-point]], fmt="o", color=color,
                   ecolor=color, elinewidth=2.4, capsize=5, markersize=8, zorder=3)
        b.text(0.0058, yi, f"{point:+.3f}", fontsize=12.1, va="center", color=color,
               weight="bold" if name == "Dual" else "normal")
    b.axvline(0, color="#8E9BA8", linewidth=1.4)
    b.set_xlim(-.017, .010)
    b.set_ylim(-.65, 3.65)
    b.set_yticks(y, names, fontsize=13.5)
    b.set_xlabel("Δ AUROC", fontsize=15)
    b.tick_params(axis="x", labelsize=12)
    b.grid(axis="x", color=GRID, linewidth=.9)
    for s in ["top", "right", "left"]: b.spines[s].set_visible(False)
    b.spines["bottom"].set_color("#B7C2CD")
    fig.text(0.547, 0.270, "E4 dual readout: −0.0076", fontsize=12.2,
             ha="center", color=RED, weight="bold")
    fig.text(0.547, 0.245, "95% CI [−0.0143, −0.0011]", fontsize=11.5,
             ha="center", color=RED, weight="bold")

    # C. age-conditioned endpoint
    panel_box(fig, [0.705, 0.19, 0.260, 0.66],
              "C  Age-conditioned endpoint",
              "AUROC after conditioning cIMT on age + sex; distinct target")
    c = fig.add_axes([0.748, 0.35, 0.170, 0.355])
    ckeys = ["E7A_core_residual", "E7B_core_z", "E7C_means_only_z", "E7D_mass_z", "E7E_factorized_z"]
    cnames = ["Core residual", "Core z", "Means only", "Mass z", "Structure z"]
    cvals = [conditioned["overall"][k]["accelerated_auroc"] for k in ckeys]
    y = np.arange(len(ckeys))[::-1]
    for yi, key, v in zip(y, ckeys, cvals):
        ci = conditioned["paired_vs_normative"][key]["accelerated_auroc"]["ci95"]
        color = VIOLET if key == "E7D_mass_z" else "#A795C5"
        c.errorbar(v, yi, xerr=[[v-ci[0]], [ci[1]-v]], fmt="o", color=color,
                   ecolor=color, elinewidth=2.3, capsize=5, markersize=8, zorder=3)
        c.text(.570, yi, f"{v:.3f}", fontsize=11.8, va="center", color=color,
               weight="bold" if key == "E7D_mass_z" else "normal")
    c.axvline(.5, color="#8E9BA8", linestyle="--", linewidth=1.4)
    c.set_xlim(.49, .585)
    c.set_ylim(-.65, 4.65)
    c.set_yticks(y, cnames, fontsize=12.4)
    c.set_xlabel("AUROC", fontsize=15)
    c.tick_params(axis="x", labelsize=12)
    c.grid(axis="x", color=GRID, linewidth=.9)
    for s in ["top", "right", "left"]: c.spines[s].set_visible(False)
    c.spines["bottom"].set_color("#B7C2CD")
    fig.text(0.835, 0.255, "Different endpoint: remaining signal is close to chance.", fontsize=11.7,
             ha="center", color=VIOLET, weight="bold")

    callout = FancyBboxPatch((0.055, .075), .89, .075, transform=fig.transFigure,
                              boxstyle="round,pad=0.012,rounding_size=.02", facecolor=VIOLET_LIGHT,
                              edgecolor="#CBBEDE", linewidth=1.3)
    fig.add_artist(callout)
    fig.text(.5, .112,
             "Interpretation: strong raw-endpoint discrimination does not, by itself, demonstrate retinal information beyond age and sex.",
             ha="center", va="center", fontsize=15.2, color="#4D3B6E", weight="bold")
    fig.text(.5, .038,
             "Panels A-B and Panel C deliberately use separate axes because raw cIMT and age-conditioned accelerated cIMT are different endpoints.",
             ha="center", va="center", fontsize=13.2, color=GRAY)

    for ext in ("svg", "png", "pdf"):
        fig.savefig(OUT / f"Figure2_performance_reversal.{ext}", dpi=320 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
