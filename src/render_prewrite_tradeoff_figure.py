#!/usr/bin/env python3
"""Render an exact, source-hashed trade-off figure from the frozen ledger."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts/prewriting_evidence_ledger_20260909_v1"
OUT = ROOT / "artifacts/figures_prewrite_v1"
PRIMARY = ("branch_density", "endpoint_density", "occupancy_entropy")
LABELS = {
    "branch_density": "Branch density",
    "endpoint_density": "Endpoint density",
    "occupancy_entropy": "Occupancy entropy",
    "vessel_dice": "Vessel Dice",
    "thin_centerline_recall": "Thin centerline recall",
    "cldice": "clDice",
}
GATE_COLORS = {
    "concordant_relative_improvement": "#8BCF8B",
    "mixed_tradeoff": "#E9C46A",
    "not_improved": "#F6CFCB",
}
GATE_SHORT = {
    "concordant_relative_improvement": "concordant\nimprovement",
    "mixed_tradeoff": "mixed\ntrade-off",
    "not_improved": "not\nimproved",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows() -> list[dict[str, str]]:
    with (LEDGER / "MASTER_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[dict[str, str]], **required: str) -> dict[str, str]:
    found = [row for row in rows if all(row[key] == value for key, value in required.items())]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one row for {required}; found {len(found)}")
    return found[0]


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure package: {OUT}")
    snapshot = json.loads((LEDGER / "snapshot_manifest.json").read_text(encoding="utf-8"))
    if snapshot["snapshot_id"] != "RMDB-prewriting-evidence-ledger-20260909-v1":
        raise ValueError("Unexpected ledger snapshot")
    for name, record in snapshot["records"].items():
        if sha256(LEDGER / name) != record["sha256"]:
            raise ValueError(f"Ledger input hash mismatch: {name}")
    rows = load_rows()

    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.3, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.72])
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[0, 1])
    c = fig.add_subplot(grid[0, 2])

    # A: Historical low-resolution repeatability is included only as a control.
    x = np.arange(len(PRIMARY))
    raw = [float(one(rows, experiment_id="historical_mbrset_test_repeatability", endpoint=endpoint, statistic="icc")["raw_vascx_value"]) for endpoint in PRIMARY]
    joint = [float(one(rows, experiment_id="historical_mbrset_test_repeatability", endpoint=endpoint, statistic="icc")["joint_readout_value"]) for endpoint in PRIMARY]
    width = 0.35
    a.bar(x - width / 2, raw, width, color="#B64342", label="Raw VascX")
    a.bar(x + width / 2, joint, width, color="#0F4D92", label="Joint readout")
    a.set_ylim(0, 1.0)
    a.set_xticks(x, ["Branch", "Endpoint", "Entropy"])
    a.set_ylabel("ICC(1,1)")
    a.set_title("A  Historical same-eye repeatability\n(mBRSET test; 475 repeated eyes)", loc="left", fontweight="bold")
    for idx, value in enumerate(joint):
        a.text(idx + width / 2, value + 0.025, f"Δ {value - raw[idx]:+.3f}", ha="center", va="bottom", fontsize=8)
    a.legend(frameon=False, loc="upper left", fontsize=8)
    a.text(0.5, -0.30, "Public 224px input; historical control only", transform=a.transAxes, ha="center", va="top", fontsize=7.7, color="#4D4D4D")
    style_axis(a)

    # B: labelled structure deliberately shows why this is not a VascX replacement claim.
    structural = ("vessel_dice", "thin_centerline_recall", "cldice")
    vals, low, high = [], [], []
    for endpoint in structural:
        item = one(rows, experiment_id="historical_fundus_avseg_reference", endpoint=endpoint)
        vals.append(float(item["effect_value"]))
        low.append(float(item["effect_ci_low"]))
        high.append(float(item["effect_ci_high"]))
    yerr = np.array([np.array(vals) - np.array(low), np.array(high) - np.array(vals)])
    b.barh(np.arange(len(structural)), vals, xerr=yerr, color="#B64342", capsize=3)
    b.axvline(0, color="#4D4D4D", linewidth=1.0)
    b.set_yticks(np.arange(len(structural)), [LABELS[key] for key in structural])
    b.invert_yaxis()
    b.set_xlabel("Joint readout − raw VascX")
    b.set_title("B  Labelled structural control\n(Fundus-AVSeg; n=100)", loc="left", fontweight="bold")
    b.set_xlim(-0.047, 0.008)
    for idx, value in enumerate(vals):
        b.text(value - 0.0011, idx, f"{value:+.3f}", va="center", ha="right", fontsize=8, color="#4D4D4D")
    b.text(0.5, -0.30, "All three reference metrics decline for joint", transform=b.transAxes, ha="center", va="top", fontsize=7.7, color="#4D4D4D")
    style_axis(b)

    # C: fresh HRF evidence makes the endpoint dependency visible, rather than averaging it away.
    experiments = (
        ("hrf_quality_stress", "Quality\nstress"),
        ("hrf_resolution_native_vs_75pct", "Native vs\n75%"),
        ("hrf_resolution_native_vs_50pct", "Native vs\n50%"),
    )
    c.set_xlim(0, len(experiments))
    c.set_ylim(0, len(PRIMARY))
    c.set_xticks(np.arange(len(experiments)) + 0.5, [label for _, label in experiments])
    c.set_yticks(np.arange(len(PRIMARY)) + 0.5, [LABELS[key] for key in PRIMARY])
    c.invert_yaxis()
    c.tick_params(axis="both", length=0)
    for side in c.spines.values():
        side.set_visible(False)
    for col, (experiment, _) in enumerate(experiments):
        for row, endpoint in enumerate(PRIMARY):
            icc = one(rows, experiment_id=experiment, endpoint=endpoint, statistic="icc")
            mad = one(rows, experiment_id=experiment, endpoint=endpoint, statistic="within_eye_mad")
            label = icc["gate_label"]
            patch = FancyBboxPatch((col + 0.04, row + 0.07), 0.92, 0.86, boxstyle="round,pad=0.02,rounding_size=0.035", linewidth=1.0, edgecolor="white", facecolor=GATE_COLORS[label])
            c.add_patch(patch)
            c.text(col + 0.5, row + 0.30, GATE_SHORT[label], ha="center", va="center", fontsize=8.1, fontweight="bold")
            c.text(col + 0.5, row + 0.63, f"ΔICC {float(icc['effect_value']):+.3f}\nMAD r {float(mad['effect_value']):.2f}", ha="center", va="center", fontsize=7.4)
    c.set_title("C  Fresh high-resolution probes (HRF IQA; 18 pairs/source eyes)", loc="left", fontweight="bold")
    c.text(0.0, -0.27, "ΔICC = joint − raw VascX; MAD r = joint/raw. Gate requires ICC together with MAD, SEM, and LoA (98.333% simultaneous bootstrap CI).", transform=c.transAxes, ha="left", va="top", fontsize=7.6, color="#4D4D4D")
    c.legend(handles=[
        Patch(facecolor=GATE_COLORS["concordant_relative_improvement"], label="Concordant relative improvement"),
        Patch(facecolor=GATE_COLORS["mixed_tradeoff"], label="Mixed trade-off"),
        Patch(facecolor=GATE_COLORS["not_improved"], label="Not improved"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.43), ncol=3, fontsize=7.4)

    fig.suptitle("Readout effects on frozen VascX: improvements are endpoint- and condition-dependent", x=0.01, ha="left", fontsize=13, fontweight="bold")
    OUT.mkdir(parents=True)
    base = OUT / "Figure_1_VascX_readout_tradeoffs"
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (OUT / "README.md").write_text(
        "# Figure 1: frozen VascX readout trade-offs\n\n"
        "Generated only from `artifacts/prewriting_evidence_ledger_20260909_v1/MASTER_RESULTS.csv`. "
        "Panel A is historical 224px mBRSET repeatability; Panel B is labelled structural control; Panel C is the fresh HRF high-resolution evidence. "
        "The figure does not imply cross-camera validation, clinical interchangeability, or VascX replacement.\n",
        encoding="utf-8",
    )
    (OUT / "figure_manifest.json").write_text(json.dumps({
        "figure_id": "RMDB-prewriting-figure-1-v1",
        "source_ledger_snapshot": snapshot["snapshot_id"],
        "source_ledger_manifest_sha256": sha256(LEDGER / "snapshot_manifest.json"),
        "source_results_sha256": sha256(LEDGER / "MASTER_RESULTS.csv"),
        "formats": ["png", "pdf", "svg"],
        "numeric_source_only": True,
        "interpretation_boundary": "No cross-camera, clinical interchangeability, or VascX-replacement claim.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(base)


if __name__ == "__main__":
    main()
