#!/usr/bin/env python3
"""Render exact post-optimization evidence figures and audit locked outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"R0_clinical": "#64748B", "R2_reliable_core": "#0F766E"}


def save(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=320, bbox_inches="tight")


def render_permutation(output: Path) -> None:
    base = ROOT / "artifacts/mbrset_external/permutation_negative_control"
    null = pd.read_csv(base / "permutation_auroc.csv")
    report = json.loads((base / "metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    for axis, profile in zip(axes, ["R0_clinical", "R2_reliable_core"]):
        values = null.loc[null.profile.eq(profile), "auroc"]
        observed = report["summary"][profile]["observed_auroc"]
        axis.hist(values, bins=22, color="#D8E7F0", edgecolor="white", density=True)
        axis.axvline(observed, color=COLORS[profile], lw=2.6, label=f"Observed {observed:.3f}")
        axis.axvline(.5, color="#475569", lw=1.1, ls="--", label="Chance")
        axis.set(title=("Clinical baseline" if profile == "R0_clinical" else "Clinical + reliable core"),
                 xlabel="Nested-CV AUROC after patient-level label permutation", ylabel="Density")
        axis.legend(frameon=False, loc="lower right")
        axis.grid(axis="y", alpha=.18)
        p = report["summary"][profile]["empirical_p_one_sided"]
        axis.text(.02, .95, f"200 permutations\nempirical p={p:.3f}\nouter-fold patient overlap=0",
                  transform=axis.transAxes, va="top", fontsize=10,
                  bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#CBD5E1"})
    fig.suptitle("Patient-level permutation negative control", fontsize=15, fontweight="bold")
    save(fig, output, "supplementary_patient_permutation_negative_control")
    plt.close(fig)


def render_qc_transport(output: Path) -> None:
    qc = pd.read_csv(ROOT / "artifacts/fundus_avseg_external_validation/blind_qc_gate/qc_gate_summary.csv")
    transport = pd.read_csv(ROOT / "artifacts/quality_overlap_transport_rerun/overlap_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    labels = {"vascx": "VascX", "mass_udct": "Mass-UDCT", "factorized_mass_udct": "Gated factorized"}
    mass = qc.loc[qc.metric.eq("mean_av_dice")].copy()
    x = np.arange(len(mass))
    axes[0].bar(x, mass.all_images, color=["#64748B", "#0F766E", "#2563EB"], width=.62)
    axes[0].set(xticks=x, xticklabels=[labels[value] for value in mass.model], ylabel="Mean A/V Dice", title="Frozen external measurement performance")
    axes[0].set_ylim(.45, .60)
    for index, row in mass.reset_index(drop=True).iterrows():
        axes[0].text(index, row.all_images + .004, f"{row.all_images:.3f}", ha="center", fontsize=10)
    axes[0].text(.02, .95, "Predefined CIMT-train QC gate retained 100/100 images\n→ no externally valid threshold separation", transform=axes[0].transAxes, va="top", fontsize=9.3,
                 bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#CBD5E1"})
    cohorts = transport.cohort.tolist(); x = np.arange(len(cohorts)); width=.36
    axes[1].bar(x-width/2, transport.min_ess_fraction, width, label="Min. effective-sample fraction", color="#7DD3C7")
    axes[1].bar(x+width/2, transport.max_abs_smd_after, width, label="Max. |SMD| after weighting", color="#FB9A99")
    axes[1].axhline(.20, color="#0F766E", lw=1.1, ls="--", label="Overlap target")
    axes[1].axhline(.10, color="#B91C1C", lw=1.1, ls=":", label="Balance target")
    axes[1].set(xticks=x, xticklabels=cohorts, ylabel="Value", title="Measured-QC overlap remains inadequate")
    axes[1].legend(frameon=False, fontsize=8.1, ncol=2, loc="upper right")
    axes[1].grid(axis="y", alpha=.18)
    fig.suptitle("Quality constraints limit, rather than repair, cross-cohort transport", fontsize=14.5, fontweight="bold")
    save(fig, output, "supplementary_quality_gate_and_transport")
    plt.close(fig)


def audit(output: Path) -> None:
    checks = {
        "fundus_qc_gate": (ROOT / "artifacts/fundus_avseg_external_validation/blind_qc_gate/metrics.json", 100),
        "mbrset_permutation": (ROOT / "artifacts/mbrset_external/permutation_negative_control/metrics.json", 1276),
        "e12": (ROOT / "artifacts/e12_repeatability_gated_centile_full/metrics.json", 2603),
        "qc_transport": (ROOT / "artifacts/quality_overlap_transport_rerun/metadata.json", None),
    }
    rows = []
    for name, (path, expected) in checks.items():
        data = json.loads(path.read_text())
        n = data.get("n_images", data.get("patients"))
        rows.append({"artifact": name, "path": str(path.relative_to(ROOT)), "exists": path.is_file(),
                     "expected_n": expected, "observed_n": n, "count_pass": expected is None or expected == n,
                     "cimt_test_used": data.get("cimt_test_used", data.get("test_used", data.get("raw_cimt_validation_or_test_read", False)))})
    frame = pd.DataFrame(rows)
    if not frame.exists.all() or not frame.count_pass.all() or frame.cimt_test_used.astype(bool).any():
        raise RuntimeError("Post-optimization integrity audit failed.")
    frame.to_csv(output / "postoptimization_integrity_audit.csv", index=False)
    (output / "postoptimization_integrity_audit.json").write_text(json.dumps({"checks": rows, "complete": True}, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manuscript_completion/08_postoptimization_figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold", "figure.facecolor": "white"})
    render_permutation(args.output)
    render_qc_transport(args.output)
    audit(args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
