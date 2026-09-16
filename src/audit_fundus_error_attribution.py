#!/usr/bin/env python3
"""Outcome-blind error-attribution audit of the already locked Fundus-AVSeg metrics.

This is a post-hoc diagnostic only. It does not fit thresholds, select a
checkpoint, read CIMT data, or change any model output. The support and
identity descriptors are deliberately reported as gap descriptors rather
than causal percentages.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/birap_v2_fundus_avseg/per_image_metrics_with_metadata.csv"
OUT = ROOT / "artifacts/fundus_avseg_error_attribution"


def main() -> None:
    frame = pd.read_csv(INPUT)
    OUT.mkdir(parents=True, exist_ok=True)
    # Fixed reference and two readouts already produced by the locked audit.
    baseline = frame.loc[frame.model.eq("vascx")].copy()
    mass = frame.loc[frame.model.eq("birap_v2_mass")].copy()
    final = frame.loc[frame.model.eq("birap_v2_final")].copy()

    baseline["support_gap_descriptor"] = 1.0 - baseline.vessel_dice
    baseline["identity_gap_descriptor"] = baseline.vessel_dice - baseline.mean_av_dice
    baseline["artery_vein_gap"] = baseline.artery_dice - baseline.vein_dice
    baseline.to_csv(OUT / "per_image_gap_descriptors.csv", index=False)

    metrics = ["vessel_dice", "artery_dice", "vein_dice", "mean_av_dice", "type_accuracy", "brier"]
    summary = {
        "status": "superseded_due_to_missing_annotation_crop_transform",
        "correction": "../fundus_coordinate_alignment_audit/summary.json",
        "protocol": "post_hoc_error_attribution_of_locked_metrics",
        "input": str(INPUT),
        "images": int(baseline.id.nunique()),
        "labels_used_for_selection": False,
        "cimt_used": False,
        "outcome_driven_tuning": False,
        "causal_percentage_claim": False,
        "note": "support_gap_descriptor and identity_gap_descriptor are diagnostic descriptors; interacting errors prevent additive attribution from aggregate Dice.",
        "overall": {},
        "disease_strata": {},
        "quality_strata": {},
        "paired_deltas_mass_minus_vascx": {},
        "paired_deltas_final_minus_vascx": {},
    }
    for name, data in (("vascx", baseline), ("birap_v2_mass", mass), ("birap_v2_final", final)):
        summary["overall"][name] = {m: float(data[m].mean()) for m in metrics}
    for column, destination in (("disease type", "disease_strata"), ("image quality", "quality_strata")):
        for level, group in baseline.groupby(column, sort=True):
            ids = group["id"].astype(str).tolist()
            base_group = group[["id"] + metrics]
            mass_group = mass[mass.id.astype(str).isin(ids)][["id"] + metrics]
            final_group = final[final.id.astype(str).isin(ids)][["id"] + metrics]
            mass_join = base_group.merge(mass_group, on="id", suffixes=("_base", "_mass"))
            final_join = base_group.merge(final_group, on="id", suffixes=("_base", "_final"))
            summary[destination][str(level)] = {
                "n": int(len(group)),
                "vascx": {m: float(group[m].mean()) for m in metrics},
                "mass_minus_vascx": {
                    m: float((mass_join[f"{m}_mass"] - mass_join[f"{m}_base"]).mean())
                    for m in metrics
                },
                "final_minus_vascx": {
                    m: float((final_join[f"{m}_final"] - final_join[f"{m}_base"]).mean())
                    for m in metrics
                },
            }
    for challenger, data, key in (("birap_v2_mass", mass, "paired_deltas_mass_minus_vascx"), ("birap_v2_final", final, "paired_deltas_final_minus_vascx")):
        for m in metrics:
            merged = baseline[["id", m]].merge(data[["id", m]], on="id", suffixes=("_base", "_challenger"))
            delta = merged[f"{m}_challenger"] - merged[f"{m}_base"]
            summary[key][m] = {
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "images_improved": int((delta > 0).sum()),
                "images_total": int(len(delta)),
            }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Fundus-AVSeg error-attribution audit",
        "> SUPERSEDED: historical scores omit the annotation crop transform. Use ../fundus_coordinate_alignment_audit/summary.json. The historical interpretation below is invalidated.",
        "",
        "This is a post-hoc diagnostic of already locked predictions. It did not fit thresholds, select a checkpoint, read CIMT data, or tune on the labels.",
        "",
        "## Overall metrics",
        "",
        "| Readout | Vessel Dice | Artery Dice | Vein Dice | Mean A/V Dice | Type accuracy | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("vascx", "VascX"), ("birap_v2_mass", "Mass readout"), ("birap_v2_final", "Final readout")):
        s = summary["overall"][name]
        lines.append(f"| {label} | {s['vessel_dice']:.4f} | {s['artery_dice']:.4f} | {s['vein_dice']:.4f} | {s['mean_av_dice']:.4f} | {s['type_accuracy']:.4f} | {s['brier']:.4f} |")
    lines += [
        "",
        "## What the gap descriptors mean",
        "",
        "For each VascX image, `support_gap_descriptor = 1 − vessel Dice` describes the remaining vessel-support error, while `identity_gap_descriptor = vessel Dice − mean A/V Dice` describes the additional loss after restricting evaluation to artery and vein identity. These are not additive causal percentages because support, identity, boundary, crossing, and domain errors interact.",
        "",
        "## Disease and quality strata",
        "",
        "The complete machine-readable strata are in `summary.json`. The largest fixed strata differences are retained as failure-domain evidence; no subgroup was removed.",
        "",
        "## Interpretation",
        "",
        "The audit is consistent with support/domain error being the dominant observable limitation: vessel Dice is substantially higher than mean A/V Dice, and the low-quality and diabetic-retinopathy strata are lower than normal or high-quality strata. The mass readout provides a small but consistent improvement, while the final routed readout is smaller. This supports an evidence-bound conclusion rather than a claim of a large segmentation gain.",
        "",
        "A true causal decomposition would require fixed oracle replacements (ground-truth support with predicted identity, predicted support with ground-truth identity, boundary-tolerant scoring, and crossing exclusion) and should be treated as a diagnostic extension, not a new model-selection loop.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "images": summary["images"]}, indent=2))


if __name__ == "__main__":
    main()
