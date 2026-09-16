#!/usr/bin/env python3
"""Evaluate a pre-specified, label-blind Fundus-AVSeg measurement QC gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260810
RELIABLE = [
    "soft_vessel_mass_fraction", "mass_vessel_skeleton_length_density",
    "factorized_vessel_skeleton_length_density", "vessel_occupancy_disagreement",
]
METRICS = ["vessel_dice", "artery_dice", "vein_dice", "mean_av_dice", "type_accuracy", "av_ratio_log_error"]


def bootstrap_delta(frame: pd.DataFrame, metric: str, repetitions: int, rng: np.random.Generator) -> dict[str, object]:
    values = frame[metric].to_numpy(float)
    selected = frame.qc_gate.to_numpy(bool)
    point = float(values[selected].mean() - values.mean())
    samples = []
    for _ in range(repetitions):
        take = rng.integers(0, len(frame), len(frame))
        gate = selected[take]
        if gate.sum() >= 5:
            samples.append(float(values[take][gate].mean() - values[take].mean()))
    return {"point": point, "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cimt-per-eye", type=Path, required=True)
    parser.add_argument("--fundus-per-eye", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Frozen image manifest used only to map internal IDs to source image stems.")
    parser.add_argument("--frozen-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    reference = pd.read_csv(args.cimt_per_eye)
    if not reference.split.eq("train").all():
        raise ValueError("QC thresholds must use CIMT split=train only.")
    fundus = pd.read_csv(args.fundus_per_eye)
    manifest = pd.read_csv(args.manifest)
    frozen = pd.read_csv(args.frozen_metrics)
    thresholds = {
        "zone_b_valid_pixels_p05": float(reference.zone_b_valid_pixels.quantile(.05)),
        "vessel_occupancy_disagreement_p95": float(reference.vessel_occupancy_disagreement.quantile(.95)),
    }
    required = {"id", "disc_detected", "zone_b_valid_pixels", *RELIABLE}
    missing = sorted(required - set(fundus.columns))
    if missing:
        raise KeyError(f"Missing Fundus phenotype fields: {missing}")
    id_map = manifest[["id", "path"]].copy()
    id_map["source_id"] = id_map.path.map(lambda value: Path(value).stem)
    if id_map.source_id.duplicated().any():
        raise ValueError("Manifest source image stems must be unique.")
    gate = fundus[["id", "disc_detected", "zone_b_valid_pixels", *RELIABLE]].copy()
    finite = np.isfinite(gate[RELIABLE].to_numpy(float)).all(axis=1)
    gate["qc_gate"] = (
        gate.disc_detected.astype(bool)
        & gate.zone_b_valid_pixels.ge(thresholds["zone_b_valid_pixels_p05"])
        & gate.vessel_occupancy_disagreement.le(thresholds["vessel_occupancy_disagreement_p95"])
        & finite
    )
    gate = gate.merge(id_map[["id", "source_id"]], on="id", validate="one_to_one")
    # Frozen labels use source image stems; the gate is one row per internal image ID.
    merged = frozen.merge(gate[["source_id", "qc_gate"]], left_on="id", right_on="source_id", validate="many_to_one")
    if len(merged) != len(frozen):
        raise ValueError("Incomplete frozen-metric / phenotype merge.")
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "per_image_qc_gate_metrics.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows, bootstrap = [], {}
    for model, group in merged.groupby("model", sort=True):
        for metric in METRICS:
            all_value = float(group[metric].mean())
            gated_value = float(group.loc[group.qc_gate, metric].mean())
            rows.append({"model": model, "metric": metric, "all_images": all_value,
                         "qc_gated": gated_value, "delta_gated_minus_all": gated_value - all_value,
                         "coverage": float(group.qc_gate.mean()), "n_all": int(len(group)), "n_gated": int(group.qc_gate.sum())})
            bootstrap[f"{model}__{metric}"] = bootstrap_delta(group, metric, args.bootstrap_repetitions, rng)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "qc_gate_summary.csv", index=False)
    report = {
        "protocol": "Fundus_AVSeg_结局盲质量门控敏感性协议.md",
        "threshold_reference": "CIMT split=train per-eye phenotype only",
        "thresholds": thresholds,
        "gate_definition": ["disc_detected", "zone_b_valid_pixels>=CIMT_p05", "occupancy_disagreement<=CIMT_p95", "four_reliable_features_finite"],
        "frozen_external_labels_used_only_after_gate_fixed": True,
        "cimt_fixed_validation_used": False, "cimt_test_used": False,
        "coverage": float(gate.qc_gate.mean()), "n_images": int(len(gate)),
        "bootstrap": bootstrap,
        "interpretation_boundary": "Frozen external segmentation sensitivity analysis; not threshold tuning or clinical validation.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
