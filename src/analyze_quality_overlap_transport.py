#!/usr/bin/env python3
"""Outcome-blind QC overlap-weighted phenotype transport sensitivity analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260804
FEATURES = [
    "soft_vessel_mass_fraction",
    "mass_vessel_skeleton_length_density",
    "factorized_vessel_skeleton_length_density",
    "vessel_occupancy_disagreement",
]
SHORT = {
    "soft_vessel_mass_fraction": "Soft vessel mass",
    "mass_vessel_skeleton_length_density": "Mass skeleton density",
    "factorized_vessel_skeleton_length_density": "Factorized skeleton",
    "vessel_occupancy_disagreement": "Readout disagreement",
}
QC = [
    "disc_detected",
    "disc_radius_px",
    "zone_b_valid_pixels",
    "occupancy_entropy",
    "identity_entropy",
    "mean_gate",
]
INPUTS = {
    "CIMT train": ROOT / "artifacts/cimt_dual_readout_development/phenotypes_v2/per_eye.csv",
    "mBRSET": ROOT / "artifacts/mbrset_external/phenotypes/per_eye.csv",
    "RAV": ROOT / "artifacts/external_transfer/RAV/phenotypes/per_eye.csv",
    "Fundus-AVSeg": ROOT / "artifacts/external_transfer/FundusAVSeg/phenotypes/per_eye.csv",
}
COLORS = {"mBRSET": "#E45756", "RAV": "#54A24B", "Fundus-AVSeg": "#E6B94A"}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    return float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))


def effective_sample_size(weights: np.ndarray) -> float:
    return float(np.sum(weights) ** 2 / np.sum(weights ** 2))


def load(name: str, path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"patient_id", "eye", *FEATURES, *QC}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    data = data.copy()
    data["patient_id"] = data["patient_id"].astype(str)
    data["eye"] = data["eye"].astype(str)
    data = data.groupby(["patient_id", "eye"], as_index=False, dropna=False)[FEATURES + QC].median()
    data["cohort"] = name
    return data


def qc_overlap(reference: pd.DataFrame, external: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    merged = pd.concat([reference.assign(domain=0), external.assign(domain=1)], ignore_index=True)
    truth = merged["domain"].to_numpy(int)
    groups = (merged["cohort"] + "::" + merged["patient_id"]).to_numpy(str)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        LogisticRegression(C=1.0, max_iter=3000, solver="liblinear", random_state=SEED),
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = cross_val_predict(
        model, merged[QC], truth, groups=groups, cv=splitter, method="predict_proba", n_jobs=1
    )[:, 1]
    merged["propensity_external"] = np.clip(probability, 1e-5, 1 - 1e-5)
    merged["overlap_weight"] = np.where(truth == 0, merged["propensity_external"], 1 - merged["propensity_external"])
    return merged, float(roc_auc_score(truth, probability))


def balance_table(merged: pd.DataFrame, external_name: str) -> pd.DataFrame:
    fill = merged.loc[merged["domain"] == 0, QC].median()
    table = merged.copy()
    table[QC] = table[QC].fillna(fill)
    rows = []
    for variable in QC:
        ref = table.loc[table["domain"] == 0, variable].to_numpy(float)
        ext = table.loc[table["domain"] == 1, variable].to_numpy(float)
        ref_w = table.loc[table["domain"] == 0, "overlap_weight"].to_numpy(float)
        ext_w = table.loc[table["domain"] == 1, "overlap_weight"].to_numpy(float)
        pooled_before = np.sqrt((np.var(ref) + np.var(ext)) / 2)
        pooled_after = np.sqrt((weighted_variance(ref, ref_w) + weighted_variance(ext, ext_w)) / 2)
        rows.append({
            "cohort": external_name,
            "qc_variable": variable,
            "smd_before": float((np.mean(ext) - np.mean(ref)) / max(pooled_before, np.finfo(float).eps)),
            "smd_after": float((weighted_mean(ext, ext_w) - weighted_mean(ref, ref_w)) / max(pooled_after, np.finfo(float).eps)),
        })
    return pd.DataFrame(rows)


def grouped_arrays(frame: pd.DataFrame, feature: str) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = []
    for _, group in frame.groupby("patient_id"):
        mask = group[feature].notna().to_numpy()
        groups.append((group.loc[mask, feature].to_numpy(float), group.loc[mask, "overlap_weight"].to_numpy(float)))
    return groups


def bootstrap_distances(
    reference: pd.DataFrame,
    external: pd.DataFrame,
    feature: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    ref_groups = grouped_arrays(reference, feature)
    ext_groups = grouped_arrays(external, feature)
    reference_iqr = max(
        float(reference[feature].quantile(0.75) - reference[feature].quantile(0.25)), np.finfo(float).eps
    )
    before = np.empty(repetitions)
    after = np.empty(repetitions)
    attenuation = np.empty(repetitions)
    for index in range(repetitions):
        ref_sample = [ref_groups[i] for i in rng.integers(0, len(ref_groups), len(ref_groups))]
        ext_sample = [ext_groups[i] for i in rng.integers(0, len(ext_groups), len(ext_groups))]
        ref_values = np.concatenate([item[0] for item in ref_sample])
        ref_weights = np.concatenate([item[1] for item in ref_sample])
        ext_values = np.concatenate([item[0] for item in ext_sample])
        ext_weights = np.concatenate([item[1] for item in ext_sample])
        before[index] = wasserstein_distance(ref_values, ext_values) / reference_iqr
        after[index] = wasserstein_distance(ref_values, ext_values, ref_weights, ext_weights) / reference_iqr
        attenuation[index] = 1 - after[index] / max(before[index], np.finfo(float).eps)
    return {
        "before_ci": np.quantile(before, [0.025, 0.975]),
        "after_ci": np.quantile(after, [0.025, 0.975]),
        "attenuation_ci": np.quantile(attenuation, [0.025, 0.975]),
    }


def cluster_auc_ci(merged: pd.DataFrame, repetitions: int, rng: np.random.Generator) -> tuple[float, float]:
    truth = merged["domain"].to_numpy(int)
    probability = merged["propensity_external"].to_numpy(float)
    groups = (merged["cohort"] + "::" + merged["patient_id"]).to_numpy(str)
    unique = np.unique(groups)
    index = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([index[group] for group in sampled])
        if np.unique(truth[take]).size == 2:
            values.append(roc_auc_score(truth[take], probability[take]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def make_figure(metrics: pd.DataFrame, summary: pd.DataFrame, balance: pd.DataFrame, output: Path) -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5, "axes.titlesize": 14,
        "axes.labelsize": 12, "axes.titleweight": "bold", "axes.grid": True,
        "grid.alpha": 0.22, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    cohorts = ["mBRSET", "RAV", "Fundus-AVSeg"]
    features = [SHORT[value] for value in FEATURES]

    for axis, column, title in [
        (axes[0, 0], "distance_before", "A  Shift before QC overlap weighting"),
        (axes[0, 1], "distance_after", "B  Shift after QC overlap weighting"),
    ]:
        matrix = metrics.pivot(index="cohort", columns="feature_short", values=column).reindex(index=cohorts, columns=features)
        image = axis.imshow(matrix.to_numpy(float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1.0, metrics[["distance_before", "distance_after"]].to_numpy().max()))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iloc[row, col]
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontweight="bold", color="white" if value > 2.8 else "#111111")
        axis.set_xticks(np.arange(len(features)), features, rotation=18)
        axis.set_yticks(np.arange(len(cohorts)), cohorts)
        axis.set_title(title)
        axis.grid(False)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Wasserstein / CIMT IQR")

    attenuation = metrics.pivot(index="cohort", columns="feature_short", values="attenuation_fraction").reindex(index=cohorts, columns=features)
    image = axes[1, 0].imshow(attenuation.to_numpy(float), aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    for row in range(attenuation.shape[0]):
        for col in range(attenuation.shape[1]):
            value = attenuation.iloc[row, col]
            axes[1, 0].text(col, row, f"{100 * value:.0f}%", ha="center", va="center", fontweight="bold")
    axes[1, 0].set_xticks(np.arange(len(features)), features, rotation=18)
    axes[1, 0].set_yticks(np.arange(len(cohorts)), cohorts)
    axes[1, 0].set_title("C  Fraction of shift explained by measured QC")
    axes[1, 0].grid(False)
    colorbar = fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("Attenuation fraction")

    x = np.arange(len(cohorts)); width = 0.34
    before = balance.groupby("cohort")["smd_before"].apply(lambda values: values.abs().max()).reindex(cohorts)
    after = balance.groupby("cohort")["smd_after"].apply(lambda values: values.abs().max()).reindex(cohorts)
    axes[1, 1].bar(x - width / 2, before, width, color="#A7A7A7", edgecolor="#333333", label="Before weighting")
    axes[1, 1].bar(x + width / 2, after, width, color="#4C78A8", edgecolor="#333333", label="After weighting")
    axes[1, 1].axhline(0.10, color="#B64342", linestyle="--", linewidth=1.3, label="Balance target 0.10")
    axes[1, 1].set_xticks(x, cohorts)
    axes[1, 1].set_ylabel("Maximum absolute QC SMD")
    axes[1, 1].set_title("D  Covariate balance and QC-only domain AUROC")
    axes[1, 1].legend(frameon=False, fontsize=10)
    top = max(float(before.max()), float(after.max()))
    axes[1, 1].set_ylim(0, max(0.4, top * 1.28))
    for xpos, cohort in zip(x, cohorts):
        row = summary.loc[summary["cohort"] == cohort].iloc[0]
        axes[1, 1].text(xpos, top * 1.12, f"QC AUROC {row.qc_domain_auroc:.3f}\nESS {row.min_ess_fraction:.0%}", ha="center", va="center", fontsize=10.5, fontweight="bold")

    fig.suptitle("QC overlap-weighted sensitivity analysis of cross-cohort phenotype shift", fontsize=17, fontweight="bold")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"quality_overlap_transport.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/quality_overlap_transport")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = {name: load(name, path) for name, path in INPUTS.items()}
    reference = data["CIMT train"]

    metric_rows = []
    summary_rows = []
    balance_frames = []
    weight_frames = []
    for cohort in ["mBRSET", "RAV", "Fundus-AVSeg"]:
        merged, auc = qc_overlap(reference, data[cohort])
        auc_ci = cluster_auc_ci(merged, args.bootstrap_repetitions, rng)
        balance = balance_table(merged, cohort)
        balance_frames.append(balance)
        ref = merged[merged["domain"] == 0].copy()
        ext = merged[merged["domain"] == 1].copy()
        ref_ess = effective_sample_size(ref["overlap_weight"].to_numpy(float))
        ext_ess = effective_sample_size(ext["overlap_weight"].to_numpy(float))
        max_before = float(balance["smd_before"].abs().max())
        max_after = float(balance["smd_after"].abs().max())
        min_ess_fraction = min(ref_ess / len(ref), ext_ess / len(ext))
        summary_rows.append({
            "cohort": cohort,
            "qc_domain_auroc": auc,
            "auroc_ci95_low": auc_ci[0],
            "auroc_ci95_high": auc_ci[1],
            "reference_ess": ref_ess,
            "external_ess": ext_ess,
            "min_ess_fraction": min_ess_fraction,
            "max_abs_smd_before": max_before,
            "max_abs_smd_after": max_after,
            "balance_achieved": max_after <= 0.10,
            "minimum_overlap_achieved": min_ess_fraction >= 0.20,
        })
        keep = merged[["cohort", "patient_id", "eye", "domain", "propensity_external", "overlap_weight"]].copy()
        keep["comparison"] = cohort
        weight_frames.append(keep)
        for feature in FEATURES:
            ref_values = ref[feature].to_numpy(float); ext_values = ext[feature].to_numpy(float)
            ref_weights = ref["overlap_weight"].to_numpy(float); ext_weights = ext["overlap_weight"].to_numpy(float)
            reference_iqr = max(float(reference[feature].quantile(0.75) - reference[feature].quantile(0.25)), np.finfo(float).eps)
            before = wasserstein_distance(ref_values, ext_values) / reference_iqr
            after = wasserstein_distance(ref_values, ext_values, ref_weights, ext_weights) / reference_iqr
            intervals = bootstrap_distances(ref, ext, feature, args.bootstrap_repetitions, rng)
            metric_rows.append({
                "cohort": cohort,
                "feature": feature,
                "feature_short": SHORT[feature],
                "distance_before": before,
                "before_ci95_low": intervals["before_ci"][0],
                "before_ci95_high": intervals["before_ci"][1],
                "distance_after": after,
                "after_ci95_low": intervals["after_ci"][0],
                "after_ci95_high": intervals["after_ci"][1],
                "attenuation_fraction": 1 - after / max(before, np.finfo(float).eps),
                "attenuation_ci95_low": intervals["attenuation_ci"][0],
                "attenuation_ci95_high": intervals["attenuation_ci"][1],
            })

    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    balance = pd.concat(balance_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    metrics.to_csv(output / "weighted_shift_metrics.csv", index=False)
    summary.to_csv(output / "overlap_summary.csv", index=False)
    balance.to_csv(output / "qc_balance.csv", index=False)
    weights.to_csv(output / "oof_overlap_weights.csv", index=False)
    make_figure(metrics, summary, balance, output)
    metadata = {
        "protocol": "质量重叠加权迁移_锁定协议.md",
        "analysis": "outcome-blind QC overlap-weighted transport sensitivity",
        "reference": "CIMT split=train phenotype artifact only",
        "raw_cimt_validation_or_test_read": False,
        "quality_covariates": QC,
        "features": FEATURES,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "random_seed": SEED,
        "complete": True,
        "interpretation_boundary": "Exploratory measured-QC sensitivity; not causal decomposition or clinical external validation.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nWeighted phenotype shift:")
    print(metrics[["cohort", "feature_short", "distance_before", "distance_after", "attenuation_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
