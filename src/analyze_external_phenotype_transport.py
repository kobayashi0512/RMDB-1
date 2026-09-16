#!/usr/bin/env python3
"""Label-blind phenotype transport audit across retinal cohorts.

The reference distribution is restricted to the frozen CIMT development split.
No raw CIMT validation or test rows are read by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
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
SHORT_NAMES = {
    "soft_vessel_mass_fraction": "Soft vessel mass",
    "mass_vessel_skeleton_length_density": "Mass skeleton density",
    "factorized_vessel_skeleton_length_density": "Factorized skeleton",
    "vessel_occupancy_disagreement": "Readout disagreement",
}
DEFAULT_INPUTS = {
    "CIMT train": ROOT / "artifacts/cimt_dual_readout_development/phenotypes_v2/per_eye.csv",
    "mBRSET": ROOT / "artifacts/mbrset_external/phenotypes/per_eye.csv",
    "RAV": ROOT / "artifacts/external_transfer/RAV/phenotypes/per_eye.csv",
    "Fundus-AVSeg": ROOT / "artifacts/external_transfer/FundusAVSeg/phenotypes/per_eye.csv",
}
PALETTE = {
    "CIMT train": "#4C78A8",
    "mBRSET": "#E45756",
    "RAV": "#54A24B",
    "Fundus-AVSeg": "#E6B94A",
}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_and_aggregate(name: str, path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = {"patient_id", "eye", "disc_detected", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["eye"] = frame["eye"].astype(str)
    numeric = FEATURES + ["disc_detected"]
    aggregated = frame.groupby(["patient_id", "eye"], as_index=False, dropna=False)[numeric].median()
    aggregated["cohort"] = name
    audit = {
        "cohort": name,
        "source": str(path.relative_to(ROOT)),
        "raw_images": int(len(frame)),
        "patient_eyes": int(len(aggregated)),
        "patients": int(aggregated["patient_id"].nunique()),
        "repeat_views_collapsed": int(len(frame) - len(aggregated)),
        "disc_detection_rate": float(aggregated["disc_detected"].mean()),
        "complete_core_fraction": float(aggregated[FEATURES].notna().all(axis=1).mean()),
    }
    return aggregated, audit


def psi(reference: np.ndarray, external: np.ndarray) -> float:
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(quantiles) < 3:
        return float("nan")
    bins = np.r_[-np.inf, quantiles[1:-1], np.inf]
    ref_hist = np.histogram(reference, bins=bins)[0].astype(float)
    ext_hist = np.histogram(external, bins=bins)[0].astype(float)
    ref_prop = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)
    ext_prop = np.clip(ext_hist / ext_hist.sum(), 1e-6, None)
    return float(np.sum((ext_prop - ref_prop) * np.log(ext_prop / ref_prop)))


def cluster_values(frame: pd.DataFrame, feature: str) -> list[np.ndarray]:
    return [group[feature].dropna().to_numpy(float) for _, group in frame.groupby("patient_id")]


def bootstrap_shift(
    reference: pd.DataFrame,
    external: pd.DataFrame,
    feature: str,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ref_groups = cluster_values(reference, feature)
    ext_groups = cluster_values(external, feature)
    ref_iqr = np.subtract(*np.percentile(reference[feature].dropna(), [75, 25]))
    ref_iqr = max(float(ref_iqr), np.finfo(float).eps)
    shifts = np.empty(repetitions, dtype=float)
    distances = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        ref_sample = np.concatenate([ref_groups[i] for i in rng.integers(0, len(ref_groups), len(ref_groups))])
        ext_sample = np.concatenate([ext_groups[i] for i in rng.integers(0, len(ext_groups), len(ext_groups))])
        shifts[index] = (np.median(ext_sample) - np.median(ref_sample)) / ref_iqr
        distances[index] = wasserstein_distance(ref_sample, ext_sample) / ref_iqr
    return tuple(np.quantile(shifts, [0.025, 0.975])), tuple(np.quantile(distances, [0.025, 0.975]))


def cluster_bootstrap_auc(
    truth: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[group] for group in sampled])
        if np.unique(truth[take]).size == 2:
            values.append(roc_auc_score(truth[take], probability[take]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def domain_auc(
    reference: pd.DataFrame,
    external: pd.DataFrame,
    features: list[str],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float], int]:
    merged = pd.concat([reference.assign(domain=0), external.assign(domain=1)], ignore_index=True)
    complete = merged[features].notna().any(axis=1)
    merged = merged.loc[complete].reset_index(drop=True)
    truth = merged["domain"].to_numpy(int)
    groups = (merged["cohort"] + "::" + merged["patient_id"]).to_numpy(str)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=SEED),
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = cross_val_predict(
        model, merged[features], truth, groups=groups, cv=splitter, method="predict_proba", n_jobs=1
    )[:, 1]
    auc = float(roc_auc_score(truth, probability))
    interval = cluster_bootstrap_auc(truth, probability, groups, repetitions, rng)
    return auc, interval, len(merged)


def correlation_drift(reference: pd.DataFrame, external: pd.DataFrame) -> tuple[float, float]:
    ref_corr = reference[FEATURES].corr(method="spearman").to_numpy(float)
    ext_corr = external[FEATURES].corr(method="spearman").to_numpy(float)
    upper = np.triu_indices(len(FEATURES), 1)
    difference = np.abs(ext_corr - ref_corr)
    return float(np.nanmean(difference[upper])), float(np.linalg.norm(ext_corr - ref_corr, ord="fro"))


def make_figure(
    all_data: pd.DataFrame,
    shift: pd.DataFrame,
    discrimination: pd.DataFrame,
    drift: pd.DataFrame,
    audits: pd.DataFrame,
    output: Path,
) -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5, "axes.titlesize": 14,
        "axes.labelsize": 12, "axes.titleweight": "bold", "figure.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
    })
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)

    reference = all_data[all_data["cohort"] == "CIMT train"]
    domains = ["CIMT train", "mBRSET", "RAV", "Fundus-AVSeg"]
    offsets = np.linspace(-0.30, 0.30, len(domains))
    for feature_index, feature in enumerate(FEATURES):
        median = reference[feature].median()
        iqr = reference[feature].quantile(0.75) - reference[feature].quantile(0.25)
        for domain_index, domain in enumerate(domains):
            values = all_data.loc[all_data["cohort"] == domain, feature].dropna().to_numpy(float)
            robust_z = (values - median) / max(float(iqr), np.finfo(float).eps)
            box = axes[0, 0].boxplot(
                [robust_z], positions=[feature_index + offsets[domain_index]], widths=0.17,
                patch_artist=True, showfliers=False, whis=(5, 95), manage_ticks=False,
                medianprops={"color": "#222222", "linewidth": 1.4},
                whiskerprops={"color": "#444444", "linewidth": 0.9},
                capprops={"color": "#444444", "linewidth": 0.9},
                boxprops={"edgecolor": "#333333", "linewidth": 0.9},
            )
            box["boxes"][0].set_facecolor(PALETTE[domain])
            box["boxes"][0].set_alpha(0.82)
    axes[0, 0].axhline(0, color="#555555", linewidth=1, linestyle="--")
    axes[0, 0].set_ylim(-5, 8)
    axes[0, 0].set_xticks(np.arange(len(FEATURES)), [SHORT_NAMES[x] for x in FEATURES])
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Robust z vs CIMT-train (median/IQR)")
    axes[0, 0].tick_params(axis="x", rotation=12)
    axes[0, 0].legend(
        handles=[Patch(facecolor=PALETTE[x], edgecolor="#333333", label=x) for x in domains],
        title="Cohort", frameon=False, ncol=2, fontsize=10, title_fontsize=11,
    )
    axes[0, 0].set_title("A  Reliable-core distributions")

    heat = shift.pivot(index="cohort", columns="feature_short", values="normalized_wasserstein")
    heat = heat.reindex(index=["mBRSET", "RAV", "Fundus-AVSeg"], columns=[SHORT_NAMES[x] for x in FEATURES])
    image = axes[0, 1].imshow(heat.to_numpy(float), aspect="auto", cmap="YlOrRd")
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            value = heat.iloc[row, column]
            axes[0, 1].text(column, row, f"{value:.2f}", ha="center", va="center", color="#111111" if value < heat.to_numpy().max() * 0.65 else "white", fontweight="bold")
    axes[0, 1].set_xticks(np.arange(heat.shape[1]), heat.columns)
    axes[0, 1].set_yticks(np.arange(heat.shape[0]), heat.index)
    colorbar = figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)
    colorbar.set_label("Wasserstein / CIMT IQR")
    axes[0, 1].grid(False)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    axes[0, 1].tick_params(axis="x", rotation=18)
    axes[0, 1].tick_params(axis="y", rotation=0)
    axes[0, 1].set_title("B  Magnitude of phenotype shift")

    combined = discrimination[discrimination["feature_set"] == "Reliable core"].copy()
    order = ["mBRSET", "RAV", "Fundus-AVSeg"]
    combined["cohort"] = pd.Categorical(combined["cohort"], order, ordered=True)
    combined = combined.sort_values("cohort")
    x = np.arange(len(combined))
    yerr = np.vstack([combined["auroc"] - combined["ci95_low"], combined["ci95_high"] - combined["auroc"]])
    axes[1, 0].bar(x, combined["auroc"], color=[PALETTE[str(x)] for x in combined["cohort"]], edgecolor="#333333", linewidth=1, yerr=yerr, capsize=5)
    axes[1, 0].axhline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="Chance")
    axes[1, 0].set_xticks(x, [str(value) for value in combined["cohort"]])
    axes[1, 0].set_ylim(0.45, 1.075)
    axes[1, 0].set_ylabel("Patient-grouped OOF AUROC")
    axes[1, 0].set_title("C  Domain separability (higher = more shift)")
    axes[1, 0].legend(frameon=False, loc="lower right")
    for xpos, value in zip(x, combined["auroc"]):
        axes[1, 0].text(xpos, min(value + 0.018, 1.045), f"{value:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    merged = drift.merge(audits[["cohort", "disc_detection_rate"]], on="cohort")
    merged["disc_failure_rate"] = 1 - merged["disc_detection_rate"]
    merged["cohort"] = pd.Categorical(merged["cohort"], order, ordered=True)
    merged = merged.sort_values("cohort")
    x = np.arange(len(merged)); width = 0.36
    axes[1, 1].bar(x - width / 2, merged["mean_abs_spearman_delta"], width, color="#7A5195", edgecolor="#333333", label="Mean |Δ Spearman ρ|")
    axes[1, 1].bar(x + width / 2, merged["disc_failure_rate"], width, color="#B7B7B7", edgecolor="#333333", label="Disc failure rate")
    axes[1, 1].set_xticks(x, [str(value) for value in merged["cohort"]])
    axes[1, 1].set_ylim(0, max(0.45, float(merged[["mean_abs_spearman_delta", "disc_failure_rate"]].max().max()) * 1.2))
    axes[1, 1].set_ylabel("Proportion / absolute correlation change")
    axes[1, 1].set_title("D  Correlation structure and QC drift")
    axes[1, 1].legend(frameon=False, fontsize=10)

    figure.suptitle("Label-blind transport audit of the RMDB–BiRAP reliable phenotype core", fontsize=17, fontweight="bold")
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(output / f"external_phenotype_transport.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/external_phenotype_transport")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cohorts: dict[str, pd.DataFrame] = {}
    audits = []
    for name, path in DEFAULT_INPUTS.items():
        cohorts[name], audit = load_and_aggregate(name, path)
        audits.append(audit)
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(output / "cohort_audit.csv", index=False)

    reference = cohorts["CIMT train"]
    shift_rows = []
    for cohort_name in ["mBRSET", "RAV", "Fundus-AVSeg"]:
        external = cohorts[cohort_name]
        for feature in FEATURES:
            ref = reference[feature].dropna().to_numpy(float)
            ext = external[feature].dropna().to_numpy(float)
            reference_iqr = max(float(np.subtract(*np.percentile(ref, [75, 25]))), np.finfo(float).eps)
            shift_ci, distance_ci = bootstrap_shift(reference, external, feature, args.bootstrap_repetitions, rng)
            test = ks_2samp(ref, ext, alternative="two-sided", method="auto")
            shift_rows.append({
                "cohort": cohort_name,
                "feature": feature,
                "feature_short": SHORT_NAMES[feature],
                "n_reference_eyes": len(ref),
                "n_external_eyes": len(ext),
                "reference_median": float(np.median(ref)),
                "external_median": float(np.median(ext)),
                "robust_median_shift": float((np.median(ext) - np.median(ref)) / reference_iqr),
                "median_shift_ci95_low": shift_ci[0],
                "median_shift_ci95_high": shift_ci[1],
                "normalized_wasserstein": float(wasserstein_distance(ref, ext) / reference_iqr),
                "wasserstein_ci95_low": distance_ci[0],
                "wasserstein_ci95_high": distance_ci[1],
                "ks_statistic": float(test.statistic),
                "ks_pvalue": float(test.pvalue),
                "psi": psi(ref, ext),
            })
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(output / "phenotype_shift_metrics.csv", index=False)

    domain_rows = []
    feature_sets = [(SHORT_NAMES[feature], [feature]) for feature in FEATURES] + [("Reliable core", FEATURES)]
    for cohort_name in ["mBRSET", "RAV", "Fundus-AVSeg"]:
        for feature_set, columns in feature_sets:
            auc, interval, n = domain_auc(reference, cohorts[cohort_name], columns, args.bootstrap_repetitions, rng)
            domain_rows.append({"cohort": cohort_name, "feature_set": feature_set, "n_eyes": n, "auroc": auc, "ci95_low": interval[0], "ci95_high": interval[1]})
    discrimination = pd.DataFrame(domain_rows)
    discrimination.to_csv(output / "domain_discrimination.csv", index=False)

    drift_rows = []
    for cohort_name in ["mBRSET", "RAV", "Fundus-AVSeg"]:
        mean_delta, frobenius = correlation_drift(reference, cohorts[cohort_name])
        drift_rows.append({"cohort": cohort_name, "mean_abs_spearman_delta": mean_delta, "spearman_frobenius_distance": frobenius})
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(output / "correlation_drift.csv", index=False)

    all_data = pd.concat(cohorts.values(), ignore_index=True)
    make_figure(all_data, shift, discrimination, drift, audit_frame, output)
    metadata = {
        "protocol": "外部表型迁移_锁定协议.md",
        "analysis": "label-blind phenotype transport and failure-domain audit",
        "reference": "CIMT split=train phenotype artifact only",
        "raw_cimt_validation_or_test_read": False,
        "features": FEATURES,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "patient_eye_aggregation": "median across repeated views",
        "random_seed": SEED,
        "complete": True,
        "interpretation_boundary": "Measurement-domain transport only; not external clinical-outcome validation.",
    }
    write_json(output / "metadata.json", metadata)
    print(audit_frame.to_string(index=False))
    print("\nReliable-core domain discrimination:")
    print(discrimination[discrimination["feature_set"] == "Reliable core"].to_string(index=False))


if __name__ == "__main__":
    main()
