#!/usr/bin/env python3
"""Outcome-blind transport stress test across all available image cohorts.

This analysis deliberately uses cohort identity as the only target.  It is a
measurement-domain audit, not clinical validation, and is restricted to the
frozen CIMT development phenotype artifact.  No CIMT validation or test rows
are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
SEED = 20260813
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
INPUTS = {
    "CIMT train": ROOT / "artifacts/cimt_dual_readout_development/phenotypes_v2/per_eye.csv",
    "mBRSET": ROOT / "artifacts/mbrset_external/phenotypes/per_eye.csv",
    "AFIO": ROOT / "artifacts/afio_mendeley/phenotypes/per_eye.csv",
    "RAV": ROOT / "artifacts/external_transfer/RAV/phenotypes/per_eye.csv",
    "Fundus-AVSeg": ROOT / "artifacts/external_transfer/FundusAVSeg/phenotypes/per_eye.csv",
}
COLORS = {"CIMT train": "#6C5B9B", "mBRSET": "#B56576", "AFIO": "#E09F3E", "RAV": "#3A7D75", "Fundus-AVSeg": "#457B9D"}


def load_cohort(name: str, path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    required = {"patient_id", "eye", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["eye"] = frame["eye"].astype(str)
    # Collapse repeated views only; each patient-eye contributes one row.
    agg = frame.groupby(["patient_id", "eye"], as_index=False, dropna=False)[FEATURES].median()
    agg["cohort"] = name
    audit = {
        "cohort": name,
        "source": str(path.relative_to(ROOT)),
        "raw_rows": int(len(frame)),
        "patient_eyes": int(len(agg)),
        "patients": int(agg["patient_id"].nunique()),
        "repeat_views_collapsed": int(len(frame) - len(agg)),
        "complete_core_fraction": float(agg[FEATURES].notna().all(axis=1).mean()),
    }
    return agg, audit


def cluster_auc_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float, float]:
    auc = float(roc_auc_score(y, p))
    unique = np.unique(groups)
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    boot = []
    for _ in range(reps):
        sample = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([positions[g] for g in sample])
        if np.unique(y[take]).size == 2:
            boot.append(roc_auc_score(y[take], p[take]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return auc, float(lo), float(hi)


def domain_auc(a: pd.DataFrame, b: pd.DataFrame, reps: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    merged = pd.concat([a.assign(domain=0), b.assign(domain=1)], ignore_index=True)
    merged = merged.loc[merged[FEATURES].notna().any(axis=1)].reset_index(drop=True)
    y = merged["domain"].to_numpy(int)
    groups = (merged["cohort"] + "::" + merged["patient_id"]).to_numpy(str)
    n_groups = min(5, *(merged.loc[merged.domain == c, "patient_id"].nunique() for c in [0, 1]))
    if n_groups < 2:
        return float("nan"), float("nan"), float("nan"), int(len(merged))
    model = make_pipeline(
        SimpleImputer(strategy="median"), RobustScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=SEED),
    )
    cv = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=SEED)
    p = cross_val_predict(model, merged[FEATURES], y, groups=groups, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    auc, lo, hi = cluster_auc_ci(y, p, groups, reps, rng)
    return auc, lo, hi, int(len(merged))


def normalized_wasserstein(a: pd.DataFrame, b: pd.DataFrame, feature: str) -> float:
    x = a[feature].dropna().to_numpy(float)
    y = b[feature].dropna().to_numpy(float)
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    return float(wasserstein_distance(x, y) / max(iqr, np.finfo(float).eps))


def correlation_distance(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float]:
    ca = a[FEATURES].corr(method="spearman").to_numpy(float)
    cb = b[FEATURES].corr(method="spearman").to_numpy(float)
    upper = np.triu_indices(len(FEATURES), 1)
    return float(np.nanmean(np.abs(ca[upper] - cb[upper]))), float(np.linalg.norm(ca - cb, ord="fro"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/four_cohort_transport")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cohorts: dict[str, pd.DataFrame] = {}
    audits = []
    for name, path in INPUTS.items():
        cohorts[name], audit = load_cohort(name, path)
        audits.append(audit)
    pd.DataFrame(audits).to_csv(output / "cohort_audit.csv", index=False)

    reference = cohorts["CIMT train"]
    shift_rows = []
    for name, data in cohorts.items():
        if name == "CIMT train":
            continue
        for feature in FEATURES:
            ref = reference[feature].dropna().to_numpy(float)
            ext = data[feature].dropna().to_numpy(float)
            iqr = max(float(np.subtract(*np.percentile(ref, [75, 25]))), np.finfo(float).eps)
            test = ks_2samp(ref, ext, method="auto")
            shift_rows.append({
                "cohort": name,
                "feature": feature,
                "feature_short": SHORT[feature],
                "n_reference": int(len(ref)),
                "n_external": int(len(ext)),
                "reference_median": float(np.median(ref)),
                "external_median": float(np.median(ext)),
                "robust_median_shift": float((np.median(ext) - np.median(ref)) / iqr),
                "normalized_wasserstein": normalized_wasserstein(reference, data, feature),
                "ks_statistic": float(test.statistic),
                "ks_pvalue": float(test.pvalue),
            })
    pd.DataFrame(shift_rows).to_csv(output / "reference_shift.csv", index=False)

    pair_rows = []
    names = list(cohorts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            auc, lo, hi, n = domain_auc(cohorts[left], cohorts[right], args.bootstrap_repetitions, rng)
            mean_w = float(np.mean([normalized_wasserstein(cohorts[left], cohorts[right], f) for f in FEATURES]))
            mean_delta, frob = correlation_distance(cohorts[left], cohorts[right])
            pair_rows.append({
                "left": left, "right": right, "n_eyes": n,
                "domain_auroc": auc, "ci95_low": lo, "ci95_high": hi,
                "mean_normalized_wasserstein": mean_w,
                "mean_abs_spearman_delta": mean_delta,
                "spearman_frobenius_distance": frob,
            })
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(output / "pairwise_domain_transport.csv", index=False)

    # Compact figure: pairwise domain separability and reference-normalized shifts.
    order = ["CIMT train", "mBRSET", "AFIO", "RAV", "Fundus-AVSeg"]
    auc_mat = np.full((len(order), len(order)), np.nan)
    shift_mat = np.full((len(order), len(order)), np.nan)
    for row in pair_rows:
        i, j = order.index(row["left"]), order.index(row["right"])
        auc_mat[i, j] = auc_mat[j, i] = row["domain_auroc"]
        shift_mat[i, j] = shift_mat[j, i] = row["mean_normalized_wasserstein"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for ax, mat, title, cmap, fmt in [
        (axes[0], auc_mat, "A  Pairwise domain separability", "PuBuGn", ".3f"),
        (axes[1], shift_mat, "B  Mean phenotype transport distance", "OrRd", ".2f"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0.5 if ax is axes[0] else 0, vmax=1.0 if ax is axes[0] else None)
        for i in range(len(order)):
            for j in range(len(order)):
                if i != j and np.isfinite(mat[i, j]):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_yticks(range(len(order)), order)
        ax.set_title(title, fontweight="bold")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Outcome-blind five-cohort transport stress test of the RMDB–BiRAP reliable core", fontweight="bold", fontsize=15)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output / f"four_cohort_transport.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "analysis": "outcome-blind five-cohort phenotype transport stress test",
        "cohorts": order,
        "features": FEATURES,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "random_seed": SEED,
        "patient_eye_aggregation": "median across repeated views",
        "reference": "CIMT split=train phenotype artifact only",
        "raw_cimt_validation_or_test_read": False,
        "clinical_outcomes_used": False,
        "interpretation_boundary": "measurement-domain transport only; not independent clinical validation",
        "complete": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(audits).to_string(index=False))
    print("\nPairwise domain transport:")
    print(pair.to_string(index=False))


if __name__ == "__main__":
    main()
