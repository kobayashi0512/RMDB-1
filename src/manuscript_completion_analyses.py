#!/usr/bin/env python3
"""Analyses that complete the RMDB--BiRAP manuscript evidence package.

This module deliberately consumes frozen artifacts.  It never reads the raw
CIMT test split and never retunes a model on the frozen validation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260803
CORE_BASES = [
    "soft_vessel_mass_fraction",
    "mass_vessel_skeleton_length_density",
    "factorized_vessel_skeleton_length_density",
    "vessel_occupancy_disagreement",
]
CORE_PATIENT = [f"{name}__bilateral_mean" for name in CORE_BASES]
CLINICAL = ["age", "sex", "dm_time"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_ci(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return [float(np.quantile(array, .025)), float(np.quantile(array, .975))]


def audit_access(output: Path) -> None:
    """Record the existing split-access evidence without opening raw val/test rows."""
    clinical_test = ROOT / "artifacts/clinical_test_metrics.json"
    e9 = ROOT / "artifacts/cimt_dual_readout_validation/e9_locked/metrics.json"
    locked_metrics = [
        ROOT / "artifacts/cimt_dual_readout_development/outer_cv/metrics.json",
        ROOT / "artifacts/cimt_dual_readout_development/normative_e7/metrics.json",
        ROOT / "artifacts/cimt_dual_readout_development/e8_double_orthogonal/metrics.json",
        e9,
        ROOT / "artifacts/e10_distributionally_robust_full/metrics.json",
        ROOT / "artifacts/e11_dual_head_full/metrics.json",
    ]
    locked_flags = []
    for path in locked_metrics:
        if not path.is_file():
            locked_flags.append({"path": str(path.relative_to(ROOT)), "exists": False})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        locked_flags.append({
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "test_used": data.get("test_used", data.get("held_out_test_used")),
            "sha256": sha256(path),
        })
    report = {
        "audit_scope": "artifact and code provenance only; raw CIMT val/test rows were not opened",
        "clinical_test_artifact_exists": clinical_test.is_file(),
        "clinical_test_artifact": {
            "path": str(clinical_test.relative_to(ROOT)),
            "sha256": sha256(clinical_test) if clinical_test.is_file() else None,
            "mtime": clinical_test.stat().st_mtime if clinical_test.is_file() else None,
        },
        "conclusion": (
            "The CIMT test split is not pristine because an early clinical-baseline test artifact exists. "
            "It must not be described as unopened or used as confirmatory evidence."
            if clinical_test.is_file()
            else "No clinical test artifact was found, but absence is not proof of non-access."
        ),
        "locked_dual_readout_runs": locked_flags,
        "reporting_language": (
            "The test split was accessed during an early clinical-baseline experiment but was not used for "
            "RMDB--BiRAP feature development, model selection, or the locked dual-readout analyses."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "access_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def icc_from_summaries(counts: np.ndarray, means: np.ndarray, within_ss: np.ndarray) -> float:
    grand = float(np.average(means, weights=counts))
    n, groups = float(counts.sum()), float(len(counts))
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(within_ss.sum())
    ms_between = ss_between / (groups - 1)
    ms_within = ss_within / (n - groups)
    effective_k = (n - float(np.sum(counts ** 2)) / n) / (groups - 1)
    return float((ms_between - ms_within) / (ms_between + (effective_k - 1) * ms_within))


def repeat_summaries(frame: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame[["eye_key", feature]].dropna().groupby("eye_key", sort=False)[feature]
    count = grouped.size().to_numpy(dtype=float)
    mean = grouped.mean().to_numpy(dtype=float)
    sum_sq = grouped.apply(lambda x: float(np.square(x.to_numpy(dtype=float)).sum())).to_numpy(dtype=float)
    within = sum_sq - count * np.square(mean)
    keep = count >= 2
    return count[keep], mean[keep], within[keep]


def repeatability_metrics(frame: pd.DataFrame, feature: str, repetitions: int, rng: np.random.Generator) -> dict:
    counts, means, within = repeat_summaries(frame, feature)
    if len(counts) < 20:
        return {"feature": feature, "n_repeat_eyes": int(len(counts)), "status": "insufficient"}
    icc = icc_from_summaries(counts, means, within)
    df_within = float(counts.sum() - len(counts))
    sem = float(math.sqrt(max(float(within.sum()) / df_within, 0.0)))
    grand = float(np.average(means, weights=counts))
    pair = (
        frame[["eye_key", "view", feature]].dropna()
        .sort_values(["eye_key", "view"])
        .groupby("eye_key", sort=False)
        .head(2)
    )
    pair["rep"] = pair.groupby("eye_key").cumcount()
    wide = pair.pivot(index="eye_key", columns="rep", values=feature).dropna()
    difference = wide[1].to_numpy() - wide[0].to_numpy()
    average = wide[[0, 1]].mean(axis=1).to_numpy()
    bias = float(difference.mean())
    sd_diff = float(difference.std(ddof=1))
    icc_samples, bias_samples, loa_low_samples, loa_high_samples = [], [], [], []
    for _ in range(repetitions):
        selected = rng.integers(0, len(counts), len(counts))
        icc_samples.append(icc_from_summaries(counts[selected], means[selected], within[selected]))
        idx = rng.integers(0, len(difference), len(difference))
        sampled = difference[idx]
        sample_bias = float(sampled.mean())
        sample_sd = float(sampled.std(ddof=1))
        bias_samples.append(sample_bias)
        loa_low_samples.append(sample_bias - 1.96 * sample_sd)
        loa_high_samples.append(sample_bias + 1.96 * sample_sd)
    return {
        "feature": feature,
        "n_repeat_eyes": int(len(counts)),
        "n_bland_altman_pairs": int(len(difference)),
        "icc_1_1": icc,
        "icc_ci95_low": quantile_ci(icc_samples)[0],
        "icc_ci95_high": quantile_ci(icc_samples)[1],
        "sem": sem,
        "mdc95": float(1.96 * math.sqrt(2) * sem),
        "within_subject_cv_percent": float(100 * sem / abs(grand)) if abs(grand) > 1e-12 else np.nan,
        "bland_altman_bias": bias,
        "bias_ci95_low": quantile_ci(bias_samples)[0],
        "bias_ci95_high": quantile_ci(bias_samples)[1],
        "loa_low": float(bias - 1.96 * sd_diff),
        "loa_high": float(bias + 1.96 * sd_diff),
        "loa_low_ci95_low": quantile_ci(loa_low_samples)[0],
        "loa_low_ci95_high": quantile_ci(loa_low_samples)[1],
        "loa_high_ci95_low": quantile_ci(loa_high_samples)[0],
        "loa_high_ci95_high": quantile_ci(loa_high_samples)[1],
        "average": average,
        "difference": difference,
    }


def extended_repeatability(output: Path, repetitions: int) -> None:
    eye = pd.read_csv(ROOT / "artifacts/mbrset_external/phenotypes/per_eye.csv", dtype={"patient_id": str})
    manifest = pd.read_csv(
        ROOT / "artifacts/mbrset_external/manifest/per_image.csv",
        usecols=["id", "view", "final_quality", "final_icdr"],
    )
    frame = eye.merge(manifest, on="id", validate="one_to_one")
    frame["eye_key"] = frame["patient_id"] + "__" + frame["eye"]
    frame["good_quality"] = frame["final_quality"].astype(str).str.lower().eq("yes")
    eligibility = frame.groupby("eye_key").agg(
        images=("id", "size"),
        both_good_quality=("good_quality", "all"),
        both_disc_detected=("disc_detected", "all"),
        both_no_dr=("final_icdr", lambda x: bool(x.notna().all() and x.eq(0).all())),
    )
    eligibility = eligibility.loc[eligibility.images.ge(2)]
    strata = {
        "all": eligibility.index,
        "both_good_quality": eligibility.index[eligibility.both_good_quality],
        "both_disc_detected": eligibility.index[eligibility.both_disc_detected],
        "both_no_dr": eligibility.index[eligibility.both_no_dr],
    }
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, pair_rows = [], []
    all_metrics = {}
    for stratum, keys in strata.items():
        local = frame.loc[frame.eye_key.isin(keys)].copy()
        for feature in CORE_BASES:
            result = repeatability_metrics(local, feature, repetitions, rng)
            averages = result.pop("average", None)
            differences = result.pop("difference", None)
            result["stratum"] = stratum
            rows.append(result)
            if stratum == "all" and averages is not None:
                all_metrics[feature] = result
                for a, d in zip(averages, differences):
                    pair_rows.append({"feature": feature, "pair_average": a, "view2_minus_view1": d})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "repeatability_extended.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output / "bland_altman_pairs.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    titles = {
        "soft_vessel_mass_fraction": "Soft vessel mass fraction",
        "mass_vessel_skeleton_length_density": "Mass skeleton density",
        "factorized_vessel_skeleton_length_density": "Factorized skeleton density",
        "vessel_occupancy_disagreement": "Occupancy disagreement",
    }
    for axis, feature in zip(axes.flat, CORE_BASES):
        local = pairs.loc[pairs.feature.eq(feature)]
        row = all_metrics[feature]
        axis.scatter(local.pair_average, local.view2_minus_view1, s=9, alpha=.28, color="#2A788E", edgecolors="none")
        axis.axhline(row["bland_altman_bias"], color="#D95F02", lw=1.5, label="Bias")
        axis.axhline(row["loa_low"], color="#666666", lw=1.1, ls="--")
        axis.axhline(row["loa_high"], color="#666666", lw=1.1, ls="--", label="95% LoA")
        axis.set_title(f"{titles[feature]}\nICC={row['icc_1_1']:.3f}; MDC95={row['mdc95']:.4g}", fontsize=11, fontweight="bold")
        axis.set_xlabel("Mean of repeated views")
        axis.set_ylabel("View 2 - View 1")
        axis.grid(alpha=.15)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("RMDB--BiRAP reliable-core repeatability: Bland--Altman audit", fontsize=14, fontweight="bold")
    fig.savefig(output / "repeatability_bland_altman.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "metadata.json", {
        "label_blind_primary_selection": True,
        "systemic_outcomes_used": False,
        "features": CORE_BASES,
        "bootstrap_repetitions": repetitions,
        "pair_rule": "two earliest numbered views per same patient-eye",
        "strata": {name: int(len(keys)) for name, keys in strata.items()},
    })
    print(metrics.to_string(index=False))


def model_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])


def negative_controls(output: Path, outcome_permutations: int, pairing_permutations: int) -> None:
    base = ROOT / "artifacts/mbrset_external/cv_hypertension_reliable_core"
    oof = pd.read_csv(base / "oof_predictions.csv", dtype={"patient_id": str})
    pivot = oof.pivot(index=["patient_id", "outcome", "fold"], columns="experiment", values="probability").reset_index()
    y = pivot.outcome.to_numpy(dtype=int)
    p0 = pivot.R0_clinical.to_numpy(dtype=float)
    p2 = pivot.R2_clinical_lowdim_vascular.to_numpy(dtype=float)
    observed = {
        "delta_auroc": float(roc_auc_score(y, p2) - roc_auc_score(y, p0)),
        "delta_auprc": float(average_precision_score(y, p2) - average_precision_score(y, p0)),
        "delta_brier": float(brier_score_loss(y, p2) - brier_score_loss(y, p0)),
    }
    rng = np.random.default_rng(SEED)
    perm_rows = []
    for index in range(outcome_permutations):
        yp = rng.permutation(y)
        perm_rows.append({
            "permutation": index,
            "delta_auroc": roc_auc_score(yp, p2) - roc_auc_score(yp, p0),
            "delta_auprc": average_precision_score(yp, p2) - average_precision_score(yp, p0),
            "delta_brier": brier_score_loss(yp, p2) - brier_score_loss(yp, p0),
        })
    perm = pd.DataFrame(perm_rows)
    p_values = {
        "delta_auroc_one_sided": float((1 + perm.delta_auroc.ge(observed["delta_auroc"]).sum()) / (len(perm) + 1)),
        "delta_auprc_one_sided": float((1 + perm.delta_auprc.ge(observed["delta_auprc"]).sum()) / (len(perm) + 1)),
        "delta_brier_one_sided": float((1 + perm.delta_brier.le(observed["delta_brier"]).sum()) / (len(perm) + 1)),
    }

    patient = pd.read_csv(ROOT / "artifacts/mbrset_external/manifest/per_patient.csv", dtype={"patient_id": str})
    phenotype = pd.read_csv(ROOT / "artifacts/mbrset_external/phenotypes/per_patient.csv", dtype={"patient_id": str})
    association = pivot[["patient_id", "outcome", "R0_clinical"]].merge(
        phenotype[["patient_id", *CORE_PATIENT]], on="patient_id", validate="one_to_one"
    )
    association["clinical_residual"] = association.outcome - association.R0_clinical
    association_rows = []
    for feature in CORE_PATIENT:
        rho = float(spearmanr(association[feature], association.clinical_residual, nan_policy="omit").statistic)
        null = []
        values = association[feature].to_numpy()
        residual = association.clinical_residual.to_numpy()
        for _ in range(outcome_permutations):
            null.append(float(spearmanr(values, rng.permutation(residual), nan_policy="omit").statistic))
        association_rows.append({
            "feature": feature, "rho_with_clinical_residual": rho,
            "two_sided_permutation_p": float((1 + (np.abs(null) >= abs(rho)).sum()) / (len(null) + 1)),
        })

    eye = pd.read_csv(ROOT / "artifacts/mbrset_external/phenotypes/per_eye.csv", dtype={"patient_id": str})
    eye_mean = eye.groupby(["patient_id", "eye"], as_index=False)[CORE_BASES].mean()
    wide = eye_mean.pivot(index="patient_id", columns="eye", values=CORE_BASES)
    complete_ids = wide.dropna().index
    local = patient.loc[patient.patient_id.isin(complete_ids) & patient.systemic_hypertension.notna(), ["patient_id", *CLINICAL, "systemic_hypertension"]].copy()
    fold_map = pivot.set_index("patient_id").fold
    local["fold"] = local.patient_id.map(fold_map)
    local = local.dropna(subset=["fold"]).reset_index(drop=True)
    tuning = pd.read_csv(base / "nested_tuning.csv")
    c_by_fold = tuning.loc[tuning.experiment.eq("R2_clinical_lowdim_vascular")].set_index("fold").logistic_c.to_dict()
    left = wide.loc[local.patient_id, pd.IndexSlice[:, "left"]].copy(); left.columns = CORE_BASES
    right = wide.loc[local.patient_id, pd.IndexSlice[:, "right"]].copy(); right.columns = CORE_BASES
    left.index = local.index; right.index = local.index

    def fit_pair_features(pair_features: pd.DataFrame) -> tuple[float, float]:
        prediction = np.empty(len(local), dtype=float)
        design = pd.concat([local[CLINICAL], pair_features], axis=1)
        for fold in sorted(local.fold.astype(int).unique()):
            train = local.fold.astype(int).ne(fold)
            hold = ~train
            model = model_pipeline(float(c_by_fold[fold]))
            model.fit(design.loc[train], local.loc[train, "systemic_hypertension"].astype(int))
            prediction[hold] = model.predict_proba(design.loc[hold])[:, 1]
        truth = local.systemic_hypertension.to_numpy(dtype=int)
        return float(roc_auc_score(truth, prediction)), float(brier_score_loss(truth, prediction))

    true_features = (left + right) / 2
    true_auc, true_brier = fit_pair_features(true_features)
    pair_rows = []
    for index in range(pairing_permutations):
        shuffled = right.iloc[rng.permutation(len(right))].reset_index(drop=True)
        randomized = (left.reset_index(drop=True) + shuffled) / 2
        auc, brier = fit_pair_features(randomized)
        pair_rows.append({"permutation": index, "auroc": auc, "brier": brier})
    pair_null = pd.DataFrame(pair_rows)
    pairing_p = float((1 + pair_null.auroc.ge(true_auc).sum()) / (len(pair_null) + 1))

    output.mkdir(parents=True, exist_ok=True)
    perm.to_csv(output / "outcome_label_permutations.csv", index=False)
    pair_null.to_csv(output / "random_eye_pairing.csv", index=False)
    pd.DataFrame(association_rows).to_csv(output / "feature_residual_associations.csv", index=False)
    write_json(output / "negative_control_summary.json", {
        "locked_oof_outcome_permutation": {"observed": observed, "p_values": p_values, "repetitions": outcome_permutations},
        "random_eye_pairing": {
            "complete_bilateral_patients": int(len(local)), "true_pair_auroc": true_auc,
            "true_pair_brier": true_brier, "random_pair_auroc_mean": float(pair_null.auroc.mean()),
            "random_pair_auroc_ci95": quantile_ci(pair_null.auroc), "one_sided_p": pairing_p,
            "repetitions": pairing_permutations,
        },
        "interpretation_boundary": "Exploratory post-hoc negative controls on frozen OOF predictions and the label-blind reliable core.",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    axes[0].hist(perm.delta_auroc, bins=35, color="#91BFDB", edgecolor="white")
    axes[0].axvline(observed["delta_auroc"], color="#D73027", lw=2)
    axes[0].set_title("Outcome-label permutation")
    axes[0].set_xlabel("R2 - R0 AUROC")
    axes[1].hist(pair_null.auroc, bins=30, color="#A6D96A", edgecolor="white")
    axes[1].axvline(true_auc, color="#D73027", lw=2)
    axes[1].set_title("Random right-eye pairing")
    axes[1].set_xlabel("Reliable-core AUROC")
    fig.suptitle("RMDB--BiRAP negative controls", fontsize=14, fontweight="bold")
    fig.savefig(output / "negative_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"observed": observed, "p_values": p_values, "pairing_p": pairing_p}, indent=2))


def calibration_fit(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=5000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_utility(output: Path, repetitions: int) -> None:
    cohorts = {
        "Overall": ROOT / "artifacts/mbrset_external/cv_hypertension_reliable_core/oof_predictions.csv",
        "No DR": ROOT / "artifacts/mbrset_external/cv_hypertension_reliable_core_no_dr/oof_predictions.csv",
        "Strict disc QC": ROOT / "artifacts/mbrset_external/cv_hypertension_reliable_core_strict_disc/oof_predictions.csv",
    }
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary_rows, calibration_rows, decision_rows = [], [], []
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for row_index, (cohort, path) in enumerate(cohorts.items()):
        data = pd.read_csv(path, dtype={"patient_id": str})
        for model_name, color, label in [
            ("R0_clinical", "#666666", "Clinical"),
            ("R2_clinical_lowdim_vascular", "#1B9E77", "Clinical + reliable core"),
        ]:
            local = data.loc[data.experiment.eq(model_name)].copy()
            y = local.outcome.to_numpy(dtype=int); p = local.probability.to_numpy(dtype=float)
            intercept, slope = calibration_fit(y, p)
            boot_i, boot_s = [], []
            for _ in range(repetitions):
                sample = rng.integers(0, len(local), len(local))
                if np.unique(y[sample]).size < 2:
                    continue
                try:
                    i, s = calibration_fit(y[sample], p[sample]); boot_i.append(i); boot_s.append(s)
                except ValueError:
                    continue
            bins = pd.qcut(pd.Series(p), q=min(10, len(np.unique(p))), duplicates="drop")
            curve = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
                predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
            ).reset_index(drop=True)
            ece = float(np.average(np.abs(curve.observed - curve.predicted), weights=curve.n))
            summary_rows.append({
                "cohort": cohort, "model": label, "n": len(y), "positive": int(y.sum()),
                "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                "brier": brier_score_loss(y, p), "calibration_intercept": intercept,
                "intercept_ci95_low": quantile_ci(boot_i)[0], "intercept_ci95_high": quantile_ci(boot_i)[1],
                "calibration_slope": slope, "slope_ci95_low": quantile_ci(boot_s)[0],
                "slope_ci95_high": quantile_ci(boot_s)[1], "ece_10bin": ece,
            })
            for _, item in curve.iterrows():
                calibration_rows.append({"cohort": cohort, "model": label, **item.to_dict()})
            axes[row_index, 0].plot(curve.predicted, curve.observed, marker="o", lw=1.7, color=color, label=label)
            thresholds = np.linspace(.1, .9, 81)
            for threshold in thresholds:
                tp = float(((p >= threshold) & (y == 1)).sum())
                fp = float(((p >= threshold) & (y == 0)).sum())
                net = tp / len(y) - fp / len(y) * threshold / (1 - threshold)
                decision_rows.append({"cohort": cohort, "model": label, "threshold": threshold, "net_benefit": net})
            axes[row_index, 1].plot(thresholds, [r["net_benefit"] for r in decision_rows[-len(thresholds):]], color=color, lw=1.7, label=label)
        prevalence = float(data.loc[data.experiment.eq("R0_clinical"), "outcome"].mean())
        thresholds = np.linspace(.1, .9, 81)
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        axes[row_index, 0].plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
        axes[row_index, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency", title=f"{cohort}: calibration")
        axes[row_index, 0].legend(frameon=False, fontsize=9)
        axes[row_index, 1].plot(thresholds, treat_all, "--", color="#AAAAAA", lw=1, label="Treat all")
        axes[row_index, 1].axhline(0, color="#222222", lw=1, label="Treat none")
        axes[row_index, 1].set(xlabel="Decision threshold", ylabel="Net benefit", title=f"{cohort}: exploratory decision curve")
        axes[row_index, 1].legend(frameon=False, fontsize=8)
    pd.DataFrame(summary_rows).to_csv(output / "calibration_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_curves.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(output / "decision_curves.csv", index=False)
    fig.suptitle("Reliable-core hypertension calibration and exploratory utility", fontsize=15, fontweight="bold")
    fig.savefig(output / "calibration_decision_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "metadata.json", {
        "bootstrap_repetitions": repetitions,
        "status": "exploratory_internal_patient_level_cv",
        "warning": "Decision curves do not establish clinical utility without an externally specified decision threshold and independent confirmation.",
    })
    print(pd.DataFrame(summary_rows).to_string(index=False))


def summarize_continuous(series: pd.Series) -> str:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return f"{series.mean():.2f} ({series.std(ddof=1):.2f})" if len(series) else "NA"


def evidence_tables(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(ROOT / "artifacts/cimt_dual_readout_development/normative_e7/oof_predictions.csv", dtype={"patient_id": str})
    dev = dev.loc[dev.experiment.eq("E7_0_normative")].drop_duplicates("patient_id")
    val = pd.read_csv(ROOT / "artifacts/cimt_dual_readout_validation/e9_locked/predictions.csv", dtype={"patient_id": str})
    mbr = pd.read_csv(ROOT / "artifacts/mbrset_external/manifest/per_patient.csv", dtype={"patient_id": str})
    cohorts = [
        {"cohort": "China Fundus--CIMT development", "n": len(dev), "age_mean_sd": summarize_continuous(dev.age),
         "female_n_percent": f"{int((dev.sex == 0).sum())} ({100*(dev.sex == 0).mean():.1f}%)",
         "primary_outcome": f"CIMT {summarize_continuous(dev.cimt_mm)} mm",
         "positive_n_percent": f"{int(dev.accelerated.sum())} ({100*dev.accelerated.mean():.1f}%)",
         "source": "Frozen development OOF predictions; no raw val/test read"},
        {"cohort": "China Fundus--CIMT frozen validation", "n": len(val), "age_mean_sd": summarize_continuous(val.age),
         "female_n_percent": f"{int((val.sex == 0).sum())} ({100*(val.sex == 0).mean():.1f}%)",
         "primary_outcome": f"CIMT {summarize_continuous(val.cimt_mm)} mm",
         "positive_n_percent": f"{int(val.accelerated.sum())} ({100*val.accelerated.mean():.1f}%)",
         "source": "Previously frozen E9 prediction artifact; descriptive reporting only"},
        {"cohort": "mBRSET 224-pixel mirror", "n": len(mbr), "age_mean_sd": summarize_continuous(mbr.age),
         "female_n_percent": f"{int((mbr.sex == 0).sum())} ({100*(mbr.sex == 0).mean():.1f}%)",
         "primary_outcome": "Systemic hypertension",
         "positive_n_percent": f"{int(mbr.systemic_hypertension.fillna(0).sum())} ({100*mbr.systemic_hypertension.mean():.1f}%)",
         "source": "External mechanism cohort; internal patient-level CV"},
    ]
    pd.DataFrame(cohorts).to_csv(output / "cohort_characteristics.csv", index=False)

    flow = [
        {"stage": "CIMT development", "patients": len(dev), "images_or_pairs": 5206, "exclusions_or_qc": "24 expected disc-localization failures; global features retained"},
        {"stage": "CIMT frozen validation", "patients": len(val), "images_or_pairs": 400, "exclusions_or_qc": "One-shot E9 artifact; test not used in locked dual-readout analyses"},
        {"stage": "mBRSET manifest", "patients": len(mbr), "images_or_pairs": int(mbr.images.sum()), "exclusions_or_qc": "All images retained with explicit QC flags"},
        {"stage": "mBRSET repeatability", "patients": int(mbr.patient_id.nunique()), "images_or_pairs": 2359, "exclusions_or_qc": "Same-eye repeat pairs"},
        {"stage": "mBRSET hypertension complete-case", "patients": int(mbr.systemic_hypertension.notna().sum()), "images_or_pairs": int(mbr.systemic_hypertension.sum()), "exclusions_or_qc": "Positive count shown in images_or_pairs field"},
    ]
    pd.DataFrame(flow).to_csv(output / "participant_flow.csv", index=False)

    comparisons = []
    outer = json.loads((ROOT / "artifacts/cimt_dual_readout_development/outer_cv/metrics.json").read_text())
    for name, values in outer.get("overall", {}).items():
        comparisons.append({"cohort": "CIMT development", "model": name, **{k: values.get(k) for k in ["mae_mm", "rmse_mm", "r2", "auroc", "auprc", "brier"]}})
    for folder, cohort in [
        ("cv_hypertension_reliable_core", "mBRSET overall"),
        ("cv_hypertension_reliable_core_no_dr", "mBRSET no DR"),
        ("cv_hypertension_reliable_core_strict_disc", "mBRSET strict disc QC"),
    ]:
        values = json.loads((ROOT / "artifacts/mbrset_external" / folder / "metrics.json").read_text())
        for name, metric in values["overall"].items():
            comparisons.append({"cohort": cohort, "model": name, **{k: metric.get(k) for k in ["auroc", "auprc", "brier"]}})
    pd.DataFrame(comparisons).to_csv(output / "fair_model_comparison.csv", index=False)

    secondary = []
    for outcome in ["vascular_disease", "diabetic_foot", "acute_myocardial_infarction"]:
        data = json.loads((ROOT / "artifacts/mbrset_external" / f"cv_{outcome}" / "metrics.json").read_text())
        for name, metric in data["overall"].items():
            secondary.append({"outcome": outcome, "patients": data["patients"], "positive": data["positive"], "model": name,
                              "auroc": metric.get("auroc"), "auprc": metric.get("auprc"), "brier": metric.get("brier"),
                              "calibration_intercept": metric.get("calibration_intercept"), "calibration_slope": metric.get("calibration_slope")})
    pd.DataFrame(secondary).to_csv(output / "secondary_outcomes.csv", index=False)
    write_json(output / "metadata.json", {
        "raw_cimt_manifest_read": False,
        "cimt_test_read": False,
        "validation_source": "existing locked E9 prediction artifact only",
        "note": "These tables are manuscript-reporting artifacts, not new model selection."
    })
    print(pd.DataFrame(cohorts).to_string(index=False))


def external_preflight(output: Path, cohort_dir: Path | None) -> int:
    if cohort_dir is None:
        value = os.environ.get("RMDB_EXTERNAL_COHORT_DIR")
        cohort_dir = Path(value) if value else ROOT / "external_validation/MESA/data"
    manifest = cohort_dir / "manifest.csv"
    images = cohort_dir / "images"
    report = {
        "cohort_dir": str(cohort_dir),
        "manifest_exists": manifest.is_file(),
        "images_dir_exists": images.is_dir(),
        "required_contract": {
            "manifest_columns": ["patient_id", "image_id", "eye", "image_path", "age", "sex", "cimt_mm"],
            "image_policy": "original high-resolution fundus images; patient-level identity required",
            "evaluation_policy": "frozen preprocessing and paired patient bootstrap; no current CIMT val/test reuse",
        },
    }
    if manifest.is_file():
        columns = pd.read_csv(manifest, nrows=0).columns.tolist()
        report["manifest_columns_found"] = columns
        report["missing_columns"] = sorted(set(report["required_contract"]["manifest_columns"]) - set(columns))
    else:
        report["missing_columns"] = report["required_contract"]["manifest_columns"]
    ready = manifest.is_file() and images.is_dir() and not report["missing_columns"]
    report["status"] = "READY_FOR_FROZEN_EXTERNAL_PIPELINE" if ready else "BLOCKED_EXTERNAL_DATA"
    report["blocking_reason"] = None if ready else "Independent high-resolution images and linked patient-level outcomes are not present."
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "external_preflight.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ready else 20


def compile_summary(output: Path) -> None:
    base = ROOT / "artifacts/manuscript_completion"
    paths = {
        "access_audit": base / "01_access_audit/access_audit.json",
        "repeatability": base / "02_repeatability_extended/repeatability_extended.csv",
        "negative_controls": base / "03_negative_controls/negative_control_summary.json",
        "calibration": base / "04_calibration_utility/calibration_metrics.csv",
        "cohort_table": base / "05_evidence_tables/cohort_characteristics.csv",
        "model_comparison": base / "05_evidence_tables/fair_model_comparison.csv",
        "external_phenotype_transport": ROOT / "artifacts/external_phenotype_transport/metadata.json",
        "quality_overlap_transport": ROOT / "artifacts/quality_overlap_transport/metadata.json",
        "external_preflight": base / "06_external_confirmation/external_preflight.json",
    }
    report = {name: {"path": str(path.relative_to(ROOT)), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else 0}
              for name, path in paths.items()}
    external = json.loads(paths["external_preflight"].read_text()) if paths["external_preflight"].is_file() else {}
    report["overall_status"] = "AVAILABLE_ANALYSES_COMPLETE_EXTERNAL_BLOCKED" if external.get("status") == "BLOCKED_EXTERNAL_DATA" else "AVAILABLE_AND_EXTERNAL_PREFLIGHT_COMPLETE"
    report["claim_boundary"] = "Independent high-resolution outcome confirmation remains required before a confirmatory clinical claim."
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "completion_summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name in ["access-audit", "repeatability", "negative-controls", "calibration", "evidence-tables", "external-preflight", "summary"]:
        item = sub.add_parser(name)
        item.add_argument("--output", type=Path, required=True)
        if name in {"repeatability", "calibration"}:
            item.add_argument("--bootstrap-repetitions", type=int, default=2000)
        if name == "negative-controls":
            item.add_argument("--outcome-permutations", type=int, default=5000)
            item.add_argument("--pairing-permutations", type=int, default=250)
        if name == "external-preflight":
            item.add_argument("--cohort-dir", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "access-audit": audit_access(args.output)
    elif args.command == "repeatability": extended_repeatability(args.output, args.bootstrap_repetitions)
    elif args.command == "negative-controls": negative_controls(args.output, args.outcome_permutations, args.pairing_permutations)
    elif args.command == "calibration": calibration_utility(args.output, args.bootstrap_repetitions)
    elif args.command == "evidence-tables": evidence_tables(args.output)
    elif args.command == "external-preflight": raise SystemExit(external_preflight(args.output, args.cohort_dir))
    elif args.command == "summary": compile_summary(args.output)


if __name__ == "__main__":
    main()
