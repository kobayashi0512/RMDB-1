#!/usr/bin/env python3
"""One-shot patient-grouped M0–M6 comparison after reverse-audit freezing.

Feature manifests are constructed from phenotype availability before the
clinical outcome column is loaded. The program never reads CIMT data and is
restricted to exploratory internal mBRSET patient-level cross-validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTCOME_TOKENS = (
    "hypertension", "cimt", "vascular_disease", "diabetic_foot",
    "myocardial", "infarction", "outcome", "nephropathy", "neuropathy",
)


def make_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=c_value, class_weight="balanced", max_iter=5000,
            random_state=seed,
        )),
    ])


def choose_c_brier(
    frame: pd.DataFrame,
    columns: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    grid: list[float],
    seed: int,
) -> float:
    scores: dict[float, float] = {}
    for c_value in grid:
        values = []
        for fit, hold in splits:
            model = make_model(c_value, seed).fit(
                frame.iloc[fit][columns], frame.iloc[fit]["outcome"]
            )
            probability = model.predict_proba(frame.iloc[hold][columns])[:, 1]
            values.append(brier_score_loss(frame.iloc[hold]["outcome"], probability))
        scores[c_value] = float(np.mean(values))
    return min(scores, key=scores.get)


def fit_platt(probability: np.ndarray, outcome: np.ndarray, seed: int) -> LogisticRegression:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=5000, random_state=seed).fit(logits, outcome)


def apply_platt(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def ece_10bin(outcome: np.ndarray, probability: np.ndarray) -> float:
    bins = np.linspace(0, 1, 11)
    value = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        selected = (probability >= low) & (
            (probability < high) if high < 1 else (probability <= high)
        )
        if selected.any():
            value += selected.mean() * abs(
                float(outcome[selected].mean()) - float(probability[selected].mean())
            )
    return float(value)


def metric_summary(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, max_iter=5000).fit(logits, outcome)
    return {
        "n": int(len(outcome)), "positive": int(outcome.sum()),
        "prevalence": float(outcome.mean()),
        "auroc": float(roc_auc_score(outcome, probability)),
        "auprc": float(average_precision_score(outcome, probability)),
        "brier": float(brier_score_loss(outcome, probability)),
        "log_loss": float(log_loss(outcome, probability, labels=[0, 1])),
        "ece_10bin": ece_10bin(outcome, probability),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def paired_bootstrap(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    joined = challenger.merge(
        baseline, on=["patient_id", "outcome"],
        suffixes=("_challenger", "_baseline"), validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("auroc", "auprc", "brier", "log_loss", "ece_10bin")}
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        if sample["outcome"].nunique() != 2:
            continue
        outcome = sample["outcome"].to_numpy(dtype=int)
        left = np.clip(sample["probability_challenger"].to_numpy(float), 1e-6, 1 - 1e-6)
        right = np.clip(sample["probability_baseline"].to_numpy(float), 1e-6, 1 - 1e-6)
        left_metrics = {
            "auroc": roc_auc_score(outcome, left),
            "auprc": average_precision_score(outcome, left),
            "brier": brier_score_loss(outcome, left),
            "log_loss": log_loss(outcome, left, labels=[0, 1]),
            "ece_10bin": ece_10bin(outcome, left),
        }
        right_metrics = {
            "auroc": roc_auc_score(outcome, right),
            "auprc": average_precision_score(outcome, right),
            "brier": brier_score_loss(outcome, right),
            "log_loss": log_loss(outcome, right, labels=[0, 1]),
            "ece_10bin": ece_10bin(outcome, right),
        }
        for name in values:
            values[name].append(left_metrics[name] - right_metrics[name])
    return {
        name: {
            "mean_delta": float(np.mean(metric_values)),
            "ci95": [
                float(np.quantile(metric_values, .025)),
                float(np.quantile(metric_values, .975)),
            ],
        }
        for name, metric_values in values.items()
    }


def build_feature_manifest(
    phenotypes: pd.DataFrame,
    config: dict[str, object],
) -> tuple[dict[str, list[str]], dict[str, object]]:
    suffix = "__bilateral_mean"
    nine = [f"{feature}{suffix}" for feature in config["prespecified_eye_phenotypes"]]
    reliable = [f"{feature}{suffix}" for feature in config["previously_frozen_reliable_core"]]
    rejected = [f"{feature}{suffix}" for feature in config["previously_rejected_core"]]
    candidate_full = [
        column for column in phenotypes.columns
        if column.endswith(("__bilateral_mean", "__absolute_difference", "__within_eye_std"))
        and not any(token in column.lower() for token in OUTCOME_TOKENS)
    ]
    threshold = float(config["minimum_reverse_feature_coverage"])
    coverage = {
        column: float(phenotypes[column].notna().mean())
        for column in candidate_full
    }
    full = sorted(column for column in candidate_full if coverage[column] >= threshold)
    reverse_requested = list(config["prespecified_reverse_patient_features"])
    reverse = [
        column for column in reverse_requested
        if column in phenotypes and float(phenotypes[column].notna().mean()) >= threshold
    ]
    required = sorted(set(nine + reliable + rejected))
    missing = sorted(set(required) - set(phenotypes.columns))
    if missing:
        raise KeyError(f"Missing frozen patient-level phenotype columns: {missing}")
    clinical = list(config["clinical_covariates"])
    sets = {
        "M0_clinical": clinical,
        "M1_all9_only": nine,
        "M2_reliable4_only": reliable,
        "M3_rejected5_only": rejected,
        "M4_full_frozen_derived_only": full,
        "M5_clinical_reliable4": clinical + reliable,
        "M6_clinical_reverse": clinical + reverse,
    }
    audit = {
        "outcome_blind": True,
        "coverage_threshold": threshold,
        "coverage": coverage,
        "reverse_requested": reverse_requested,
        "reverse_retained_by_coverage": reverse,
        "reverse_stopped_for_coverage": sorted(set(reverse_requested) - set(reverse)),
        "full_derived_count": len(full),
        "feature_sets": sets,
    }
    return sets, audit


def run_profile(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    seed = int(config["seed"])
    grid = [float(value) for value in config["regularization_grid"]]
    outputs, tuning = [], []
    for fold, (fit, hold) in enumerate(outer_splits, start=1):
        train = frame.iloc[fit].reset_index(drop=True)
        validation = frame.iloc[hold].reset_index(drop=True)
        inner = StratifiedKFold(
            n_splits=int(config["inner_folds"]), shuffle=True,
            random_state=seed + fold,
        )
        inner_splits = list(inner.split(train, train["outcome"]))
        c_value = choose_c_brier(train, columns, inner_splits, grid, seed + fold)
        crossfit = np.empty(len(train), dtype=float)
        for inner_fit, inner_hold in inner_splits:
            model = make_model(c_value, seed + fold).fit(
                train.iloc[inner_fit][columns], train.iloc[inner_fit]["outcome"]
            )
            crossfit[inner_hold] = model.predict_proba(
                train.iloc[inner_hold][columns]
            )[:, 1]
        calibrator = fit_platt(crossfit, train["outcome"].to_numpy(int), seed + fold)
        fitted = make_model(c_value, seed + fold).fit(train[columns], train["outcome"])
        raw = fitted.predict_proba(validation[columns])[:, 1]
        calibrated = apply_platt(calibrator, raw)
        for variant, probability in (("raw", raw), ("platt", calibrated)):
            result = validation[["patient_id", "outcome"]].copy()
            result["experiment"] = f"{name}_{variant}"
            result["fold"] = fold
            result["probability"] = probability
            outputs.append(result)
        tuning.append({
            "profile": name, "fold": fold, "features": len(columns),
            "brier_selected_c": c_value,
            "platt_intercept": float(calibrator.intercept_[0]),
            "platt_slope": float(calibrator.coef_[0, 0]),
        })
        print(f"{name} fold={fold}/{len(outer_splits)} features={len(columns)} C={c_value}", flush=True)
    return pd.concat(outputs, ignore_index=True), tuning


def plot_summary(overall: dict[str, dict[str, float]], output: Path) -> None:
    records = []
    for experiment, values in overall.items():
        if experiment.endswith("_platt"):
            records.append({"experiment": experiment.removesuffix("_platt"), **values})
    frame = pd.DataFrame(records).sort_values("experiment")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#6E5AA6", "#149C95", "#D19A2A"]
    for axis, metric, color in zip(axes, ("auroc", "auprc", "brier"), colors):
        axis.bar(frame["experiment"], frame[metric], color=color)
        axis.set_title(metric.upper())
        axis.tick_params(axis="x", rotation=70, labelsize=8)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen M0–M6 patient-level comparison")
    fig.tight_layout()
    fig.savefig(output / "m0_m6_metrics.png", dpi=300)
    fig.savefig(output / "m0_m6_metrics.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/rmdb/reverse_audit_v1.json"),
    )
    parser.add_argument("--outcome", default="systemic_hypertension")
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--no-dr-only", action="store_true")
    parser.add_argument("--strict-disc-qc", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if "cimt" in args.outcome.lower():
        raise ValueError("CIMT fixed validation or test data are prohibited in this stage.")
    args.output.mkdir(parents=True, exist_ok=True)

    phenotype_columns = pd.read_csv(args.phenotypes, nrows=0).columns.tolist()
    phenotype_frame = pd.read_csv(
        args.phenotypes,
        usecols=[column for column in phenotype_columns if not any(
            token in column.lower() for token in OUTCOME_TOKENS
        )],
        dtype={"patient_id": str},
    ).drop(columns=["split"], errors="ignore")
    feature_sets, feature_audit = build_feature_manifest(phenotype_frame, config)
    if not feature_audit["reverse_retained_by_coverage"]:
        raise ValueError("M6 stopped: no prespecified reverse feature passed coverage.")
    (args.output / "outcome_blind_feature_manifest.json").write_text(
        json.dumps(feature_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    patient_headers = pd.read_csv(args.patients, nrows=0).columns.tolist()
    requested_patient = ["patient_id", args.outcome] + list(config["clinical_covariates"])
    for optional in ("all_images_no_dr",):
        if optional in patient_headers:
            requested_patient.append(optional)
    patients = pd.read_csv(
        args.patients, usecols=sorted(set(requested_patient)),
        dtype={"patient_id": str},
    )
    frame = patients.merge(phenotype_frame, on="patient_id", validate="one_to_one")
    if args.no_dr_only:
        if "all_images_no_dr" not in frame:
            raise KeyError("No-DR sensitivity requires all_images_no_dr.")
        frame = frame.loc[frame["all_images_no_dr"].eq(True)].copy()
    if args.strict_disc_qc:
        if "both_discs_detected" not in frame:
            raise KeyError("Strict disc sensitivity requires both_discs_detected.")
        frame = frame.loc[frame["both_discs_detected"].eq(True)].copy()
    frame = frame.loc[frame[args.outcome].notna()].copy()
    frame["outcome"] = frame[args.outcome].astype(int)
    if frame["outcome"].value_counts().min() < int(config["outer_folds"]):
        raise ValueError("Too few events for the frozen outer-fold count.")
    missing = sorted({column for columns in feature_sets.values() for column in columns} - set(frame.columns))
    if missing:
        raise KeyError(f"Missing frozen model columns: {missing}")

    outer = StratifiedKFold(
        n_splits=int(config["outer_folds"]), shuffle=True,
        random_state=int(config["seed"]),
    )
    outer_splits = list(outer.split(frame, frame["outcome"]))
    outputs, tuning = [], []
    for name, columns in feature_sets.items():
        result, rows = run_profile(frame, columns, name, outer_splits, config)
        outputs.append(result)
        tuning.extend(rows)
    oof = pd.concat(outputs, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)
    overall = {
        name: metric_summary(group["outcome"].to_numpy(int), group["probability"].to_numpy(float))
        for name, group in oof.groupby("experiment")
    }
    repetitions = args.bootstrap_repetitions or int(config["bootstrap_repetitions"])
    pairs = {
        "M2_minus_M1": ("M2_reliable4_only", "M1_all9_only"),
        "M3_minus_M2": ("M3_rejected5_only", "M2_reliable4_only"),
        "M4_minus_M2": ("M4_full_frozen_derived_only", "M2_reliable4_only"),
        "M5_minus_M0": ("M5_clinical_reliable4", "M0_clinical"),
        "M6_minus_M0": ("M6_clinical_reverse", "M0_clinical"),
        "M6_minus_M5": ("M6_clinical_reverse", "M5_clinical_reliable4"),
    }
    comparisons = {
        label: paired_bootstrap(
            oof.loc[oof["experiment"].eq(f"{challenger}_platt")],
            oof.loc[oof["experiment"].eq(f"{baseline}_platt")],
            repetitions, int(config["seed"]),
        )
        for label, (challenger, baseline) in pairs.items()
    }
    report = {
        "protocol": "RMDB_BiRAP_forward_reverse_locked_protocol_20260821.md",
        "status": "exploratory_internal_one_shot_m0_m6",
        "dataset": "mBRSET", "outcome": args.outcome,
        "no_dr_only": args.no_dr_only, "strict_disc_qc": args.strict_disc_qc,
        "patients": int(len(frame)), "positive": int(frame["outcome"].sum()),
        "cimt_fixed_validation_or_test_read": False,
        "feature_sets": feature_sets, "overall": overall,
        "paired_bootstrap": comparisons,
        "bootstrap_repetitions": repetitions,
        "permutation_control_required_separately": True,
        "interpretation_boundary": (
            "Exploratory patient-level internal CV in a previously explored portable-camera cohort; "
            "not independent high-resolution clinical confirmation."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_summary(overall, args.output)
    print(json.dumps({"patients": report["patients"], "positive": report["positive"],
                      "models": list(feature_sets), "comparisons": comparisons},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
