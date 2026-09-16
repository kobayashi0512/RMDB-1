#!/usr/bin/env python3
"""Nested, patient-level calibration audit for the fixed mBRSET reliable core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260810
CLINICAL = ["age", "sex", "dm_time"]
RELIABLE_CORE = [
    "soft_vessel_mass_fraction__bilateral_mean",
    "mass_vessel_skeleton_length_density__bilateral_mean",
    "factorized_vessel_skeleton_length_density__bilateral_mean",
    "vessel_occupancy_disagreement__bilateral_mean",
]
CS = (.001, .01, .1, 1.0, 10.0)


def base_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])


def choose_c_brier(frame: pd.DataFrame, columns: list[str], splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scores = {}
    for c_value in CS:
        values = []
        for fit, hold in splits:
            model = base_model(c_value).fit(frame.iloc[fit][columns], frame.iloc[fit].outcome)
            values.append(brier_score_loss(frame.iloc[hold].outcome, model.predict_proba(frame.iloc[hold][columns])[:, 1]))
        scores[c_value] = float(np.mean(values))
    return min(scores, key=scores.get)


def platt_model(probability: np.ndarray, outcome: np.ndarray) -> LogisticRegression:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1 - 1e-6))
    return LogisticRegression(C=1e6, max_iter=5000, random_state=SEED).fit(logits.reshape(-1, 1), outcome)


def apply_platt(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1 - 1e-6))
    return model.predict_proba(logits.reshape(-1, 1))[:, 1]


def ece_10bin(y: np.ndarray, probability: np.ndarray) -> float:
    """Fixed-bin ECE; deliberately contains no fitted calibration model."""
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        take = (probability >= low) & ((probability < high) if high < 1 else (probability <= high))
        if take.any():
            ece += take.mean() * abs(float(y[take].mean()) - float(probability[take].mean()))
    return float(ece)


def calibration_summary(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability))
    model = LogisticRegression(C=1e6, max_iter=5000, random_state=SEED).fit(logits.reshape(-1, 1), y)
    return {
        "n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "ece_10bin": ece_10bin(y, probability),
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def paired_bootstrap(challenger: pd.DataFrame, baseline: pd.DataFrame, repetitions: int) -> dict[str, dict[str, object]]:
    joined = challenger.merge(baseline, on=["patient_id", "outcome"], suffixes=("_challenger", "_baseline"), validate="one_to_one")
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in ("auroc", "auprc", "brier", "log_loss", "ece_10bin")}
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        if sample.outcome.nunique() != 2:
            continue
        y = sample.outcome.to_numpy(int)
        left = np.clip(sample.probability_challenger.to_numpy(float), 1e-6, 1 - 1e-6)
        right = np.clip(sample.probability_baseline.to_numpy(float), 1e-6, 1 - 1e-6)
        metrics_left = {
            "auroc": roc_auc_score(y, left), "auprc": average_precision_score(y, left),
            "brier": brier_score_loss(y, left), "log_loss": log_loss(y, left, labels=[0, 1]),
            "ece_10bin": ece_10bin(y, left),
        }
        metrics_right = {
            "auroc": roc_auc_score(y, right), "auprc": average_precision_score(y, right),
            "brier": brier_score_loss(y, right), "log_loss": log_loss(y, right, labels=[0, 1]),
            "ece_10bin": ece_10bin(y, right),
        }
        for name in values:
            values[name].append(metrics_left[name] - metrics_right[name])
    return {
        name: {"mean_delta": float(np.mean(value)), "ci95": [float(np.quantile(value, .025)), float(np.quantile(value, .975))]}
        for name, value in values.items()
    }


def run_profile(frame: pd.DataFrame, columns: list[str], name: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    output, tuning = [], []
    for fold, (fit, hold) in enumerate(outer.split(frame, frame.outcome), start=1):
        train, validation = frame.iloc[fit].reset_index(drop=True), frame.iloc[hold].reset_index(drop=True)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        splits = list(inner.split(train, train.outcome))
        c_value = choose_c_brier(train, columns, splits)
        crossfit_probability = np.empty(len(train), dtype=float)
        for inner_fit, inner_hold in splits:
            fitted = base_model(c_value).fit(train.iloc[inner_fit][columns], train.iloc[inner_fit].outcome)
            crossfit_probability[inner_hold] = fitted.predict_proba(train.iloc[inner_hold][columns])[:, 1]
        calibrator = platt_model(crossfit_probability, train.outcome.to_numpy(int))
        fitted = base_model(c_value).fit(train[columns], train.outcome)
        raw = fitted.predict_proba(validation[columns])[:, 1]
        calibrated = apply_platt(calibrator, raw)
        for variant, probability in (("raw", raw), ("platt", calibrated)):
            result = validation[["patient_id", "outcome"]].copy()
            result["experiment"] = f"{name}_{variant}"
            result["fold"] = fold
            result["probability"] = probability
            output.append(result)
        tuning.append({
            "profile": name, "fold": fold, "features": len(columns), "brier_selected_c": c_value,
            "platt_intercept": float(calibrator.intercept_[0]), "platt_slope": float(calibrator.coef_[0, 0]),
        })
        print(f"{name} fold={fold}/5 C={c_value}", flush=True)
    return pd.concat(output, ignore_index=True), tuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--outcome", default="systemic_hypertension")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--no-dr-only", action="store_true")
    parser.add_argument("--strict-disc-qc", action="store_true")
    args = parser.parse_args()
    patients = pd.read_csv(args.patients, dtype={"patient_id": str})
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str}).drop(columns=["split"], errors="ignore")
    frame = patients.merge(phenotypes, on="patient_id", validate="one_to_one")
    if args.no_dr_only:
        frame = frame.loc[frame.all_images_no_dr.eq(True)].copy()
    if args.strict_disc_qc:
        frame = frame.loc[frame.both_discs_detected.eq(True)].copy()
    frame = frame.loc[frame[args.outcome].notna()].copy()
    frame["outcome"] = frame[args.outcome].astype(int)
    profiles = {"R0_clinical": CLINICAL, "R2_reliable_core": CLINICAL + RELIABLE_CORE}
    missing = sorted({column for values in profiles.values() for column in values} - set(frame.columns))
    if missing:
        raise KeyError(f"Missing locked columns: {missing}")
    if frame.outcome.value_counts().min() < 5:
        raise ValueError("Too few events for five-fold patient-level CV.")
    outputs, tuning = [], []
    for name, columns in profiles.items():
        result, rows = run_profile(frame, columns, name)
        outputs.append(result); tuning.extend(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(outputs, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)
    overall = {name: calibration_summary(group.outcome.to_numpy(int), group.probability.to_numpy(float)) for name, group in oof.groupby("experiment")}
    comparisons = {
        "R0_platt__minus__R0_raw": paired_bootstrap(oof[oof.experiment.eq("R0_clinical_platt")], oof[oof.experiment.eq("R0_clinical_raw")], args.bootstrap_repetitions),
        "R2_platt__minus__R2_raw": paired_bootstrap(oof[oof.experiment.eq("R2_reliable_core_platt")], oof[oof.experiment.eq("R2_reliable_core_raw")], args.bootstrap_repetitions),
        "R2_platt__minus__R0_platt": paired_bootstrap(oof[oof.experiment.eq("R2_reliable_core_platt")], oof[oof.experiment.eq("R0_clinical_platt")], args.bootstrap_repetitions),
    }
    report = {
        "protocol": "mBRSET_可靠核心嵌套校准探索协议.md", "status": "exploratory_post_hoc_mechanism_calibration",
        "dataset": "mBRSET_224px_mirror", "outcome": args.outcome, "no_dr_only": args.no_dr_only,
        "strict_disc_qc": args.strict_disc_qc, "patients": int(len(frame)), "positive": int(frame.outcome.sum()),
        "profiles": profiles, "overall": overall, "paired_bootstrap": comparisons,
        "interpretation_boundary": "Internal patient-level CV in a repeatedly explored 224px external mechanism cohort; not independent confirmation.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
