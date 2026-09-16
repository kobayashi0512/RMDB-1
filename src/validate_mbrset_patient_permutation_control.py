#!/usr/bin/env python3
"""Patient-level permutation negative control for fixed mBRSET R0/R2 profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260810
CLINICAL = ["age", "sex", "dm_time"]
CORE = [
    "soft_vessel_mass_fraction__bilateral_mean", "mass_vessel_skeleton_length_density__bilateral_mean",
    "factorized_vessel_skeleton_length_density__bilateral_mean", "vessel_occupancy_disagreement__bilateral_mean",
]
CS = (.001, .01, .1, 1.0, 10.0)


def model(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])


def choose_c(frame: pd.DataFrame, columns: list[str], outcome: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scores = {}
    for c_value in CS:
        values = []
        for fit, hold in splits:
            fitted = model(c_value).fit(frame.iloc[fit][columns], outcome[fit])
            values.append(brier_score_loss(outcome[hold], fitted.predict_proba(frame.iloc[hold][columns])[:, 1]))
        scores[c_value] = float(np.mean(values))
    return min(scores, key=scores.get)


def nested_oof(frame: pd.DataFrame, columns: list[str], outcome: np.ndarray, split_seed: int) -> tuple[float, list[dict[str, int]]]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    prediction = np.empty(len(frame), dtype=float)
    audit = []
    for fold, (fit, hold) in enumerate(outer.split(frame, outcome), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed + fold)
        inner_splits = list(inner.split(frame.iloc[fit], outcome[fit]))
        c_value = choose_c(frame.iloc[fit].reset_index(drop=True), columns, outcome[fit], inner_splits)
        fitted = model(c_value).fit(frame.iloc[fit][columns], outcome[fit])
        prediction[hold] = fitted.predict_proba(frame.iloc[hold][columns])[:, 1]
        overlap = set(frame.iloc[fit].patient_id).intersection(frame.iloc[hold].patient_id)
        audit.append({"fold": fold, "train_patients": int(len(fit)), "hold_patients": int(len(hold)), "patient_overlap": int(len(overlap)), "selected_c": float(c_value)})
    return float(roc_auc_score(outcome, prediction)), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcome", default="systemic_hypertension")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    patients = pd.read_csv(args.patients, dtype={"patient_id": str})
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str}).drop(columns=["split"], errors="ignore")
    frame = patients.merge(phenotypes, on="patient_id", validate="one_to_one")
    frame = frame.loc[frame[args.outcome].notna()].reset_index(drop=True)
    if frame.patient_id.nunique() != len(frame):
        raise ValueError("Patient-level negative control requires exactly one row per patient.")
    outcome = frame[args.outcome].to_numpy(int)
    profiles = {"R0_clinical": CLINICAL, "R2_reliable_core": CLINICAL + CORE}
    missing = sorted({field for values in profiles.values() for field in values} - set(frame.columns))
    if missing:
        raise KeyError(f"Missing fixed profile fields: {missing}")
    observed, fold_audits = {}, []
    for name, columns in profiles.items():
        auc, audit = nested_oof(frame, columns, outcome, SEED)
        observed[name] = auc
        fold_audits.extend([{"profile": name, **row} for row in audit])
    audit_frame = pd.DataFrame(fold_audits)
    if not audit_frame.patient_overlap.eq(0).all():
        raise RuntimeError("Patient leakage detected in outer folds.")
    rng = np.random.default_rng(SEED)
    rows = []
    for repetition in range(args.permutations):
        shuffled = rng.permutation(outcome)
        for name, columns in profiles.items():
            auc, _ = nested_oof(frame, columns, shuffled, SEED + 1000 + repetition)
            rows.append({"permutation": repetition + 1, "profile": name, "auroc": auc})
        if (repetition + 1) % 20 == 0:
            print(f"permutation={repetition + 1}/{args.permutations}", flush=True)
    null = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    null.to_csv(args.output / "permutation_auroc.csv", index=False)
    audit_frame.to_csv(args.output / "outer_fold_patient_audit.csv", index=False)
    summary = {}
    for name, value in observed.items():
        values = null.loc[null.profile.eq(name), "auroc"].to_numpy(float)
        summary[name] = {
            "observed_auroc": value, "null_mean": float(values.mean()),
            "null_ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "empirical_p_one_sided": float((1 + (values >= value).sum()) / (1 + len(values))),
        }
    report = {
        "protocol": "mBRSET_患者级置换负对照协议.md", "dataset": "mBRSET_224px_mirror",
        "patients": int(len(frame)), "positive": int(outcome.sum()), "permutations": args.permutations,
        "patient_overlap_all_outer_folds_zero": True, "summary": summary,
        "interpretation_boundary": "Implementation negative control in an exploratory internal CV cohort; not independent clinical validation.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

