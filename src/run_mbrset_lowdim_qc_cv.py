#!/usr/bin/env python3
"""Exploratory, label-blind low-dimensional mBRSET phenotype CV.

This script deliberately has no outcome-derived feature screening.  It is a
post-hoc robustness experiment and must not be described as independent
external confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260731
CLINICAL = ["age", "sex", "dm_time"]
OCULAR = ["max_icdr", "mean_icdr", "any_dr", "any_edema"]
CORE_BASES = [
    "soft_vessel_mass_fraction",
    "soft_av_log_mass_ratio",
    "mass_vessel_skeleton_length_density",
    "mass_vessel_branch_cluster_density",
    "factorized_vessel_skeleton_length_density",
    "factorized_av_log_area_ratio",
    "vessel_occupancy_disagreement",
    "base_to_mass_identity_disagreement",
    "identity_entropy",
]
RELIABLE_CORE_BASES = [
    "soft_vessel_mass_fraction",
    "mass_vessel_skeleton_length_density",
    "factorized_vessel_skeleton_length_density",
    "vessel_occupancy_disagreement",
]


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000,
                                      random_state=SEED)),
    ])


def choose_c(frame: pd.DataFrame, columns: list[str], splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scores: dict[float, float] = {}
    for c_value in (.001, .01, .1, 1.0, 10.0):
        fold_scores = []
        for train, hold in splits:
            model = make_pipeline(c_value)
            model.fit(frame.iloc[train][columns], frame.iloc[train]["outcome"])
            probability = model.predict_proba(frame.iloc[hold][columns])[:, 1]
            fold_scores.append(roc_auc_score(frame.iloc[hold]["outcome"], probability))
        scores[c_value] = float(np.mean(fold_scores))
    return max(scores, key=scores.get)


def model_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["outcome"].to_numpy(dtype=int)
    probability = frame["probability"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)), "positive": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def bootstrap(challenger: pd.DataFrame, baseline: pd.DataFrame, repetitions: int) -> dict[str, object]:
    joined = challenger.merge(baseline, on=["patient_id", "outcome"], suffixes=("_challenger", "_baseline"),
                              validate="one_to_one")
    rng = np.random.default_rng(SEED)
    values = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(repetitions):
        sampled = joined.iloc[rng.integers(0, len(joined), len(joined))]
        if sampled["outcome"].nunique() != 2:
            continue
        y = sampled["outcome"]
        pc, pb = sampled["probability_challenger"], sampled["probability_baseline"]
        values["auroc"].append(roc_auc_score(y, pc) - roc_auc_score(y, pb))
        values["auprc"].append(average_precision_score(y, pc) - average_precision_score(y, pb))
        values["brier"].append(brier_score_loss(y, pc) - brier_score_loss(y, pb))
    return {name: {"mean_delta": float(np.mean(v)), "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}
            for name, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", required=True, type=Path)
    parser.add_argument("--phenotypes", required=True, type=Path)
    parser.add_argument("--outcome", default="systemic_hypertension")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--no-dr-only", action="store_true")
    parser.add_argument("--strict-disc-qc", action="store_true")
    parser.add_argument("--profile", choices=("core9", "repeatability_icc60"), default="core9")
    args = parser.parse_args()

    patients = pd.read_csv(args.patients, dtype={"patient_id": str})
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str}).drop(columns=["split"], errors="ignore")
    frame = patients.merge(phenotypes, on="patient_id", validate="one_to_one")
    if args.no_dr_only:
        frame = frame.loc[frame["all_images_no_dr"].eq(True)].copy()
    if args.strict_disc_qc:
        frame = frame.loc[frame["both_discs_detected"].eq(True)].copy()
    frame = frame.loc[frame[args.outcome].notna()].copy()
    frame["outcome"] = frame[args.outcome].astype(int)
    bases = CORE_BASES if args.profile == "core9" else RELIABLE_CORE_BASES
    core = [f"{name}__bilateral_mean" for name in bases]
    feature_sets = {
        "R0_clinical": CLINICAL,
        "R1_clinical_ocular": CLINICAL + OCULAR,
        "R2_clinical_lowdim_vascular": CLINICAL + core,
        "R3_full_lowdim_vascular": CLINICAL + OCULAR + core,
    }
    missing = [column for columns in feature_sets.values() for column in columns if column not in frame]
    if missing:
        raise KeyError(f"Missing frozen columns: {sorted(set(missing))}")
    if frame["outcome"].value_counts().min() < 5:
        raise ValueError("Too few events for five folds")

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outputs, tuning = [], []
    for name, columns in feature_sets.items():
        for fold, (train_idx, hold_idx) in enumerate(outer.split(frame, frame["outcome"]), start=1):
            train, hold = frame.iloc[train_idx], frame.iloc[hold_idx]
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
            c_value = choose_c(train, columns, list(inner_cv.split(train, train["outcome"])))
            model = make_pipeline(c_value).fit(train[columns], train["outcome"])
            result = hold[["patient_id", "outcome"]].copy()
            result["experiment"], result["fold"] = name, fold
            result["probability"] = model.predict_proba(hold[columns])[:, 1]
            outputs.append(result)
            tuning.append({"experiment": name, "fold": fold, "features": len(columns), "logistic_c": c_value})
            print(f"{name} fold={fold}/5 features={len(columns)} C={c_value}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(outputs, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)
    overall = {name: model_metrics(group) for name, group in oof.groupby("experiment")}
    pairs = {
        "R1__minus__R0": ("R1_clinical_ocular", "R0_clinical"),
        "R2__minus__R0": ("R2_clinical_lowdim_vascular", "R0_clinical"),
        "R3__minus__R1": ("R3_full_lowdim_vascular", "R1_clinical_ocular"),
        "R3__minus__R2": ("R3_full_lowdim_vascular", "R2_clinical_lowdim_vascular"),
    }
    report = {
        "status": "exploratory_post_hoc_label_blind_feature_definition",
        "dataset": "mBRSET_224px_mirror", "outcome": args.outcome,
        "no_dr_only": args.no_dr_only, "strict_disc_qc": args.strict_disc_qc, "profile": args.profile,
        "patients": int(len(frame)), "positive": int(frame["outcome"].sum()),
        "feature_sets": feature_sets, "overall": overall,
        "paired_bootstrap": {
            label: bootstrap(oof.loc[oof.experiment.eq(challenger)], oof.loc[oof.experiment.eq(baseline)], args.bootstrap_repetitions)
            for label, (challenger, baseline) in pairs.items()
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
