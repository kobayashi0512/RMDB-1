#!/usr/bin/env python3
"""E12: repeatability-gated conditional-centile CIMT development CV.

The feature core is fixed by an outcome-blind, independent mBRSET repeat-view
audit.  This script reads CIMT split=train only and never opens fixed val/test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from run_e10_distributionally_robust_cv import (
    CENTILE_THRESHOLD, SEEDS, bootstrap_delta, conditional_centiles,
    inverse_centiles, metrics, output_frame, tune_classifier, tune_ridge,
)
from run_e8_double_orthogonal_cv import MASS_MECHANISMS, residualize_features
from run_normative_deviation_outer_cv import (
    bilateral, crossfit_normative, phenotype_classifier, phenotype_regressor, strata,
)


REPEATABILITY_CORE = [
    "soft_vessel_mass_fraction",
    "mass_vessel_skeleton_length_density",
    "factorized_vessel_skeleton_length_density",
    "vessel_occupancy_disagreement",
]


def run_profile(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int,
    fold: int,
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
    inner_splits = list(inner.split(train, strata(train)))
    normative = crossfit_normative(train, validation, 3, seed + fold)
    train_orth, validation_orth = residualize_features(train, validation, features, inner_splits)
    centile = conditional_centiles(train, validation, normative["mu_train"], normative["mu_validation"])
    z_train, z_validation = centile["z_train"], centile["z_validation"]
    binary_train = (z_train >= CENTILE_THRESHOLD).astype(int)
    binary_validation = (z_validation >= CENTILE_THRESHOLD).astype(int)
    alpha = tune_ridge(train_orth, z_train, inner_splits, robust=False)
    regressor = phenotype_regressor(alpha).fit(train_orth, z_train)
    predicted_z = regressor.predict(validation_orth)
    prediction = inverse_centiles(
        predicted_z, centile["validation_groups"], centile["pools"],
        centile["global_pool"], normative["mu_validation"],
    )
    mean_count = len(features) // 2
    c_value = tune_classifier(train_orth.iloc[:, :mean_count], binary_train, inner_splits)
    classifier = phenotype_classifier(c_value).fit(train_orth.iloc[:, :mean_count], binary_train)
    probability = classifier.predict_proba(validation_orth.iloc[:, :mean_count])[:, 1]
    return output_frame(
        validation, label, seed, fold, z_validation, predicted_z, prediction,
        binary_validation, probability,
    ), {"ridge_alpha": alpha, "logistic_c": c_value, "feature_count": len(features)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    seeds = SEEDS[:args.seeds]

    outcomes = pd.read_csv(
        args.manifest, usecols=["patient_id", "split", "age", "sex", "cimt_mm"], dtype={"patient_id": str},
    )
    outcomes = outcomes.loc[outcomes.split.eq("train")].copy()
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    if not phenotypes.split.eq("train").all():
        raise ValueError("E12 is development-only: phenotype input must contain split=train only.")
    frame = outcomes.merge(phenotypes.drop(columns=["split"]), on="patient_id", validate="one_to_one").reset_index(drop=True)
    mass_features = bilateral(MASS_MECHANISMS)
    reliable_features = bilateral(REPEATABILITY_CORE)
    required = ["age", "sex", "cimt_mm", *mass_features, *reliable_features]
    if len(frame) != 2603 or frame[required].isna().any().any():
        raise ValueError("Incomplete E12 development input; expected 2603 complete train patients.")

    profiles = {
        "E12A_e10a_mass5_reference": mass_features,
        "E12B_repeatability_gated_core4": reliable_features,
    }
    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    for seed in seeds:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit, hold) in enumerate(outer.split(frame, strata(frame)), start=1):
            train, validation = frame.iloc[fit].reset_index(drop=True), frame.iloc[hold].reset_index(drop=True)
            for label, features in profiles.items():
                result, record = run_profile(train, validation, features, seed, fold, label)
                predictions.append(result)
                tuning.append({"experiment": label, "seed": seed, "fold": fold, **record})
            print(f"seed={seed} fold={fold}/5 complete", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "repeated_oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)
    per_seed = []
    for (experiment, seed), group in oof.groupby(["experiment", "seed"]):
        per_seed.append({"experiment": experiment, "seed": int(seed), **metrics(group)})
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
    summary = per_seed_frame.groupby("experiment").agg(
        {column: ["mean", "std"] for column in per_seed_frame.columns if column not in ("experiment", "seed")}
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    reference = oof.loc[oof.experiment.eq("E12A_e10a_mass5_reference")]
    challenger = oof.loc[oof.experiment.eq("E12B_repeatability_gated_core4")]
    comparison = bootstrap_delta(challenger, reference, args.bootstrap_repetitions, seeds[0])
    fold_deltas = []
    for (seed, fold), group in challenger.groupby(["seed", "fold"]):
        base = reference.loc[reference.seed.eq(seed) & reference.fold.eq(fold)]
        fold_deltas.append(mean_absolute_error(group.cimt_mm, group.prediction_mm) - mean_absolute_error(base.cimt_mm, base.prediction_mm))
    report = {
        "protocol": "E12_重复性约束条件百分位微血管偏离_锁定协议.md",
        "development_only": True,
        "fixed_val_used": False,
        "test_used": False,
        "patients": int(len(frame)),
        "seeds": seeds,
        "outer_folds_per_seed": 5,
        "profiles": profiles,
        "summary": summary.to_dict(orient="records"),
        "repeatability_gated_minus_mass5_mae_mm": comparison,
        "fold_direction": {
            "folds_improved_or_equal": int((np.asarray(fold_deltas) <= 0).sum()),
            "folds_total": len(fold_deltas),
            "mean_fold_delta_mm": float(np.mean(fold_deltas)),
        },
        "interpretation_boundary": "Development-only, outcome-blind feature constraint; requires a new independent CIMT cohort.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({"repeatability_gated_minus_mass5_mae_mm": comparison}, indent=2))


if __name__ == "__main__":
    main()

