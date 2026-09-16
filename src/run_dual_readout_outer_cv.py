#!/usr/bin/env python3
"""Leakage-safe outer-CV benchmark for locked dual-readout phenotypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260729


def regression_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def classification_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=c_value,
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


def strata(frame: pd.DataFrame) -> pd.Series:
    bins = pd.qcut(
        frame["cimt_mm"], q=4, labels=False, duplicates="drop"
    ).astype(str)
    return frame["sex"].astype(str) + "_" + bins


def tune(
    train: pd.DataFrame,
    features: list[str],
    inner_splits: int,
    seed: int,
) -> tuple[float, float]:
    inner = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )
    split_indices = list(inner.split(train, strata(train)))
    alpha_scores = {}
    for alpha in (.1, 1.0, 10.0, 100.0, 1000.0):
        scores = []
        for fit, hold in split_indices:
            model = regression_pipeline(alpha)
            model.fit(train.iloc[fit][features], train.iloc[fit]["cimt_mm"])
            prediction = model.predict(train.iloc[hold][features])
            scores.append(mean_absolute_error(
                train.iloc[hold]["cimt_mm"], prediction
            ))
        alpha_scores[alpha] = float(np.mean(scores))
    c_scores = {}
    for c_value in (.001, .01, .1, 1.0, 10.0):
        scores = []
        for fit, hold in split_indices:
            model = classification_pipeline(c_value)
            model.fit(train.iloc[fit][features], train.iloc[fit]["label"])
            probability = model.predict_proba(train.iloc[hold][features])[:, 1]
            scores.append(average_precision_score(
                train.iloc[hold]["label"], probability
            ))
        c_scores[c_value] = float(np.mean(scores))
    return min(alpha_scores, key=alpha_scores.get), max(c_scores, key=c_scores.get)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    probability = np.clip(frame["probability"].to_numpy(), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=5000
    ).fit(logit, frame["label"])
    tn, fp, fn, tp = confusion_matrix(
        frame["label"], probability >= .5, labels=[0, 1]
    ).ravel()
    return {
        "n": int(len(frame)),
        "mae_mm": float(mean_absolute_error(
            frame["cimt_mm"], frame["prediction_mm"]
        )),
        "rmse_mm": float(mean_squared_error(
            frame["cimt_mm"], frame["prediction_mm"]
        ) ** .5),
        "r2": float(r2_score(frame["cimt_mm"], frame["prediction_mm"])),
        "spearman_rho": float(frame[
            ["cimt_mm", "prediction_mm"]
        ].corr(method="spearman").iloc[0, 1]),
        "auroc": float(roc_auc_score(frame["label"], frame["probability"])),
        "auprc": float(average_precision_score(
            frame["label"], frame["probability"]
        )),
        "brier": float(brier_score_loss(
            frame["label"], frame["probability"]
        )),
        "sensitivity_at_0_5": float(tp / max(tp + fn, 1)),
        "specificity_at_0_5": float(tn / max(tn + fp, 1)),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
    }


def bootstrap_delta(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    joined = challenger.merge(
        baseline,
        on=["patient_id", "cimt_mm", "label"],
        suffixes=("_challenger", "_baseline"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    values = {"mae_mm": [], "auroc": [], "auprc": []}
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        values["mae_mm"].append(
            mean_absolute_error(
                sample.cimt_mm, sample.prediction_mm_challenger
            )
            - mean_absolute_error(
                sample.cimt_mm, sample.prediction_mm_baseline
            )
        )
        if sample.label.nunique() == 2:
            values["auroc"].append(
                roc_auc_score(sample.label, sample.probability_challenger)
                - roc_auc_score(sample.label, sample.probability_baseline)
            )
            values["auprc"].append(
                average_precision_score(
                    sample.label, sample.probability_challenger
                )
                - average_precision_score(
                    sample.label, sample.probability_baseline
                )
            )
    output = {}
    for name, samples in values.items():
        array = np.asarray(samples)
        output[name] = {
            "mean_delta": float(array.mean()),
            "ci95": [
                float(np.quantile(array, .025)),
                float(np.quantile(array, .975)),
            ],
        }
    return output


def feature_sets(
    dual: pd.DataFrame, legacy: pd.DataFrame | None
) -> dict[str, list[str]]:
    clinical = ["age", "sex"]
    candidate = [
        column for column in dual.columns
        if column.endswith("__bilateral_mean")
        or column.endswith("__absolute_difference")
    ]
    soft = [column for column in candidate if column.startswith("soft_")]
    mass = [column for column in candidate if column.startswith("mass_")]
    factorized = [
        column for column in candidate if column.startswith("factorized_")
    ]
    uncertainty_tokens = (
        "entropy", "disagreement", "mean_gate",
        "support_pixels", "disc_radius", "zone_b_valid_pixels",
    )
    uncertainty = [
        column for column in candidate
        if any(token in column for token in uncertainty_tokens)
    ]
    dual_all = sorted(set(soft + mass + factorized + uncertainty))
    sets = {
        "E0_clinical": clinical,
        "E2_mass_udct": clinical + sorted(set(soft + mass)),
        "E3_factorized": clinical + sorted(factorized),
        "E4_dual_readout": clinical + dual_all,
        "E5_without_zone_b": clinical + [
            column for column in dual_all if "zone_b" not in column
        ],
        "E6_without_uncertainty": clinical + [
            column for column in dual_all if column not in uncertainty
        ],
    }
    if legacy is not None:
        legacy_features = [
            column for column in legacy.columns
            if column.startswith("common_") or column.startswith("asymmetry_")
        ]
        sets["E1_legacy_vascx"] = clinical + legacy_features
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument(
        "--legacy", type=Path, default=Path("artifacts/cimt_vessel_phenotypes.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    outcomes = pd.read_csv(
        args.manifest,
        usecols=["patient_id", "split", "age", "sex", "cimt_mm", "label"],
        dtype={"patient_id": str},
    )
    outcomes = outcomes.loc[outcomes["split"].eq("train")].copy()
    dual = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    if not dual["split"].eq("train").all():
        raise ValueError("Phenotype input contains a non-development split.")
    frame = outcomes.merge(
        dual.drop(columns=["split"]), on="patient_id", validate="one_to_one"
    )
    if len(frame) != len(outcomes):
        raise ValueError(
            f"Expected {len(outcomes)} development patients, found {len(frame)}."
        )
    legacy = None
    if args.legacy.is_file():
        legacy = pd.read_csv(args.legacy, dtype={"patient_id": str})
        legacy = legacy.loc[legacy["split"].eq("train")].copy()
        legacy_columns = [
            "patient_id",
            *[
                column for column in legacy.columns
                if column.startswith("common_") or column.startswith("asymmetry_")
            ],
        ]
        frame = frame.merge(
            legacy[legacy_columns], on="patient_id", validate="one_to_one"
        )

    sets = feature_sets(dual, legacy)
    outer = StratifiedKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=SEED
    )
    outer_indices = list(outer.split(frame, strata(frame)))
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = []
    tuning_rows = []
    for experiment, features in sets.items():
        absent = [column for column in features if column not in frame]
        if absent:
            raise KeyError(f"{experiment} missing features: {absent[:5]}")
        for fold, (fit, hold) in enumerate(outer_indices, start=1):
            train, validation = frame.iloc[fit], frame.iloc[hold]
            alpha, c_value = tune(
                train, features, args.inner_folds, SEED + fold
            )
            regressor = regression_pipeline(alpha)
            classifier = classification_pipeline(c_value)
            regressor.fit(train[features], train["cimt_mm"])
            classifier.fit(train[features], train["label"])
            output = validation[
                ["patient_id", "cimt_mm", "label", "age", "sex"]
            ].copy()
            output["experiment"] = experiment
            output["fold"] = fold
            output["prediction_mm"] = regressor.predict(validation[features])
            output["probability"] = classifier.predict_proba(
                validation[features]
            )[:, 1]
            predictions.append(output)
            tuning_rows.append({
                "experiment": experiment,
                "fold": fold,
                "features": len(features),
                "ridge_alpha": alpha,
                "logistic_c": c_value,
            })
            print(
                f"{experiment} fold={fold}/{args.outer_folds} "
                f"features={len(features)} alpha={alpha} C={c_value}",
                flush=True,
            )

    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(
        args.output / "nested_tuning.csv", index=False
    )
    overall = {
        experiment: metrics(group)
        for experiment, group in oof.groupby("experiment")
    }
    comparisons = {}
    for challenger, baseline in (
        ("E1_legacy_vascx", "E0_clinical"),
        ("E2_mass_udct", "E0_clinical"),
        ("E3_factorized", "E0_clinical"),
        ("E4_dual_readout", "E0_clinical"),
        ("E4_dual_readout", "E2_mass_udct"),
        ("E4_dual_readout", "E3_factorized"),
        ("E4_dual_readout", "E5_without_zone_b"),
        ("E4_dual_readout", "E6_without_uncertainty"),
    ):
        if challenger not in overall or baseline not in overall:
            continue
        comparisons[f"{challenger}__minus__{baseline}"] = bootstrap_delta(
            oof.loc[oof.experiment.eq(challenger)],
            oof.loc[oof.experiment.eq(baseline)],
            args.bootstrap_repetitions,
            SEED,
        )
    report = {
        "protocol": "CIMT_双读出表型锁定协议.md",
        "development_only": True,
        "fixed_val_used": False,
        "test_used": False,
        "patients": len(frame),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "feature_sets": {
            name: {"count": len(columns), "columns": columns}
            for name, columns in sets.items()
        },
        "overall": overall,
        "paired_bootstrap": comparisons,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
