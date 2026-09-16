#!/usr/bin/env python3
"""E7: cross-fitted age/sex normative CIMT-deviation prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


SEED = 20260729


CORE_MECHANISMS = [
    "soft_vessel_mass_fraction",
    "soft_av_log_mass_ratio",
    "mass_vessel_skeleton_length_density",
    "mass_vessel_branch_cluster_density",
    "mass_av_log_area_ratio",
    "factorized_av_log_area_ratio",
    "vessel_occupancy_disagreement",
    "base_to_mass_identity_disagreement",
]


def bilateral(mechanisms: list[str], include_asymmetry: bool = True) -> list[str]:
    columns = [
        f"{name}__bilateral_mean" for name in mechanisms
    ]
    if include_asymmetry:
        columns += [
            f"{name}__absolute_difference" for name in mechanisms
        ]
    return columns


def strata(frame: pd.DataFrame) -> pd.Series:
    cimt_bin = pd.qcut(
        frame["cimt_mm"], q=4, labels=False, duplicates="drop"
    ).astype(str)
    return frame["sex"].astype(str) + "_" + cimt_bin


def normative_model(alpha: float) -> Pipeline:
    transform = ColumnTransformer([
        (
            "age",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(
                    n_knots=5, degree=3, include_bias=False
                )),
                ("scale", StandardScaler()),
            ]),
            ["age"],
        ),
        (
            "sex",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("scale", StandardScaler()),
            ]),
            ["sex"],
        ),
    ])
    return Pipeline([
        ("transform", transform),
        ("model", Ridge(alpha=alpha)),
    ])


def phenotype_regressor(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def phenotype_classifier(c_value: float) -> Pipeline:
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


def choose_normative_alpha(
    train: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
) -> float:
    scores = {}
    for alpha in (.01, .1, 1.0, 10.0, 100.0):
        errors = []
        for fit, hold in splits:
            model = normative_model(alpha)
            model.fit(train.iloc[fit][["age", "sex"]], train.iloc[fit].cimt_mm)
            errors.append(mean_absolute_error(
                train.iloc[hold].cimt_mm,
                model.predict(train.iloc[hold][["age", "sex"]]),
            ))
        scores[alpha] = float(np.mean(errors))
    return min(scores, key=scores.get)


def crossfit_normative(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    inner_folds: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    inner = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed
    )
    splits = list(inner.split(train, strata(train)))
    alpha = choose_normative_alpha(train, splits)
    mu_train = np.empty(len(train), dtype=float)
    for fit, hold in splits:
        model = normative_model(alpha)
        model.fit(train.iloc[fit][["age", "sex"]], train.iloc[fit].cimt_mm)
        mu_train[hold] = model.predict(train.iloc[hold][["age", "sex"]])
    full_mean = normative_model(alpha)
    full_mean.fit(train[["age", "sex"]], train.cimt_mm)
    mu_validation = full_mean.predict(validation[["age", "sex"]])

    residual_train = train.cimt_mm.to_numpy() - mu_train
    log_squared = np.log(residual_train ** 2 + 1e-4)
    sigma_train = np.empty(len(train), dtype=float)
    for fit, hold in splits:
        scale = normative_model(10.0)
        scale.fit(
            train.iloc[fit][["age", "sex"]],
            pd.Series(log_squared[fit], index=train.iloc[fit].index),
        )
        sigma_train[hold] = np.sqrt(np.exp(
            scale.predict(train.iloc[hold][["age", "sex"]])
        ))
    full_scale = normative_model(10.0)
    full_scale.fit(train[["age", "sex"]], log_squared)
    sigma_validation = np.sqrt(np.exp(
        full_scale.predict(validation[["age", "sex"]])
    ))
    absolute = np.abs(residual_train)
    lower = max(float(np.quantile(absolute, .10) * np.sqrt(np.pi / 2)), .03)
    upper = max(float(np.quantile(absolute, .90) * np.sqrt(np.pi / 2)), lower)
    sigma_train = np.clip(sigma_train, lower, upper)
    sigma_validation = np.clip(sigma_validation, lower, upper)
    return {
        "alpha": alpha,
        "mu_train": mu_train,
        "mu_validation": mu_validation,
        "sigma_train": sigma_train,
        "sigma_validation": sigma_validation,
    }


def tune_phenotype(
    train: pd.DataFrame,
    features: list[str],
    target: np.ndarray,
    binary: np.ndarray,
    inner_folds: int,
    seed: int,
) -> tuple[float, float]:
    inner = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed
    )
    splits = list(inner.split(train, binary))
    alpha_scores, c_scores = {}, {}
    for alpha in (.1, 1.0, 10.0, 100.0, 1000.0):
        errors = []
        for fit, hold in splits:
            model = phenotype_regressor(alpha)
            model.fit(train.iloc[fit][features], target[fit])
            errors.append(mean_absolute_error(
                target[hold], model.predict(train.iloc[hold][features])
            ))
        alpha_scores[alpha] = float(np.mean(errors))
    for c_value in (.001, .01, .1, 1.0, 10.0):
        scores = []
        for fit, hold in splits:
            model = phenotype_classifier(c_value)
            model.fit(train.iloc[fit][features], binary[fit])
            probability = model.predict_proba(train.iloc[hold][features])[:, 1]
            scores.append(average_precision_score(binary[hold], probability))
        c_scores[c_value] = float(np.mean(scores))
    return min(alpha_scores, key=alpha_scores.get), max(c_scores, key=c_scores.get)


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "n": int(len(frame)),
        "cimt_mae_mm": float(mean_absolute_error(
            frame.cimt_mm, frame.prediction_mm
        )),
        "cimt_r2": float(r2_score(frame.cimt_mm, frame.prediction_mm)),
        "deviation_mae_z": float(mean_absolute_error(
            frame.true_z, frame.predicted_z
        )),
        "accelerated_auroc": float(roc_auc_score(
            frame.accelerated, frame.probability
        )),
        "accelerated_auprc": float(average_precision_score(
            frame.accelerated, frame.probability
        )),
        "accelerated_brier": float(brier_score_loss(
            frame.accelerated, frame.probability
        )),
        "true_z_age_spearman": float(frame[
            ["true_z", "age"]
        ].corr(method="spearman").iloc[0, 1]),
    }


def bootstrap(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    joined = challenger.merge(
        baseline[["patient_id", "prediction_mm"]],
        on="patient_id",
        suffixes=("_challenger", "_baseline"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    mae_delta, auroc = [], []
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        mae_delta.append(
            mean_absolute_error(
                sample.cimt_mm, sample.prediction_mm_challenger
            )
            - mean_absolute_error(
                sample.cimt_mm, sample.prediction_mm_baseline
            )
        )
        if sample.accelerated.nunique() == 2:
            auroc.append(roc_auc_score(
                sample.accelerated, sample.probability
            ))
    return {
        "mae_delta_mm": {
            "point": float(
                mean_absolute_error(
                    joined.cimt_mm, joined.prediction_mm_challenger
                )
                - mean_absolute_error(
                    joined.cimt_mm, joined.prediction_mm_baseline
                )
            ),
            "ci95": [
                float(np.quantile(mae_delta, .025)),
                float(np.quantile(mae_delta, .975)),
            ],
        },
        "accelerated_auroc": {
            "point": float(roc_auc_score(
                joined.accelerated, joined.probability
            )),
            "ci95": [
                float(np.quantile(auroc, .025)),
                float(np.quantile(auroc, .975)),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    outcomes = pd.read_csv(
        args.manifest,
        usecols=["patient_id", "split", "age", "sex", "cimt_mm"],
        dtype={"patient_id": str},
    )
    outcomes = outcomes.loc[outcomes.split.eq("train")].copy()
    phenotype = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    if not phenotype.split.eq("train").all():
        raise ValueError("Non-development phenotype rows found.")
    frame = outcomes.merge(
        phenotype.drop(columns=["split"]), on="patient_id", validate="one_to_one"
    ).reset_index(drop=True)
    if len(frame) != 2603:
        raise ValueError(f"Expected 2603 patients, found {len(frame)}.")

    mass_mechanisms = CORE_MECHANISMS[:5]
    factorized_mechanisms = [
        "soft_vessel_mass_fraction",
        "soft_av_log_mass_ratio",
        "factorized_vessel_skeleton_length_density",
        "factorized_vessel_branch_cluster_density",
        "factorized_av_log_area_ratio",
    ]
    feature_sets = {
        "E7A_core_residual": (bilateral(CORE_MECHANISMS), "residual"),
        "E7B_core_z": (bilateral(CORE_MECHANISMS), "z"),
        "E7C_means_only_z": (bilateral(
            CORE_MECHANISMS, include_asymmetry=False
        ), "z"),
        "E7D_mass_z": (bilateral(mass_mechanisms), "z"),
        "E7E_factorized_z": (bilateral(factorized_mechanisms), "z"),
    }
    for name, (columns, _) in feature_sets.items():
        missing = [column for column in columns if column not in frame]
        if missing:
            raise KeyError(f"{name} missing: {missing}")

    outer = StratifiedKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=SEED
    )
    predictions, tuning = [], []
    for fold, (fit, hold) in enumerate(
        outer.split(frame, strata(frame)), start=1
    ):
        train = frame.iloc[fit].reset_index(drop=True)
        validation = frame.iloc[hold].reset_index(drop=True)
        normative = crossfit_normative(
            train, validation, args.inner_folds, SEED + fold
        )
        residual_train = (
            train.cimt_mm.to_numpy() - normative["mu_train"]
        )
        residual_validation = (
            validation.cimt_mm.to_numpy() - normative["mu_validation"]
        )
        z_train = residual_train / normative["sigma_train"]
        z_validation = residual_validation / normative["sigma_validation"]
        threshold = float(np.quantile(z_train, .75))
        accelerated_train = (z_train >= threshold).astype(int)
        accelerated_validation = (z_validation >= threshold).astype(int)

        baseline = validation[
            ["patient_id", "age", "sex", "cimt_mm"]
        ].copy()
        baseline["experiment"] = "E7_0_normative"
        baseline["fold"] = fold
        baseline["true_z"] = z_validation
        baseline["predicted_z"] = 0.0
        baseline["accelerated"] = accelerated_validation
        baseline["prediction_mm"] = normative["mu_validation"]
        baseline["probability"] = float(accelerated_train.mean())
        predictions.append(baseline)

        for experiment, (features, target_kind) in feature_sets.items():
            target = residual_train if target_kind == "residual" else z_train
            alpha, c_value = tune_phenotype(
                train, features, target, accelerated_train,
                args.inner_folds, SEED + 100 + fold,
            )
            regressor = phenotype_regressor(alpha)
            classifier = phenotype_classifier(c_value)
            regressor.fit(train[features], target)
            classifier.fit(train[features], accelerated_train)
            predicted_target = regressor.predict(validation[features])
            if target_kind == "residual":
                predicted_z = predicted_target / normative["sigma_validation"]
                prediction_mm = normative["mu_validation"] + predicted_target
            else:
                predicted_z = predicted_target
                prediction_mm = (
                    normative["mu_validation"]
                    + normative["sigma_validation"] * predicted_target
                )
            output = validation[
                ["patient_id", "age", "sex", "cimt_mm"]
            ].copy()
            output["experiment"] = experiment
            output["fold"] = fold
            output["true_z"] = z_validation
            output["predicted_z"] = predicted_z
            output["accelerated"] = accelerated_validation
            output["prediction_mm"] = prediction_mm
            output["probability"] = classifier.predict_proba(
                validation[features]
            )[:, 1]
            predictions.append(output)
            tuning.append({
                "experiment": experiment,
                "fold": fold,
                "features": len(features),
                "target": target_kind,
                "normative_alpha": normative["alpha"],
                "ridge_alpha": alpha,
                "logistic_c": c_value,
                "accelerated_threshold_z": threshold,
            })
        print(f"fold={fold}/{args.outer_folds} complete", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)
    overall = {
        name: evaluate(group)
        for name, group in oof.groupby("experiment")
    }
    baseline = oof.loc[oof.experiment.eq("E7_0_normative")]
    paired = {
        name: bootstrap(
            group, baseline, args.bootstrap_repetitions, SEED
        )
        for name, group in oof.groupby("experiment")
        if name != "E7_0_normative"
    }
    report = {
        "protocol": "E7_年龄正交化微血管偏离_锁定协议.md",
        "development_only": True,
        "fixed_val_used": False,
        "test_used": False,
        "patients": len(frame),
        "feature_sets": {
            name: {"target": target, "count": len(columns), "columns": columns}
            for name, (columns, target) in feature_sets.items()
        },
        "overall": overall,
        "paired_vs_normative": paired,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
