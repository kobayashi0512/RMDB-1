#!/usr/bin/env python3
"""E8 repeated cross-fitting with double orthogonalization and low-rank axes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_normative_deviation_outer_cv import (
    bilateral,
    crossfit_normative,
    normative_model,
    phenotype_classifier,
    phenotype_regressor,
    strata,
    tune_phenotype,
)


SEEDS = [20260729, 20260730, 20260731, 20260732, 20260733]
MASS_MECHANISMS = [
    "soft_vessel_mass_fraction",
    "soft_av_log_mass_ratio",
    "mass_vessel_skeleton_length_density",
    "mass_vessel_branch_cluster_density",
    "mass_av_log_area_ratio",
]
RELIABILITY = [
    "occupancy_entropy__bilateral_mean",
    "identity_entropy__bilateral_mean",
    "vessel_occupancy_disagreement__bilateral_mean",
    "base_to_mass_identity_disagreement__bilateral_mean",
]


def residualize_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_residual = np.empty((len(train), len(features)), dtype=float)
    for fit, hold in splits:
        model = normative_model(10.0)
        model.fit(
            train.iloc[fit][["age", "sex"]],
            train.iloc[fit][features],
        )
        train_residual[hold] = (
            train.iloc[hold][features].to_numpy()
            - model.predict(train.iloc[hold][["age", "sex"]])
        )
    full = normative_model(10.0)
    full.fit(train[["age", "sex"]], train[features])
    validation_residual = (
        validation[features].to_numpy()
        - full.predict(validation[["age", "sex"]])
    )
    columns = [f"orth_{column}" for column in features]
    return (
        pd.DataFrame(train_residual, columns=columns, index=train.index),
        pd.DataFrame(validation_residual, columns=columns, index=validation.index),
    )


def low_rank_regressor(components: int, alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, random_state=20260729)),
        ("model", Ridge(alpha=alpha)),
    ])


def low_rank_classifier(components: int, c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, random_state=20260729)),
        ("model", LogisticRegression(
            C=c_value,
            max_iter=5000,
            class_weight="balanced",
            random_state=20260729,
        )),
    ])


def tune_low_rank(
    features: pd.DataFrame,
    target: np.ndarray,
    binary: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[int, float, int, float]:
    regression_scores, classification_scores = {}, {}
    for components in (2, 4, 6):
        for alpha in (1.0, 10.0, 100.0):
            errors = []
            for fit, hold in splits:
                model = low_rank_regressor(components, alpha)
                model.fit(features.iloc[fit], target[fit])
                errors.append(mean_absolute_error(
                    target[hold], model.predict(features.iloc[hold])
                ))
            regression_scores[(components, alpha)] = float(np.mean(errors))
        for c_value in (.01, .1, 1.0):
            scores = []
            for fit, hold in splits:
                model = low_rank_classifier(components, c_value)
                model.fit(features.iloc[fit], binary[fit])
                probability = model.predict_proba(features.iloc[hold])[:, 1]
                scores.append(average_precision_score(binary[hold], probability))
            classification_scores[(components, c_value)] = float(np.mean(scores))
    reg_components, alpha = min(regression_scores, key=regression_scores.get)
    cls_components, c_value = max(
        classification_scores, key=classification_scores.get
    )
    return reg_components, alpha, cls_components, c_value


def reliability_percentile(
    train: pd.DataFrame, validation: pd.DataFrame
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(
        imputer.fit_transform(train[RELIABILITY])
    ).mean(1)
    validation_values = scaler.transform(
        imputer.transform(validation[RELIABILITY])
    ).mean(1)
    sorted_train = np.sort(train_values)
    return np.searchsorted(
        sorted_train, validation_values, side="right"
    ) / len(sorted_train)


def model_output(
    validation: pd.DataFrame,
    experiment: str,
    seed: int,
    fold: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    true_z: np.ndarray,
    accelerated: np.ndarray,
    predicted_z: np.ndarray,
    probability: np.ndarray,
    reliability: np.ndarray,
) -> pd.DataFrame:
    output = validation[
        ["patient_id", "age", "sex", "cimt_mm"]
    ].copy()
    output["experiment"] = experiment
    output["seed"] = seed
    output["fold"] = fold
    output["true_z"] = true_z
    output["predicted_z"] = predicted_z
    output["accelerated"] = accelerated
    output["prediction_mm"] = mu + sigma * predicted_z
    output["probability"] = probability
    output["reliability_percentile"] = reliability
    return output


def seed_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "mae_mm": float(mean_absolute_error(frame.cimt_mm, frame.prediction_mm)),
        "r2": float(r2_score(frame.cimt_mm, frame.prediction_mm)),
        "auroc": float(roc_auc_score(frame.accelerated, frame.probability)),
        "auprc": float(average_precision_score(
            frame.accelerated, frame.probability
        )),
    }


def ensemble_mae_bootstrap(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    left = challenger.groupby("patient_id", as_index=False).agg(
        cimt_mm=("cimt_mm", "first"),
        prediction=("prediction_mm", "mean"),
    )
    right = baseline.groupby("patient_id", as_index=False).agg(
        baseline=("prediction_mm", "mean")
    )
    joined = left.merge(right, on="patient_id", validate="one_to_one")
    point = (
        mean_absolute_error(joined.cimt_mm, joined.prediction)
        - mean_absolute_error(joined.cimt_mm, joined.baseline)
    )
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        deltas.append(
            mean_absolute_error(sample.cimt_mm, sample.prediction)
            - mean_absolute_error(sample.cimt_mm, sample.baseline)
        )
    return {
        "point": float(point),
        "ci95": [
            float(np.quantile(deltas, .025)),
            float(np.quantile(deltas, .975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    outcomes = pd.read_csv(
        args.manifest,
        usecols=["patient_id", "split", "age", "sex", "cimt_mm"],
        dtype={"patient_id": str},
    )
    outcomes = outcomes.loc[outcomes.split.eq("train")].copy()
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    if not phenotypes.split.eq("train").all():
        raise ValueError("Non-development phenotype rows found.")
    frame = outcomes.merge(
        phenotypes.drop(columns=["split"]), on="patient_id", validate="one_to_one"
    ).reset_index(drop=True)
    full_features = bilateral(MASS_MECHANISMS)
    mean_features = bilateral(MASS_MECHANISMS, include_asymmetry=False)
    missing = [
        column for column in full_features + RELIABILITY if column not in frame
    ]
    if missing:
        raise KeyError(f"Missing locked fields: {missing}")

    predictions, tuning = [], []
    for repeat_seed in SEEDS:
        outer = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=repeat_seed
        )
        for fold, (fit, hold) in enumerate(
            outer.split(frame, strata(frame)), start=1
        ):
            train = frame.iloc[fit].reset_index(drop=True)
            validation = frame.iloc[hold].reset_index(drop=True)
            inner = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=repeat_seed + fold
            )
            inner_splits = list(inner.split(train, strata(train)))
            normative = crossfit_normative(
                train, validation, 3, repeat_seed + fold
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
            binary_train = (z_train >= threshold).astype(int)
            binary_validation = (z_validation >= threshold).astype(int)
            reliability = reliability_percentile(train, validation)

            predictions.append(model_output(
                validation, "E8_0_normative", repeat_seed, fold,
                normative["mu_validation"], normative["sigma_validation"],
                z_validation, binary_validation, np.zeros(len(validation)),
                np.full(len(validation), binary_train.mean()), reliability,
            ))

            train_orth, validation_orth = residualize_features(
                train, validation, full_features, inner_splits
            )
            datasets = {
                "E8A_raw_mass": (
                    train[full_features], validation[full_features], "ridge"
                ),
                "E8B_double_orthogonal": (
                    train_orth, validation_orth, "ridge"
                ),
                "E8C_low_rank": (
                    train_orth, validation_orth, "low_rank"
                ),
                "E8D_means_only": (
                    train_orth.iloc[:, :len(mean_features)],
                    validation_orth.iloc[:, :len(mean_features)],
                    "ridge",
                ),
            }
            for experiment, (
                train_features, validation_features, kind
            ) in datasets.items():
                feature_names = list(train_features.columns)
                # tune_phenotype expects columns to exist in `train`.
                if kind == "ridge":
                    local_train = train.copy()
                    local_validation = validation.copy()
                    for column in feature_names:
                        local_train[column] = train_features[column].to_numpy()
                        local_validation[column] = validation_features[column].to_numpy()
                    alpha, c_value = tune_phenotype(
                        local_train, feature_names, z_train, binary_train,
                        3, repeat_seed + 100 + fold,
                    )
                    regressor = phenotype_regressor(alpha)
                    classifier = phenotype_classifier(c_value)
                    regressor.fit(train_features, z_train)
                    classifier.fit(train_features, binary_train)
                    reg_components = cls_components = None
                else:
                    reg_components, alpha, cls_components, c_value = tune_low_rank(
                        train_features, z_train, binary_train, inner_splits
                    )
                    regressor = low_rank_regressor(reg_components, alpha)
                    classifier = low_rank_classifier(cls_components, c_value)
                    regressor.fit(train_features, z_train)
                    classifier.fit(train_features, binary_train)
                predicted_z = regressor.predict(validation_features)
                probability = classifier.predict_proba(validation_features)[:, 1]
                predictions.append(model_output(
                    validation, experiment, repeat_seed, fold,
                    normative["mu_validation"], normative["sigma_validation"],
                    z_validation, binary_validation, predicted_z, probability,
                    reliability,
                ))
                tuning.append({
                    "seed": repeat_seed,
                    "fold": fold,
                    "experiment": experiment,
                    "features": len(feature_names),
                    "ridge_alpha": alpha,
                    "logistic_c": c_value,
                    "regression_pca_components": reg_components,
                    "classification_pca_components": cls_components,
                })
            print(f"seed={repeat_seed} fold={fold}/5 complete", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "repeated_oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)

    per_seed = []
    for (experiment, seed), group in oof.groupby(["experiment", "seed"]):
        per_seed.append({
            "experiment": experiment,
            "seed": int(seed),
            **seed_metrics(group),
        })
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
    summary = (
        per_seed_frame.groupby("experiment")
        .agg({
            "mae_mm": ["mean", "std"],
            "r2": ["mean", "std"],
            "auroc": ["mean", "std"],
            "auprc": ["mean", "std"],
        })
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    comparisons = {}
    pairs = [
        ("E8A_raw_mass", "E8_0_normative"),
        ("E8B_double_orthogonal", "E8A_raw_mass"),
        ("E8C_low_rank", "E8B_double_orthogonal"),
        ("E8D_means_only", "E8B_double_orthogonal"),
    ]
    for challenger, baseline in pairs:
        comparisons[f"{challenger}__minus__{baseline}"] = ensemble_mae_bootstrap(
            oof.loc[oof.experiment.eq(challenger)],
            oof.loc[oof.experiment.eq(baseline)],
            args.bootstrap_repetitions,
            SEEDS[0],
        )

    baseline = oof.loc[oof.experiment.eq("E8_0_normative")]
    direction = {}
    for experiment in [
        "E8A_raw_mass", "E8B_double_orthogonal",
        "E8C_low_rank", "E8D_means_only",
    ]:
        challenger = oof.loc[oof.experiment.eq(experiment)]
        deltas = []
        for (seed, fold), group in challenger.groupby(["seed", "fold"]):
            reference = baseline.loc[
                baseline.seed.eq(seed) & baseline.fold.eq(fold)
            ]
            deltas.append(
                mean_absolute_error(group.cimt_mm, group.prediction_mm)
                - mean_absolute_error(reference.cimt_mm, reference.prediction_mm)
            )
        direction[experiment] = {
            "folds_improved": int((np.asarray(deltas) < 0).sum()),
            "folds_total": len(deltas),
            "mean_fold_delta_mm": float(np.mean(deltas)),
        }

    coverage_rows = []
    for experiment, group in oof.groupby("experiment"):
        if experiment == "E8_0_normative":
            continue
        for coverage in (1.0, .9, .8, .7):
            for seed, seed_group in group.groupby("seed"):
                selected = seed_group.loc[
                    seed_group.reliability_percentile <= coverage
                ]
                coverage_rows.append({
                    "experiment": experiment,
                    "coverage_target": coverage,
                    "seed": int(seed),
                    "n": len(selected),
                    "mae_mm": mean_absolute_error(
                        selected.cimt_mm, selected.prediction_mm
                    ),
                    "auroc": roc_auc_score(
                        selected.accelerated, selected.probability
                    ),
                })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "coverage_risk.csv", index=False)

    report = {
        "protocol": "E8_双重正交低秩微血管偏离_锁定协议.md",
        "development_only": True,
        "fixed_val_used": False,
        "test_used": False,
        "seeds": SEEDS,
        "outer_folds_per_seed": 5,
        "summary": summary.to_dict(orient="records"),
        "ensemble_mae_comparisons": comparisons,
        "fold_direction": direction,
        "coverage_summary": (
            coverage.groupby(["experiment", "coverage_target"])
            .agg(n_mean=("n", "mean"), mae_mean=("mae_mm", "mean"),
                 auroc_mean=("auroc", "mean"))
            .reset_index().to_dict(orient="records")
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
