#!/usr/bin/env python3
"""E10: conditional-centile normalization and group-DRO microvascular risk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from run_e8_double_orthogonal_cv import (
    MASS_MECHANISMS,
    residualize_features,
)
from run_normative_deviation_outer_cv import (
    bilateral,
    crossfit_normative,
    phenotype_classifier,
    phenotype_regressor,
    strata,
)


SEEDS = [20260729, 20260730, 20260731, 20260732, 20260733]
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
GROUP_DRO_STEPS = 8
GROUP_DRO_ETA = 0.5
CENTILE_THRESHOLD = float(norm.ppf(0.75))


def age_sex_groups(
    train: pd.DataFrame, other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(train.age, [.25, .50, .75]))
    train_age = np.digitize(train.age.to_numpy(), edges, right=True)
    other_age = np.digitize(other.age.to_numpy(), edges, right=True)
    train_group = np.asarray([
        f"{int(sex)}_{age_bin}"
        for sex, age_bin in zip(train.sex, train_age)
    ])
    other_group = np.asarray([
        f"{int(sex)}_{age_bin}"
        for sex, age_bin in zip(other.sex, other_age)
    ])
    return train_group, other_group, edges


def conditional_centiles(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    mu_train: np.ndarray,
    mu_validation: np.ndarray,
) -> dict[str, object]:
    train_group, validation_group, edges = age_sex_groups(train, validation)
    train_residual = train.cimt_mm.to_numpy() - mu_train
    validation_residual = validation.cimt_mm.to_numpy() - mu_validation
    z_train = np.empty(len(train), dtype=float)
    z_validation = np.empty(len(validation), dtype=float)
    pools: dict[str, np.ndarray] = {}
    global_pool = np.sort(train_residual)
    for group in np.unique(train_group):
        pool = np.sort(train_residual[train_group == group])
        if len(pool) < 30:
            pool = global_pool
        pools[group] = pool
        indices = np.flatnonzero(train_group == group)
        ranks = rankdata(train_residual[indices], method="average")
        probability = (ranks - .5) / len(indices)
        probability = np.clip(probability, 1 / (2 * len(indices)),
                              1 - 1 / (2 * len(indices)))
        z_train[indices] = norm.ppf(probability)
    for group in np.unique(validation_group):
        indices = np.flatnonzero(validation_group == group)
        pool = pools.get(group, global_pool)
        left = np.searchsorted(
            pool, validation_residual[indices], side="left"
        )
        right = np.searchsorted(
            pool, validation_residual[indices], side="right"
        )
        probability = ((left + right) / 2 + .5) / (len(pool) + 1)
        probability = np.clip(
            probability, 1 / (2 * len(pool)), 1 - 1 / (2 * len(pool))
        )
        z_validation[indices] = norm.ppf(probability)
    return {
        "z_train": z_train,
        "z_validation": z_validation,
        "train_groups": train_group,
        "validation_groups": validation_group,
        "age_edges": edges,
        "pools": pools,
        "global_pool": global_pool,
    }


def inverse_centiles(
    predicted_z: np.ndarray,
    groups: np.ndarray,
    pools: dict[str, np.ndarray],
    global_pool: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    prediction = np.empty(len(predicted_z), dtype=float)
    probability = np.clip(norm.cdf(predicted_z), .001, .999)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        pool = pools.get(group, global_pool)
        prediction[indices] = (
            mu[indices]
            + np.quantile(pool, probability[indices], method="linear")
        )
    return prediction


def z_tertile_groups(
    reference_z: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(reference_z, [1 / 3, 2 / 3])
    return (
        np.digitize(reference_z, edges, right=True),
        np.digitize(values, edges, right=True),
    )


def fit_group_dro(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> object:
    unique = np.unique(groups)
    group_weight = np.full(len(unique), 1 / len(unique), dtype=float)
    model = None
    for _ in range(GROUP_DRO_STEPS):
        sample_weight = np.zeros(len(target), dtype=float)
        for index, group in enumerate(unique):
            mask = groups == group
            sample_weight[mask] = group_weight[index] / mask.sum()
        sample_weight *= len(sample_weight) / sample_weight.sum()
        model = phenotype_regressor(alpha)
        model.fit(features, target, model__sample_weight=sample_weight)
        prediction = model.predict(features)
        losses = np.asarray([
            np.mean((target[groups == group] - prediction[groups == group]) ** 2)
            for group in unique
        ])
        scaled = losses / max(float(losses.mean()), 1e-8)
        group_weight *= np.exp(GROUP_DRO_ETA * scaled)
        group_weight /= group_weight.sum()
    return model


def tune_ridge(
    features: pd.DataFrame,
    target: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    robust: bool,
) -> float:
    scores: dict[float, float] = {}
    for alpha in ALPHAS:
        fold_scores = []
        for fit, hold in splits:
            if robust:
                fit_group, hold_group = z_tertile_groups(
                    target[fit], target[hold]
                )
                model = fit_group_dro(
                    features.iloc[fit], target[fit], fit_group, alpha
                )
                prediction = model.predict(features.iloc[hold])
                fold_scores.append(max(
                    mean_absolute_error(
                        target[hold][hold_group == group],
                        prediction[hold_group == group],
                    )
                    for group in np.unique(hold_group)
                ))
            else:
                model = phenotype_regressor(alpha)
                model.fit(features.iloc[fit], target[fit])
                fold_scores.append(mean_absolute_error(
                    target[hold], model.predict(features.iloc[hold])
                ))
        scores[alpha] = float(np.mean(fold_scores))
    return min(scores, key=scores.get)


def tune_classifier(
    features: pd.DataFrame,
    binary: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    scores = {}
    for c_value in (.001, .01, .1, 1.0, 10.0):
        values = []
        for fit, hold in splits:
            model = phenotype_classifier(c_value)
            model.fit(features.iloc[fit], binary[fit])
            probability = model.predict_proba(features.iloc[hold])[:, 1]
            values.append(average_precision_score(binary[hold], probability))
        scores[c_value] = float(np.mean(values))
    return max(scores, key=scores.get)


def output_frame(
    validation: pd.DataFrame,
    experiment: str,
    seed: int,
    fold: int,
    true_z: np.ndarray,
    predicted_z: np.ndarray,
    prediction_mm: np.ndarray,
    accelerated: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    output = validation[["patient_id", "age", "sex", "cimt_mm"]].copy()
    output["experiment"] = experiment
    output["seed"] = seed
    output["fold"] = fold
    output["true_z"] = true_z
    output["predicted_z"] = predicted_z
    output["prediction_mm"] = prediction_mm
    output["accelerated"] = accelerated
    output["probability"] = probability
    return output


def add_stress_groups(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["age_group"] = pd.qcut(
        output.age, 4, labels=False, duplicates="drop"
    )
    output["cimt_group"] = pd.qcut(
        output.cimt_mm, 3, labels=False, duplicates="drop"
    )
    return output


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    evaluated = add_stress_groups(frame)
    result = {
        "mae_mm": float(mean_absolute_error(
            evaluated.cimt_mm, evaluated.prediction_mm
        )),
        "r2": float(r2_score(evaluated.cimt_mm, evaluated.prediction_mm)),
        "auroc": float(roc_auc_score(
            evaluated.accelerated, evaluated.probability
        )),
        "auprc": float(average_precision_score(
            evaluated.accelerated, evaluated.probability
        )),
        "z_age_spearman": float(evaluated[
            ["true_z", "age"]
        ].corr(method="spearman").iloc[0, 1]),
    }
    for variable in ("age_group", "sex", "cimt_group"):
        group_mae = evaluated.groupby(variable, observed=True).apply(
            lambda group: mean_absolute_error(
                group.cimt_mm, group.prediction_mm
            ),
            include_groups=False,
        )
        result[f"worst_{variable}_mae_mm"] = float(group_mae.max())
    return result


def bootstrap_delta(
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
        baseline=("prediction_mm", "mean"),
    )
    joined = left.merge(right, on="patient_id", validate="one_to_one")
    point = (
        mean_absolute_error(joined.cimt_mm, joined.prediction)
        - mean_absolute_error(joined.cimt_mm, joined.baseline)
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        values.append(
            mean_absolute_error(sample.cimt_mm, sample.prediction)
            - mean_absolute_error(sample.cimt_mm, sample.baseline)
        )
    return {
        "point": float(point),
        "ci95": [
            float(np.quantile(values, .025)),
            float(np.quantile(values, .975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.csv")
    )
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    seeds = SEEDS[:args.seeds]

    outcomes = pd.read_csv(
        args.manifest,
        usecols=["patient_id", "split", "age", "sex", "cimt_mm"],
        dtype={"patient_id": str},
    )
    outcomes = outcomes.loc[outcomes.split.eq("train")].copy()
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    if not phenotypes.split.eq("train").all():
        raise ValueError("E10 is development-only.")
    frame = outcomes.merge(
        phenotypes.drop(columns=["split"]),
        on="patient_id",
        validate="one_to_one",
    ).reset_index(drop=True)
    if len(frame) != 2603:
        raise ValueError(f"Expected 2603 development patients, got {len(frame)}")
    features = bilateral(MASS_MECHANISMS)
    mean_features = bilateral(MASS_MECHANISMS, include_asymmetry=False)
    if frame[["age", "sex", "cimt_mm", *features]].isna().any().any():
        raise ValueError("Missing locked E10 input.")

    predictions: list[pd.DataFrame] = []
    tuning = []
    for seed in seeds:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit, hold) in enumerate(
            outer.split(frame, strata(frame)), start=1
        ):
            train = frame.iloc[fit].reset_index(drop=True)
            validation = frame.iloc[hold].reset_index(drop=True)
            inner = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=seed + fold
            )
            inner_splits = list(inner.split(train, strata(train)))
            normative = crossfit_normative(train, validation, 3, seed + fold)
            train_orth, validation_orth = residualize_features(
                train, validation, features, inner_splits
            )

            gaussian_train = (
                train.cimt_mm.to_numpy() - normative["mu_train"]
            ) / normative["sigma_train"]
            gaussian_validation = (
                validation.cimt_mm.to_numpy() - normative["mu_validation"]
            ) / normative["sigma_validation"]
            gaussian_binary_train = (
                gaussian_train >= np.quantile(gaussian_train, .75)
            ).astype(int)
            gaussian_binary_validation = (
                gaussian_validation >= np.quantile(gaussian_train, .75)
            ).astype(int)
            baseline_alpha = tune_ridge(
                train_orth, gaussian_train, inner_splits, robust=False
            )
            baseline_c = tune_classifier(
                train_orth.iloc[:, :len(mean_features)],
                gaussian_binary_train,
                inner_splits,
            )
            baseline_regressor = phenotype_regressor(baseline_alpha)
            baseline_regressor.fit(train_orth, gaussian_train)
            baseline_classifier = phenotype_classifier(baseline_c)
            baseline_classifier.fit(
                train_orth.iloc[:, :len(mean_features)],
                gaussian_binary_train,
            )
            baseline_z = baseline_regressor.predict(validation_orth)
            baseline_probability = baseline_classifier.predict_proba(
                validation_orth.iloc[:, :len(mean_features)]
            )[:, 1]
            baseline_prediction = (
                normative["mu_validation"]
                + normative["sigma_validation"] * baseline_z
            )
            predictions.append(output_frame(
                validation, "E10_0_gaussian_e8b", seed, fold,
                gaussian_validation, baseline_z, baseline_prediction,
                gaussian_binary_validation, baseline_probability,
            ))

            centile = conditional_centiles(
                train,
                validation,
                normative["mu_train"],
                normative["mu_validation"],
            )
            z_train = centile["z_train"]
            z_validation = centile["z_validation"]
            binary_train = (z_train >= CENTILE_THRESHOLD).astype(int)
            binary_validation = (z_validation >= CENTILE_THRESHOLD).astype(int)
            c_value = tune_classifier(
                train_orth.iloc[:, :len(mean_features)],
                binary_train,
                inner_splits,
            )
            classifier = phenotype_classifier(c_value)
            classifier.fit(
                train_orth.iloc[:, :len(mean_features)], binary_train
            )
            probability = classifier.predict_proba(
                validation_orth.iloc[:, :len(mean_features)]
            )[:, 1]

            standard_alpha = tune_ridge(
                train_orth, z_train, inner_splits, robust=False
            )
            standard = phenotype_regressor(standard_alpha)
            standard.fit(train_orth, z_train)
            standard_z = standard.predict(validation_orth)
            standard_prediction = inverse_centiles(
                standard_z,
                centile["validation_groups"],
                centile["pools"],
                centile["global_pool"],
                normative["mu_validation"],
            )
            predictions.append(output_frame(
                validation, "E10A_conditional_centile", seed, fold,
                z_validation, standard_z, standard_prediction,
                binary_validation, probability,
            ))

            robust_alpha = tune_ridge(
                train_orth, z_train, inner_splits, robust=True
            )
            train_dro_group, _ = z_tertile_groups(z_train, z_train)
            robust = fit_group_dro(
                train_orth, z_train, train_dro_group, robust_alpha
            )
            robust_z = robust.predict(validation_orth)
            robust_prediction = inverse_centiles(
                robust_z,
                centile["validation_groups"],
                centile["pools"],
                centile["global_pool"],
                normative["mu_validation"],
            )
            predictions.append(output_frame(
                validation, "E10B_centile_group_dro", seed, fold,
                z_validation, robust_z, robust_prediction,
                binary_validation, probability,
            ))
            tuning.extend([
                {
                    "seed": seed, "fold": fold,
                    "experiment": "E10_0_gaussian_e8b",
                    "ridge_alpha": baseline_alpha,
                    "logistic_c": baseline_c,
                },
                {
                    "seed": seed, "fold": fold,
                    "experiment": "E10A_conditional_centile",
                    "ridge_alpha": standard_alpha,
                    "logistic_c": c_value,
                },
                {
                    "seed": seed, "fold": fold,
                    "experiment": "E10B_centile_group_dro",
                    "ridge_alpha": robust_alpha,
                    "logistic_c": c_value,
                },
            ])
            print(f"seed={seed} fold={fold}/5 complete", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "repeated_oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "nested_tuning.csv", index=False)
    per_seed = []
    for (experiment, seed), group in oof.groupby(["experiment", "seed"]):
        per_seed.append({
            "experiment": experiment,
            "seed": int(seed),
            **metrics(group),
        })
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
    summary = per_seed_frame.groupby("experiment").agg(
        {column: ["mean", "std"] for column in per_seed_frame.columns
         if column not in ("experiment", "seed")}
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    baseline = oof.loc[oof.experiment.eq("E10_0_gaussian_e8b")]
    comparisons = {}
    directions = {}
    for experiment in (
        "E10A_conditional_centile", "E10B_centile_group_dro"
    ):
        challenger = oof.loc[oof.experiment.eq(experiment)]
        comparisons[f"{experiment}__minus__E10_0_gaussian_e8b"] = (
            bootstrap_delta(
                challenger, baseline, args.bootstrap_repetitions, seeds[0]
            )
        )
        deltas = []
        for (seed, fold), group in challenger.groupby(["seed", "fold"]):
            reference = baseline.loc[
                baseline.seed.eq(seed) & baseline.fold.eq(fold)
            ]
            deltas.append(
                mean_absolute_error(group.cimt_mm, group.prediction_mm)
                - mean_absolute_error(
                    reference.cimt_mm, reference.prediction_mm
                )
            )
        directions[experiment] = {
            "folds_improved_or_equal": int((np.asarray(deltas) <= 0).sum()),
            "folds_total": len(deltas),
            "mean_fold_delta_mm": float(np.mean(deltas)),
        }
    report = {
        "protocol": "E10_条件百分位分布稳健微血管偏离_锁定协议.md",
        "development_only": True,
        "fixed_val_used": False,
        "test_used": False,
        "seeds": seeds,
        "outer_folds_per_seed": 5,
        "summary": summary.to_dict(orient="records"),
        "ensemble_mae_comparisons": comparisons,
        "fold_direction": directions,
        "locked": {
            "centile_threshold_z": CENTILE_THRESHOLD,
            "group_dro_steps": GROUP_DRO_STEPS,
            "group_dro_eta": GROUP_DRO_ETA,
            "alphas": ALPHAS,
            "features": features,
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
