#!/usr/bin/env python3
"""E11: nested dual-estimand fusion and disagreement uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from run_e8_double_orthogonal_cv import (
    MASS_MECHANISMS,
    residualize_features,
)
from run_e10_distributionally_robust_cv import (
    SEEDS,
    conditional_centiles,
    fit_group_dro,
    inverse_centiles,
    tune_ridge,
    z_tertile_groups,
)
from run_normative_deviation_outer_cv import (
    bilateral,
    crossfit_normative,
    phenotype_regressor,
    strata,
)


WEIGHTS = np.round(np.linspace(0, 1, 11), 1)
RMSE_MARGIN_MM = 0.0005


def fit_heads(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> dict[str, np.ndarray | float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(train, strata(train)))
    features = bilateral(MASS_MECHANISMS)
    normative = crossfit_normative(train, validation, 3, seed)
    train_orth, validation_orth = residualize_features(
        train, validation, features, inner_splits
    )

    gaussian_train = (
        train.cimt_mm.to_numpy() - normative["mu_train"]
    ) / normative["sigma_train"]
    gaussian_alpha = tune_ridge(
        train_orth, gaussian_train, inner_splits, robust=False
    )
    gaussian = phenotype_regressor(gaussian_alpha)
    gaussian.fit(train_orth, gaussian_train)
    gaussian_z = gaussian.predict(validation_orth)
    mean_prediction = (
        normative["mu_validation"]
        + normative["sigma_validation"] * gaussian_z
    )

    centile = conditional_centiles(
        train,
        validation,
        normative["mu_train"],
        normative["mu_validation"],
    )
    centile_train = centile["z_train"]
    robust_alpha = tune_ridge(
        train_orth, centile_train, inner_splits, robust=True
    )
    dro_group, _ = z_tertile_groups(centile_train, centile_train)
    robust = fit_group_dro(
        train_orth, centile_train, dro_group, robust_alpha
    )
    robust_z = robust.predict(validation_orth)
    centile_prediction = inverse_centiles(
        robust_z,
        centile["validation_groups"],
        centile["pools"],
        centile["global_pool"],
        normative["mu_validation"],
    )
    return {
        "mean_prediction": mean_prediction,
        "centile_prediction": centile_prediction,
        "sigma": normative["sigma_validation"],
        "gaussian_alpha": gaussian_alpha,
        "robust_alpha": robust_alpha,
    }


def nested_weight(
    train: pd.DataFrame,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    mean_oof = np.empty(len(train), dtype=float)
    centile_oof = np.empty(len(train), dtype=float)
    for inner_fold, (fit, hold) in enumerate(
        inner.split(train, strata(train)), start=1
    ):
        result = fit_heads(
            train.iloc[fit].reset_index(drop=True),
            train.iloc[hold].reset_index(drop=True),
            seed + 1000 + inner_fold,
        )
        mean_oof[hold] = result["mean_prediction"]
        centile_oof[hold] = result["centile_prediction"]
    truth = train.cimt_mm.to_numpy()
    baseline_rmse = mean_squared_error(truth, mean_oof) ** .5
    rows = []
    for weight in WEIGHTS:
        prediction = (1 - weight) * mean_oof + weight * centile_oof
        rows.append({
            "weight": float(weight),
            "mae_mm": float(mean_absolute_error(truth, prediction)),
            "rmse_mm": float(mean_squared_error(truth, prediction) ** .5),
            "eligible": bool(
                mean_squared_error(truth, prediction) ** .5
                <= baseline_rmse + RMSE_MARGIN_MM
            ),
        })
    candidates = pd.DataFrame(rows)
    eligible = candidates.loc[candidates.eligible].copy()
    if eligible.empty:
        selected = 0.0
    else:
        selected = float(
            eligible.sort_values(["mae_mm", "weight"]).iloc[0].weight
        )
    return selected, candidates


def stress_groups(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["age_group"] = pd.qcut(
        output.age, 4, labels=False, duplicates="drop"
    )
    output["cimt_group"] = pd.qcut(
        output.cimt_mm, 3, labels=False, duplicates="drop"
    )
    return output


def prediction_metrics(
    frame: pd.DataFrame, column: str
) -> dict[str, float]:
    evaluated = stress_groups(frame)
    truth = evaluated.cimt_mm.to_numpy()
    prediction = evaluated[column].to_numpy()
    result = {
        "mae_mm": float(mean_absolute_error(truth, prediction)),
        "rmse_mm": float(mean_squared_error(truth, prediction) ** .5),
        "r2": float(r2_score(truth, prediction)),
        "bias_mm": float(np.mean(prediction - truth)),
        "prediction_variance_ratio": float(
            np.var(prediction, ddof=1) / np.var(truth, ddof=1)
        ),
    }
    for variable in ("age_group", "sex", "cimt_group"):
        values = evaluated.groupby(variable, observed=True).apply(
            lambda group: mean_absolute_error(
                group.cimt_mm, group[column]
            ),
            include_groups=False,
        )
        result[f"worst_{variable}_mae_mm"] = float(values.max())
    return result


def uncertainty_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = np.abs(
        frame.cimt_mm.to_numpy() - frame.fused_prediction_mm.to_numpy()
    )
    uncertainty = frame.disagreement_uncertainty.to_numpy()
    threshold = np.quantile(error, .80)
    high_error = (error >= threshold).astype(int)
    return {
        "uncertainty_error_spearman": float(pd.DataFrame({
            "uncertainty": uncertainty, "error": error
        }).corr(method="spearman").iloc[0, 1]),
        "high_error_auroc": float(roc_auc_score(high_error, uncertainty)),
    }


def bootstrap_deltas(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    patient = frame.groupby("patient_id", as_index=False).agg(
        cimt_mm=("cimt_mm", "first"),
        mean_prediction_mm=("mean_prediction_mm", "mean"),
        fused_prediction_mm=("fused_prediction_mm", "mean"),
    )

    def delta(sample: pd.DataFrame) -> tuple[float, float]:
        mae = (
            mean_absolute_error(sample.cimt_mm, sample.fused_prediction_mm)
            - mean_absolute_error(sample.cimt_mm, sample.mean_prediction_mm)
        )
        rmse = (
            mean_squared_error(
                sample.cimt_mm, sample.fused_prediction_mm
            ) ** .5
            - mean_squared_error(
                sample.cimt_mm, sample.mean_prediction_mm
            ) ** .5
        )
        return float(mae), float(rmse)

    point_mae, point_rmse = delta(patient)
    rng = np.random.default_rng(seed)
    mae_values, rmse_values = [], []
    for _ in range(repetitions):
        sample = patient.iloc[
            rng.integers(0, len(patient), len(patient))
        ]
        mae, rmse = delta(sample)
        mae_values.append(mae)
        rmse_values.append(rmse)
    return {
        "fused_minus_mean_mae_mm": {
            "point": point_mae,
            "ci95": [
                float(np.quantile(mae_values, .025)),
                float(np.quantile(mae_values, .975)),
            ],
        },
        "fused_minus_mean_rmse_mm": {
            "point": point_rmse,
            "ci95": [
                float(np.quantile(rmse_values, .025)),
                float(np.quantile(rmse_values, .975)),
            ],
        },
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
        raise ValueError("E11 is development-only.")
    frame = outcomes.merge(
        phenotypes.drop(columns=["split"]),
        on="patient_id",
        validate="one_to_one",
    ).reset_index(drop=True)
    required = ["age", "sex", "cimt_mm", *bilateral(MASS_MECHANISMS)]
    if len(frame) != 2603 or frame[required].isna().any().any():
        raise ValueError("Incomplete E11 development inputs.")

    predictions = []
    tuning_rows = []
    inner_weight_rows = []
    for seed in seeds:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (fit, hold) in enumerate(
            outer.split(frame, strata(frame)), start=1
        ):
            train = frame.iloc[fit].reset_index(drop=True)
            validation = frame.iloc[hold].reset_index(drop=True)
            weight, candidates = nested_weight(
                train, seed + 100 * fold
            )
            candidates["seed"] = seed
            candidates["fold"] = fold
            candidates["selected"] = candidates.weight.eq(weight)
            inner_weight_rows.append(candidates)
            heads = fit_heads(train, validation, seed + fold)
            mean_prediction = heads["mean_prediction"]
            centile_prediction = heads["centile_prediction"]
            fused_prediction = (
                (1 - weight) * mean_prediction
                + weight * centile_prediction
            )
            uncertainty = (
                np.abs(mean_prediction - centile_prediction)
                / (heads["sigma"] + 1e-8)
            )
            output = validation[
                ["patient_id", "age", "sex", "cimt_mm"]
            ].copy()
            output["seed"] = seed
            output["fold"] = fold
            output["selected_centile_weight"] = weight
            output["mean_prediction_mm"] = mean_prediction
            output["centile_prediction_mm"] = centile_prediction
            output["fused_prediction_mm"] = fused_prediction
            output["disagreement_uncertainty"] = uncertainty
            predictions.append(output)
            tuning_rows.append({
                "seed": seed,
                "fold": fold,
                "selected_centile_weight": weight,
                "gaussian_alpha": heads["gaussian_alpha"],
                "robust_alpha": heads["robust_alpha"],
            })
            print(
                f"seed={seed} fold={fold}/5 weight={weight:.1f} complete",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "repeated_oof_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(
        args.output / "outer_tuning.csv", index=False
    )
    pd.concat(inner_weight_rows, ignore_index=True).to_csv(
        args.output / "inner_weight_curves.csv", index=False
    )

    per_seed = []
    coverage_rows = []
    columns = {
        "H0_mean": "mean_prediction_mm",
        "H1_centile": "centile_prediction_mm",
        "H2_nested_fusion": "fused_prediction_mm",
    }
    for seed, group in oof.groupby("seed"):
        uncertainty = uncertainty_metrics(group)
        for experiment, column in columns.items():
            per_seed.append({
                "experiment": experiment,
                "seed": int(seed),
                **prediction_metrics(group, column),
                **(uncertainty if experiment == "H2_nested_fusion" else {}),
            })
        ordered = group.sort_values("disagreement_uncertainty")
        for coverage in (1.0, .9, .8, .7):
            selected = ordered.iloc[:int(np.ceil(len(ordered) * coverage))]
            coverage_rows.append({
                "seed": int(seed),
                "coverage": coverage,
                "n": len(selected),
                "mae_mm": float(mean_absolute_error(
                    selected.cimt_mm, selected.fused_prediction_mm
                )),
            })
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "coverage_risk.csv", index=False)
    summary = per_seed_frame.groupby("experiment").agg(
        {column: ["mean", "std"] for column in per_seed_frame.columns
         if column not in ("experiment", "seed")}
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    deltas = []
    for (seed, fold), group in oof.groupby(["seed", "fold"]):
        deltas.append(
            mean_absolute_error(group.cimt_mm, group.fused_prediction_mm)
            - mean_absolute_error(group.cimt_mm, group.mean_prediction_mm)
        )
    coverage_summary = (
        coverage.groupby("coverage")
        .agg(mae_mean=("mae_mm", "mean"), mae_std=("mae_mm", "std"),
             n_mean=("n", "mean"))
        .reset_index()
    )
    report = {
        "protocol": "E11_双估计头分歧不确定性_锁定协议.md",
        "development_only": True,
        "fixed_val_used": False,
        "test_used": False,
        "seeds": seeds,
        "outer_folds_per_seed": 5,
        "weight_grid": WEIGHTS.tolist(),
        "rmse_margin_mm": RMSE_MARGIN_MM,
        "summary": summary.to_dict(orient="records"),
        "selected_weight_counts": (
            pd.Series([row["selected_centile_weight"] for row in tuning_rows])
            .value_counts().sort_index().to_dict()
        ),
        "fold_direction": {
            "folds_improved_or_equal": int((np.asarray(deltas) <= 0).sum()),
            "folds_total": len(deltas),
            "mean_fold_delta_mm": float(np.mean(deltas)),
        },
        "bootstrap": bootstrap_deltas(
            oof, args.bootstrap_repetitions, seeds[0]
        ),
        "coverage_summary": coverage_summary.to_dict(orient="records"),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(report["bootstrap"], indent=2))


if __name__ == "__main__":
    main()
