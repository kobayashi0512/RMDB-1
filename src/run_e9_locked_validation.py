#!/usr/bin/env python3
"""One-shot fixed-validation evaluation of frozen E8-B/E8-D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
from run_normative_deviation_outer_cv import (
    bilateral,
    crossfit_normative,
    phenotype_classifier,
    phenotype_regressor,
    strata,
    tune_phenotype,
)


SEED = 20260729


def continuous_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    return {
        "mae_mm": float(mean_absolute_error(truth, prediction)),
        "rmse_mm": float(mean_squared_error(truth, prediction) ** .5),
        "r2": float(r2_score(truth, prediction)),
        "spearman_rho": float(pd.DataFrame({
            "truth": truth, "prediction": prediction
        }).corr(method="spearman").iloc[0, 1]),
    }


def classification_metrics(
    truth: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    return {
        "prevalence": float(truth.mean()),
        "auroc": float(roc_auc_score(truth, probability)),
        "auprc": float(average_precision_score(truth, probability)),
        "brier": float(brier_score_loss(truth, probability)),
    }


def bootstrap(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mae_delta, auroc, auprc = [], [], []
    for _ in range(repetitions):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        mae_delta.append(
            mean_absolute_error(sample.cimt_mm, sample.e8b_prediction_mm)
            - mean_absolute_error(sample.cimt_mm, sample.normative_prediction_mm)
        )
        if sample.accelerated.nunique() == 2:
            auroc.append(roc_auc_score(
                sample.accelerated, sample.e8d_probability
            ))
            auprc.append(average_precision_score(
                sample.accelerated, sample.e8d_probability
            ))
    return {
        "e8b_minus_normative_mae_mm": {
            "point": float(
                mean_absolute_error(frame.cimt_mm, frame.e8b_prediction_mm)
                - mean_absolute_error(
                    frame.cimt_mm, frame.normative_prediction_mm
                )
            ),
            "ci95": [
                float(np.quantile(mae_delta, .025)),
                float(np.quantile(mae_delta, .975)),
            ],
        },
        "e8d_auroc": {
            "point": float(roc_auc_score(
                frame.accelerated, frame.e8d_probability
            )),
            "ci95": [
                float(np.quantile(auroc, .025)),
                float(np.quantile(auroc, .975)),
            ],
        },
        "e8d_auprc": {
            "point": float(average_precision_score(
                frame.accelerated, frame.e8d_probability
            )),
            "ci95": [
                float(np.quantile(auprc, .025)),
                float(np.quantile(auprc, .975)),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument(
        "--train-phenotypes", type=Path, required=True
    )
    parser.add_argument(
        "--validation-phenotypes", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    outcomes = pd.read_csv(
        args.manifest,
        usecols=["patient_id", "split", "age", "sex", "cimt_mm"],
        dtype={"patient_id": str},
    )
    train_outcomes = outcomes.loc[outcomes.split.eq("train")].copy()
    validation_outcomes = outcomes.loc[outcomes.split.eq("val")].copy()
    if len(train_outcomes) != 2603 or len(validation_outcomes) != 200:
        raise ValueError(
            f"Expected train=2603,val=200; found "
            f"{len(train_outcomes)},{len(validation_outcomes)}"
        )
    train_phenotypes = pd.read_csv(
        args.train_phenotypes, dtype={"patient_id": str}
    )
    validation_phenotypes = pd.read_csv(
        args.validation_phenotypes, dtype={"patient_id": str}
    )
    if not train_phenotypes.split.eq("train").all():
        raise ValueError("Train phenotypes contain another split.")
    if not validation_phenotypes.split.eq("val").all():
        raise ValueError("Validation phenotypes contain another split.")
    train = train_outcomes.merge(
        train_phenotypes.drop(columns=["split"]),
        on="patient_id", validate="one_to_one",
    ).reset_index(drop=True)
    validation = validation_outcomes.merge(
        validation_phenotypes.drop(columns=["split"]),
        on="patient_id", validate="one_to_one",
    ).reset_index(drop=True)
    if len(train) != 2603 or len(validation) != 200:
        raise ValueError(
            "Phenotype/outcome merge was incomplete: expected "
            f"train=2603,val=200; found train={len(train)},val={len(validation)}"
        )
    required = ["age", "sex", "cimt_mm", *bilateral(MASS_MECHANISMS)]
    missing = {
        "train": int(train[required].isna().sum().sum()),
        "validation": int(validation[required].isna().sum().sum()),
    }
    if any(missing.values()):
        raise ValueError(f"Required E9 inputs contain missing values: {missing}")

    features = bilateral(MASS_MECHANISMS)
    mean_features = bilateral(MASS_MECHANISMS, include_asymmetry=False)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(inner.split(train, strata(train)))
    normative = crossfit_normative(train, validation, 3, SEED)
    residual_train = train.cimt_mm.to_numpy() - normative["mu_train"]
    residual_validation = (
        validation.cimt_mm.to_numpy() - normative["mu_validation"]
    )
    z_train = residual_train / normative["sigma_train"]
    z_validation = residual_validation / normative["sigma_validation"]
    threshold = float(np.quantile(z_train, .75))
    accelerated_train = (z_train >= threshold).astype(int)
    accelerated_validation = (z_validation >= threshold).astype(int)

    train_orth, validation_orth = residualize_features(
        train, validation, features, splits
    )
    orthogonal_names = list(train_orth.columns)
    local_train = train.copy()
    for column in orthogonal_names:
        local_train[column] = train_orth[column].to_numpy()
    alpha, _ = tune_phenotype(
        local_train, orthogonal_names, z_train, accelerated_train, 3, SEED + 100
    )
    regressor = phenotype_regressor(alpha)
    regressor.fit(train_orth, z_train)
    predicted_z = regressor.predict(validation_orth)
    e8b_prediction = (
        normative["mu_validation"]
        + normative["sigma_validation"] * predicted_z
    )

    mean_orthogonal_names = orthogonal_names[:len(mean_features)]
    _, c_value = tune_phenotype(
        local_train, mean_orthogonal_names, z_train, accelerated_train,
        3, SEED + 200,
    )
    classifier = phenotype_classifier(c_value)
    classifier.fit(train_orth[mean_orthogonal_names], accelerated_train)
    probability = classifier.predict_proba(
        validation_orth[mean_orthogonal_names]
    )[:, 1]

    predictions = validation[
        ["patient_id", "age", "sex", "cimt_mm"]
    ].copy()
    predictions["true_z"] = z_validation
    predictions["accelerated"] = accelerated_validation
    predictions["normative_prediction_mm"] = normative["mu_validation"]
    predictions["e8b_predicted_z"] = predicted_z
    predictions["e8b_prediction_mm"] = e8b_prediction
    predictions["e8d_probability"] = probability
    predictions["age_band"] = pd.cut(
        predictions.age,
        [-np.inf, 39, 49, 59, np.inf],
        labels=["<40", "40-49", "50-59", ">=60"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)

    subgroup = []
    for variable in ("sex", "age_band"):
        for level, group in predictions.groupby(variable, observed=True):
            row = {
                "variable": variable,
                "level": str(level),
                "n": len(group),
                "normative_mae": mean_absolute_error(
                    group.cimt_mm, group.normative_prediction_mm
                ),
                "e8b_mae": mean_absolute_error(
                    group.cimt_mm, group.e8b_prediction_mm
                ),
            }
            row["mae_delta"] = row["e8b_mae"] - row["normative_mae"]
            if group.accelerated.nunique() == 2:
                row["e8d_auroc"] = roc_auc_score(
                    group.accelerated, group.e8d_probability
                )
            else:
                row["e8d_auroc"] = np.nan
            subgroup.append(row)
    subgroup_frame = pd.DataFrame(subgroup)
    subgroup_frame.to_csv(args.output / "subgroups.csv", index=False)

    report = {
        "protocol": "E9_固定验证集一次性验证_锁定协议.md",
        "one_shot_validation": True,
        "validation_patients": len(validation),
        "test_used": False,
        "normative": continuous_metrics(
            validation.cimt_mm.to_numpy(), normative["mu_validation"]
        ),
        "e8b": continuous_metrics(
            validation.cimt_mm.to_numpy(), e8b_prediction
        ),
        "e8d": classification_metrics(
            accelerated_validation, probability
        ),
        "true_z_age_spearman": float(predictions[
            ["true_z", "age"]
        ].corr(method="spearman").iloc[0, 1]),
        "bootstrap": bootstrap(
            predictions, args.bootstrap_repetitions, SEED
        ),
        "locked_choices": {
            "normative_alpha": normative["alpha"],
            "e8b_ridge_alpha": alpha,
            "e8d_logistic_c": c_value,
            "accelerated_threshold_z": threshold,
            "regression_features": features,
            "classification_features": mean_features,
        },
        "subgroups": subgroup,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
