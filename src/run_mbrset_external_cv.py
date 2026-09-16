#!/usr/bin/env python3
"""Patient-level nested CV for frozen retinal phenotypes on mBRSET."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260730


def pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )),
    ])


def tune(
    frame: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    scores: dict[float, float] = {}
    for c_value in (.001, .01, .1, 1.0, 10.0):
        fold_scores = []
        for fit, hold in splits:
            model = pipeline(c_value)
            model.fit(frame.iloc[fit][features], frame.iloc[fit]["outcome"])
            probability = model.predict_proba(frame.iloc[hold][features])[:, 1]
            fold_scores.append(roc_auc_score(
                frame.iloc[hold]["outcome"], probability
            ))
        scores[c_value] = float(np.mean(fold_scores))
    return max(scores, key=scores.get)


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=5000).fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["outcome"].to_numpy(dtype=int)
    probability = frame["probability"].to_numpy(dtype=float)
    intercept, slope = calibration(y, probability)
    return {
        "n": int(len(frame)),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(
            y, probability >= .5
        )),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def paired_bootstrap(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    repetitions: int,
) -> dict[str, object]:
    joined = challenger.merge(
        baseline,
        on=["patient_id", "outcome"],
        suffixes=("_challenger", "_baseline"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(SEED)
    samples = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(repetitions):
        sample = joined.iloc[rng.integers(0, len(joined), len(joined))]
        if sample["outcome"].nunique() != 2:
            continue
        y = sample["outcome"]
        pc = sample["probability_challenger"]
        pb = sample["probability_baseline"]
        samples["auroc"].append(
            roc_auc_score(y, pc) - roc_auc_score(y, pb)
        )
        samples["auprc"].append(
            average_precision_score(y, pc)
            - average_precision_score(y, pb)
        )
        samples["brier"].append(
            brier_score_loss(y, pc) - brier_score_loss(y, pb)
        )
    report = {}
    for name, values in samples.items():
        array = np.asarray(values)
        report[name] = {
            "mean_delta": float(array.mean()),
            "ci95": [
                float(np.quantile(array, .025)),
                float(np.quantile(array, .975)),
            ],
        }
    return report


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    clinical = ["age", "sex", "dm_time"]
    ocular = ["max_icdr", "mean_icdr", "any_dr", "any_edema"]
    candidates = [
        column for column in frame.columns
        if column.endswith("__bilateral_mean")
        or column.endswith("__absolute_difference")
        or column.endswith("__within_eye_std")
    ]
    phenotype = sorted(set(candidates))
    without_zone = [column for column in phenotype if "zone_b" not in column]
    uncertainty_tokens = (
        "entropy", "disagreement", "mean_gate", "support_pixels",
        "disc_radius", "zone_b_valid_pixels",
    )
    without_uncertainty = [
        column for column in phenotype
        if not any(token in column for token in uncertainty_tokens)
    ]
    return {
        "M0_clinical": clinical,
        "M1_clinical_ocular": clinical + ocular,
        "M2_clinical_vascular": clinical + phenotype,
        "M3_full": clinical + ocular + phenotype,
        "M4_full_without_zone_b": clinical + ocular + without_zone,
        "M5_full_without_uncertainty": (
            clinical + ocular + without_uncertainty
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--phenotypes", type=Path, required=True)
    parser.add_argument("--outcome", default="systemic_hypertension")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--no-dr-only", action="store_true")
    args = parser.parse_args()

    patients = pd.read_csv(args.patients, dtype={"patient_id": str})
    phenotypes = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    frame = patients.merge(
        phenotypes.drop(columns=["split"], errors="ignore"),
        on="patient_id",
        validate="one_to_one",
    )
    if args.no_dr_only:
        frame = frame.loc[frame["all_images_no_dr"].eq(True)].copy()
    frame = frame.loc[frame[args.outcome].notna()].copy()
    frame["outcome"] = frame[args.outcome].astype(int)
    if frame["outcome"].value_counts().min() < args.outer_folds:
        raise ValueError("Too few outcome events for the requested folds.")

    sets = feature_sets(frame)
    outer = StratifiedKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=SEED
    )
    predictions = []
    tuning = []
    for experiment, features in sets.items():
        missing = [column for column in features if column not in frame]
        if missing:
            raise KeyError(f"{experiment} missing columns: {missing[:5]}")
        for fold, (fit, hold) in enumerate(
            outer.split(frame, frame["outcome"]), start=1
        ):
            train, validation = frame.iloc[fit], frame.iloc[hold]
            inner = StratifiedKFold(
                n_splits=args.inner_folds,
                shuffle=True,
                random_state=SEED + fold,
            )
            inner_splits = list(inner.split(train, train["outcome"]))
            c_value = tune(train, features, inner_splits)
            model = pipeline(c_value)
            model.fit(train[features], train["outcome"])
            output = validation[["patient_id", "outcome"]].copy()
            output["experiment"] = experiment
            output["fold"] = fold
            output["probability"] = model.predict_proba(
                validation[features]
            )[:, 1]
            predictions.append(output)
            tuning.append({
                "experiment": experiment,
                "fold": fold,
                "features": len(features),
                "logistic_c": c_value,
            })
            print(
                f"{experiment} fold={fold}/{args.outer_folds} "
                f"features={len(features)} C={c_value}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(
        args.output / "nested_tuning.csv", index=False
    )
    overall = {
        experiment: metrics(group)
        for experiment, group in oof.groupby("experiment")
    }
    comparisons = {}
    for challenger, baseline in (
        ("M1_clinical_ocular", "M0_clinical"),
        ("M2_clinical_vascular", "M0_clinical"),
        ("M3_full", "M0_clinical"),
        ("M3_full", "M1_clinical_ocular"),
        ("M3_full", "M2_clinical_vascular"),
        ("M3_full", "M4_full_without_zone_b"),
        ("M3_full", "M5_full_without_uncertainty"),
    ):
        comparisons[f"{challenger}__minus__{baseline}"] = paired_bootstrap(
            oof.loc[oof["experiment"].eq(challenger)],
            oof.loc[oof["experiment"].eq(baseline)],
            args.bootstrap_repetitions,
        )
    report = {
        "dataset": "mBRSET",
        "outcome": args.outcome,
        "no_dr_only": args.no_dr_only,
        "patients": int(len(frame)),
        "positive": int(frame["outcome"].sum()),
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
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
