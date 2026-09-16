#!/usr/bin/env python3
"""Leakage-safe outer-CV evaluation of the two stable A/V phenotypes.

This deliberately tests the smallest hypothesis that survived the feature
stability audit. Official validation and test sets are not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CORE = ["common_av_density_ratio", "common_artery_density"]


def score(frame: pd.DataFrame) -> dict:
    y = frame.accelerated_cimt.to_numpy()
    delta = frame.delta_cimt_mm.to_numpy()
    return {
        "n": len(frame),
        "auroc": roc_auc_score(y, frame.probability),
        "auprc": average_precision_score(y, frame.probability),
        "delta_mae": mean_absolute_error(delta, frame.prediction_delta_mm),
        "delta_r2": r2_score(delta, frame.prediction_delta_mm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("artifacts/topology_transport_features/features.csv"))
    parser.add_argument("--manifests", type=Path, default=Path("artifacts/bimfo_v2_outer_cv/manifests"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/av_core_cv"))
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    phenotypes = pd.read_csv(args.features, dtype={"patient_id": str}).set_index("patient_id")
    outputs, fold_rows = [], []
    for fold in range(1, 6):
        manifest = pd.read_csv(args.manifests / f"fold_{fold}.csv", dtype={"patient_id": str})
        train = manifest.query("split == 'train'").copy()
        valid = manifest.query("split == 'val'").copy()
        for model in ["av_core", "av_core_clinical"]:
            columns = CORE + (["age", "sex"] if model.endswith("clinical") else [])
            x_train = pd.concat(
                [phenotypes.reindex(train.patient_id)[CORE].reset_index(drop=True),
                 train[["age", "sex"]].reset_index(drop=True)],
                axis=1,
            )[columns].to_numpy(float)
            x_valid = pd.concat(
                [phenotypes.reindex(valid.patient_id)[CORE].reset_index(drop=True),
                 valid[["age", "sex"]].reset_index(drop=True)],
                axis=1,
            )[columns].to_numpy(float)
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=np.logspace(-4, 1, 8), cv=4, scoring="roc_auc",
                    class_weight="balanced", max_iter=4000,
                    random_state=args.seed + fold, n_jobs=-1,
                ),
            ).fit(x_train, train.accelerated_cimt)
            regressor = make_pipeline(
                StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))
            ).fit(x_train, train.delta_cimt_mm)
            out = valid[["patient_id", "age", "sex", "accelerated_cimt", "delta_cimt_mm"]].copy()
            out["fold"] = fold
            out["model"] = model
            out["probability"] = classifier.predict_proba(x_valid)[:, 1]
            out["prediction_delta_mm"] = regressor.predict(x_valid)
            outputs.append(out)
            fold_rows.append({"fold": fold, "model": model, **score(out)})

    oof = pd.concat(outputs, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    overall = pd.DataFrame([{"model": name, **score(frame)} for name, frame in oof.groupby("model")])
    oof.to_csv(args.output / "oof_predictions.csv", index=False)
    folds.to_csv(args.output / "fold_metrics.csv", index=False)
    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    summary = {
        "development_patients": int(oof.patient_id.nunique()),
        "features": CORE,
        "outer_folds": 5,
        "inner_cv_only": True,
        "official_validation_used": False,
        "official_test_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(overall.to_string(index=False))
    print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
