#!/usr/bin/env python3
"""Age/sex-only baselines. These quantify incremental value from retinal images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def metrics(y_true, prob, target, prediction):
    return {
        "auroc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "mae_mm": float(mean_absolute_error(target, prediction)),
        "rmse_mm": float(mean_squared_error(target, prediction) ** 0.5),
        "r2": float(r2_score(target, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str})
    train = frame.query("split == 'train'")
    evaluate = frame.query("split == @args.split")
    features = ["age", "sex"]
    preprocessor = ColumnTransformer([("clinical", Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())
    ]), features)])

    classifier = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026))])
    regressor = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))])
    classifier.fit(train[features], train.label)
    regressor.fit(train[features], train.cimt_mm)
    probability = classifier.predict_proba(evaluate[features])[:, 1]
    cimt_prediction = regressor.predict(evaluate[features])
    result = metrics(evaluate.label, probability, evaluate.cimt_mm, cimt_prediction)
    output = args.output or Path(f"artifacts/clinical_{args.split}_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
