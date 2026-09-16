#!/usr/bin/env python3
"""Create leakage-safe outer-CV manifests for the BiMFO signal validation stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from prepare_bimfo_v1 import expected_cimt_model


def add_fold_labels(train: pd.DataFrame, val: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """OOF expected-CIMT labels for outer training and train-only labels for outer validation."""
    train = train.copy().reset_index(drop=True)
    val = val.copy().reset_index(drop=True)
    oof_expected = np.empty(len(train), dtype=np.float64)
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, hold_index in inner.split(train):
        model = expected_cimt_model()
        model.fit(train.loc[fit_index, ["age", "sex"]], train.loc[fit_index, "cimt_mm"])
        oof_expected[hold_index] = model.predict(train.loc[hold_index, ["age", "sex"]])
    full = expected_cimt_model()
    full.fit(train[["age", "sex"]], train.cimt_mm)
    train["expected_cimt_mm"] = oof_expected
    val["expected_cimt_mm"] = full.predict(val[["age", "sex"]])
    for frame in (train, val):
        frame["delta_cimt_mm"] = frame.cimt_mm - frame.expected_cimt_mm
    threshold = float(train.delta_cimt_mm.quantile(.75))
    for frame in (train, val):
        frame["accelerated_cimt"] = (frame.delta_cimt_mm >= threshold).astype(int)
    return train, val, threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, default=Path("artifacts/bimfo_v1_1_cache/patients.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bimfo_v2_outer_cv/manifests"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    patients = pd.read_csv(args.patients, dtype={"patient_id": str})
    patients["cache_index"] = np.arange(len(patients))
    development = patients.loc[patients.split.eq("train")].copy().reset_index(drop=True)
    # Stratification is used only to balance outer folds; no outcome values are
    # ever used by a training fold to construct the held-out labels.
    cimt_bin = pd.qcut(development.cimt_mm, q=4, labels=False, duplicates="drop").astype(str)
    strata = development.sex.astype(str) + "_" + cimt_bin
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for fold, (train_index, val_index) in enumerate(outer.split(development, strata), start=1):
        train, val, threshold = add_fold_labels(
            development.iloc[train_index], development.iloc[val_index], args.seed + fold
        )
        train["split"] = "train"
        val["split"] = "val"
        manifest = pd.concat([train, val], ignore_index=True)
        path = args.output / f"fold_{fold}.csv"
        manifest.to_csv(path, index=False)
        summary.append({
            "fold": fold, "manifest": str(path), "train_n": len(train), "val_n": len(val),
            "train_positive_rate": float(train.accelerated_cimt.mean()),
            "val_positive_rate": float(val.accelerated_cimt.mean()),
            "threshold_mm": threshold,
            "train_delta_mean": float(train.delta_cimt_mm.mean()),
            "val_delta_mean": float(val.delta_cimt_mm.mean()),
        })
    (args.output / "summary.json").write_text(json.dumps({
        "n_development_patients": len(development), "outer_folds": args.folds,
        "outer_seed": args.seed,
        "label_protocol": "outer_train_only_expected_CIMT; OOF_within_outer_train",
        "test_set_used": False, "folds": summary,
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
