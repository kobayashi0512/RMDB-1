#!/usr/bin/env python3
"""Outcome-aware but leakage-controlled stability screen for frozen BiRAP phenotypes.

Patients belonging to the held-out China-Fundus-CIMT test group are never read.
Age/sex conditional CIMT residuals are learned separately inside each outer fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Dependency-free Benjamini--Hochberg adjusted q-values."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order] * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def bootstrap_ci(x: np.ndarray, y: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    values = []
    n = len(x)
    for _ in range(repeats):
        index = rng.integers(0, n, n)
        values.append(spearmanr(x[index], y[index]).statistic)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotypes", type=Path, default=Path("artifacts/birap_phenotypes/patient_level.csv"))
    parser.add_argument("--patients", type=Path, default=Path("artifacts/bimfo_v1_1_cache/patients.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/birap_stability"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pheno = pd.read_csv(args.phenotypes, dtype={"patient_id": str})
    cohort = pd.read_csv(args.patients, dtype={"patient_id": str})
    # group=1 is the original development cohort; group=2 remains locked test data.
    cohort = cohort.loc[cohort.group == 1, ["patient_id", "age", "sex", "cimt_mm", "accelerated_cimt"]]
    data = cohort.merge(pheno, on="patient_id", how="inner", validate="one_to_one").reset_index(drop=True)
    feature_cols = [c for c in pheno.columns if c.startswith(("common_", "discordance_"))]
    # Remove administrative/measurement flags from hypothesis family.
    feature_cols = [c for c in feature_cols if not c.endswith(("fov_fraction", "landmark_valid"))]
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_id = np.full(len(data), -1, dtype=int)
    residual = np.full(len(data), np.nan)
    for fold, (train, val) in enumerate(splitter.split(data, data.accelerated_cimt)):
        baseline = LinearRegression().fit(data.loc[train, ["age", "sex"]], data.loc[train, "cimt_mm"])
        residual[val] = data.loc[val, "cimt_mm"] - baseline.predict(data.loc[val, ["age", "sex"]])
        fold_id[val] = fold
    data["outer_fold"] = fold_id
    data["age_sex_oof_residual"] = residual
    rng = np.random.default_rng(args.seed)
    rows = []
    for feature in feature_cols:
        valid = data[[feature, "age_sex_oof_residual"]].dropna()
        if len(valid) < 500 or valid[feature].nunique() < 20:
            continue
        fold_rhos = []
        for fold in range(args.folds):
            part = valid.loc[data.loc[valid.index, "outer_fold"] == fold]
            fold_rhos.append(float(spearmanr(part[feature], part.age_sex_oof_residual).statistic))
        rho, p = spearmanr(valid[feature], valid.age_sex_oof_residual)
        lo, hi = bootstrap_ci(valid[feature].to_numpy(), valid.age_sex_oof_residual.to_numpy(), args.bootstrap, rng)
        signs = np.sign(fold_rhos)
        rows.append({
            "feature": feature, "n": len(valid), "spearman_rho": float(rho), "p_value": float(p),
            "ci95_low": lo, "ci95_high": hi, "fold_rhos": json.dumps(fold_rhos),
            "same_sign_folds": int(max((signs > 0).sum(), (signs < 0).sum())),
        })
    report = pd.DataFrame(rows).sort_values("p_value")
    report["fdr_q"] = benjamini_hochberg(report.p_value.to_numpy())
    report["stable"] = (report.same_sign_folds >= 4) & (report.fdr_q < 0.10) & ((report.ci95_low > 0) | (report.ci95_high < 0))
    report.to_csv(args.output / "phenotype_stability.csv", index=False)
    data.to_csv(args.output / "development_oof_residuals.csv", index=False)
    summary = {
        "development_patients": int(len(data)), "candidate_features": int(len(feature_cols)),
        "screened_features": int(len(report)), "stable_features": int(report.stable.sum()),
        "folds": args.folds, "bootstrap_repetitions": args.bootstrap,
        "held_out_test_used": False,
        "stability_rule": "same association direction in >=4/5 folds, BH-FDR q<0.10, and bootstrap 95% CI excluding zero",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(report.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
