"""Paper-oriented evaluation and artifact generation for CIMT experiments."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, r2_score,
                             roc_auc_score, roc_curve)


def _ci(values: np.ndarray) -> dict[str, float]:
    return {"point": float(values[0]), "ci95_low": float(np.quantile(values[1:], 0.025)), "ci95_high": float(np.quantile(values[1:], 0.975))}


def _classification(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, probability)), "auprc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)), "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"), "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"), "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Choose the Youden threshold from validation data only."""
    fpr, tpr, thresholds = roc_curve(y, probability)
    return float(thresholds[np.argmax(tpr - fpr)])


def bootstrap(y, probability, target, prediction, threshold, repetitions=1000, seed=2026) -> dict:
    rng = np.random.default_rng(seed)
    point = _classification(y, probability, threshold)
    point.update({"mae_mm": float(mean_absolute_error(target, prediction)), "rmse_mm": float(mean_squared_error(target, prediction) ** .5), "r2": float(r2_score(target, prediction))})
    samples = {name: [value] for name, value in point.items() if isinstance(value, float) and np.isfinite(value)}
    for _ in range(repetitions):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        run = _classification(y[index], probability[index], threshold)
        run.update({"mae_mm": float(mean_absolute_error(target[index], prediction[index])), "rmse_mm": float(mean_squared_error(target[index], prediction[index]) ** .5), "r2": float(r2_score(target[index], prediction[index]))})
        for name in samples:
            if name in run and np.isfinite(run[name]):
                samples[name].append(run[name])
    return {name: _ci(np.asarray(values)) for name, values in samples.items()}


def calibration(y: np.ndarray, probability: np.ndarray) -> dict:
    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    model = LogisticRegression(C=1e6, max_iter=2000).fit(np.log(clipped / (1 - clipped)).reshape(-1, 1), y)
    return {"predicted": predicted.tolist(), "observed": observed.tolist(), "intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def make_figures(frame: pd.DataFrame, report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    y, probability = frame.label.to_numpy(), frame.probability.to_numpy()
    fpr, tpr, _ = roc_curve(y, probability)
    plt.figure(figsize=(5, 5)); plt.plot(fpr, tpr, label=f"AUROC={report['metrics']['auroc']['point']:.3f}"); plt.plot([0, 1], [0, 1], "--", color="grey"); plt.xlabel("1 − Specificity"); plt.ylabel("Sensitivity"); plt.legend(); plt.tight_layout(); plt.savefig(output / "roc.png", dpi=220); plt.close()
    recall, precision, _ = precision_recall_curve(y, probability)
    plt.figure(figsize=(5, 5)); plt.plot(recall, precision, label=f"AUPRC={report['metrics']['auprc']['point']:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout(); plt.savefig(output / "pr.png", dpi=220); plt.close()
    cal = report["calibration"]
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "--", color="grey"); plt.plot(cal["predicted"], cal["observed"], "o-"); plt.xlabel("Predicted risk"); plt.ylabel("Observed frequency"); plt.tight_layout(); plt.savefig(output / "calibration.png", dpi=220); plt.close()
    plt.figure(figsize=(5, 5)); plt.scatter(frame.cimt_mm, frame.prediction_mm, alpha=.6, s=14); limits = [min(frame.cimt_mm.min(), frame.prediction_mm.min()), max(frame.cimt_mm.max(), frame.prediction_mm.max())]; plt.plot(limits, limits, "--", color="grey"); plt.xlabel("Observed CIMT (mm)"); plt.ylabel("Predicted CIMT (mm)"); plt.tight_layout(); plt.savefig(output / "regression_scatter.png", dpi=220); plt.close()


def write_report(frame: pd.DataFrame, output: Path, threshold: float | None = None, bootstrap_repetitions: int = 1000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    y, probability, target, prediction = (frame[key].to_numpy() for key in ("label", "probability", "cimt_mm", "prediction_mm"))
    threshold = choose_threshold(y, probability) if threshold is None else threshold
    report = {"n_patients": int(len(frame)), "prevalence": float(y.mean()), "threshold": threshold,
              "metrics": bootstrap(y, probability, target, prediction, threshold, repetitions=bootstrap_repetitions), "calibration": calibration(y, probability)}
    subgroups = {}
    for column in ("sex", "age_band"):
        for value, subgroup in frame.groupby(column):
            if len(subgroup) >= 10 and subgroup.label.nunique() == 2:
                subgroups[f"{column}={value}"] = _classification(subgroup.label.to_numpy(), subgroup.probability.to_numpy(), threshold)
                subgroups[f"{column}={value}"]["n"] = int(len(subgroup))
    report["subgroups"] = subgroups
    frame.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figures(frame, report, output / "figures")
    return report
