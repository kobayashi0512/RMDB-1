#!/usr/bin/env python3
"""Reverse audit: explain when a frozen retinal phenotype should not advance.

This script treats pixel error and dataset identity as measurement-audit labels,
never as clinical outcomes.  All feature sets are read from the locked protocol;
no feature search is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


SEED = 20260821


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, received {value}")
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def normalized_id(frame: pd.DataFrame) -> pd.Series:
    source = frame["id"] if "id" in frame.columns else frame["image"]
    return source.astype(str).map(lambda value: Path(value).stem)


def holm(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    delta = first - second
    indices = rng.integers(0, len(delta), size=(repetitions, len(delta)))
    means = delta[indices].mean(axis=1)
    return {
        "point": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def oof_ridge(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    folds: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = frame[features].to_numpy(float)
    y = frame[target].to_numpy(float)
    outer = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    predictions = np.full(len(frame), np.nan)
    tuning = []
    for fold, (train, test) in enumerate(outer.split(x), 1):
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED + fold)
        pipeline = make_pipeline(StandardScaler(), Ridge())
        search = GridSearchCV(
            pipeline,
            {"ridge__alpha": np.logspace(-3, 3, 13)},
            scoring="neg_mean_absolute_error",
            cv=inner,
            n_jobs=-1,
        )
        search.fit(x[train], y[train])
        predictions[test] = search.predict(x[test])
        tuning.append({"fold": fold, "alpha": float(search.best_params_["ridge__alpha"])})
    return predictions, tuning


def load_pixel_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame = frame.loc[frame.model.eq("vascx")].copy()
    frame["id"] = normalized_id(frame)
    required = {"id", "mean_av_dice"}
    if missing := sorted(required - set(frame.columns)):
        raise KeyError(f"{path}: missing {missing}")
    frame["pixel_error"] = 1.0 - frame["mean_av_dice"].astype(float)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-eye", action="append", required=True, help="DATASET=per_eye.csv")
    parser.add_argument("--pixel-metrics", action="append", required=True, help="DATASET=per_image_metrics.csv")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    per_eye_paths = parse_mapping(args.per_eye)
    metric_paths = parse_mapping(args.pixel_metrics)
    all_features = config["prespecified_eye_phenotypes"]
    reliable = config["previously_frozen_reliable_core"]
    rejected = config["previously_rejected_core"]
    reverse_qc = [
        "vessel_occupancy_disagreement",
        "base_to_mass_identity_disagreement",
        "identity_entropy",
    ]
    feature_sets = {
        "Q0_reliable_core": reliable,
        "Q1_rejected_core": rejected,
        "Q2_reverse_qc_triad": reverse_qc,
        "Q3_all_prespecified": all_features,
    }

    phenotype_parts = []
    for dataset, path in per_eye_paths.items():
        frame = pd.read_csv(path, dtype={"id": str, "patient_id": str})
        frame["id"] = normalized_id(frame)
        frame["dataset"] = dataset
        missing = sorted(set(all_features) - set(frame.columns))
        if missing:
            raise KeyError(f"{dataset} phenotype table missing {missing}")
        phenotype_parts.append(frame[["id", "dataset", *all_features]])
    phenotypes = pd.concat(phenotype_parts, ignore_index=True)
    finite = np.isfinite(phenotypes[all_features].to_numpy(float)).all(axis=1)
    phenotypes = phenotypes.loc[finite].reset_index(drop=True)

    encoder = LabelEncoder()
    y_domain = encoder.fit_transform(phenotypes.dataset)
    x_domain = phenotypes[reliable].to_numpy(float)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=SEED),
    )
    domain_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    domain_probability = cross_val_predict(
        classifier, x_domain, y_domain, cv=domain_cv, method="predict_proba", n_jobs=-1,
    )
    domain_prediction = domain_probability.argmax(axis=1)
    y_binary = label_binarize(y_domain, classes=np.arange(len(encoder.classes_)))
    domain_metrics = {
        "n": len(phenotypes),
        "classes": encoder.classes_.tolist(),
        "accuracy": float(accuracy_score(y_domain, domain_prediction)),
        "macro_ovr_auroc": float(roc_auc_score(y_binary, domain_probability, average="macro", multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_domain, domain_prediction).tolist(),
        "features": reliable,
        "clinical_outcomes_used": False,
    }
    domain_oof = phenotypes[["id", "dataset"]].copy()
    domain_oof["predicted_dataset"] = encoder.inverse_transform(domain_prediction)
    for index, name in enumerate(encoder.classes_):
        domain_oof[f"probability_{name}"] = domain_probability[:, index]
    domain_oof.to_csv(args.output / "domain_oof_predictions.csv", index=False)

    pixel_parts = []
    for dataset, path in metric_paths.items():
        metrics = load_pixel_metrics(path)
        eyes = phenotypes.loc[phenotypes.dataset.eq(dataset)].drop(columns="dataset")
        merged = eyes.merge(metrics, on="id", validate="one_to_one")
        if len(merged) != len(metrics):
            raise ValueError(f"{dataset}: phenotype and pixel metric tables do not align")
        merged["dataset"] = dataset
        pixel_parts.append(merged)
    pixel = pd.concat(pixel_parts, ignore_index=True)
    pixel.to_csv(args.output / "pixel_failure_analysis_table.csv", index=False)

    model_rows = []
    oof_parts = []
    tuning_rows = []
    rng = np.random.default_rng(SEED)
    absolute_errors: dict[tuple[str, str], np.ndarray] = {}
    for dataset, frame in pixel.groupby("dataset", sort=True):
        for model_name, features in feature_sets.items():
            prediction, tuning = oof_ridge(frame.reset_index(drop=True), features, "pixel_error", folds=5)
            observed = frame.pixel_error.to_numpy(float)
            mae = np.abs(observed - prediction)
            rho = spearmanr(observed, prediction)
            model_rows.append({
                "dataset": dataset,
                "model": model_name,
                "n": len(frame),
                "features": ";".join(features),
                "mae": float(mean_absolute_error(observed, prediction)),
                "r2": float(r2_score(observed, prediction)),
                "spearman_rho": float(rho.statistic),
                "spearman_p": float(rho.pvalue),
            })
            absolute_errors[(dataset, model_name)] = mae
            output = frame[["id", "dataset", "mean_av_dice", "pixel_error"]].copy()
            output["model"] = model_name
            output["oof_predicted_pixel_error"] = prediction
            oof_parts.append(output)
            tuning_rows.extend({"dataset": dataset, "model": model_name, **item} for item in tuning)
    model_metrics = pd.DataFrame(model_rows)
    model_metrics.to_csv(args.output / "failure_model_metrics.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(args.output / "failure_model_oof.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(args.output / "failure_model_tuning.csv", index=False)

    comparisons = []
    for dataset in sorted(pixel.dataset.unique()):
        for challenger, baseline in (
            ("Q1_rejected_core", "Q0_reliable_core"),
            ("Q2_reverse_qc_triad", "Q0_reliable_core"),
            ("Q3_all_prespecified", "Q0_reliable_core"),
        ):
            result = bootstrap_difference(
                absolute_errors[(dataset, challenger)],
                absolute_errors[(dataset, baseline)],
                args.bootstrap_repetitions,
                rng,
            )
            comparisons.append({
                "dataset": dataset,
                "comparison": f"{challenger}_minus_{baseline}",
                "metric": "absolute_error",
                **result,
            })
    (args.output / "failure_model_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8",
    )

    correlation_rows = []
    for dataset, frame in pixel.groupby("dataset", sort=True):
        local = []
        for feature in reverse_qc:
            result = spearmanr(frame[feature], frame.pixel_error)
            local.append({
                "dataset": dataset,
                "feature": feature,
                "rho": float(result.statistic),
                "p": float(result.pvalue),
            })
        adjusted = holm([row["p"] for row in local])
        for row, value in zip(local, adjusted):
            row["holm_p"] = value
        correlation_rows.extend(local)
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(args.output / "reverse_qc_correlations.csv", index=False)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), constrained_layout=True)
    feature_labels = {
        "soft_vessel_mass_fraction": "Soft vessel mass",
        "mass_vessel_skeleton_length_density": "Mass skeleton density",
        "factorized_vessel_skeleton_length_density": "Gated skeleton density",
        "vessel_occupancy_disagreement": "Occupancy disagreement",
    }
    model_labels = {
        "Q0_reliable_core": "Reliable core",
        "Q1_rejected_core": "Rejected core",
        "Q2_reverse_qc_triad": "Reverse QC triad",
        "Q3_all_prespecified": "All 9 features",
    }
    qc_labels = {
        "vessel_occupancy_disagreement": "Occupancy disagreement",
        "base_to_mass_identity_disagreement": "Identity disagreement",
        "identity_entropy": "Identity entropy",
    }
    palette = ["#6D4C9A", "#C58A2B", "#148F8B", "#C85B53"]
    confusion = np.asarray(domain_metrics["confusion_matrix"], float)
    confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    image = axes[0, 0].imshow(confusion, vmin=0, vmax=1, cmap="Purples")
    axes[0, 0].set_xticks(range(len(encoder.classes_)), encoder.classes_, rotation=25, ha="right")
    axes[0, 0].set_yticks(range(len(encoder.classes_)), encoder.classes_)
    axes[0, 0].set_title(f"A  Domain separability, macro AUROC {domain_metrics['macro_ovr_auroc']:.3f}", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted imaging domain")
    axes[0, 0].set_ylabel("Observed imaging domain")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046)
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = confusion[row, column]
            axes[0, 0].text(
                column, row, f"{value:.2f}", ha="center", va="center",
                color="white" if value > 0.55 else "#222222", fontweight="bold",
            )

    positions = np.arange(len(encoder.classes_))
    for offset, feature in enumerate(reliable):
        values = []
        for dataset in encoder.classes_:
            group = phenotypes.loc[phenotypes.dataset.eq(dataset), feature].astype(float)
            values.append((group.median(), group.quantile(0.25), group.quantile(0.75)))
        med = np.array([item[0] for item in values])
        lo = med - np.array([item[1] for item in values])
        hi = np.array([item[2] for item in values]) - med
        axes[0, 1].errorbar(
            positions + (offset - 1.5) * 0.08, med, yerr=[lo, hi], marker="o",
            capsize=3, linestyle="none", color=palette[offset],
            label=feature_labels[feature],
        )
    axes[0, 1].set_xticks(positions, encoder.classes_, rotation=25, ha="right")
    axes[0, 1].set_title("B  Reliable scales remain domain dependent", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Phenotype value, median and IQR")
    axes[0, 1].legend(frameon=False, ncol=2, loc="upper center")

    pivot = model_metrics.pivot(index="model", columns="dataset", values="mae").reindex(feature_sets)
    width = 0.8 / max(len(pivot.columns), 1)
    for index, dataset in enumerate(pivot.columns):
        axes[1, 0].bar(
            np.arange(len(pivot)) + index * width, pivot[dataset], width=width,
            label=dataset, color=palette[index],
        )
    axes[1, 0].set_xticks(
        np.arange(len(pivot)) + width * (len(pivot.columns) - 1) / 2,
        [model_labels[name] for name in pivot.index], rotation=18, ha="right",
    )
    axes[1, 0].set_ylabel("OOF MAE for pixel error")
    axes[1, 0].set_title("C  Outcome blind prediction of pixel failure", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    for index, dataset in enumerate(sorted(correlations.dataset.unique())):
        group = correlations.loc[correlations.dataset.eq(dataset)]
        axes[1, 1].scatter(
            group.rho, np.arange(len(group)) + index * 0.15,
            label=dataset, s=60, color=palette[index],
        )
    axes[1, 1].axvline(0, color="#777777", linewidth=1)
    axes[1, 1].set_yticks(np.arange(len(reverse_qc)), [qc_labels[name] for name in reverse_qc])
    axes[1, 1].set_xlabel("Spearman correlation with pixel error")
    axes[1, 1].set_title("D  Prespecified QC signals versus pixel error", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, loc="lower right")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "RMDB–BiRAP reverse audit: a measurable phenotype can still be unsafe",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(args.output / "reverse_failure_backtracking.png", dpi=300)
    fig.savefig(args.output / "reverse_failure_backtracking.svg")
    plt.close(fig)

    summary = {
        "status": "complete",
        "protocol": config["protocol_version"],
        "domain_separability": domain_metrics,
        "failure_models": model_metrics.to_dict("records"),
        "paired_bootstrap": comparisons,
        "reverse_qc_correlations": correlations.to_dict("records"),
        "cimt_fixed_validation_or_test_used": False,
        "clinical_outcomes_used": False,
        "interpretation_boundary": (
            "Reverse variables are measurement-failure indicators. They are not promoted "
            "to biomarkers and are not selected using a systemic clinical outcome."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
