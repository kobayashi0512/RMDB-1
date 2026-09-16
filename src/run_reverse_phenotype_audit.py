#!/usr/bin/env python3
"""Outcome-blind reverse audit of frozen RMDB–BiRAP eye phenotypes.

The script never loads systemic outcomes. It partitions descriptive variation,
rechecks same-eye ICC, and asks whether prespecified readout-disagreement
measures track same-eye measurement instability or acquisition covariates.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OUTCOME_TOKENS = (
    "hypertension", "cimt", "vascular_disease", "diabetic_foot",
    "myocardial", "infarction", "outcome", "nephropathy", "neuropathy",
)


def load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    mapping = {
        "true": 1.0, "yes": 1.0, "y": 1.0, "1": 1.0,
        "false": 0.0, "no": 0.0, "n": 0.0, "0": 0.0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def icc_from_stats(
    count: np.ndarray, mean: np.ndarray, within: np.ndarray
) -> float:
    if len(count) < 2:
        return float("nan")
    grand = float(np.average(mean, weights=count))
    observations, eyes = float(count.sum()), float(len(count))
    ss_between = float(np.sum(count * np.square(mean - grand)))
    ss_within = float(within.sum())
    ms_between = ss_between / (eyes - 1)
    ms_within = ss_within / (observations - eyes)
    effective_k = (
        observations - float(np.sum(np.square(count))) / observations
    ) / (eyes - 1)
    denominator = ms_between + (effective_k - 1) * ms_within
    return float(
        (ms_between - ms_within) / denominator
        if denominator > 0 else float("nan")
    )


def icc_stats(frame: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = frame[["eye_key", feature]].dropna()
    groups = data.groupby("eye_key", sort=False)[feature]
    count = groups.size().to_numpy(dtype=float)
    mean = groups.mean().to_numpy(dtype=float)
    sum_sq = groups.apply(
        lambda x: float(np.sum(np.square(x.to_numpy(dtype=float))))
    ).to_numpy(dtype=float)
    within = sum_sq - count * np.square(mean)
    keep = count >= 2
    return count[keep], mean[keep], within[keep]


def icc_1_1(frame: pd.DataFrame, feature: str) -> tuple[float, int]:
    count, mean, within = icc_stats(frame, feature)
    return icc_from_stats(count, mean, within), int(len(count))


def bootstrap_icc(
    frame: pd.DataFrame,
    feature: str,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    count, mean, within = icc_stats(frame, feature)
    if len(count) < 2:
        return float("nan"), float("nan")
    samples: list[float] = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(count), len(count))
        estimate = icc_from_stats(
            count[selected], mean[selected], within[selected]
        )
        if np.isfinite(estimate):
            samples.append(estimate)
    if not samples:
        return float("nan"), float("nan")
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def pairwise_absolute_difference(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return float("nan")
    differences = [abs(left - right) for left, right in itertools.combinations(array, 2)]
    return float(np.mean(differences))


def nested_ss_partition(frame: pd.DataFrame, feature: str) -> dict[str, float]:
    """Descriptive nested sum-of-squares partition, not a causal variance model."""
    data = frame[["patient_id", "eye_key", feature]].dropna().copy()
    if len(data) < 3 or data[feature].nunique() < 2:
        return {
            "total_ss": float("nan"), "patient_share": float("nan"),
            "eye_within_patient_share": float("nan"),
            "view_within_eye_share": float("nan"),
        }
    values = data[feature].to_numpy(dtype=float)
    grand = float(values.mean())
    total_ss = float(np.sum(np.square(values - grand)))
    patient_mean = data.groupby("patient_id")[feature].transform("mean")
    eye_mean = data.groupby("eye_key")[feature].transform("mean")
    patient_ss = float(
        data.groupby("patient_id")[feature].size().mul(
            np.square(data.groupby("patient_id")[feature].mean() - grand)
        ).sum()
    )
    eye_table = data.groupby("eye_key", as_index=False).agg(
        patient_id=("patient_id", "first"), count=(feature, "size"),
        eye_mean=(feature, "mean"),
    )
    patient_lookup = data.groupby("patient_id")[feature].mean()
    eye_ss = float(np.sum(
        eye_table["count"].to_numpy(dtype=float)
        * np.square(
            eye_table["eye_mean"].to_numpy(dtype=float)
            - eye_table["patient_id"].map(patient_lookup).to_numpy(dtype=float)
        )
    ))
    view_ss = float(np.sum(np.square(
        data[feature].to_numpy(dtype=float) - eye_mean.to_numpy(dtype=float)
    )))
    if total_ss <= 0:
        return {
            "total_ss": total_ss, "patient_share": float("nan"),
            "eye_within_patient_share": float("nan"),
            "view_within_eye_share": float("nan"),
        }
    return {
        "total_ss": total_ss,
        "patient_share": patient_ss / total_ss,
        "eye_within_patient_share": eye_ss / total_ss,
        "view_within_eye_share": view_ss / total_ss,
        "partition_residual": (total_ss - patient_ss - eye_ss - view_ss) / total_ss,
        "n_images": int(len(data)),
        "n_patients": int(data["patient_id"].nunique()),
        "n_eyes": int(data["eye_key"].nunique()),
    }


def eta_squared(frame: pd.DataFrame, feature: str, covariate: str) -> dict[str, float]:
    data = frame[[feature, covariate]].dropna().copy()
    if len(data) < 3 or data[feature].nunique() < 2 or data[covariate].nunique() < 2:
        return {"eta_squared": float("nan"), "groups": int(data[covariate].nunique())}
    grand = float(data[feature].mean())
    total = float(np.sum(np.square(data[feature] - grand)))
    grouped = data.groupby(covariate)[feature]
    between = float(np.sum(
        grouped.size().to_numpy(dtype=float)
        * np.square(grouped.mean().to_numpy(dtype=float) - grand)
    ))
    return {
        "eta_squared": between / total if total > 0 else float("nan"),
        "groups": int(data[covariate].nunique()),
        "n": int(len(data)),
    }


def load_eye_tables(
    paths: list[Path],
    dataset_names: list[str],
    features: list[str],
) -> pd.DataFrame:
    frames = []
    required = ["id", "patient_id", "eye"] + features
    optional = [
        "disc_detected", "av_log_ratio_absolute_disagreement",
        "split", "dataset",
    ]
    for path, dataset_name in zip(paths, dataset_names):
        available = read_columns(path)
        missing = sorted(set(required) - set(available))
        if missing:
            raise KeyError(f"{path} is missing frozen columns: {missing}")
        use = required + [column for column in optional if column in available]
        frame = pd.read_csv(
            path, usecols=sorted(set(use)),
            dtype={"id": str, "patient_id": str, "eye": str},
        )
        frame["dataset"] = dataset_name
        frame["source_id"] = dataset_name + "__" + frame["id"].astype(str)
        frame["patient_id"] = dataset_name + "__" + frame["patient_id"].astype(str)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["eye"] = combined["eye"].astype(str).str.lower()
    combined["eye_key"] = combined["patient_id"] + "__" + combined["eye"]
    return combined


def merge_metadata(
    frame: pd.DataFrame,
    path: Path | None,
    allowed: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if path is None:
        return frame, []
    columns = read_columns(path)
    ignored_outcomes = [
        column for column in columns
        if any(token in column.lower() for token in OUTCOME_TOKENS)
    ]
    id_column = "id" if "id" in columns else "file" if "file" in columns else None
    if id_column is None:
        raise KeyError("Metadata requires an id or file column for label-blind merging.")
    selected = [id_column] + [column for column in allowed if column in columns and column != id_column]
    metadata = pd.read_csv(path, usecols=sorted(set(selected)), dtype={id_column: str})
    metadata = metadata.rename(columns={id_column: "id"})
    if metadata["id"].duplicated().any():
        metadata = metadata.drop_duplicates("id", keep="first")
    merged = frame.merge(metadata, on="id", how="left", suffixes=("", "_metadata"), validate="many_to_one")
    for column in ("disc_detected", "final_quality", "final_artifacts"):
        if column in merged:
            merged[column] = normalize_bool(merged[column])
    return merged, ignored_outcomes


def plot_results(
    variance: pd.DataFrame,
    repeatability: pd.DataFrame,
    nuisance: pd.DataFrame,
    disagreement: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    colors = ["#6E5AA6", "#149C95", "#D19A2A"]
    ordered = variance.sort_values("view_within_eye_share", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    bottom = np.zeros(len(ordered))
    for column, label, color in zip(
        ("patient_share", "eye_within_patient_share", "view_within_eye_share"),
        ("Patient", "Eye within patient", "View within eye"), colors,
    ):
        values = ordered[column].fillna(0).to_numpy(float)
        ax.barh(ordered["feature"], values, left=bottom, label=label, color=color)
        bottom += values
    ax.set_xlabel("Descriptive share of total sum of squares")
    ax.set_title("Where phenotype variation enters the frozen measurement chain")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "reverse_variance_partition.png", dpi=300)
    fig.savefig(output / "reverse_variance_partition.svg")
    plt.close(fig)

    nuisance_max = nuisance.groupby("feature")["eta_squared"].max().fillna(0).rename("max_nuisance_eta2")
    scatter = repeatability.merge(nuisance_max, on="feature", how="left")
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for reliable, group in scatter.groupby("retained_by_locked_rule"):
        ax.scatter(
            group["max_nuisance_eta2"], group["icc_1_1"], s=90,
            color="#149C95" if reliable else "#C65D57",
            label="Locked reliable core" if reliable else "Previously rejected",
            edgecolor="white", linewidth=.8,
        )
        for row in group.itertuples():
            ax.annotate(row.feature, (row.max_nuisance_eta2, row.icc_1_1),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(.60, color="#6E5AA6", linestyle="--", linewidth=1.5, label="ICC threshold")
    ax.set_xlabel("Largest outcome-blind acquisition association, η²")
    ax.set_ylabel("Same-eye ICC(1,1)")
    ax.set_title("Repeatability and acquisition dependence are separate properties")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "reverse_reliability_nuisance_map.png", dpi=300)
    fig.savefig(output / "reverse_reliability_nuisance_map.svg")
    plt.close(fig)

    if len(disagreement):
        shown = disagreement.loc[disagreement["predictor"].eq(
            "vessel_occupancy_disagreement"
        )].sort_values("spearman_rho")
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.barh(shown["feature"], shown["spearman_rho"], color="#149C95")
        ax.axvline(0, color="#223042", linewidth=1)
        ax.set_xlabel("Spearman correlation with absolute same-eye deviation")
        ax.set_title("Does dual-readout disagreement identify unstable measurements?")
        fig.tight_layout()
        fig.savefig(output / "reverse_disagreement_instability.png", dpi=300)
        fig.savefig(output / "reverse_disagreement_instability.svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-eye", type=Path, action="append", required=True)
    parser.add_argument("--dataset-name", action="append", required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--per-patient", type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/rmdb/reverse_audit_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if len(args.per_eye) != len(args.dataset_name):
        raise ValueError("Repeat --dataset-name once for each --per-eye table.")

    config = load_config(args.config)
    features = list(config["prespecified_eye_phenotypes"])
    reliable = set(config["previously_frozen_reliable_core"])
    frame = load_eye_tables(args.per_eye, args.dataset_name, features)
    frame, ignored_outcomes = merge_metadata(
        frame, args.metadata,
        list(config["allowed_outcome_blind_audit_covariates"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    safe_columns = [
        column for column in frame.columns
        if not any(token in column.lower() for token in OUTCOME_TOKENS)
    ]
    frame[safe_columns].to_csv(args.output / "outcome_blind_audit_table.csv", index=False)

    rng = np.random.default_rng(int(config["seed"]))
    repeat_rows, variance_rows = [], []
    for feature in features:
        estimate, repeat_eyes = icc_1_1(frame, feature)
        low, high = bootstrap_icc(frame, feature, args.bootstrap_repetitions, rng)
        grouped = frame.groupby("eye_key")[feature]
        within_std = grouped.std(ddof=0)
        pair_delta = grouped.apply(pairwise_absolute_difference)
        repeat_rows.append({
            "feature": feature, "icc_1_1": estimate,
            "ci95_low": low, "ci95_high": high,
            "n_repeat_eyes": repeat_eyes,
            "mean_within_eye_std": float(within_std.dropna().mean()),
            "median_pairwise_absolute_difference": float(pair_delta.dropna().median()),
            "retained_by_locked_rule": feature in reliable,
            "passes_current_icc_threshold": bool(
                np.isfinite(estimate) and estimate >= float(config["icc_threshold"])
            ),
        })
        variance_rows.append({"feature": feature, **nested_ss_partition(frame, feature)})
    repeatability = pd.DataFrame(repeat_rows).sort_values("icc_1_1", ascending=False)
    variance = pd.DataFrame(variance_rows)
    repeatability.to_csv(args.output / "repeatability_recheck.csv", index=False)
    variance.to_csv(args.output / "nested_ss_variance_partition.csv", index=False)

    nuisance_rows = []
    nuisance_covariates = [
        column for column in ("dataset", "disc_detected", "final_quality", "final_artifacts", "final_icdr")
        if column in frame.columns
    ]
    for feature in features:
        for covariate in nuisance_covariates:
            nuisance_rows.append({
                "feature": feature, "covariate": covariate,
                **eta_squared(frame, feature, covariate),
            })
    nuisance = pd.DataFrame(nuisance_rows)
    nuisance.to_csv(args.output / "outcome_blind_nuisance_associations.csv", index=False)

    disagreement_predictors = [
        column for column in (
            "vessel_occupancy_disagreement",
            "base_to_mass_identity_disagreement",
            "av_log_ratio_absolute_disagreement",
        ) if column in frame.columns
    ]
    disagreement_rows = []
    for feature in features:
        deviation = (frame[feature] - frame.groupby("eye_key")[feature].transform("mean")).abs()
        for predictor in disagreement_predictors:
            data = pd.DataFrame({"deviation": deviation, "predictor": frame[predictor]}).dropna()
            if len(data) < 3 or data["deviation"].nunique() < 2 or data["predictor"].nunique() < 2:
                rho, pvalue = float("nan"), float("nan")
            else:
                result = spearmanr(data["predictor"], data["deviation"])
                rho, pvalue = float(result.statistic), float(result.pvalue)
            disagreement_rows.append({
                "feature": feature, "predictor": predictor,
                "spearman_rho": rho, "pvalue_descriptive": pvalue,
                "n": int(len(data)),
            })
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(args.output / "disagreement_vs_same_eye_instability.csv", index=False)

    coverage_rows = []
    if args.per_patient is not None:
        headers = read_columns(args.per_patient)
        selected = [
            column for column in config["prespecified_reverse_patient_features"]
            if column in headers
        ]
        patients = pd.read_csv(args.per_patient, usecols=["patient_id"] + selected,
                               dtype={"patient_id": str})
        for feature in config["prespecified_reverse_patient_features"]:
            coverage = float(patients[feature].notna().mean()) if feature in patients else 0.0
            coverage_rows.append({
                "feature": feature, "coverage": coverage,
                "passes_locked_coverage": coverage >= float(
                    config["minimum_reverse_feature_coverage"]
                ),
            })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "reverse_patient_feature_coverage.csv", index=False)

    core_audit = repeatability[[
        "feature", "icc_1_1", "ci95_low", "ci95_high",
        "retained_by_locked_rule", "passes_current_icc_threshold",
    ]].copy()
    core_audit["locked_rule_consistent"] = (
        core_audit["retained_by_locked_rule"]
        == core_audit["passes_current_icc_threshold"]
    )
    core_audit.to_csv(args.output / "locked_core_consistency.csv", index=False)

    plot_results(variance, repeatability, nuisance, disagreement, args.output)
    metadata = {
        "protocol": "RMDB_BiRAP_forward_reverse_locked_protocol_20260821.md",
        "config": str(args.config), "outcome_blind": True,
        "cimt_fixed_validation_or_test_read": False,
        "datasets": args.dataset_name,
        "images": int(len(frame)), "patients": int(frame["patient_id"].nunique()),
        "eyes": int(frame["eye_key"].nunique()),
        "ignored_outcome_columns_from_metadata": ignored_outcomes,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "descriptive_variance_warning": (
            "Nested sum-of-squares shares are descriptive and do not identify biological causes."
        ),
        "locked_core_fully_consistent": bool(core_audit["locked_rule_consistent"].all()),
        "patient_reverse_features_ready": (
            bool(len(coverage)) and bool(coverage["passes_locked_coverage"].any())
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
