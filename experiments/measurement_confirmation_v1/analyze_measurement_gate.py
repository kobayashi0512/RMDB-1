#!/usr/bin/env python3
"""Run the locked, patient-clustered RMDB measurement Gate on a new cohort.

Input values must be long-format CSV: image_id,arm,feature,value.  This script is
deliberately model-agnostic: it analyses frozen outputs, not model selection.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


PRIMARY = ("branch_density", "endpoint_density", "occupancy_entropy")
RAW, JOINT = "raw_vascx", "joint"


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], probability: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    index = (len(values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def metric_summary(records: list[dict[str, Any]], arm: str) -> dict[str, float | None]:
    """ICC(1,1), MAD, SEM and Bland--Altman values for two captures per eye-session."""
    groups = [[record[arm][0], record[arm][1]] for record in records]
    number_groups = len(groups)
    observations = [item for group in groups for item in group]
    if number_groups < 2 or len(observations) <= number_groups:
        return {name: None for name in ("icc_1_1", "within_eye_mad", "sem", "bias", "loa_low", "loa_high", "loa_halfwidth")}
    grand = fmean(observations)
    means = [fmean(group) for group in groups]
    ms_within = sum((value - mean) ** 2 for group, mean in zip(groups, means) for value in group) / (len(observations) - number_groups)
    ms_between = sum(len(group) * (mean - grand) ** 2 for group, mean in zip(groups, means)) / (number_groups - 1)
    effective_k = (len(observations) - sum(len(group) ** 2 for group in groups) / len(observations)) / (number_groups - 1)
    denominator = ms_between + (effective_k - 1.0) * ms_within
    icc = None if denominator == 0 else (ms_between - ms_within) / denominator
    mads = [fmean([abs(value - mean) for value in group]) for group, mean in zip(groups, means)]
    differences = [group[1] - group[0] for group in groups]
    bias = fmean(differences)
    if len(differences) > 1:
        standard_deviation = math.sqrt(sum((item - bias) ** 2 for item in differences) / (len(differences) - 1))
    else:
        standard_deviation = float("nan")
    halfwidth = 1.96 * standard_deviation
    return {
        "icc_1_1": icc,
        "within_eye_mad": fmean(mads),
        "sem": math.sqrt(ms_within),
        "bias": bias,
        "loa_low": bias - halfwidth,
        "loa_high": bias + halfwidth,
        "loa_halfwidth": halfwidth,
    }


def patient_bootstrap(records: list[dict[str, Any]], repeats: int, seed: int, alpha: float) -> dict[str, list[float | None]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clusters[record["patient_id"]].append(record)
    patients = sorted(clusters)
    rng = random.Random(seed)
    outputs: dict[str, list[float]] = defaultdict(list)
    for _ in range(repeats):
        sample: list[dict[str, Any]] = []
        for _ in patients:
            sample.extend(clusters[rng.choice(patients)])
        raw = metric_summary(sample, RAW)
        joint = metric_summary(sample, JOINT)
        values = {
            "delta_icc": None if raw["icc_1_1"] is None or joint["icc_1_1"] is None else joint["icc_1_1"] - raw["icc_1_1"],
            "mad_ratio": None if not raw["within_eye_mad"] else joint["within_eye_mad"] / raw["within_eye_mad"],
            "sem_ratio": None if not raw["sem"] else joint["sem"] / raw["sem"],
            "loa_halfwidth_ratio": None if not raw["loa_halfwidth"] else joint["loa_halfwidth"] / raw["loa_halfwidth"],
        }
        for name, value in values.items():
            if value is not None and math.isfinite(value):
                outputs[name].append(value)
    result: dict[str, list[float | None]] = {}
    for name in ("delta_icc", "mad_ratio", "sem_ratio", "loa_halfwidth_ratio"):
        result[name] = [percentile(outputs[name], alpha / 2.0), percentile(outputs[name], 1.0 - alpha / 2.0)]
    return result


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["patient_id"], row["eye"].lower(), row["session_id"], row["camera_configuration"]


def sort_capture(row: dict[str, str]) -> tuple[str, str, str]:
    return row["capture_time"], row["acquisition_id"], row["image_id"]


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "patient_id", "eye", "session_id", "camera_configuration", "capture_time", "acquisition_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Confirmation manifest must contain: {', '.join(sorted(required))}")
    return {row["image_id"]: row for row in rows}


def load_values(path: Path, image_ids: set[str]) -> dict[tuple[str, str, str], float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "arm", "feature", "value"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Feature CSV must contain: {', '.join(sorted(required))}")
        values: dict[tuple[str, str, str], float] = {}
        for row in reader:
            if row["image_id"] not in image_ids:
                raise ValueError(f"Feature value refers to image outside locked manifest: {row['image_id']}")
            number = finite(row["value"])
            if number is None:
                continue
            identity = row["image_id"], row["arm"], row["feature"]
            if identity in values:
                raise ValueError(f"Duplicate feature value: {identity}")
            values[identity] = number
    return values


def within_config_records(manifest: dict[str, dict[str, str]], values: dict[tuple[str, str, str], float], feature: str, config: str, arms: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest.values():
        if row["camera_configuration"] == config:
            grouped[row_key(row)].append(row)
    records: list[dict[str, Any]] = []
    for group_key, rows in grouped.items():
        rows = sorted(rows, key=sort_capture)
        # Select the first two native captures deterministically.  A third capture remains
        # in the locked data and is available for a planned sensitivity analysis, not tuning.
        rows = rows[:2]
        if len(rows) != 2:
            continue
        paired: dict[str, tuple[float, float]] = {}
        for arm in arms:
            current = [values.get((row["image_id"], arm, feature)) for row in rows]
            if any(value is None for value in current):
                break
            paired[arm] = (current[0], current[1])  # type: ignore[arg-type]
        if len(paired) == len(arms):
            records.append({"patient_id": group_key[0], "eye_session_key": "|".join(group_key[:3]), **paired})
    return records


def cross_camera_records(manifest: dict[str, dict[str, str]], values: dict[tuple[str, str, str], float], feature: str, left: str, right: str, arms: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest.values():
        grouped[row["patient_id"], row["eye"].lower(), row["session_id"]].append(row)
    records: list[dict[str, Any]] = []
    for group_key, rows in grouped.items():
        pairs: dict[str, tuple[float, float]] = {}
        for arm in arms:
            per_config: list[float] = []
            for config in (left, right):
                items = [values.get((row["image_id"], arm, feature)) for row in rows if row["camera_configuration"] == config]
                items = [value for value in items if value is not None]
                if not items:
                    break
                per_config.append(fmean(items))
            if len(per_config) == 2:
                pairs[arm] = (per_config[0], per_config[1])
        if len(pairs) == len(arms):
            records.append({"patient_id": group_key[0], "eye_session_key": "|".join(group_key), **pairs})
    return records


def contrast_point(records: list[dict[str, Any]]) -> dict[str, float | None]:
    raw, joint = metric_summary(records, RAW), metric_summary(records, JOINT)
    return {
        "delta_icc": None if raw["icc_1_1"] is None or joint["icc_1_1"] is None else joint["icc_1_1"] - raw["icc_1_1"],
        "mad_ratio": None if not raw["within_eye_mad"] else joint["within_eye_mad"] / raw["within_eye_mad"],
        "sem_ratio": None if not raw["sem"] else joint["sem"] / raw["sem"],
        "loa_halfwidth_ratio": None if not raw["loa_halfwidth"] else joint["loa_halfwidth"] / raw["loa_halfwidth"],
    }


def label_gate(point: dict[str, float | None], interval: dict[str, list[float | None]]) -> tuple[str, dict[str, bool | None]]:
    checks: dict[str, bool | None] = {
        "icc_improved": None if interval["delta_icc"][0] is None else interval["delta_icc"][0] > 0,
        "mad_reduced": None if interval["mad_ratio"][1] is None else interval["mad_ratio"][1] < 1,
        "sem_reduced": None if interval["sem_ratio"][1] is None else interval["sem_ratio"][1] < 1,
    }
    values = list(checks.values())
    if any(value is None for value in values):
        return "insufficient_data", checks
    if all(values):
        return "concordant_relative_improvement", checks
    if any(values):
        return "mixed_tradeoff", checks
    return "not_improved", checks


def serialise_metric(values: dict[str, float | None]) -> dict[str, float | None]:
    return {key: (None if value is None or not math.isfinite(value) else value) for key, value in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Locked confirmation_manifest.csv")
    parser.add_argument("--feature-values", required=True, help="Long-format feature values CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--features", nargs="+", default=list(PRIMARY), help="Features to analyse; default is the three primary endpoints")
    parser.add_argument("--arms", nargs="+", default=[RAW, "mass", JOINT, "smooth", "cleanup"], help="All locked arms to report on common finite rows")
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap repetitions; protocol default is 10000")
    if RAW not in args.arms or JOINT not in args.arms:
        raise ValueError(f"--arms must include {RAW} and {JOINT} for the primary contrast")
    if len(args.arms) != len(set(args.arms)):
        raise ValueError("--arms contains a duplicate")
    manifest = load_manifest(Path(args.manifest))
    values = load_values(Path(args.feature_values), set(manifest))
    configurations = sorted({row["camera_configuration"] for row in manifest.values()})
    alpha = 0.05 / len(PRIMARY) if set(args.features) == set(PRIMARY) else 0.05
    within: list[dict[str, Any]] = []
    for feature in args.features:
        for config_index, config in enumerate(configurations):
            records = within_config_records(manifest, values, feature, config, args.arms)
            point = contrast_point(records)
            interval = patient_bootstrap(records, args.bootstrap, args.seed + config_index, alpha) if records else {name: [None, None] for name in ("delta_icc", "mad_ratio", "sem_ratio", "loa_halfwidth_ratio")}
            label, checks = label_gate(point, interval)
            within.append({
                "feature": feature, "camera_configuration": config, "eye_sessions": len(records), "participants": len({item['patient_id'] for item in records}),
                "arm_metrics": {arm: serialise_metric(metric_summary(records, arm)) for arm in args.arms},
                "raw_vascx": serialise_metric(metric_summary(records, RAW)), "joint": serialise_metric(metric_summary(records, JOINT)),
                "joint_minus_raw_vascx": serialise_metric(point), "bootstrap_interval": interval,
                "gate_label": label, "gate_checks": checks,
            })
    cross: list[dict[str, Any]] = []
    for feature in args.features:
        for pair_index, (left, right) in enumerate(itertools.combinations(configurations, 2)):
            records = cross_camera_records(manifest, values, feature, left, right, args.arms)
            point = contrast_point(records)
            interval = patient_bootstrap(records, args.bootstrap, args.seed + 1000 + pair_index, alpha) if records else {name: [None, None] for name in ("delta_icc", "mad_ratio", "sem_ratio", "loa_halfwidth_ratio")}
            cross.append({
                "feature": feature, "left_camera_configuration": left, "right_camera_configuration": right,
                "eye_sessions": len(records), "participants": len({item['patient_id'] for item in records}),
                "arm_metrics": {arm: serialise_metric(metric_summary(records, arm)) for arm in args.arms},
                "raw_vascx": serialise_metric(metric_summary(records, RAW)), "joint": serialise_metric(metric_summary(records, JOINT)),
                "joint_minus_raw_vascx": serialise_metric(point), "bootstrap_interval": interval,
                "interpretation": "descriptive device/resolution sensitivity; not a within-camera Gate",
            })
    report = {
        "analysis_id": "RMDB-measurement-gate-v1",
        "manifest": str(Path(args.manifest).resolve()),
        "feature_values": str(Path(args.feature_values).resolve()),
        "features": args.features,
        "arms": args.arms,
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "interval": f"{(1 - alpha) * 100:.3f}% central percentile interval",
        "within_camera": within,
        "cross_camera": cross,
        "interpretation_boundary": "ICC is never interpreted alone. Relative Gate labels are not clinical interchangeability claims; report MAD, SEM, bias and LoA with every label.",
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
