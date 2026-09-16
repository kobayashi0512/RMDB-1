#!/usr/bin/env python3
"""Create one immutable, pre-writing evidence ledger for the RMDB project.

This program does not run models, tune an operator, or alter source results.  It
only collates the already frozen historical controls and the two audited HRF
experiments into a source-hashed results table, a data/split registry, and an
explicit claim boundary.  Its purpose is to prevent a later manuscript from
mixing denominators, data roles, or VascX-level claims.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/prewriting_evidence_ledger_20260909_v1"
HISTORICAL = ROOT / "experiments/measurement_confirmation_v1/frozen_historical_evidence_v1"
CONTROLS = ROOT / "artifacts/final_measurement_controls"
HRF_QUALITY = ROOT / "artifacts/hrf_iqa_readout_stress_test_v1"
HRF_RESOLUTION = ROOT / "artifacts/hrf_iqa_resolution_sensitivity_v1"
HRF_SNAPSHOT = ROOT / "artifacts/frozen_external_experiments_20260909_v1"
PRIMARY = ("branch_density", "endpoint_density", "occupancy_entropy")
# The historical control predates the confirmation gate and therefore has no
# SEM/LoA.  Do not silently fabricate them; the new HRF gate carries those.
REPEAT_METRICS = ("icc", "within_eye_mad", "mean_pairwise_abs_diff", "within_eye_sd")
GATE_METRICS = (
    ("icc", "icc_1_1", "delta_icc", "difference"),
    ("within_eye_mad", "within_eye_mad", "mad_ratio", "ratio"),
    ("sem", "sem", "sem_ratio", "ratio"),
    ("loa_halfwidth", "loa_halfwidth", "loa_halfwidth_ratio", "ratio"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def check_source_freezes() -> dict[str, str]:
    historical_manifest = load_json(HISTORICAL / "snapshot_manifest.json")
    if historical_manifest.get("snapshot_id") != "RMDB-historical-evidence-snapshot-20260909-v1":
        raise ValueError("Unexpected historical snapshot ID")
    for record in historical_manifest["files"].values():
        path = HISTORICAL / record["filename"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Historical snapshot hash mismatch: {record['filename']}")

    hrf_manifest = load_json(HRF_SNAPSHOT / "snapshot_manifest.json")
    if hrf_manifest.get("snapshot_id") != "RMDB-external-HRF-experiments-20260909-v1":
        raise ValueError("Unexpected HRF snapshot ID")
    for name, record in hrf_manifest["records"].items():
        path = HRF_SNAPSHOT / name
        if sha256(path) != record["sha256"]:
            raise ValueError(f"HRF snapshot hash mismatch: {name}")

    for path in (CONTROLS / "audit.json", HRF_QUALITY / "audit.json", HRF_RESOLUTION / "audit.json"):
        audit = load_json(path)
        if not audit.get("passed", audit.get("complete", False)):
            raise ValueError(f"Source audit did not pass: {path}")
    return {
        "historical_snapshot_manifest": sha256(HISTORICAL / "snapshot_manifest.json"),
        "hrf_snapshot_manifest": sha256(HRF_SNAPSHOT / "snapshot_manifest.json"),
        "historical_controls_audit": sha256(CONTROLS / "audit.json"),
        "hrf_quality_audit": sha256(HRF_QUALITY / "audit.json"),
        "hrf_resolution_audit": sha256(HRF_RESOLUTION / "audit.json"),
    }


def historical_repeat_rows() -> list[dict[str, object]]:
    estimates = pd.read_csv(CONTROLS / "repeat_estimates.csv")
    contrasts = pd.read_csv(CONTROLS / "repeat_contrasts.csv")
    rows: list[dict[str, object]] = []
    for feature in PRIMARY:
        for metric in REPEAT_METRICS:
            select = (
                (estimates["split"] == "test") & (estimates["feature"] == feature)
                & (estimates["metric"] == metric) & estimates["arm"].isin(["raw", "joint"])
            )
            subset = estimates.loc[select].set_index("arm")
            if set(subset.index) != {"raw", "joint"}:
                raise ValueError(f"Missing historical estimate: {feature}/{metric}")
            contrast = contrasts.loc[
                (contrasts["split"] == "test") & (contrasts["feature"] == feature)
                & (contrasts["contrast"] == "joint_minus_raw") & (contrasts["metric"] == metric)
            ]
            if len(contrast) != 1:
                raise ValueError(f"Missing historical contrast: {feature}/{metric}")
            c = contrast.iloc[0]
            raw, joint = subset.loc["raw"], subset.loc["joint"]
            rows.append({
                "experiment_id": "historical_mbrset_test_repeatability",
                "evidence_tier": "historical_exploratory_control",
                "analysis_role": "same-eye repeatability under public 224px inputs; not high-resolution evidence",
                "cohort": "mBRSET test",
                "pairing_or_reference": "475 repeated eyes; patient-cluster bootstrap across 255 patients",
                "camera_or_input": "public 224x224 mirror",
                "comparison": "joint_readout_minus_raw_vascx",
                "endpoint": feature,
                "statistic": metric,
                "effect_representation": "difference",
                "raw_vascx_value": float(raw.estimate),
                "joint_readout_value": float(joint.estimate),
                "effect_value": float(c.delta),
                "effect_ci_low": float(c.ci_low),
                "effect_ci_high": float(c.ci_high),
                "n_images": 950,
                "n_eyes_or_pairs": int(c.eyes),
                "n_patients_or_clusters": int(c.patients),
                "gate_label": "historical_descriptive_only",
                "allowed_interpretation": "Historical within-source readout contrast only; do not pool with HRF.",
            })
    return rows


def historical_reference_rows() -> list[dict[str, object]]:
    estimates = pd.read_csv(CONTROLS / "reference_estimates.csv")
    contrasts = pd.read_csv(CONTROLS / "reference_contrasts.csv")
    endpoints = ("vessel_dice", "thin_centerline_recall", "cldice", "abs_error_branch_density", "abs_error_endpoint_density")
    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        contrast = contrasts.loc[(contrasts["metric"] == endpoint) & (contrasts["contrast"] == "joint_minus_raw")]
        if len(contrast) != 1:
            raise ValueError(f"Missing reference contrast: {endpoint}")
        subset = estimates.loc[(estimates["metric"] == endpoint) & estimates["arm"].isin(["raw", "joint"])].set_index("arm")
        if set(subset.index) != {"raw", "joint"}:
            raise ValueError(f"Missing reference estimates: {endpoint}")
        c = contrast.iloc[0]
        rows.append({
            "experiment_id": "historical_fundus_avseg_reference",
            "evidence_tier": "historical_exploratory_control",
            "analysis_role": "labelled structural reference comparison of readouts applied on VascX",
            "cohort": "Fundus-AVSeg reference",
            "pairing_or_reference": "100 manually labelled images",
            "camera_or_input": "dataset-specific images; no camera transportability claim",
            "comparison": "joint_readout_minus_raw_vascx",
            "endpoint": endpoint,
            "statistic": "bootstrap_mean_contrast",
            "effect_representation": "difference",
            "raw_vascx_value": float(subset.loc["raw", "estimate"]),
            "joint_readout_value": float(subset.loc["joint", "estimate"]),
            "effect_value": float(c.delta),
            "effect_ci_low": float(c.ci_low),
            "effect_ci_high": float(c.ci_high),
            "n_images": int(c.n),
            "n_eyes_or_pairs": "",
            "n_patients_or_clusters": "",
            "gate_label": "structural_tradeoff_not_segmentation_replacement",
            "allowed_interpretation": "Joint readout is assessed on top of VascX; this cannot support a claim of replacing or improving VascX segmentation.",
        })
    return rows


def gate_rows(path: Path, experiment_id: str, tier: str, role: str, input_label: str) -> list[dict[str, object]]:
    report = load_json(path)
    if tuple(report.get("features", ())) != PRIMARY:
        raise ValueError(f"Unexpected primary endpoint order: {path}")
    rows: list[dict[str, object]] = []
    for entry in report["within_camera"]:
        feature = entry["feature"]
        for statistic, value_key, effect_key, representation in GATE_METRICS:
            interval = entry["bootstrap_interval"][effect_key]
            rows.append({
                "experiment_id": experiment_id,
                "evidence_tier": tier,
                "analysis_role": role,
                "cohort": "HRF IQA",
                "pairing_or_reference": "published same-eye pair" if "quality" in experiment_id else "same source image at two digital resolutions",
                "camera_or_input": entry["camera_configuration"] + "; " + input_label,
                "comparison": "joint_readout_vs_raw_vascx",
                "endpoint": feature,
                "statistic": statistic,
                "effect_representation": representation,
                "raw_vascx_value": float(entry["raw_vascx"][value_key]),
                "joint_readout_value": float(entry["joint"][value_key]),
                "effect_value": float(entry["joint_minus_raw_vascx"][effect_key]),
                "effect_ci_low": float(interval[0]),
                "effect_ci_high": float(interval[1]),
                "n_images": 36,
                "n_eyes_or_pairs": int(entry["eye_sessions"]),
                "n_patients_or_clusters": int(entry["participants"]),
                "gate_label": entry["gate_label"],
                "allowed_interpretation": report["analysis_role"] + "; not a clinical interchangeability claim.",
            })
    return rows


def split_rows() -> list[dict[str, object]]:
    return [
        {
            "data_role_id": "mbrset_development_historical",
            "dataset": "mBRSET",
            "role": "historical development control",
            "images": 978, "patients": 257, "eyes": 505, "paired_eyes": 472,
            "data_relationship": "patient-level development partition; historical only",
            "independence": "separate from frozen mBRSET test partition",
            "eligible_for_cross_camera_claim": "no",
            "eligible_for_high_resolution_claim": "no; public 224x224 mirror",
            "pooling_rule": "never pool with HRF experiments",
        },
        {
            "data_role_id": "mbrset_test_historical",
            "dataset": "mBRSET",
            "role": "historical test repeatability control",
            "images": 982, "patients": 258, "eyes": 507, "paired_eyes": 475,
            "data_relationship": "test partition; 255 patients contribute repeated eyes",
            "independence": "separate from development at patient level",
            "eligible_for_cross_camera_claim": "no",
            "eligible_for_high_resolution_claim": "no; public 224x224 mirror",
            "pooling_rule": "never pool with HRF experiments",
        },
        {
            "data_role_id": "fundus_avseg_historical_reference",
            "dataset": "Fundus-AVSeg",
            "role": "historical labelled structural reference",
            "images": 100, "patients": "not released", "eyes": "not released", "paired_eyes": "",
            "data_relationship": "manual vessel labels; VascX-associated benchmark context",
            "independence": "not external to VascX for a VascX replacement claim",
            "eligible_for_cross_camera_claim": "no",
            "eligible_for_high_resolution_claim": "no",
            "pooling_rule": "structural reference only; never pool with repeatability or HRF",
        },
        {
            "data_role_id": "hrf_iqa_quality_stress_fresh",
            "dataset": "HRF IQA allQuality",
            "role": "fresh high-resolution acquisition-quality stress test",
            "images": 36, "patients": 18, "eyes": 18, "paired_eyes": 18,
            "data_relationship": "18 published bad/good same-eye pairs; Canon CR-1; 36 native images",
            "independence": "not pooled with historical controls; resolution experiment shares its good-capture source images",
            "eligible_for_cross_camera_claim": "no; one camera",
            "eligible_for_high_resolution_claim": "yes; native 3888x2592 or 5184x3456",
            "pooling_rule": "report as a quality perturbation stress test, not test-retest",
        },
        {
            "data_role_id": "hrf_iqa_digital_resolution_sensitivity",
            "dataset": "HRF IQA allQuality",
            "role": "controlled digital-resolution sensitivity",
            "images": 54, "patients": 18, "eyes": 18, "paired_eyes": 18,
            "data_relationship": "same 18 HRF good captures at native, 75%, and 50% pixels",
            "independence": "not independent from HRF quality stress; derived images share the good-capture source",
            "eligible_for_cross_camera_claim": "no; digital resampling is not a camera change",
            "eligible_for_high_resolution_claim": "partly; probes input-pixel sensitivity only",
            "pooling_rule": "keep native-vs-75% and native-vs-50% contrasts separate",
        },
    ]


def claim_rows() -> list[dict[str, object]]:
    return [
        {
            "claim_id": "C1",
            "claim": "The readout layer separates or exposes endpoint-specific measurement trade-offs on top of VascX.",
            "support": "Historical structural and mBRSET controls plus two HRF stress/sensitivity analyses.",
            "support_level": "supported with bounded scope",
            "allowed_wording": "separates/exposes the trade-off",
            "prohibited_wording": "resolves the trade-off; replaces VascX; universally improves measurement",
        },
        {
            "claim_id": "C2",
            "claim": "Under high-resolution same-camera quality perturbation, endpoint-density agreement improves for joint versus raw VascX.",
            "support": "HRF IQA quality-stress gate; ICC/MAD/SEM/LoA all favour joint for endpoint_density.",
            "support_level": "supported only as quality-perturbation sensitivity",
            "allowed_wording": "improves relative agreement for this endpoint under this stress test",
            "prohibited_wording": "clinical test-retest reliability; cross-camera robustness",
        },
        {
            "claim_id": "C3",
            "claim": "The joint readout is not uniformly robust: entropy worsens in both high-resolution probes.",
            "support": "Both HRF gate reports have occupancy_entropy=not_improved.",
            "support_level": "supported",
            "allowed_wording": "not uniformly improved; endpoint-dependent",
            "prohibited_wording": "all readouts benefit",
        },
        {
            "claim_id": "C4",
            "claim": "No cross-camera or clinical interchangeability conclusion is currently supported.",
            "support": "All available confirmation data lack a same-eye cross-camera design; HRF has one camera and digital downsampling is synthetic.",
            "support_level": "explicit limitation",
            "allowed_wording": "unresolved external-validation gap",
            "prohibited_wording": "cross-camera validation; clinical deployability",
        },
        {
            "claim_id": "C5",
            "claim": "The contribution is a measurement/readout study using VascX as the frozen segmentation base, not a VascX replacement.",
            "support": "All five-arm controls share frozen VascX probabilities; structural-reference contrasts show trade-offs.",
            "support_level": "supported by design and evidence",
            "allowed_wording": "increment beyond VascX lies in readout selection and quantified trade-offs",
            "prohibited_wording": "outperforms/replaces VascX segmentation",
        },
    ]


def increment_note() -> str:
    return """# VascX 增量定位（实验前固定）

## 比较对象

所有表中 `raw_vascx` 都是冻结的 VascX A/V 概率输出。`mass`、`joint`、`smooth`、`cleanup` 是其上游概率不变时的读出层/控制臂；它们没有重新训练或替换 VascX。

## 已经能说什么

1. 在历史 mBRSET 测试重复眼中，joint 相对 raw VascX 的三个预先指定读出（branch density、endpoint density、occupancy entropy）均有较高 ICC 和较小 MAD。这是历史、224px 同源重复性证据。
2. 在 100 张带标注的 Fundus-AVSeg 控制中，joint 的 vessel Dice 相对 raw VascX 下降；这排除了“joint 改善/替代 VascX 分割”的表述，并直接显示结构保真与测量读出存在取舍。
3. 在新完成的 HRF 同眼高分辨率质量压力测试中，endpoint density 是一致相对改善，branch density 为 mixed tradeoff，而 occupancy entropy 为 not improved。
4. 在同一 HRF good-capture 的数字降采样实验中，没有出现所有终点一致改善；尤其 occupancy entropy 两个比较均为 not improved。

## 结论边界

唯一可用的核心结论是：**在冻结 VascX 分割上，候选 readout 会分离并暴露不同测量终点的 trade-off。**

不得使用 “Resolves”、 “replaces VascX”、 “improves VascX segmentation”、 “cross-camera validated” 或 “clinically interchangeable”。跨相机、真实高分辨率重复采集以及临床可互换性仍未得到证据支持。
"""


def audit(rows: list[dict[str, object]], splits: list[dict[str, object]], source_hashes: dict[str, str]) -> dict[str, object]:
    expected_ids = {
        "historical_mbrset_test_repeatability", "historical_fundus_avseg_reference",
        "hrf_quality_stress", "hrf_resolution_native_vs_75pct", "hrf_resolution_native_vs_50pct",
    }
    actual_ids = {str(row["experiment_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise ValueError(f"Unexpected experiment IDs: {actual_ids}")
    if len(rows) != 53:
        raise ValueError(f"Unexpected result row count: {len(rows)}")
    for row in rows:
        for key in ("raw_vascx_value", "joint_readout_value", "effect_value", "effect_ci_low", "effect_ci_high"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite ledger value: {row['experiment_id']}/{row['endpoint']}/{key}")
        if float(row["effect_ci_low"]) > float(row["effect_ci_high"]):
            raise ValueError("Invalid confidence interval")
    for exp in ("hrf_quality_stress", "hrf_resolution_native_vs_75pct", "hrf_resolution_native_vs_50pct"):
        endpoint_labels = {
            (row["endpoint"], row["gate_label"])
            for row in rows if row["experiment_id"] == exp and row["statistic"] == "icc"
        }
        if {x[0] for x in endpoint_labels} != set(PRIMARY):
            raise ValueError(f"Missing gate endpoint: {exp}")
    if any("cross-camera" in str(row["allowed_interpretation"]).lower() and "not" not in str(row["allowed_interpretation"]).lower() for row in rows):
        raise ValueError("A ledger result inappropriately permits a cross-camera claim")
    if len(splits) != 5 or any(not str(item["eligible_for_cross_camera_claim"]).startswith("no") for item in splits):
        raise ValueError("Split registry has an invalid cross-camera eligibility flag")
    return {
        "passed": True,
        "result_rows": len(rows),
        "experiment_ids": sorted(actual_ids),
        "data_role_rows": len(splits),
        "all_result_values_finite": True,
        "primary_gate_endpoints_present": True,
        "all_data_roles_cross_camera_ineligible": True,
        "source_freezes_verified": True,
        "source_hashes": source_hashes,
        "no_training_or_tuning_performed_by_ledger": True,
    }


def snapshot() -> dict[str, object]:
    records = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "snapshot_manifest.json":
            records[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "snapshot_id": "RMDB-prewriting-evidence-ledger-20260909-v1",
        "overwrite_policy": "The build refuses an existing output directory. Create a new dated version for any new evidence.",
        "records": records,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing ledger: {OUT}")
    source_hashes = check_source_freezes()
    rows = (
        historical_repeat_rows()
        + historical_reference_rows()
        + gate_rows(
            HRF_QUALITY / "measurement_stress_gate.json", "hrf_quality_stress", "fresh_high_resolution_stress_test",
            "same-camera acquisition-quality sensitivity; not standard test-retest or structural validation",
            "native 3888x2592 or 5184x3456",
        )
        + gate_rows(
            HRF_RESOLUTION / "gate_native_vs_75pct.json", "hrf_resolution_native_vs_75pct", "fresh_high_resolution_sensitivity_probe",
            "controlled digital-resolution sensitivity; not optical/cross-camera validation", "native versus 75% LANCZOS input",
        )
        + gate_rows(
            HRF_RESOLUTION / "gate_native_vs_50pct.json", "hrf_resolution_native_vs_50pct", "fresh_high_resolution_sensitivity_probe",
            "controlled digital-resolution sensitivity; not optical/cross-camera validation", "native versus 50% LANCZOS input",
        )
    )
    splits = split_rows()
    audit_result = audit(rows, splits, source_hashes)
    fields = [
        "experiment_id", "evidence_tier", "analysis_role", "cohort", "pairing_or_reference", "camera_or_input",
        "comparison", "endpoint", "statistic", "effect_representation", "raw_vascx_value", "joint_readout_value",
        "effect_value", "effect_ci_low", "effect_ci_high", "n_images", "n_eyes_or_pairs", "n_patients_or_clusters",
        "gate_label", "allowed_interpretation",
    ]
    split_fields = [
        "data_role_id", "dataset", "role", "images", "patients", "eyes", "paired_eyes", "data_relationship",
        "independence", "eligible_for_cross_camera_claim", "eligible_for_high_resolution_claim", "pooling_rule",
    ]
    claim_fields = ["claim_id", "claim", "support", "support_level", "allowed_wording", "prohibited_wording"]
    OUT.mkdir(parents=True)
    write_csv(OUT / "MASTER_RESULTS.csv", rows, fields)
    write_csv(OUT / "DATA_SPLIT_REGISTRY.csv", splits, split_fields)
    write_csv(OUT / "CLAIM_EVIDENCE_MATRIX.csv", claim_rows(), claim_fields)
    (OUT / "VASCX_INCREMENT_BOUNDARY.md").write_text(increment_note(), encoding="utf-8")
    write_json(OUT / "provenance.json", {
        "ledger_id": "RMDB-prewriting-evidence-ledger-20260909-v1",
        "purpose": "Single source of truth before manuscript writing; collates frozen results without re-analysis.",
        "source_hashes": source_hashes,
        "source_paths": {
            "historical_snapshot": str(HISTORICAL.relative_to(ROOT)),
            "hrf_snapshot": str(HRF_SNAPSHOT.relative_to(ROOT)),
            "historical_controls": str(CONTROLS.relative_to(ROOT)),
            "hrf_quality": str(HRF_QUALITY.relative_to(ROOT)),
            "hrf_resolution": str(HRF_RESOLUTION.relative_to(ROOT)),
        },
        "prohibited_pooling": [
            "Do not pool historical mBRSET and Fundus-AVSeg results with HRF.",
            "Do not pool the HRF quality stress test with the digital-resolution sensitivity experiment.",
            "Do not treat the HRF resolution experiment as independent from the HRF quality test.",
        ],
    })
    write_json(OUT / "audit.json", audit_result)
    write_json(OUT / "snapshot_manifest.json", snapshot())
    print(OUT)


if __name__ == "__main__":
    main()
