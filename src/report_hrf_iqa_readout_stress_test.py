#!/usr/bin/env python3
"""Render a compact Chinese experiment report from locked HRF IQA outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/hrf_iqa_readout_stress_test_v1"


def number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def interval(values: list[float | None], digits: int = 4) -> str:
    return "—" if any(value is None for value in values) else f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "measurement_stress_gate.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    rows = gate["within_camera"]
    lines = [
        "# HRF IQA 高分辨率读出压力测试",
        "",
        "## 状态",
        "",
        f"已完成并通过完整性审计：{audit['passed']}。本报告不是论文结果表，也不与既往 mBRSET 或 Fundus-AVSeg 数值合并。",
        "",
        "## 数据与边界",
        "",
        f"- 官方 HRF Image Quality Assessment 子集：{protocol['source']['native_images']} 张原图、{protocol['source']['pairs']} 个已公布同眼配对；原始分辨率为 {', '.join(protocol['source']['native_resolutions'])}。",
        "- 每对均为 Canon CR-1 的低质/失焦首拍与改善后的复拍。它是采集质量扰动测试，而非同条件 test–retest；没有跨相机配对。",
        "- 没有血管人工标注，因此本轮不评价结构准确性；只测试冻结 readout 对采集质量变化的测量敏感性。",
        "- 36 图、5 个读出、12 个特征，未训练、未调参、未读取临床结局。",
        "",
        "## 预先指定的三项测量端点",
        "",
        "数值为 bad→good 配对差异。CI 为以 18 个配对 ID 为聚类单位的 10,000 次 bootstrap、三端点 Bonferroni 后的 98.333% 中心区间。LoA 半宽比、MAD 比和 SEM 比均为 `joint/raw VascX`，小于 1 表示 joint 对质量扰动更不敏感。",
        "",
        "|端点|raw ICC|joint ICC|ΔICC [CI]|MAD 比 [CI]|SEM 比 [CI]|LoA 半宽比 [CI]|Gate|",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        raw, joint, contrast, ci = row["raw_vascx"], row["joint"], row["joint_minus_raw_vascx"], row["bootstrap_interval"]
        lines.append(
            f"|{row['feature']}|{number(raw['icc_1_1'])}|{number(joint['icc_1_1'])}|"
            f"{number(contrast['delta_icc'])} {interval(ci['delta_icc'])}|"
            f"{number(contrast['mad_ratio'])} {interval(ci['mad_ratio'])}|"
            f"{number(contrast['sem_ratio'])} {interval(ci['sem_ratio'])}|"
            f"{number(contrast['loa_halfwidth_ratio'])} {interval(ci['loa_halfwidth_ratio'])}|{row['gate_label']}|"
        )
    lines.extend([
        "",
        "## 读法",
        "",
        "- `endpoint_density`：joint 的 ΔICC 为正，MAD、SEM、LoA 半宽比均显著小于 1，属于在该质量扰动下的协同相对改善。",
        "- `branch_density`：ICC 的区间改善，但 MAD、SEM、LoA 没有同时改善，属于 mixed trade-off，不能声称稳定性整体提升。",
        "- `occupancy_entropy`：joint 的 ICC、MAD、SEM 和 LoA 都更差，明确属于 not improved。",
        "",
        "因此，这个独立高分辨率压力测试支持“不同 readout 暴露/分离 trade-off”，不支持“RMDB 解决了稳定性问题”或“joint 全面优于 VascX”。",
        "",
        "所有五个读出、全部 12 个特征及原始逐图数值见 `per_image_features.csv`、`feature_values_long.csv` 和 `measurement_stress_gate.json`；审计见 `audit.json`。",
        "",
    ])
    (OUT / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "REPORT_ZH.md")


if __name__ == "__main__":
    main()

