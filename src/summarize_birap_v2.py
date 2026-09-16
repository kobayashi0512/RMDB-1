#!/usr/bin/env python3
"""Create a fixed, outcome-blind comparison table for the BiRAP V2 queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = (
    "occupancy", "joint", "topology", "phenotype", "full",
    "residual_joint", "residual_topology", "residual_phenotype", "residual_full",
)


def flatten(summary: dict, split: str, prefix: str) -> dict:
    row = {"variant": summary["variant"], "split": split}
    metrics = summary.get(split, {})
    for key in (
        "n", "vessel_dice", "artery_dice", "vein_dice", "mean_av_dice",
        "type_accuracy", "av_ratio_log_error", "brier", "route", "bypass",
    ):
        if key in metrics:
            row[f"{prefix}{key}"] = metrics[key]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for variant in VARIANTS:
        path = args.root / variant / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        row = {"variant": summary["variant"]}
        for split, prefix in (("rav_validation", "rav_"), ("fives_validation", "fives_"),
                              ("rav_device_holdout_retrospective", "rav_holdout_")):
            row.update({k: v for k, v in flatten(summary, split, prefix).items() if k != "variant" and k != "split"})
        row["best_epoch"] = summary.get("best_epoch")
        row["best_selection_score"] = summary.get("best_selection_score")
        row["fundus_avseg_used"] = summary["development_boundaries"]["fundus_avseg_used"]
        row["cimt_used"] = summary["development_boundaries"]["cimt_used"]
        row["fives_official_test_used"] = summary["development_boundaries"]["fives_official_test_used"]
        row["rav_device_holdout_used_for_selection"] = summary["development_boundaries"]["rav_device_holdout_used_for_selection"]
        rows.append(row)
    if not rows:
        raise SystemExit("No completed BiRAP V2 summaries found")
    frame = pd.DataFrame(rows).sort_values("variant")
    args.root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.root / "ablation_summary.csv", index=False)
    (args.root / "ablation_summary.json").write_text(frame.to_json(orient="records", indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
