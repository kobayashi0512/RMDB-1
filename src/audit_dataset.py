#!/usr/bin/env python3
"""Create a reproducible, patient-level manifest for the China-Fundus-CIMT data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


GROUP_NAMES = {1: "train", 2: "val", 3: "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("../data/China-Fundus-CIMT"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    image_root = data_root / "Fundus_CIMT_2903 Dataset"
    with (data_root / "data_info.json").open(encoding="utf-8") as handle:
        source = json.load(handle)

    rows = []
    for patient_id, record in source.items():
        left_path = image_root / record["left_eye"]
        right_path = image_root / record["right_eye"]
        rows.append({
            "patient_id": str(patient_id),
            "split": GROUP_NAMES[record["group"]],
            "group": record["group"],
            "age": record["True_age"],
            "age_normalized": record["age"],
            "sex": record["gender"],
            "cimt_mm": record["thickness"],
            "label": record["label"],
            "left_path": str(left_path),
            "right_path": str(right_path),
            "left_exists": left_path.is_file(),
            "right_exists": right_path.is_file(),
        })

    manifest = pd.DataFrame(rows).sort_values(["group", "patient_id"])
    if not manifest.patient_id.is_unique:
        raise RuntimeError("Patient IDs must be unique in a bilateral manifest.")
    if not manifest[["left_exists", "right_exists"]].all(axis=None):
        raise FileNotFoundError("At least one referenced fundus image is missing.")

    if args.verify_images:
        for column in ("left_path", "right_path"):
            for path in manifest[column]:
                with Image.open(path) as image:
                    image.verify()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    summary = manifest.groupby("split").agg(
        patients=("patient_id", "count"),
        thickened=("label", "sum"),
        cimt_mean=("cimt_mm", "mean"),
        age_mean=("age", "mean"),
    ).round(3)
    print(summary)
    print(f"Wrote {len(manifest)} patient records to {args.output}")


if __name__ == "__main__":
    main()
