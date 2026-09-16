#!/usr/bin/env python3
"""Build an audited patient-eye-view manifest for the mBRSET pilot mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    required = {
        "patient", "file", "laterality", "age", "sex", "dm_time",
        "systemic_hypertension", "vascular_disease",
        "acute_myocardial_infarction", "nephropathy", "neuropathy",
        "diabetic_foot", "obesity", "smoking", "final_quality",
        "final_icdr", "final_edema",
    }
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        raise KeyError(f"Missing mBRSET columns: {missing_columns}")

    image_paths = {
        path.name: path.resolve()
        for path in args.images.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    metadata["file"] = metadata["file"].astype(str)
    available = metadata.loc[metadata["file"].isin(image_paths)].copy()
    available["patient_id"] = available["patient"].map(
        lambda value: f"MBRSET_{int(value):04d}"
    )
    available["eye"] = available["laterality"].str.lower().map(
        {"left": "left", "right": "right"}
    )
    if available["eye"].isna().any():
        raise ValueError("Unknown laterality values in matched mBRSET rows.")
    available["view"] = available["file"].str.extract(
        r"\.(\d+)\.[^.]+$", expand=False
    ).fillna("0")
    available["id"] = (
        available["patient_id"]
        + "_"
        + available["eye"].str[0].str.upper()
        + "_V"
        + available["view"].astype(str)
    )
    if available["id"].duplicated().any():
        duplicates = available.loc[available["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate image identifiers: {duplicates[:5]}")
    available["path"] = available["file"].map(
        lambda name: str(image_paths[name])
    )

    dimensions = []
    for row in available.itertuples(index=False):
        with Image.open(row.path) as image:
            dimensions.append((image.width, image.height, image.mode))
    available[["width", "height", "mode"]] = pd.DataFrame(
        dimensions, index=available.index
    )
    ordered = [
        "id", "patient_id", "patient", "eye", "view", "file", "path",
        "width", "height", "mode", "age", "sex", "dm_time", "insulin",
        "insulin_time", "oraltreatment_dm", "systemic_hypertension",
        "vascular_disease", "acute_myocardial_infarction", "nephropathy",
        "neuropathy", "diabetic_foot", "obesity", "smoking",
        "final_quality", "final_icdr", "final_edema",
    ]
    image_manifest = available[ordered].sort_values(
        ["patient_id", "eye", "view"]
    )
    image_manifest.to_csv(args.output / "per_image.csv", index=False)

    patient_columns = [
        "age", "sex", "dm_time", "insulin", "insulin_time",
        "oraltreatment_dm", "systemic_hypertension", "vascular_disease",
        "acute_myocardial_infarction", "nephropathy", "neuropathy",
        "diabetic_foot", "obesity", "smoking",
    ]
    patient_rows = []
    for patient_id, group in image_manifest.groupby("patient_id", sort=True):
        record: dict[str, object] = {"patient_id": patient_id}
        for column in patient_columns:
            values = group[column].dropna().unique()
            if len(values) > 1:
                raise ValueError(
                    f"Inconsistent patient field {column} for {patient_id}."
                )
            record[column] = values[0] if len(values) else float("nan")
        icdr = group["final_icdr"].dropna()
        record.update({
            "images": int(len(group)),
            "left_images": int(group["eye"].eq("left").sum()),
            "right_images": int(group["eye"].eq("right").sum()),
            "both_eyes": bool(group["eye"].nunique() == 2),
            "good_quality_images": int(group["final_quality"].eq("yes").sum()),
            "gradable_icdr_images": int(len(icdr)),
            "max_icdr": float(icdr.max()) if len(icdr) else float("nan"),
            "mean_icdr": float(icdr.mean()) if len(icdr) else float("nan"),
            "any_dr": bool((icdr > 0).any()) if len(icdr) else float("nan"),
            "all_images_no_dr": bool((icdr == 0).all()) if len(icdr) else False,
            "any_edema": bool(group["final_edema"].eq("yes").any()),
        })
        patient_rows.append(record)
    patients = pd.DataFrame(patient_rows).sort_values("patient_id")
    patients.to_csv(args.output / "per_patient.csv", index=False)

    metadata_report = {
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": sha256(args.metadata),
        "source_images": str(args.images.resolve()),
        "metadata_rows": int(len(metadata)),
        "matched_images": int(len(image_manifest)),
        "unmatched_metadata_rows": int(len(metadata) - len(image_manifest)),
        "archive_images": int(len(image_paths)),
        "patients": int(len(patients)),
        "patients_with_both_eyes": int(patients["both_eyes"].sum()),
        "dimensions": {
            f"{width}x{height}_{mode}": int(count)
            for (width, height, mode), count in image_manifest[
                ["width", "height", "mode"]
            ].value_counts().items()
        },
        "primary_outcome": {
            "name": "systemic_hypertension",
            "known": int(patients["systemic_hypertension"].notna().sum()),
            "positive": int(patients["systemic_hypertension"].eq(1).sum()),
        },
        "no_dr_subgroup": {
            "patients": int(patients["all_images_no_dr"].sum()),
            "known_primary_outcome": int(
                patients.loc[
                    patients["all_images_no_dr"],
                    "systemic_hypertension",
                ].notna().sum()
            ),
        },
        "complete": bool(
            len(image_manifest) == len(image_paths)
        and image_manifest["path"].map(lambda value: Path(value).is_file()).all()
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
