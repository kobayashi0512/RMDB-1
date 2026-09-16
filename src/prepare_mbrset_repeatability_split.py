#!/usr/bin/env python3
"""Create a patient and eye grouped, outcome-blind mBRSET repeatability split.

The split is intended for the RMDB-BiRAP V2 paired-view experiment.  A patient,
including both eyes and every repeated view, is assigned to exactly one split.
Clinical columns are carried through for later analysis but are never used to
choose the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--patient-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--development-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1 or not 0 < args.development_fraction < 1:
        raise ValueError("Fractions must be in (0, 1).")
    if args.train_fraction + args.development_fraction >= 1:
        raise ValueError("Train plus development fractions must be below one.")

    image = pd.read_csv(args.image_manifest, dtype={"patient_id": str})
    patient = pd.read_csv(args.patient_manifest, dtype={"patient_id": str})
    required_image = {"id", "patient_id", "eye", "path"}
    required_patient = {"patient_id"}
    if not required_image.issubset(image.columns):
        raise KeyError(f"Image manifest lacks {sorted(required_image - set(image.columns))}")
    if not required_patient.issubset(patient.columns):
        raise KeyError(f"Patient manifest lacks {sorted(required_patient - set(patient.columns))}")
    if image["patient_id"].isna().any() or patient["patient_id"].isna().any():
        raise ValueError("Missing patient identifiers are not allowed.")
    if image["id"].duplicated().any():
        raise ValueError("Duplicate image identifiers detected.")
    if set(image.patient_id) != set(patient.patient_id):
        raise ValueError("Image and patient manifests contain different patient sets.")

    # Use a deterministic hash rather than clinical labels or outcome fields.
    # Sorting before hashing makes the assignment independent of CSV row order.
    identifiers = sorted(patient.patient_id.astype(str).unique())
    keyed = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{args.seed}:{value}".encode()).hexdigest(),
    )
    n = len(keyed)
    n_train = int(round(args.train_fraction * n))
    n_dev = int(round(args.development_fraction * n))
    split_map = {
        value: ("train" if index < n_train else "development" if index < n_train + n_dev else "test")
        for index, value in enumerate(keyed)
    }
    image = image.copy()
    patient = patient.copy()
    image["split"] = image.patient_id.map(split_map)
    patient["split"] = patient.patient_id.map(split_map)
    if image.split.isna().any() or patient.split.isna().any():
        raise AssertionError("Incomplete split assignment.")

    # All repeated views of an eye must remain together in one patient group.
    consistency = image.groupby("patient_id", sort=False)["split"].nunique()
    if int(consistency.max()) != 1:
        raise AssertionError("Patient leakage detected in split assignment.")

    args.output.mkdir(parents=True, exist_ok=True)
    image.sort_values(["split", "patient_id", "eye", "id"]).to_csv(
        args.output / "per_image.csv", index=False
    )
    patient.sort_values(["split", "patient_id"]).to_csv(
        args.output / "per_patient.csv", index=False
    )
    report = {
        "protocol": "outcome_blind_patient_grouped_repeatability_v2",
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "development_fraction": args.development_fraction,
        "patients": int(len(patient)),
        "images": int(len(image)),
        "split_patients": patient.groupby("split").size().astype(int).to_dict(),
        "split_images": image.groupby("split").size().astype(int).to_dict(),
        "repeat_eyes": int(
            image.groupby(["patient_id", "eye"]).size().ge(2).sum()
        ),
        "clinical_columns_not_used_for_assignment": [
            column for column in patient.columns
            if column not in {"patient_id", "split"}
        ],
        "cimt_used": False,
        "outcomes_used": False,
    }
    (args.output / "split_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
