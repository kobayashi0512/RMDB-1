#!/usr/bin/env python3
"""Build compact, canonical VascX anatomy inputs for BiMFO V2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def resize_label(path: Path, size: int, mirror: bool) -> np.ndarray:
    image = Image.open(path).resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(array[:, ::-1] if mirror else array)


def canonical_coordinate(value: float, size: int, mirror: bool) -> float:
    scaled = float(value) * (size - 1) / 1023.0
    return float(size - 1 - scaled) if mirror else scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/bimfo_v1_1_cache"))
    parser.add_argument("--anatomy", type=Path, default=Path("artifacts/vascx_anatomy"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bimfo_v2_anatomy_cache"))
    parser.add_argument("--map-size", type=int, default=256)
    args = parser.parse_args()

    patients = pd.read_csv(args.cache / "patients.csv", dtype={"patient_id": str})
    audit = pd.read_csv(args.anatomy / "anatomy_audit.csv", dtype={"patient_id": str})
    audit = audit.set_index("id")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = np.lib.format.open_memmap(
        args.output / "anatomy_labels.npy", mode="w+", dtype=np.uint8,
        shape=(len(patients), 2, 2, args.map_size, args.map_size),
    )
    records = []
    for patient_index, row in enumerate(tqdm(
        patients.itertuples(), total=len(patients), desc="BiMFO V2 anatomy"
    )):
        for eye_index, (eye, suffix) in enumerate((("left", "L"), ("right", "R"))):
            identifier = f"{row.patient_id}_{suffix}"
            source = audit.loc[identifier]
            mirror = eye == "right"
            av = resize_label(
                args.anatomy / "artery_vein" / f"{identifier}.png",
                args.map_size, mirror,
            )
            disc = resize_label(
                args.anatomy / "disc" / f"{identifier}.png",
                args.map_size, mirror,
            )
            detected = bool(source.disc_detected)
            laterality_valid = (
                (eye == "left" and float(source.disc_x) < float(source.fovea_x))
                or (eye == "right" and float(source.disc_x) > float(source.fovea_x))
            )
            distance = float(np.hypot(
                float(source.disc_x) - float(source.fovea_x),
                float(source.disc_y) - float(source.fovea_y),
            ))
            geometry_valid = (
                detected
                and laterality_valid
                and 0.003 <= float(source.disc_fraction) <= 0.05
                and 150 <= distance <= 650
            )
            labels[patient_index, eye_index, 0] = av
            labels[patient_index, eye_index, 1] = (
                (disc > 0).astype(np.uint8) if geometry_valid else np.zeros_like(disc)
            )
            diameter = (
                2 * np.sqrt(float(source.disc_fraction) * 1024 * 1024 / np.pi)
                * (args.map_size - 1) / 1023.0
            )
            records.append({
                "cache_index": patient_index,
                "patient_id": str(row.patient_id),
                "eye": eye,
                "id": identifier,
                "disc_detected": detected,
                "landmark_geometry_valid": geometry_valid,
                "landmark_confidence": float(geometry_valid),
                "disc_x": canonical_coordinate(source.disc_x, args.map_size, mirror),
                "disc_y": canonical_coordinate(source.disc_y, args.map_size, False),
                "disc_diameter": float(diameter) if geometry_valid else np.nan,
                "fovea_x": canonical_coordinate(source.fovea_x, args.map_size, mirror),
                "fovea_y": canonical_coordinate(source.fovea_y, args.map_size, False),
                "artery_fraction": float(source.artery_fraction),
                "vein_fraction": float(source.vein_fraction),
                "crossing_fraction": float(source.crossing_fraction),
            })
    labels.flush()
    anatomy = pd.DataFrame(records)
    median_diameter = float(
        anatomy.loc[anatomy.landmark_geometry_valid, "disc_diameter"].median()
    )
    anatomy["disc_diameter_imputed"] = ~anatomy["landmark_geometry_valid"]
    anatomy["disc_diameter"] = anatomy["disc_diameter"].fillna(median_diameter)
    anatomy.to_csv(args.output / "anatomy.csv", index=False)
    metadata = {
        "patients": len(patients),
        "eyes": len(anatomy),
        "shape": list(labels.shape),
        "dtype": "uint8",
        "map_size": args.map_size,
        "channels": ["artery_vein_class_0_3", "optic_disc_binary"],
        "canonicalization": "right_eye_horizontal_mirror",
        "disc_failures": int((~anatomy.disc_detected).sum()),
        "landmark_geometry_failures": int((~anatomy.landmark_geometry_valid).sum()),
        "disc_failure_policy": (
            "empty_or_anatomically_implausible_disc_gets_landmark_confidence_zero; "
            "no fabricated disc mask; median diameter only stabilizes coordinate "
            "scaling and must be gated downstream"
        ),
        "median_detected_disc_diameter_map_px": median_diameter,
        "test_set_used": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
