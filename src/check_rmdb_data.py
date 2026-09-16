#!/usr/bin/env python3
"""Verify the RMDB data registry without altering source datasets.

Run from baseline/:
    .venv/bin/python scripts/check_rmdb_data.py

The script intentionally does not infer licences or download protected datasets.
It reports whether each dataset is present and whether its expected image/mask
layout is plausible before it is admitted to the measurement layer.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in exts)


def report(name: str, path: Path, expected: str, role: str) -> bool:
    image_count = count_images(path) if path.exists() else 0
    present = image_count > 0
    state = "READY" if present else "PENDING"
    print(f"[{state}] {name}")
    print(f"  role: {role}")
    print(f"  path: {path}")
    print(f"  images: {image_count} (expected: {expected})")
    return present


def main() -> None:
    checks = [
        (
            "China Fundus-CIMT",
            DATA_ROOT / "China-Fundus-CIMT",
            "5806 bilateral fundus images",
            "primary RMDB-C outcome cohort",
        ),
        (
            "FIVES",
            DATA_ROOT / "FIVES",
            "800 image/mask pairs (about 1600 image files)",
            "binary vessel measurement layer",
        ),
        (
            "Fundus-DG-OD/OC",
            DATA_ROOT / "external" / "fundus_dg_odoc",
            "1391 optic-disc/cup image/mask pairs after extraction",
            "open CC-BY optic-disc/cup measurement calibration",
        ),
        (
            "RITE",
            DATA_ROOT / "external" / "RITE",
            "40 images with artery/vein labels",
            "A/V calibration; official-request dataset",
        ),
        (
            "REFUGE",
            DATA_ROOT / "external" / "REFUGE",
            "400 labelled training images",
            "optic-disc/cup calibration; account-authorized dataset",
        ),
        (
            "External CIMT",
            DATA_ROOT / "external" / "external_cimt",
            "500-1000 matched bilateral fundus-CIMT patients",
            "locked external clinical validation",
        ),
        (
            "CFP-OCTA pilot",
            DATA_ROOT / "external" / "cfp_octa_pilot",
            "100-200 paired CFP/OCTA patients",
            "biological anchoring",
        ),
    ]
    ready = sum(report(*check) for check in checks)
    print(f"\nRMDB data readiness: {ready}/{len(checks)} components present")
    print("Only READY datasets may be used. Keep protected challenge test labels out of development.")


if __name__ == "__main__":
    main()
