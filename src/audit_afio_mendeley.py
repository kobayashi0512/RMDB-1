#!/usr/bin/env python3
"""Audit the public AFIO/Mendeley fundus-vessel dataset without reading labels.

The audit is deliberately outcome-blind: it inventories image/annotation files,
dimensions, finite values and pairing only.  Clinical interpretation is kept in
separate downstream tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def kind(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".ai":
        return "illustrator_annotation"
    if path.suffix.lower() != ".jpg":
        return "other"
    if name.endswith(".jpg") and path.stem.lower().endswith("--vessels"):
        return "vessel_mask"
    if "both" in name:
        return "av_mask"
    if "arter" in name or "atertr" in name or "artry" in name:
        return "artery_mask"
    if "vein" in name:
        return "vein_mask"
    if "map" in name:
        return "map_annotation"
    if path.suffix == ".JPG":
        return "fundus_image"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    files = sorted(x for x in args.root.rglob("*") if x.is_file())
    rows = []
    for path in files:
        role = kind(path)
        row = {
            "path": str(path),
            "relative_path": str(path.relative_to(args.root)),
            "id": path.parent.name,
            "filename": path.name,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "width": np.nan,
            "height": np.nan,
            "channels": np.nan,
            "finite": True,
        }
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            try:
                with Image.open(path) as im:
                    row["width"], row["height"] = im.size
                    row["channels"] = len(im.getbands())
                    arr = np.asarray(im)
                    row["finite"] = bool(np.isfinite(arr).all())
            except Exception as exc:  # keep a visible audit failure
                row["finite"] = False
                row["error"] = repr(exc)
        rows.append(row)
    manifest = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output / "file_manifest.csv", index=False)
    images = manifest[manifest.role.eq("fundus_image")]
    ids = sorted(images.id.unique())
    paired = []
    for image_id in ids:
        sub = manifest[manifest.id.eq(image_id)]
        paired.append({
            "id": image_id,
            "image": int((sub.role == "fundus_image").sum()),
            "vessel_mask": int((sub.role == "vessel_mask").sum()),
            "av_mask": int((sub.role == "av_mask").sum()),
            "artery_mask": int((sub.role == "artery_mask").sum()),
            "vein_mask": int((sub.role == "vein_mask").sum()),
            "illustrator_annotation": int((sub.role == "illustrator_annotation").sum()),
        })
    pd.DataFrame(paired).to_csv(args.output / "pairing.csv", index=False)
    summary = {
        "source": "Mendeley Data 3csr652p9y, version 2",
        "license": "CC BY 4.0",
        "outcome_blind": True,
        "files": int(len(manifest)),
        "images": int(len(images)),
        "unique_image_ids": int(len(ids)),
        "annotation_files": int((manifest.role != "fundus_image").sum()),
        "all_files_finite_or_nonpixel": bool(manifest.finite.all()),
        "paired_vessel_masks": int((pd.DataFrame(paired).vessel_mask > 0).sum()) if paired else 0,
        "paired_av_masks": int((pd.DataFrame(paired).av_mask > 0).sum()) if paired else 0,
        "paired_illustrator_annotations": int((pd.DataFrame(paired).illustrator_annotation > 0).sum()) if paired else 0,
        "complete_for_measurement": bool(len(images) == 100 and len(ids) == 100 and manifest.finite.all()),
    }
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
