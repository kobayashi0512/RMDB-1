#!/usr/bin/env python3
"""Create a label-free image manifest for external anatomy transfer audits."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    paths = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(f"No images under {args.images}")
    rows = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            width, height, mode = image.width, image.height, image.mode
        image_id = f"{args.dataset}_{index:05d}"
        rows.append({
            "id": image_id, "patient_id": image_id, "eye": "unknown",
            "path": str(path.resolve()), "dataset": args.dataset,
            "width": width, "height": height, "mode": mode,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"dataset={args.dataset} images={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
