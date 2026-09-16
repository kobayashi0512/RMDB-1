#!/usr/bin/env python3
"""Outcome-blind structural migration audit for open vessel datasets.

It inventories image/label pairs and runs no model or clinical label search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def audit_dataset(name: str, root: Path) -> dict:
    if not root.exists():
        return {"dataset": name, "status": "not_found", "root": str(root)}
    files = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm", ".tif", ".tiff"}]
    labels = [p for p in files if any(s in p.name.lower() for s in ("mask", "label", "ground", ".ah.", ".vk."))]
    dims = []
    finite = True
    for p in images[:2000]:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im)
                dims.append(tuple(arr.shape))
                finite = finite and bool(np.isfinite(arr).all())
        except Exception:
            finite = False
    return {
        "dataset": name,
        "status": "ready" if images else "no_images",
        "root": str(root),
        "files": len(files),
        "images": len(images),
        "labels_or_masks": len(labels),
        "finite_sample": bool(finite),
        "unique_dimensions_sample": sorted({str(x) for x in dims})[:20],
        "sha256_first_image": digest(images[0]) if images else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "ROOT"), required=True)
    args = p.parse_args()
    rows = [audit_dataset(name, Path(root)) for name, root in args.dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
