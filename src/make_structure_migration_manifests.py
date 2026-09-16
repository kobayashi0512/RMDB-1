#!/usr/bin/env python3
"""Build image-only manifests for frozen phenotype inference on FIVES/STARE.

No labels are read into the manifests; annotations remain in a separate tree
for measurement-only scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fives", "stare"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.dataset == "fives":
        # Accept either the dataset root (…/FIVES) or a split root
        # (…/FIVES/test).  FIVES stores images directly under Original/.
        image_root = args.root / "Original" if (args.root / "Original").exists() else args.root / "test" / "Original"
        images = sorted(image_root.glob("*"))
    else:
        images = sorted((args.root / "images").glob("*.ppm"))
    rows = [{"id": x.stem, "patient_id": x.stem, "eye": "left", "path": str(x), "split": f"external_{args.dataset}"} for x in images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"{args.dataset}: images={len(rows)}")


if __name__ == "__main__":
    main()
