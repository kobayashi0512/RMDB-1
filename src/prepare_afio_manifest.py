#!/usr/bin/env python3
"""Create an outcome-blind AFIO image manifest from public extracted files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = sorted({p.parent for p in args.root.rglob("*.JPG")})
    rows = []
    for directory in paths:
        image = directory / f"{directory.name}.JPG"
        if not image.is_file():
            nested = directory / directory.name / f"{directory.name}.JPG"
            image = nested if nested.is_file() else image
        if image.is_file():
            rows.append({
                "id": directory.name,
                "patient_id": directory.name,
                "eye": "left",
                "path": str(image),
                "split": "external_afio",
            })
    if len(rows) != len(set(r["id"] for r in rows)):
        raise ValueError("duplicate AFIO IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"images={len(rows)} patients={len(set(r['patient_id'] for r in rows))}")


if __name__ == "__main__":
    main()
