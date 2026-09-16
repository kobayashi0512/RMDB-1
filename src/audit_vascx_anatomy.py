#!/usr/bin/env python3
"""Quantitative and visual quality control for VascX anatomy extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def overlay_eye(root: Path, row: pd.Series, size: int = 256) -> Image.Image:
    identifier = str(row["id"])
    rgb = np.asarray(Image.open(root / "preprocessed_rgb" / f"{identifier}.png").convert("RGB"))
    av = np.asarray(Image.open(root / "artery_vein" / f"{identifier}.png"))
    disc = np.asarray(Image.open(root / "disc" / f"{identifier}.png"))
    canvas = rgb.astype(np.float32)
    colors = {
        1: np.array([255, 40, 40], dtype=np.float32),
        2: np.array([40, 100, 255], dtype=np.float32),
        3: np.array([255, 220, 20], dtype=np.float32),
    }
    for label, color in colors.items():
        mask = av == label
        canvas[mask] = 0.58 * canvas[mask] + 0.42 * color
    mask = disc > 0
    canvas[mask] = 0.50 * canvas[mask] + 0.50 * np.array([40, 255, 80], dtype=np.float32)
    image = Image.fromarray(np.uint8(np.clip(canvas, 0, 255)))
    draw = ImageDraw.Draw(image)
    fx, fy = float(row["fovea_x"]), float(row["fovea_y"])
    radius = 10
    draw.ellipse((fx - radius, fy - radius, fx + radius, fy + radius), outline=(0, 255, 255), width=4)
    if not bool(row["disc_detected"]):
        dx, dy = float(row["disc_x"]), float(row["disc_y"])
        draw.rectangle((dx - radius, dy - radius, dx + radius, dy + radius), outline=(255, 0, 255), width=4)
    image.thumbnail((size, size))
    tile = Image.new("RGB", (size, size + 34), "white")
    tile.paste(image, ((size - image.width) // 2, 0))
    label = f"{identifier}  {row['qc_reason']}"
    ImageDraw.Draw(tile).text((4, size + 5), label[:46], fill="black")
    return tile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/vascx_anatomy"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vascx_anatomy_qc"))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--random-eyes", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input / "anatomy_audit.csv")
    frame["vessel_fraction"] = (
        frame["artery_fraction"] + frame["vein_fraction"] + frame["crossing_fraction"]
    )
    frame["disc_fovea_distance"] = np.hypot(
        frame["disc_x"] - frame["fovea_x"], frame["disc_y"] - frame["fovea_y"]
    )
    detected = frame["disc_detected"].astype(bool)
    frame["laterality_mismatch"] = detected & (
        ((frame["eye"] == "left") & (frame["disc_x"] >= frame["fovea_x"]))
        | ((frame["eye"] == "right") & (frame["disc_x"] <= frame["fovea_x"]))
    )
    frame["disc_size_flag"] = detected & (
        (frame["disc_fraction"] < 0.003) | (frame["disc_fraction"] > 0.05)
    )
    frame["disc_distance_flag"] = detected & (
        (frame["disc_fovea_distance"] < 150) | (frame["disc_fovea_distance"] > 650)
    )
    frame["fovea_bounds_flag"] = (
        (frame["fovea_x"] < 0) | (frame["fovea_x"] >= 1024)
        | (frame["fovea_y"] < 0) | (frame["fovea_y"] >= 1024)
    )
    vessel_low, vessel_high = frame["vessel_fraction"].quantile([0.005, 0.995])
    frame["vessel_extreme_flag"] = (
        (frame["vessel_fraction"] < vessel_low) | (frame["vessel_fraction"] > vessel_high)
    )
    flag_columns = [
        "laterality_mismatch", "disc_size_flag", "disc_distance_flag",
        "fovea_bounds_flag", "vessel_extreme_flag",
    ]
    frame["qc_flag"] = (~detected) | frame[flag_columns].any(axis=1)
    frame["qc_reason"] = ""
    reasons = [
        (~detected, "disc_empty"),
        (frame["laterality_mismatch"], "laterality"),
        (frame["disc_size_flag"], "disc_size"),
        (frame["disc_distance_flag"], "disc_distance"),
        (frame["fovea_bounds_flag"], "fovea_bounds"),
        (frame["vessel_extreme_flag"], "vessel_extreme"),
    ]
    for condition, reason in reasons:
        frame.loc[condition, "qc_reason"] += reason + ";"

    wide = frame.pivot(index="patient_id", columns="eye", values=[
        "artery_fraction", "vein_fraction", "crossing_fraction", "vessel_fraction",
    ])
    bilateral_correlations = {}
    for metric in ["artery_fraction", "vein_fraction", "crossing_fraction", "vessel_fraction"]:
        bilateral_correlations[metric] = float(wide[metric]["left"].corr(wide[metric]["right"]))

    quantile_columns = [
        "artery_fraction", "vein_fraction", "crossing_fraction",
        "vessel_fraction", "disc_fraction", "disc_fovea_distance",
    ]
    quantiles = {
        column: {
            str(q): float(value)
            for q, value in frame[column].quantile([0, .005, .01, .05, .5, .95, .99, .995, 1]).items()
        }
        for column in quantile_columns
    }
    report = {
        "eyes": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "missing_cells": int(frame.isna().sum().sum()),
        "empty_av_masks": int((frame["vessel_fraction"] <= 0).sum()),
        "empty_disc_masks": int((~detected).sum()),
        "laterality_mismatches": int(frame["laterality_mismatch"].sum()),
        "disc_size_flags": int(frame["disc_size_flag"].sum()),
        "disc_distance_flags": int(frame["disc_distance_flag"].sum()),
        "fovea_out_of_bounds": int(frame["fovea_bounds_flag"].sum()),
        "vessel_extreme_flags": int(frame["vessel_extreme_flag"].sum()),
        "unique_flagged_eyes": int(frame["qc_flag"].sum()),
        "bilateral_correlations": bilateral_correlations,
        "quantiles": quantiles,
        "visual_legend": {
            "artery": "red", "vein": "blue", "crossing": "yellow",
            "disc_mask": "green", "fovea": "cyan_circle",
            "imputed_disc_center": "magenta_square",
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    frame[frame["qc_flag"]].to_csv(args.output / "flagged_eyes.csv", index=False)

    flagged = frame[frame["qc_flag"]].copy()
    if len(flagged) > 48:
        priorities = pd.concat([
            flagged[~flagged["disc_detected"].astype(bool)],
            flagged[flagged["laterality_mismatch"]],
            flagged.nlargest(8, "vessel_fraction"),
            flagged.nsmallest(8, "vessel_fraction"),
        ]).drop_duplicates("id")
        flagged = pd.concat([priorities, flagged]).drop_duplicates("id").head(48)
    random_rows = frame[~frame["qc_flag"]].sample(
        n=min(args.random_eyes, int((~frame["qc_flag"]).sum())),
        random_state=args.seed,
    ).copy()
    random_rows["qc_reason"] = "random"
    selected = pd.concat([flagged, random_rows], ignore_index=True)
    tiles = [overlay_eye(args.input, row) for _, row in selected.iterrows()]
    columns = 6
    rows = int(np.ceil(len(tiles) / columns))
    montage = Image.new("RGB", (columns * 256, rows * 290), (235, 235, 235))
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * 256, (index // columns) * 290))
    montage.save(args.output / "montage.png")
    selected[["id", "qc_reason"]].to_csv(args.output / "montage_manifest.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
