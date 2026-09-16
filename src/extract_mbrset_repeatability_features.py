#!/usr/bin/env python3
"""Outcome-blind, restartable feature extraction for public mBRSET repeats.

This extractor intentionally uses only RGB images, frozen VascX logits and a
frozen occupancy probability.  It produces global features suitable for the
paired-view repeatability benchmark; no clinical column is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize
from tqdm import tqdm


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits -= logits.max(axis=-1, keepdims=True)
    value = np.exp(logits)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-8)


def box_dimension(mask: np.ndarray) -> float:
    points = np.argwhere(mask)
    if len(points) < 20:
        return float("nan")
    sizes = np.asarray([2, 4, 8, 16, 32, 64])
    counts = []
    for size in sizes:
        h, w = mask.shape
        hh, ww = h - h % size, w - w % size
        blocks = mask[:hh, :ww].reshape(hh // size, size, ww // size, size)
        counts.append(max(int(blocks.any(axis=(1, 3)).sum()), 1))
    return float(np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)[0])


def readout_features(probability: np.ndarray, fov: np.ndarray, prefix: str) -> dict[str, float]:
    probability = np.clip(probability.astype(np.float32), 1e-7, 1.0)
    vessel = probability[1:].sum(axis=0)
    conditional = probability[1:] / np.maximum(vessel[None], 1e-7)
    artery = probability[1] + probability[3]
    vein = probability[2] + probability[3]
    mask = (vessel >= 0.5) & fov
    small = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST), dtype=bool)
    skeleton = skeletonize(small)
    neighbours = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    components = label(small, connectivity=2)
    component_sizes = np.bincount(components.ravel())[1:]
    area = max(int(fov.sum()), 1)
    # Clamp probabilities before entropy so exact 0/1 outputs remain finite.
    vessel_safe = np.clip(vessel, 1e-7, 1.0 - 1e-7)
    vessel_entropy = -(vessel_safe * np.log(vessel_safe) + (1 - vessel_safe) * np.log(1 - vessel_safe))
    identity_entropy = -(conditional * np.log(conditional)).sum(axis=0) / np.log(3.0)
    out = {
        f"{prefix}vessel_mass": float(vessel[fov].mean()),
        f"{prefix}artery_mass": float(artery[fov].mean()),
        f"{prefix}vein_mass": float(vein[fov].mean()),
        f"{prefix}av_ratio": float(artery[fov].mean() / max(vein[fov].mean(), 1e-7)),
        f"{prefix}vessel_density": float(mask.sum() / area),
        f"{prefix}skeleton_density": float(skeleton.mean()),
        f"{prefix}branch_density": float((skeleton & (neighbours >= 3)).mean()),
        f"{prefix}endpoint_density": float((skeleton & (neighbours == 1)).mean()),
        f"{prefix}component_count": float((component_sizes >= 8).sum()),
        f"{prefix}fractal_dimension": box_dimension(skeleton),
        f"{prefix}occupancy_entropy": float(vessel_entropy[fov].mean()),
        f"{prefix}identity_entropy": float(identity_entropy[fov].mean()),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--occupancy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"id": str, "patient_id": str})
    expected = set(manifest.id.astype(str))
    args.output.mkdir(parents=True, exist_ok=True)
    parts = args.output / "per_image_parts"
    parts.mkdir(exist_ok=True)
    pending = [identifier for identifier in manifest.id.astype(str) if not (parts / f"{identifier}.json").is_file()]
    for identifier in tqdm(pending, desc="mBRSET repeatability features"):
        row = manifest.loc[manifest.id.astype(str).eq(identifier)].iloc[0]
        # The manifest keeps stable semantic IDs, whereas the frozen
        # inference export uses the source filename stem (for example 9.3).
        artifact_id = Path(str(row["file"])).stem
        rgb = np.asarray(Image.open(args.rgb / f"{artifact_id}.png").convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR), dtype=np.uint8)
        logits = np.load(args.logits / f"{artifact_id}.npy").astype(np.float32)
        probability = np.moveaxis(softmax(logits), -1, 0)
        if probability.shape[1:] != (args.size, args.size):
            probability = np.stack([
                np.asarray(Image.fromarray(channel).resize((args.size, args.size), Image.Resampling.BILINEAR))
                for channel in probability
            ])
            probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-7)
        occupancy = np.load(args.occupancy / f"{artifact_id}.npy").astype(np.float32)
        if occupancy.shape != (args.size, args.size):
            occupancy = np.asarray(Image.fromarray(occupancy).resize((args.size, args.size), Image.Resampling.BILINEAR), dtype=np.float32)
        fov = rgb.max(axis=2) > 10
        conditional = probability[1:] / np.maximum(probability[1:].sum(axis=0, keepdims=True), 1e-7)
        joint = np.concatenate([1.0 - occupancy[None], occupancy[None] * conditional], axis=0)
        result = {
            "id": identifier,
            "patient_id": str(row.patient_id),
            "eye": str(row.eye),
            "split": str(row.split),
            "view": str(row["view"]),
            "quality": str(row.final_quality) if "final_quality" in row else "",
            "icdr": str(row.final_icdr) if "final_icdr" in row else "",
            "occupancy_mean": float(occupancy[fov].mean()),
            "occupancy_support": float((occupancy[fov] >= 0.5).mean()),
        }
        result.update(readout_features(probability, fov, "raw_"))
        result.update(readout_features(joint, fov, "joint_"))
        # Cross-readout disagreements are retained as explicit measurements.
        result["readout_vessel_mass_abs_diff"] = abs(result["raw_vessel_mass"] - result["joint_vessel_mass"])
        result["readout_av_ratio_log_abs_diff"] = abs(np.log((result["raw_av_ratio"] + 1e-7) / (result["joint_av_ratio"] + 1e-7)))
        temporary = parts / f"{identifier}.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(parts / f"{identifier}.json")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(parts.glob("*.json"))]
    frame = pd.DataFrame(rows).sort_values(["patient_id", "eye", "view", "id"])
    frame.to_csv(args.output / "per_image.csv", index=False)
    expected_missing = sorted(expected - set(frame.id.astype(str)))
    report = {
        "images_expected": int(len(expected)),
        "images_completed": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "repeat_eyes": int(frame.groupby(["patient_id", "eye"]).size().ge(2).sum()),
        "missing": expected_missing,
        "label_blind": True,
        "cimt_used": False,
        "outcomes_used": False,
        "complete": not expected_missing and len(frame) == len(expected),
    }
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
