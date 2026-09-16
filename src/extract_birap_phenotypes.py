#!/usr/bin/env python3
"""Extract quality-gated, bilateral retinal microvascular phenotypes from VascX.

No outcome is read by this script.  It only uses frozen anatomy masks and QC
metadata, making phenotype construction independent of CIMT labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm


def load_label(path: Path, size: int = 256) -> np.ndarray:
    return np.asarray(Image.open(path).resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)


def fractal_dimension(binary: np.ndarray) -> float:
    """Box-counting dimension, NaN for degenerate masks."""
    points = np.argwhere(binary)
    if len(points) < 20:
        return np.nan
    sizes = np.array([2, 4, 8, 16, 32, 64], dtype=int)
    counts = []
    for size in sizes:
        bins = np.floor(points / size).astype(int)
        counts.append(len(np.unique(bins, axis=0)))
    slope = np.polyfit(np.log(1.0 / sizes), np.log(np.maximum(counts, 1)), 1)[0]
    return float(slope)


def skeleton_metrics(mask: np.ndarray, region: np.ndarray) -> dict[str, float]:
    mask = mask & region
    skel = skeletonize(mask)
    if skel.sum() < 10:
        return {k: np.nan for k in ("density", "skeleton_density", "width", "branch_density", "endpoint_density", "fractal")}
    # Number of 8-neighbours on a skeleton. Branches >=3; endpoints ==1.
    neighbours = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skel
    area = max(int(region.sum()), 1)
    distance = ndi.distance_transform_edt(mask)
    widths = 2.0 * distance[skel]
    return {
        "density": float(mask.sum() / area),
        "skeleton_density": float(skel.sum() / area),
        "width": float(np.mean(widths)) if len(widths) else np.nan,
        "branch_density": float(((skel) & (neighbours >= 3)).sum() / area),
        "endpoint_density": float(((skel) & (neighbours == 1)).sum() / area),
        "fractal": fractal_dimension(skel & region),
    }


def orientation_entropy(mask: np.ndarray, region: np.ndarray) -> float:
    skel = skeletonize(mask & region)
    if skel.sum() < 20:
        return np.nan
    gy, gx = np.gradient(skel.astype(float))
    angles = np.mod(np.arctan2(gy[skel], gx[skel]), np.pi)
    hist, _ = np.histogram(angles, bins=12, range=(0, np.pi))
    p = hist[hist > 0] / max(hist.sum(), 1)
    return float(-(p * np.log(p)).sum() / np.log(12))


def eye_features(av: np.ndarray, rgb: np.ndarray, meta: pd.Series) -> dict[str, float]:
    yy, xx = np.indices(av.shape)
    # Black outside the rendered fundus is excluded. It is intentionally outcome-free.
    fov = rgb.max(axis=2) > 10
    vessel = (av >= 1) & (av <= 3)
    artery, vein, crossing = av == 1, av == 2, av == 3
    diameter = float(meta.disc_diameter)
    landmark_ok = bool(meta.landmark_geometry_valid) and np.isfinite(diameter) and diameter > 4
    if landmark_ok:
        radius = np.hypot(xx - float(meta.disc_x), yy - float(meta.disc_y)) / (diameter / 2.0)
        inner = fov & (radius >= 1.0) & (radius < 2.0)
        outer = fov & (radius >= 2.0) & (radius < 4.0)
    else:
        inner = outer = np.zeros_like(fov, dtype=bool)

    out: dict[str, float] = {
        "fov_fraction": float(fov.mean()),
        "landmark_valid": float(landmark_ok),
        "crossing_density": float(crossing[fov].mean()) if fov.any() else np.nan,
    }
    for name, mask in (("artery", artery), ("vein", vein), ("vessel", vessel)):
        for suffix, region in (("global", fov), ("inner_dd", inner), ("outer_dd", outer)):
            values = skeleton_metrics(mask, region)
            for key, value in values.items():
                out[f"{name}_{suffix}_{key}"] = value
        out[f"{name}_global_orientation_entropy"] = orientation_entropy(mask, fov)
    out["av_density_ratio"] = out["artery_global_density"] / max(out["vein_global_density"], 1e-6)
    out["av_width_ratio"] = out["artery_global_width"] / max(out["vein_global_width"], 1e-6)
    return out


def reliability(features: dict[str, float]) -> float:
    # Predeclared, outcome-free rejection rule. Anatomical regions require a valid landmark.
    density = features.get("vessel_global_density", np.nan)
    fov = features.get("fov_fraction", np.nan)
    plausible = np.isfinite(density) and 0.01 <= density <= 0.20
    coverage = np.isfinite(fov) and fov >= 0.50
    return float(bool(plausible and coverage))


def paired_patient_features(eye: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in eye.columns if c not in {"cache_index", "patient_id", "eye", "id", "measurement_reliability"}]
    records = []
    for patient_id, group in eye.groupby("patient_id", sort=False):
        row = {"patient_id": patient_id}
        left = group.loc[group.eye == "left"].iloc[0]
        right = group.loc[group.eye == "right"].iloc[0]
        wl, wr = float(left.measurement_reliability), float(right.measurement_reliability)
        row["bilateral_reliability"] = min(wl, wr)
        row["bilateral_available"] = float(wl > 0 and wr > 0)
        for feature in features:
            lv, rv = float(left[feature]), float(right[feature])
            valid_l, valid_r = np.isfinite(lv) and wl > 0, np.isfinite(rv) and wr > 0
            if valid_l and valid_r:
                row[f"common_{feature}"] = (wl * lv + wr * rv) / (wl + wr)
                row[f"discordance_{feature}"] = abs(lv - rv)
            elif valid_l:
                row[f"common_{feature}"] = lv
                row[f"discordance_{feature}"] = np.nan
            elif valid_r:
                row[f"common_{feature}"] = rv
                row[f"discordance_{feature}"] = np.nan
            else:
                row[f"common_{feature}"] = np.nan
                row[f"discordance_{feature}"] = np.nan
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy", type=Path, default=Path("artifacts/vascx_anatomy"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/bimfo_v2_anatomy_cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/birap_phenotypes"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test first N patients; 0 means all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    anatomy = pd.read_csv(args.cache / "anatomy.csv", dtype={"patient_id": str})
    patient_ids = anatomy.cache_index.drop_duplicates().sort_values().tolist()
    if args.limit:
        patient_ids = patient_ids[:args.limit]
        anatomy = anatomy[anatomy.cache_index.isin(patient_ids)].copy()
    records = []
    for meta in tqdm(anatomy.itertuples(index=False), total=len(anatomy), desc="BiRAP phenotypes"):
        identifier = meta.id
        av = load_label(args.anatomy / "artery_vein" / f"{identifier}.png")
        rgb = np.asarray(
            Image.open(args.anatomy / "preprocessed_rgb" / f"{identifier}.png").resize((256, 256), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        if meta.eye == "right":
            av = np.ascontiguousarray(av[:, ::-1])
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        feature = eye_features(av, rgb, pd.Series(meta._asdict()))
        feature.update({"cache_index": int(meta.cache_index), "patient_id": str(meta.patient_id), "eye": meta.eye, "id": identifier})
        feature["measurement_reliability"] = reliability(feature)
        records.append(feature)
    eye = pd.DataFrame(records)
    patient = paired_patient_features(eye)
    eye.to_csv(args.output / "eye_level.csv", index=False)
    patient.to_csv(args.output / "patient_level.csv", index=False)
    audit = {
        "patients": int(patient.patient_id.nunique()), "eyes": int(len(eye)),
        "reliable_eyes": int(eye.measurement_reliability.sum()),
        "bilateral_available": int(patient.bilateral_available.sum()),
        "coordinate_valid_eyes": int(eye.landmark_valid.sum()),
        "test_set_used": False,
        "rules": "Outcome-free A/V masks; coordinate metrics require trusted disc/fovea landmarks; patient common=quality-weighted mean and discordance=absolute bilateral difference.",
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
