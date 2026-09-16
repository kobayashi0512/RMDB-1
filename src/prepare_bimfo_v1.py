#!/usr/bin/env python3
"""Create leakage-safe labels and cached bilateral vessel probability maps for BiMFO V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm


def run_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def canonicalize_fundus(rgb: np.ndarray, eye: str, output_size: int) -> tuple[np.ndarray, dict]:
    """Crop the retinal field and mirror right eyes into a common nasal-temporal frame."""
    gray = rgb.mean(axis=2)
    mask = gray > 10
    ys, xs = np.where(mask)
    if len(xs) == 0:
        box = (0, 0, rgb.shape[1], rgb.shape[0])
    else:
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = int(max(x1 - x0, y1 - y0) * 1.04)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, y0 = cx - side // 2, cy - side // 2
        x1, y1 = x0 + side, y0 + side
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - rgb.shape[1]), max(0, y1 - rgb.shape[0])
        if any((pad_l, pad_t, pad_r, pad_b)):
            rgb = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), constant_values=0)
            x0, x1, y0, y1 = x0 + pad_l, x1 + pad_l, y0 + pad_t, y1 + pad_t
        box = (int(x0), int(y0), int(x1), int(y1))
    canonical = np.asarray(
        Image.fromarray(rgb).crop(box).resize((output_size, output_size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    mirrored = eye == "right"
    if mirrored:
        canonical = np.ascontiguousarray(canonical[:, ::-1])
    return canonical, {"crop_box": list(box), "mirrored": mirrored}


def quality_metrics(rgb: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    gray = rgb.mean(axis=2) / 255.0
    fov_full = gray > 0.04
    if gray.shape != probability.shape:
        gray_measure = np.asarray(
            Image.fromarray((gray * 255).astype(np.uint8)).resize((probability.shape[1], probability.shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        fov = np.asarray(
            Image.fromarray(fov_full).resize((probability.shape[1], probability.shape[0]), Image.Resampling.NEAREST),
            dtype=bool,
        )
    else:
        gray_measure, fov = gray, fov_full
    gy, gx = np.gradient(gray_measure)
    sharpness = float(np.mean((gx * gx + gy * gy)[fov])) if fov.any() else 0.0
    active = fov & (probability > 0.02)
    p = np.clip(probability[active], 1e-6, 1 - 1e-6)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
    structure_confidence = float(1 - entropy.mean()) if len(entropy) else 0.0
    illumination = float(np.std(gray_measure[fov])) if fov.any() else 0.0
    return {
        "fov_coverage": float(fov.mean()),
        "sharpness": sharpness,
        "illumination_std": illumination,
        "vessel_density": float(np.mean(probability[fov] >= 0.5)) if fov.any() else 0.0,
        "vessel_signal_q95": float(np.quantile(probability[fov], 0.95)) if fov.any() else 0.0,
        "structure_confidence": structure_confidence,
        "uncertain_fraction": float(np.mean((probability[fov] > 0.1) & (probability[fov] < 0.9))) if fov.any() else 1.0,
    }


def expected_cimt_model() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("age_spline", SplineTransformer(n_knots=5, degree=3, include_bias=False), ["age"]),
        ("sex", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), ["sex"]),
    ])
    return Pipeline([("features", preprocessor), ("ridge", Ridge(alpha=1.0))])


def fit_delta_labels(frame: pd.DataFrame, seed: int = 20260727) -> tuple[pd.DataFrame, dict]:
    """Use OOF residuals for train and a full-train model for untouched val/test."""
    train_mask = frame.split.eq("train").to_numpy()
    train = frame.loc[train_mask].reset_index()
    oof = np.empty(len(train), dtype=np.float64)
    fold_id = np.full(len(frame), -1, dtype=np.int16)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(train)):
        model = expected_cimt_model()
        model.fit(train.loc[fit_idx, ["age", "sex"]], train.loc[fit_idx, "cimt_mm"])
        oof[hold_idx] = model.predict(train.loc[hold_idx, ["age", "sex"]])
        fold_id[train.loc[hold_idx, "index"].to_numpy()] = fold
    expected = expected_cimt_model()
    expected.fit(train[["age", "sex"]], train.cimt_mm)
    result = frame.copy()
    result["expected_cimt_mm"] = expected.predict(result[["age", "sex"]])
    result.loc[train_mask, "expected_cimt_mm"] = oof
    result["label_fold"] = fold_id
    result["delta_cimt_mm"] = result.cimt_mm - result.expected_cimt_mm
    threshold = float(result.loc[result.split == "train", "delta_cimt_mm"].quantile(0.75))
    result["accelerated_cimt"] = (result.delta_cimt_mm >= threshold).astype(int)
    metadata = {
        "label_model": "five_fold_oof_train_age_sex_spline_ridge_full_train_for_val_test",
        "label_seed": seed, "label_folds": 5, "n_knots": 5, "ridge_alpha": 1.0,
        "accelerated_threshold_train_q75_mm": threshold,
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--weights", type=Path, default=Path("artifacts/fives_unet_resnet34/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bimfo_v1_cache"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; never use for reported results.")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest, dtype={"patient_id": str})
    labelled, label_metadata = fit_delta_labels(frame)
    selected = labelled if args.limit is None else labelled.head(args.limit).copy()
    args.output.mkdir(parents=True, exist_ok=True)
    map_path = args.output / "vessel_probability_maps.npy"
    maps = np.lib.format.open_memmap(map_path, mode="w+", dtype=np.float16, shape=(len(selected), 2, args.map_size, args.map_size))

    dev = run_device()
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    metadata = []
    for start in tqdm(range(0, len(selected), args.batch_size), desc="BiMFO V1 cache"):
        batch = selected.iloc[start:start + args.batch_size]
        tensors, images, row_eyes = [], [], []
        for row_index, row in enumerate(batch.itertuples(), start=start):
            for eye_index, eye in enumerate(("left", "right")):
                raw = np.asarray(Image.open(getattr(row, f"{eye}_path")).convert("RGB"), dtype=np.uint8)
                image, canonical = canonicalize_fundus(raw, eye, args.image_size)
                tensors.append(normalize(transforms.ToTensor()(Image.fromarray(image))))
                images.append(image)
                row_eyes.append((row_index, eye_index, canonical))
        with torch.inference_mode():
            probability = model(torch.stack(tensors).to(dev)).sigmoid()
            probability = F.interpolate(probability, size=(args.map_size, args.map_size), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        for image, prob, (row_index, eye_index, canonical) in zip(images, probability, row_eyes):
            maps[row_index, eye_index] = prob.astype(np.float16)
            q = quality_metrics(image, prob)
            target = selected.iloc[row_index]
            record = {
                "cache_index": row_index, "patient_id": str(target.patient_id),
                "eye": "left" if eye_index == 0 else "right", **canonical, **q,
            }
            metadata.append(record)
    maps.flush()
    selected.to_csv(args.output / "patients.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output / "image_quality.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps({
        **label_metadata, "map_shape": list(maps.shape), "map_dtype": "float16",
        "source_weights": str(args.weights), "smoke_test_limit": args.limit,
        "canonicalization": "fov_square_crop_4pct_margin_right_eye_horizontal_mirror",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"patients": len(selected), "map_shape": list(maps.shape), **label_metadata}, indent=2))


if __name__ == "__main__":
    main()
