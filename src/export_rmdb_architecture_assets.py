#!/usr/bin/env python3
"""Export exact, data-derived example assets for the RMDB architecture figure.

The exported maps are visualizations of one project Fundus-AVSeg case under the
frozen implementations used by ``final_measurement_controls.py``.  They are not
new experimental results and no parameter is tuned here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from skimage.morphology import skeletonize

from train_udct_av_pilot import Corrector, hessian_field, softmax_np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/scientific_figures/rmdb_birap_architecture_20260910_111506"
SAMPLE_ID = "001_G"
RGB_PATH = ROOT / "data/fundus_avseg_vascx/preprocessed_rgb" / f"{SAMPLE_ID}.png"
CE_PATH = ROOT / "data/fundus_avseg_vascx/preprocessed_ce" / f"{SAMPLE_ID}.png"
LOGITS_PATH = ROOT / "data/fundus_avseg_vascx/logits" / f"{SAMPLE_ID}.npy"
MASS_WEIGHT = ROOT / "models/baseline/artifacts/udct_av_mass_calibrated/best.pt"
OCC_WEIGHT = ROOT / "models/baseline/artifacts/fives_unet_resnet34/best.pt"
STYLE_REFERENCE = Path("/Users/tongyue/Documents/ppf/papers/current_embedded_figure1_actual.png")
DATABASE_ORIGINALS = {
    "Fundus_AVSeg_001_G_original.png": ROOT / "data/raw/Fundus_AVSeg/Fundus-AVSeg/images/001_G.png",
    "FIVES_train_100_A_original.png": ROOT / "data/raw/FIVES/train/Original/100_A.png",
    "mBRSET_9_3_original.jpg": ROOT / "data/raw/mBRSET_Kaggle/mBRSET Data/0/9.3.jpg",
    "HRF_IQA_1_good_original.JPG": ROOT / "data/raw/HRF_IQA_20260909/images/1_good.JPG",
    "Fundus_AVSeg_002_N_original.png": ROOT / "data/raw/Fundus_AVSeg/Fundus-AVSeg/images/002_N.png",
    "Fundus_AVSeg_002_N_manual_AV.png": ROOT / "data/raw/Fundus_AVSeg/Fundus-AVSeg/annotation/002_N.png",
    "mBRSET_test_613_3_original.jpg": ROOT / "data/raw/mBRSET_Kaggle/mBRSET Data/0/613.3.jpg",
    "mBRSET_test_613_4_original.jpg": ROOT / "data/raw/mBRSET_Kaggle/mBRSET Data/0/613.4.jpg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def probability_rgb(probability: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    p = np.clip(probability.astype(np.float32), 0.0, 1.0)
    # A dark navy floor preserves faint probability structure on white pages.
    floor = np.array([8, 17, 29], dtype=np.float32)
    peak = np.array(colour, dtype=np.float32)
    return np.clip(floor[None, None] + p[..., None] ** 0.72 * (peak - floor), 0, 255).astype(np.uint8)


def av_label_rgb(probability: np.ndarray, fov: np.ndarray) -> np.ndarray:
    label_map = probability.argmax(0)
    palette = np.array(
        [
            [8, 17, 29],      # background
            [226, 70, 70],    # artery
            [47, 125, 204],   # vein
            [206, 91, 157],   # crossing
        ],
        dtype=np.uint8,
    )
    rendered = palette[label_map]
    rendered[~fov] = 255
    return rendered


def feature_tensor(rgb: np.ndarray, logits: np.ndarray) -> np.ndarray:
    probability = softmax_np(logits)
    entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(-1) / np.log(4)
    return np.concatenate(
        [
            np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0),
            np.moveaxis(np.clip(logits, -12, 12) / 6.0, -1, 0),
            entropy[None].astype(np.float32),
            hessian_field(rgb),
        ],
        axis=0,
    ).astype(np.float32)


def correct_mass(rgb: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = torch.load(MASS_WEIGHT, map_location="cpu")
    model = Corrector(channels=24, iterations=2).eval()
    model.load_state_dict(checkpoint["model"])
    features = torch.from_numpy(feature_tensor(rgb, logits)[None])
    base = torch.from_numpy(np.moveaxis(logits, -1, 0).copy()[None])
    with torch.inference_mode():
        probability = model(features, base)[0].softmax(0).cpu().numpy()
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "validation_mean_av_dice": checkpoint.get("validation_mean_av_dice"),
    }
    return probability.astype(np.float32), metadata


def occupancy(rgb: np.ndarray) -> np.ndarray:
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).eval()
    checkpoint = torch.load(OCC_WEIGHT, map_location="cpu")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    tensor = (np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0) - mean) / std
    with torch.inference_mode():
        result = model(torch.from_numpy(tensor[None])).sigmoid()[0, 0].cpu().numpy()
    # Replay the cached-occupancy precision used by the final bounded experiment.
    return result.astype(np.float16).astype(np.float32)


def topology_rgb(probability: np.ndarray, fov: np.ndarray) -> np.ndarray:
    vessel = (probability[1:].sum(0) >= 0.5) & fov
    small = np.asarray(
        Image.fromarray(vessel.astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST),
        dtype=bool,
    )
    skeleton = skeletonize(small)
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - skeleton
    branches = skeleton & (neighbours >= 3)
    endpoints = skeleton & (neighbours == 1)
    canvas = np.full((256, 256, 3), [8, 17, 29], dtype=np.uint8)
    canvas[skeleton] = [82, 210, 167]
    canvas[branches] = [206, 91, 157]
    canvas[endpoints] = [244, 174, 35]
    canvas = cv2.resize(canvas, (512, 512), interpolation=cv2.INTER_NEAREST)
    canvas[~fov] = 255
    return canvas


def main() -> None:
    target = PACKAGE / "01_inputs/user_assets"
    style_target = PACKAGE / "01_inputs/source_images"
    generated = PACKAGE / "03_assets_raw/generated_single_assets"
    for folder in [target, style_target, generated]:
        folder.mkdir(parents=True, exist_ok=True)
    if any(generated.iterdir()):
        raise RuntimeError(f"Refusing to overwrite generated assets in {generated}")

    shutil.copy2(STYLE_REFERENCE, style_target / STYLE_REFERENCE.name)
    shutil.copy2(RGB_PATH, target / f"Fundus_AVSeg_{SAMPLE_ID}_RGB.png")
    shutil.copy2(CE_PATH, target / f"Fundus_AVSeg_{SAMPLE_ID}_CE.png")
    for name, source in DATABASE_ORIGINALS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)

    rgb = np.asarray(Image.open(RGB_PATH).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
    ce = np.asarray(Image.open(CE_PATH).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
    logits = np.load(LOGITS_PATH).astype(np.float32)
    logits = cv2.resize(logits, (512, 512), interpolation=cv2.INTER_LINEAR)
    raw = np.moveaxis(softmax_np(logits), -1, 0)
    mass, checkpoint_metadata = correct_mass(rgb, logits)
    occ = occupancy(rgb)
    conditional = raw[1:] / np.maximum(raw[1:].sum(0, keepdims=True), 1e-7)
    joint = np.concatenate([1.0 - occ[None], occ[None] * conditional], axis=0)
    fov = rgb.max(2) > 10

    smooth = gaussian_filter(raw, sigma=(0, 1.0, 1.0), mode="reflect", truncate=4.0)
    smooth /= np.maximum(smooth.sum(0, keepdims=True), 1e-12)
    vessel = (raw[1:].sum(0) >= 0.5) & fov
    components, count = label(vessel, structure=np.ones((3, 3), int))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    remove = (components > 0) & (sizes[components] < 32)
    cleanup = raw.copy()
    cleanup[:, remove] = 0.0
    cleanup[0, remove] = 1.0

    write_png(rgb, generated / "fundus_rgb_512.png")
    write_png(ce, generated / "fundus_ce_512.png")
    channel_colours = {
        "background": (224, 231, 238),
        "artery": (226, 70, 70),
        "vein": (47, 125, 204),
        "crossing": (206, 91, 157),
    }
    for index, (name, colour) in enumerate(channel_colours.items()):
        write_png(probability_rgb(raw[index], colour), generated / f"raw_probability_{name}.png")
    write_png(av_label_rgb(raw, fov), generated / "raw_av_map.png")
    write_png(av_label_rgb(mass, fov), generated / "mass_av_map.png")
    write_png(probability_rgb(occ, (70, 205, 172)), generated / "occupancy_probability.png")
    write_png(av_label_rgb(joint, fov), generated / "joint_av_map.png")
    write_png(av_label_rgb(smooth, fov), generated / "smooth_control_map.png")
    write_png(av_label_rgb(cleanup, fov), generated / "cleanup_control_map.png")
    write_png(topology_rgb(joint, fov), generated / "joint_topology_map.png")

    stats = {
        "purpose": "data-derived visual assets for an architecture figure; not a new experiment",
        "sample_id": SAMPLE_ID,
        "size": 512,
        "fov_rule": "RGB max > 10",
        "raw_probability_shape": list(raw.shape),
        "probability_sum_max_error": {
            "raw": float(np.abs(raw.sum(0) - 1).max()),
            "mass": float(np.abs(mass.sum(0) - 1).max()),
            "joint": float(np.abs(joint.sum(0) - 1).max()),
        },
        "mass_checkpoint_metadata": checkpoint_metadata,
        "readout_semantics": {
            "raw": "softmax of frozen VascX four-class logits",
            "mass": "two residual updates then four-class softmax",
            "joint": "frozen FIVES occupancy multiplied by original/raw VascX conditional identity",
            "smooth": "per-channel Gaussian sigma=1 px then simplex normalization",
            "cleanup": "remove 8-connected raw-vessel components smaller than 32 px",
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [RGB_PATH, CE_PATH, LOGITS_PATH, MASS_WEIGHT, OCC_WEIGHT, *DATABASE_ORIGINALS.values()]
        },
        "database_originals": [
            {
                "copied_as": name,
                "source": str(source.relative_to(ROOT)),
                "sha256": sha256(source),
                "role": (
                    "input-domain example"
                    if name in {
                        "Fundus_AVSeg_001_G_original.png",
                        "FIVES_train_100_A_original.png",
                        "mBRSET_9_3_original.jpg",
                        "HRF_IQA_1_good_original.JPG",
                    }
                    else "evaluation-evidence example"
                ),
            }
            for name, source in DATABASE_ORIGINALS.items()
        ],
    }
    (generated / "asset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"package": str(PACKAGE), "assets": len(list(generated.iterdir())), **stats}, indent=2))


if __name__ == "__main__":
    main()
