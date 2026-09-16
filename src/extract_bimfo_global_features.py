#!/usr/bin/env python3
"""Cache label-free DINOv2 global features from AAIU-canonical bilateral images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torchvision import transforms
from tqdm import tqdm

from prepare_bimfo_v1 import canonicalize_fundus


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/bimfo_v1_1_cache"))
    parser.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    patients = pd.read_csv(args.cache / "patients.csv", dtype={"patient_id": str})
    run_device = device()
    encoder = timm.create_model(
        args.backbone, pretrained=True, num_classes=0, img_size=args.image_size
    ).to(run_device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    output = np.lib.format.open_memmap(
        args.cache / "global_features.npy", mode="w+", dtype=np.float16,
        shape=(len(patients), 2, encoder.num_features),
    )
    pending, locations = [], []
    for row_index, row in enumerate(tqdm(patients.itertuples(), total=len(patients), desc="DINOv2 features")):
        for eye_index, eye in enumerate(("left", "right")):
            raw = np.asarray(Image.open(getattr(row, f"{eye}_path")).convert("RGB"))
            image, _ = canonicalize_fundus(raw, eye, args.image_size)
            pending.append(normalize(transforms.ToTensor()(Image.fromarray(image))))
            locations.append((row_index, eye_index))
        if len(pending) >= args.batch_size or row_index == len(patients) - 1:
            with torch.inference_mode():
                feature = encoder(torch.stack(pending).to(run_device)).cpu().numpy()
            for vector, (patient_index, eye_index) in zip(feature, locations):
                output[patient_index, eye_index] = vector.astype(np.float16)
            pending, locations = [], []
    output.flush()
    (args.cache / "global_features_metadata.json").write_text(json.dumps({
        "backbone": args.backbone, "image_size": args.image_size,
        "shape": list(output.shape), "dtype": "float16",
        "pretraining": "official_timm_DINOv2_label_free_frozen",
        "canonicalization": "same_fov_crop_and_right_eye_mirror_as_vessel_cache",
    }, indent=2))
    print(json.dumps({"shape": list(output.shape), "finite": bool(np.isfinite(output).all())}))


if __name__ == "__main__":
    main()
