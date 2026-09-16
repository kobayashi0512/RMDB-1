#!/usr/bin/env python3
"""Export frozen vessel-occupancy probabilities for an image folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--manifest", type=Path,
        help="Optional patient manifest. When set, only IDs from --split are exported.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in args.images.glob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if args.manifest is not None:
        manifest = pd.read_csv(
            args.manifest,
            usecols=["patient_id", "split"],
            dtype={"patient_id": str},
        )
        patient_ids = set(
            manifest.loc[manifest["split"].eq(args.split), "patient_id"].astype(str)
        )
        files = [
            path for path in files
            if path.stem.rsplit("_", 1)[0] in patient_ids
        ]
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    completed = sum((args.output / f"{path.stem}.npy").is_file() for path in files)
    with torch.inference_mode():
        pending = [
            path for path in files
            if not (args.output / f"{path.stem}.npy").is_file()
        ]
        for start in range(0, len(pending), args.batch_size):
            batch_paths = pending[start:start + args.batch_size]
            if not batch_paths:
                continue
            batch = torch.stack([
                normalize(
                    Image.open(path).convert("RGB").resize(
                        (args.size, args.size), Image.Resampling.BILINEAR
                    )
                )
                for path in batch_paths
            ]).to(device)
            probabilities = model(batch).sigmoid()[:, 0].cpu().numpy()
            for path, probability in zip(batch_paths, probabilities):
                destination = args.output / f"{path.stem}.npy"
                temporary = destination.with_suffix(".tmp.npy")
                np.save(temporary, probability.astype(np.float16))
                temporary.replace(destination)
                completed += 1
            if completed % 40 < len(batch_paths):
                print(f"completed={completed}/{len(files)}", flush=True)
    print(f"complete images={len(files)} probabilities={len(list(args.output.glob('*.npy')))}")


if __name__ == "__main__":
    main()
