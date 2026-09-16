#!/usr/bin/env python3
"""Restartable VascX A/V inference for a directory of fundus images."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rtnls_fundusprep.cli import _run_preprocessing
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.ensembles.predict_output import decollate_predict_full, restore_array_to_preprocessed
from PIL import Image


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def chunks(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("../third_party/vascx_weights/av_july24.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--preprocess-jobs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-logits", action="store_true", help="Also save ensemble-mean pre-softmax logits.")
    parser.add_argument("--disable-tta", action="store_true", help="Disable test-time flips for a faster deterministic pass.")
    parser.add_argument("--inference-patch-size", type=int, help="Override the sliding-window patch size (square).")
    parser.add_argument("--inference-overlap", type=float, help="Override sliding-window overlap.")
    args = parser.parse_args()

    files = sorted(path for path in args.images.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm", ".tif", ".tiff"})
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No images found in {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, ce_dir, av_dir, logits_dir = (
        args.output / name for name in ("preprocessed_rgb", "preprocessed_ce", "artery_vein", "logits")
    )
    for directory in (rgb_dir, ce_dir, av_dir, logits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    missing_preprocess = [
        path for path in files
        if not (rgb_dir / f"{path.stem}.png").is_file() or not (ce_dir / f"{path.stem}.png").is_file()
    ]
    if missing_preprocess:
        _run_preprocessing(
            files=missing_preprocess,
            ids=[path.stem for path in missing_preprocess],
            rgb_path=rgb_dir,
            ce_path=ce_dir,
            bounds_path=args.output / "bounds_latest_batch.csv",
            n_jobs=args.preprocess_jobs,
        )

    missing = [
        path for path in files
        if not (av_dir / f"{path.stem}.png").is_file()
        or (args.save_logits and not (logits_dir / f"{path.stem}.npy").is_file())
    ]
    if missing:
        runtime_device = device()
        model = SegmentationEnsemble.from_torchscript(args.weights).to(runtime_device).eval()
        if args.disable_tta:
            model.config.setdefault("inference", {})["tta"] = False
        if args.inference_patch_size:
            model.config.setdefault("inference", {})["tracing_input_size"] = [args.inference_patch_size, args.inference_patch_size]
        if args.inference_overlap is not None:
            model.config.setdefault("inference", {})["overlap"] = args.inference_overlap
        for group in chunks(missing, args.chunk_size):
            entries = [
                {
                    "id": path.stem,
                    "image": str(rgb_dir / f"{path.stem}.png"),
                    "contrast_enhanced": str(ce_dir / f"{path.stem}.png"),
                }
                for path in group
            ]
            hard_entries = [entry for entry in entries if not (av_dir / f"{entry['id']}.png").is_file()]
            # One forward pass writes both the hard mask and logits. This
            # avoids the previous duplicate sliding-window inference pass.
            need_mask_ids = {entry["id"] for entry in hard_entries}
            need_logit_ids = {entry["id"] for entry in entries if not (logits_dir / f"{entry['id']}.npy").is_file()}
            if need_mask_ids or (args.save_logits and need_logit_ids):
                dataloader = model._make_inference_dataloader(
                    {"images": entries}, num_workers=0,
                    preprocess=False, batch_size=args.batch_size,
                )
                for infer_batch in dataloader:
                    full = model.predict_step_full(infer_batch)
                    for item in decollate_predict_full(full):
                        identifier = str(item["id"])
                        if identifier not in need_mask_ids and identifier not in need_logit_ids:
                            continue
                        aggregate = restore_array_to_preprocessed(
                            np.asarray(item["aggregate"]), item["geometry"], "bilinear"
                        )
                        if identifier in need_mask_ids:
                            Image.fromarray(np.argmax(aggregate, axis=-1).astype(np.uint8)).save(
                                av_dir / f"{identifier}.png"
                            )
                        if args.save_logits and identifier in need_logit_ids:
                            np.save(logits_dir / f"{identifier}.npy", np.asarray(item["logits"], dtype=np.float16))
            if runtime_device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif runtime_device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"complete images={len(files)} predictions={len(list(av_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
