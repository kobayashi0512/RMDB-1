#!/usr/bin/env python3
"""Domain-randomized vessel-occupancy adaptation with device holdout."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

ANALOG = "TOPCON TRC-50VT - Analog Fundus Camera"


class RAVVessel(Dataset):
    def __init__(self, identifiers, images, masks, size=512, augment=False):
        self.identifiers, self.images, self.masks = list(identifiers), images, masks
        self.size, self.augment = size, augment
        self.normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        identifier = self.identifiers[index]
        image = Image.open(self.images / f"RAV_{identifier}.png").convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask_rgb = np.asarray(Image.open(self.masks / f"RAV_{identifier}.png").convert("RGB").resize((self.size, self.size), Image.Resampling.NEAREST))
        mask = Image.fromarray(((mask_rgb.max(-1) > 0) * 255).astype(np.uint8))
        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            image = TF.adjust_brightness(image, random.uniform(0.55, 1.35))
            image = TF.adjust_contrast(image, random.uniform(0.55, 1.45))
            image = TF.adjust_saturation(image, random.uniform(0.25, 1.35))
            image = TF.adjust_gamma(image, random.uniform(0.65, 1.65))
            if random.random() < 0.65:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 2.2)))
            if random.random() < 0.65:
                reduced = random.choice([128, 160, 192, 256, 320, 384])
                image = image.resize((reduced, reduced), Image.Resampling.BILINEAR).resize((self.size, self.size), Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(image)
            if random.random() < 0.5:
                tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.005, 0.035)).clamp(0, 1)
        else:
            tensor = TF.to_tensor(image)
        return identifier, self.normalize(tensor), (torch.from_numpy(np.asarray(mask, dtype=np.float32).copy()) / 255.0)[None]


def dice_from_logits(logits, target, threshold=0.5):
    prediction = logits.sigmoid() >= threshold
    target = target > 0.5
    return (2 * (prediction & target).sum((1, 2, 3)) + 1) / (prediction.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); rows = []
    for identifiers, image, target in loader:
        logits = model(image.to(device)).cpu()
        values = dice_from_logits(logits, target)
        for identifier, value in zip(identifiers, values):
            rows.append({"id": identifier, "vessel_dice": float(value)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rav", type=Path, default=Path("artifacts/rav_vascx_full"))
    parser.add_argument("--ground-truth", type=Path, default=Path("../data/external/RAV/dataset/av_masks"))
    parser.add_argument("--metadata", type=Path, default=Path("../data/external/RAV/metadata.csv"))
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("artifacts/fives_unet_resnet34/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/factorized_occupancy"))
    parser.add_argument("--holdout-device", default=ANALOG)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    metadata = pd.read_csv(args.metadata, dtype={"id": str})
    holdout = metadata.loc[metadata.device == args.holdout_device, "id"].tolist()
    development = metadata.loc[metadata.device != args.holdout_device, "id"].tolist()
    random.Random(args.seed).shuffle(development)
    validation, training = development[:20], development[20:]
    split = {"training": training, "validation": validation, "holdout": holdout, "holdout_device": args.holdout_device}
    (args.output / "split.json").write_text(json.dumps(split, indent=2))

    train_loader = DataLoader(RAVVessel(training, args.rav / "preprocessed_rgb", args.ground_truth, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(RAVVessel(validation, args.rav / "preprocessed_rgb", args.ground_truth), batch_size=args.batch_size, num_workers=0)
    holdout_loader = DataLoader(RAVVessel(holdout, args.rav / "preprocessed_rgb", args.ground_truth), batch_size=1, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(args.initial_checkpoint, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    history, best = [], -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for _, image, target in train_loader:
            image, target = image.to(device), target.to(device)
            logits = model(image)
            soft_dice = (2 * (logits.sigmoid() * target).sum((1, 2, 3)) + 1) / (logits.sigmoid().sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
            loss = bce(logits, target) + 1 - soft_dice.mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item())
        validation_frame = evaluate(model, validation_loader, device)
        score = float(validation_frame.vessel_dice.mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_vessel_dice": score}
        history.append(row); (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_vessel_dice": best}, args.output / "best.pt")

    checkpoint = torch.load(args.output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    holdout_frame = evaluate(model, holdout_loader, device)
    holdout_frame.to_csv(args.output / "holdout_metrics.csv", index=False)
    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_vessel_dice": float(checkpoint["validation_vessel_dice"]),
        "holdout_device": args.holdout_device,
        "holdout_images": len(holdout),
        "holdout_vessel_dice_mean": float(holdout_frame.vessel_dice.mean()),
        "holdout_vessel_dice_median": float(holdout_frame.vessel_dice.median()),
        "cimt_validation_used": False,
        "cimt_test_used": False,
        "fives_official_test_used_in_this_experiment": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
