#!/usr/bin/env python3
"""Pilot uncertainty-gated directional A/V residual corrector.

The worst RAV camera domain is held out entirely. The model only learns residual
logits on top of frozen VascX outputs; no CIMT outcomes are read.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ANALOG = "TOPCON TRC-50VT - Analog Fundus Camera"


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def target_labels(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[..., 0] > 0, rgb[..., 1] > 0, rgb[..., 2] > 0
    target = np.zeros(red.shape, np.int64)
    target[red & ~blue] = 1
    target[blue & ~red] = 2
    target[red & blue] = 3
    target[green] = -100
    return target


def hessian_field(rgb: np.ndarray) -> np.ndarray:
    green = rgb[..., 1].astype(np.float32) / 255.0
    image = cv2.GaussianBlur(1.0 - green, (0, 0), 1.5)
    hxx = cv2.Sobel(image, cv2.CV_32F, 2, 0, ksize=5)
    hyy = cv2.Sobel(image, cv2.CV_32F, 0, 2, ksize=5)
    hxy = cv2.Sobel(image, cv2.CV_32F, 1, 1, ksize=5)
    angle = 0.5 * np.arctan2(2 * hxy, hxx - hyy + 1e-8)
    root = np.sqrt((hxx - hyy) ** 2 + 4 * hxy**2)
    lambda_plus = 0.5 * (hxx + hyy + root)
    lambda_minus = 0.5 * (hxx + hyy - root)
    use_plus = np.abs(lambda_plus) < np.abs(lambda_minus)
    tangent = angle + np.where(use_plus, 0.0, np.pi / 2)
    strength = np.maximum(np.abs(lambda_plus), np.abs(lambda_minus))
    scale = np.quantile(strength, 0.99) + 1e-6
    vesselness = np.clip(strength / scale, 0, 1)
    return np.stack([np.cos(tangent), np.sin(tangent), vesselness], axis=0).astype(np.float32)


def load_case(identifier: str, root: Path, gt_root: Path, crop: int | None = None):
    rgb = np.asarray(Image.open(root / "preprocessed_rgb" / f"RAV_{identifier}.png").convert("RGB"))
    logits = np.load(root / "logits" / f"RAV_{identifier}.npy").astype(np.float32)
    target_rgb = np.asarray(Image.open(gt_root / f"RAV_{identifier}.png").convert("RGB"))
    if crop is None:
        size = 512
        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))
        logits = cv2.resize(logits, (size, size), interpolation=cv2.INTER_LINEAR)
        target_rgb = np.asarray(Image.fromarray(target_rgb).resize((size, size), Image.Resampling.NEAREST))
    else:
        height, width = rgb.shape[:2]
        y = random.randint(0, height - crop)
        x = random.randint(0, width - crop)
        rgb, logits, target_rgb = (
            rgb[y:y + crop, x:x + crop],
            logits[y:y + crop, x:x + crop],
            target_rgb[y:y + crop, x:x + crop],
        )
    probability = softmax_np(logits)
    entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(-1) / np.log(probability.shape[-1])
    direction = hessian_field(rgb)
    image = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
    base = np.moveaxis(np.clip(logits, -12, 12) / 6.0, -1, 0)
    features = np.concatenate([image, base, entropy[None].astype(np.float32), direction], axis=0)
    return torch.from_numpy(features), torch.from_numpy(logits).permute(2, 0, 1), torch.from_numpy(target_labels(target_rgb))


class RAVDataset(Dataset):
    def __init__(self, identifiers, root, gt_root, crop=None, repeats=1):
        self.identifiers, self.root, self.gt_root = list(identifiers), root, gt_root
        self.crop, self.repeats = crop, repeats

    def __len__(self):
        return len(self.identifiers) * self.repeats

    def __getitem__(self, index):
        identifier = self.identifiers[index % len(self.identifiers)]
        features, logits, target = load_case(identifier, self.root, self.gt_root, self.crop)
        return identifier, features, logits, target


class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Corrector(nn.Module):
    def __init__(self, channels=24, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.encoder1 = Block(11 + 4, channels)
        self.encoder2 = Block(channels, channels * 2)
        self.bottleneck = Block(channels * 2, channels * 4)
        self.decoder2 = Block(channels * 6, channels * 2)
        self.decoder1 = Block(channels * 3, channels)
        self.output = nn.Conv2d(channels, 4, 1)

    def step(self, features, probability):
        x1 = self.encoder1(torch.cat([features, probability], 1))
        x2 = self.encoder2(F.max_pool2d(x1, 2))
        z = self.bottleneck(F.max_pool2d(x2, 2))
        z = self.decoder2(torch.cat([F.interpolate(z, x2.shape[-2:], mode="bilinear", align_corners=False), x2], 1))
        z = self.decoder1(torch.cat([F.interpolate(z, x1.shape[-2:], mode="bilinear", align_corners=False), x1], 1))
        return self.output(z)

    def forward(self, features, base_logits):
        refined = base_logits
        for _ in range(self.iterations):
            refined = refined + self.step(features, refined.softmax(1))
        return refined


def soft_dice(probability, target, valid):
    losses = []
    for label in [1, 2, 3]:
        truth = (target == label).float()
        prediction = probability[:, label] * valid
        intersection = (prediction * truth).sum((1, 2))
        losses.append(1 - ((2 * intersection + 1) / (prediction.sum((1, 2)) + truth.sum((1, 2)) + 1)).mean())
    return torch.stack(losses).mean()


def loss_fn(
    refined,
    base,
    features,
    target,
    av_mass_weight: float = 0.0,
    vessel_mass_weight: float = 0.0,
):
    weights = torch.tensor([0.20, 1.0, 1.0, 2.0], device=refined.device)
    ce = F.cross_entropy(refined, target, weight=weights, ignore_index=-100)
    probability = refined.softmax(1)
    valid = (target != -100).float()
    dice = soft_dice(probability, target, valid)
    tx, ty, vesselness = features[:, 8], features[:, 9], features[:, 10]
    entropy = features[:, 7]
    direction_weight = vesselness * (1 - entropy) * valid * (target != 3)
    directional = 0.0
    for label in [1, 2]:
        field = probability[:, label]
        gx = F.pad((field[:, :, 2:] - field[:, :, :-2]) * 0.5, (1, 1, 0, 0))
        gy = F.pad((field[:, 2:, :] - field[:, :-2, :]) * 0.5, (0, 0, 1, 1))
        directional = directional + (direction_weight * (tx * gx + ty * gy).abs()).sum() / (direction_weight.sum() + 1)
    vessel_target = ((target > 0) & (target != -100)).float()
    vessel_probability = 1 - probability[:, 0]
    vessel = F.binary_cross_entropy(vessel_probability[valid.bool()], vessel_target[valid.bool()])
    valid_sum = valid.sum((1, 2)).clamp_min(1)
    predicted_artery_mass = ((probability[:, 1] + probability[:, 3]) * valid).sum((1, 2))
    predicted_vein_mass = ((probability[:, 2] + probability[:, 3]) * valid).sum((1, 2))
    target_artery_mass = (((target == 1) | (target == 3)).float() * valid).sum((1, 2))
    target_vein_mass = (((target == 2) | (target == 3)).float() * valid).sum((1, 2))
    predicted_log_ratio = torch.log((predicted_artery_mass + 1) / (predicted_vein_mass + 1))
    target_log_ratio = torch.log((target_artery_mass + 1) / (target_vein_mass + 1))
    av_mass = F.smooth_l1_loss(predicted_log_ratio, target_log_ratio)
    predicted_vessel_fraction = (vessel_probability * valid).sum((1, 2)) / valid_sum
    target_vessel_fraction = (vessel_target * valid).sum((1, 2)) / valid_sum
    vessel_mass = F.smooth_l1_loss(
        torch.log(predicted_vessel_fraction + 1e-4),
        torch.log(target_vessel_fraction + 1e-4),
    )
    confidence = 1 - entropy
    identity = (confidence[:, None] * (refined - base).square()).mean()
    total = (
        ce + 0.5 * dice + 0.10 * directional + 0.10 * vessel + 0.01 * identity
        + av_mass_weight * av_mass + vessel_mass_weight * vessel_mass
    )
    return total, {
        "ce": ce.item(),
        "dice": dice.item(),
        "direction": float(directional),
        "vessel": vessel.item(),
        "identity": identity.item(),
        "av_mass": av_mass.item(),
        "vessel_mass": vessel_mass.item(),
    }


def dice(binary_prediction, binary_target):
    denominator = binary_prediction.sum() + binary_target.sum()
    return float(2 * (binary_prediction & binary_target).sum() / denominator) if denominator else np.nan


def metrics(prediction, target):
    valid = target != -100
    pa, pv = np.isin(prediction, [1, 3]) & valid, np.isin(prediction, [2, 3]) & valid
    ta, tv = np.isin(target, [1, 3]) & valid, np.isin(target, [2, 3]) & valid
    typed = np.isin(target, [1, 2])
    return {
        "artery_dice": dice(pa, ta),
        "vein_dice": dice(pv, tv),
        "mean_av_dice": np.nanmean([dice(pa, ta), dice(pv, tv)]),
        "type_accuracy": float((prediction[typed] == target[typed]).mean()),
        "av_ratio_log_error": float(abs(np.log((pa.sum() + 1) / (pv.sum() + 1)) - np.log((ta.sum() + 1) / (tv.sum() + 1)))),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for identifiers, features, base, target in loader:
        features, base = features.to(device), base.to(device)
        corrected = model(features, base).argmax(1).cpu().numpy()
        baseline = base.argmax(1).cpu().numpy()
        for index, identifier in enumerate(identifiers):
            truth = target[index].numpy()
            rows.append({"id": identifier, "model": "baseline", **metrics(baseline[index], truth)})
            rows.append({"id": identifier, "model": "udct", **metrics(corrected[index], truth)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rav", type=Path, default=Path("artifacts/rav_vascx_full"))
    parser.add_argument("--ground-truth", type=Path, default=Path("../data/external/RAV/dataset/av_masks"))
    parser.add_argument("--metadata", type=Path, default=Path("../data/external/RAV/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/udct_av_pilot"))
    parser.add_argument("--holdout-device", default=ANALOG)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--av-mass-weight", type=float, default=0.0)
    parser.add_argument("--vessel-mass-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    metadata = pd.read_csv(args.metadata, dtype={"id": str})
    holdout = metadata.loc[metadata.device == args.holdout_device, "id"].tolist()
    development = metadata.loc[metadata.device != args.holdout_device, "id"].tolist()
    random.Random(args.seed).shuffle(development)
    validation, training = development[:20], development[20:]
    split = {
        "training": training, "validation": validation, "holdout": holdout,
        "holdout_device": args.holdout_device, "cimt_data_used": False,
    }
    (args.output / "split.json").write_text(json.dumps(split, indent=2))
    train_loader = DataLoader(RAVDataset(training, args.rav, args.ground_truth, crop=256, repeats=2), batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(RAVDataset(validation, args.rav, args.ground_truth), batch_size=1, num_workers=0)
    holdout_loader = DataLoader(RAVDataset(holdout, args.rav, args.ground_truth), batch_size=1, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = Corrector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history, best = [], -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for _, features, base, target in train_loader:
            features, base, target = features.to(device), base.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            refined = model(features, base)
            loss, parts = loss_fn(
                refined,
                base,
                features,
                target,
                av_mass_weight=args.av_mass_weight,
                vessel_mass_weight=args.vessel_mass_weight,
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item())
        validation_frame = evaluate(model, validation_loader, device)
        validation_score = float(validation_frame.query("model == 'udct'").mean_av_dice.mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mean_av_dice": validation_score}
        history.append(row); (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if validation_score > best:
            best = validation_score
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mean_av_dice": best}, args.output / "best.pt")

    checkpoint = torch.load(args.output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    holdout_frame = evaluate(model, holdout_loader, device)
    holdout_frame.to_csv(args.output / "holdout_metrics.csv", index=False)
    overall = holdout_frame.groupby("model").agg(
        n=("id", "size"), artery_dice=("artery_dice", "mean"), vein_dice=("vein_dice", "mean"),
        mean_av_dice=("mean_av_dice", "mean"), type_accuracy=("type_accuracy", "mean"),
        av_ratio_log_error=("av_ratio_log_error", "mean"),
    ).reset_index()
    overall.to_csv(args.output / "holdout_summary.csv", index=False)
    summary = {
        "best_epoch": int(checkpoint["epoch"]), "best_validation_mean_av_dice": float(best),
        "holdout_device": args.holdout_device, "holdout_images": len(holdout),
        "loss_weights": {
            "av_mass": args.av_mass_weight,
            "vessel_mass": args.vessel_mass_weight,
        },
        "results": overall.to_dict("records"), "cimt_validation_used": False, "cimt_test_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
