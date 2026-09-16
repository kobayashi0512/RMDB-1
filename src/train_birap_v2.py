#!/usr/bin/env python3
"""Outcome-blind RMDB-BiRAP V2 development on FIVES and RAV.

The script deliberately refuses Fundus-AVSeg paths. FIVES supplies vessel
occupancy supervision, while RAV supplies harmonised vessel and A/V identity
supervision. The analogue-camera RAV domain is excluded from model selection.

The model combines four prespecified ideas:

1. native-resolution, multi-scale occupancy training;
2. conditional A/V identity with a crossing class and a VascX bypass;
3. centreline, direction, mass, ratio, and calibration-aligned objectives;
4. a fixed uncertainty-derived route between the joint and mass readouts.

This is a development experiment. Its RAV device holdout is retrospective and
must not be described as a new independent external confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from train_udct_av_pilot import Block, Corrector, hessian_field, target_labels
from backbone_runtime import atomic_checkpoint, atomic_json, progress, restore_rng, rng_state


ANALOG = "TOPCON TRC-50VT - Analog Fundus Camera"
FORBIDDEN_TARGET_TOKENS = ("fundus-avseg", "fundus_avseg", "fundusavseg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def runtime_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbid_target_paths(values: Iterable[Path | None]) -> None:
    for value in values:
        if value is None:
            continue
        lowered = str(value.resolve()).lower()
        if any(token in lowered for token in FORBIDDEN_TARGET_TOKENS):
            raise ValueError(
                "Fundus-AVSeg is a previously inspected external evaluation set and "
                "cannot enter BiRAP V2 development."
            )


def locate_fives(root: Path) -> tuple[Path, Path, Path, Path]:
    """Locate official-style FIVES train/test image and mask directories."""
    image_names = ("Original", "original", "images", "Images")
    mask_names = ("Ground truth", "Ground Truth", "ground_truth", "masks", "Masks")
    candidates = [root, *[p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "fives"]]
    for base in candidates:
        for train_name in ("train", "Train", "training", "Training"):
            train = base / train_name
            if not train.is_dir():
                continue
            train_images = next((train / name for name in image_names if (train / name).is_dir()), None)
            train_masks = next((train / name for name in mask_names if (train / name).is_dir()), None)
            for test_name in ("test", "Test", "testing", "Testing"):
                test = base / test_name
                test_images = next((test / name for name in image_names if (test / name).is_dir()), None)
                test_masks = next((test / name for name in mask_names if (test / name).is_dir()), None)
                if all(x is not None for x in (train_images, train_masks, test_images, test_masks)):
                    return train_images, train_masks, test_images, test_masks
    raise FileNotFoundError(f"Could not locate official-style FIVES folders below {root}")


def paired_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks = {p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}}
    pairs = []
    for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}):
        if image.stem in masks:
            pairs.append((image, masks[image.stem]))
    if not pairs:
        raise FileNotFoundError(f"No paired images and masks in {image_dir} and {mask_dir}")
    return pairs


def image_quality_features(rgb: np.ndarray, base_probability: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    field = (gray > 0.02).astype(np.float32)
    count = max(float(field.sum()), 1.0)
    mean = float((gray * field).sum() / count)
    variance = float((((gray - mean) ** 2) * field).sum() / count)
    contrast = min(math.sqrt(max(variance, 0.0)) / 0.25, 1.0)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = min(float(((laplacian**2) * field).sum() / count) / 0.02, 1.0)
    illumination = 1.0 - min(abs(mean - 0.45) / 0.45, 1.0)
    entropy = -(base_probability * np.log(np.maximum(base_probability, 1e-8))).sum(-1) / np.log(4)
    certainty = 1.0 - float((entropy * field).sum() / count)
    return np.asarray([contrast, sharpness, illumination, certainty], dtype=np.float32)


def geometric_transform(
    rgb: np.ndarray,
    target: np.ndarray,
    vessel: np.ndarray,
    base_logits: np.ndarray,
    crop_size: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    side = min(crop_size, height, width)
    vessel_pixels = np.argwhere(vessel > 0.5)
    if len(vessel_pixels):
        # Outcome blind sampling of thin vessel support.  The local count is
        # a geometry-only proxy; it does not use artery/vein or clinical
        # labels and makes small-calibre segments appear in training crops.
        local_count = cv2.boxFilter(
            (vessel > 0.5).astype(np.float32), -1, (11, 11), normalize=False
        )
        thin_pixels = vessel_pixels[local_count[vessel_pixels[:, 0], vessel_pixels[:, 1]] <= 12]
        if len(thin_pixels) and random.random() < 0.60:
            vessel_pixels = thin_pixels
    if len(vessel_pixels) and random.random() < 0.75:
        cy, cx = vessel_pixels[random.randrange(len(vessel_pixels))]
        y0 = int(np.clip(cy - random.randint(side // 4, 3 * side // 4), 0, height - side))
        x0 = int(np.clip(cx - random.randint(side // 4, 3 * side // 4), 0, width - side))
    else:
        y0 = random.randint(0, max(height - side, 0))
        x0 = random.randint(0, max(width - side, 0))
    rgb = rgb[y0:y0 + side, x0:x0 + side]
    target = target[y0:y0 + side, x0:x0 + side]
    vessel = vessel[y0:y0 + side, x0:x0 + side]
    base_logits = base_logits[y0:y0 + side, x0:x0 + side]
    rgb = cv2.resize(rgb, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    target = cv2.resize(target, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    vessel = cv2.resize(vessel, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    base_logits = cv2.resize(base_logits, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        target = np.ascontiguousarray(target[:, ::-1])
        vessel = np.ascontiguousarray(vessel[:, ::-1])
        base_logits = np.ascontiguousarray(base_logits[:, ::-1])
    if random.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[::-1])
        target = np.ascontiguousarray(target[::-1])
        vessel = np.ascontiguousarray(vessel[::-1])
        base_logits = np.ascontiguousarray(base_logits[::-1])
    return rgb, target, vessel, base_logits


def photometric_transform(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.55, 1.40))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.55, 1.50))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.30, 1.45))
    gamma = random.uniform(0.60, 1.70)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.power(array, gamma)
    image = Image.fromarray(np.uint8(np.clip(array * 255.0, 0, 255)))
    if random.random() < 0.65:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.25, 2.5)))
    if random.random() < 0.55:
        side = random.choice((96, 128, 160, 192, 256, 320))
        image = image.resize((side, side), Image.Resampling.BILINEAR).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
        )
    array = np.asarray(image).astype(np.float32) / 255.0
    if random.random() < 0.55:
        array += np.random.normal(0.0, random.uniform(0.004, 0.035), array.shape).astype(np.float32)
    if random.random() < 0.35:
        # Smooth synthetic lesion occlusions, outcome blind and geometry agnostic.
        yy, xx = np.ogrid[: array.shape[0], : array.shape[1]]
        for _ in range(random.randint(1, 4)):
            cy, cx = random.randrange(array.shape[0]), random.randrange(array.shape[1])
            radius = random.randint(max(4, array.shape[0] // 80), max(8, array.shape[0] // 25))
            alpha = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))[..., None]
            colour = np.asarray(random.choice(((0.35, 0.05, 0.02), (0.95, 0.75, 0.20), (0.55, 0.45, 0.30))))
            array = array * (1 - 0.55 * alpha) + colour * (0.55 * alpha)
    return np.uint8(np.clip(array * 255.0, 0, 255))


class RetinalDevelopmentDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        train: bool,
        output_size: int,
        crop_sizes: tuple[int, ...],
    ):
        self.records = records
        self.train = train
        self.output_size = output_size
        self.crop_sizes = crop_sizes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        rgb = np.asarray(Image.open(record["image"]).convert("RGB"))
        if record["source"] == "rav":
            target_rgb = np.asarray(Image.open(record["mask"]).convert("RGB"))
            if target_rgb.shape[:2] != rgb.shape[:2]:
                target_rgb = cv2.resize(target_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            if record.get("label_convention") == "fundus_avseg":
                # Fundus green is a crossing, unlike the RAV colour convention.
                from evaluate_fundus_avseg_external import target_labels as fundus_labels
                target = fundus_labels(target_rgb)
            else:
                target = target_labels(target_rgb)
            vessel = (target_rgb.max(-1) > 0).astype(np.float32)
            base_logits = np.load(record["logits"]).astype(np.float32)
            if base_logits.shape[:2] != rgb.shape[:2]:
                base_logits = cv2.resize(base_logits, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            has_identity = 1.0
        else:
            mask = np.asarray(Image.open(record["mask"]).convert("L"))
            if mask.shape != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            vessel = (mask > 127).astype(np.float32)
            target = np.full(vessel.shape, -100, dtype=np.int64)
            base_logits = np.zeros((*vessel.shape, 4), dtype=np.float32)
            has_identity = 0.0

        if self.train:
            crop_size = random.choice(self.crop_sizes)
            rgb, target, vessel, base_logits = geometric_transform(
                rgb, target, vessel, base_logits, crop_size, self.output_size
            )
            rgb = photometric_transform(rgb)
        else:
            rgb = cv2.resize(rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(target, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            vessel = cv2.resize(vessel, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            base_logits = cv2.resize(base_logits, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)

        base_probability = torch.from_numpy(base_logits).softmax(-1).numpy()
        direction = hessian_field(rgb)
        quality = image_quality_features(rgb, base_probability)
        rgb_tensor = torch.from_numpy(np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0).copy())
        normalized = (rgb_tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        return {
            "id": record["id"],
            "source": record["source"],
            "rgb": rgb_tensor,
            "normalized": normalized,
            "base_logits": torch.from_numpy(np.moveaxis(base_logits, -1, 0).copy()),
            "direction": torch.from_numpy(direction.copy()),
            "target": torch.from_numpy(target.astype(np.int64).copy()),
            "vessel": torch.from_numpy(vessel[None].astype(np.float32).copy()),
            "quality": torch.from_numpy(quality),
            "has_identity": torch.tensor(has_identity, dtype=torch.float32),
        }


class IdentityRouter(nn.Module):
    """Light residual identity head plus a per-pixel VascX occupancy bypass."""
    def __init__(self, input_channels: int = 16, channels: int = 24, residual_logit: bool = False):
        super().__init__()
        self.encoder1 = Block(input_channels, channels)
        self.encoder2 = Block(channels, channels * 2)
        self.bottleneck = Block(channels * 2, channels * 4)
        self.decoder2 = Block(channels * 6, channels * 2)
        self.decoder1 = Block(channels * 3, channels)
        self.output = nn.Conv2d(channels, 4, 1)
        # The V2 head is a residual correction, not a replacement for the
        # frozen identity estimator.  Starting at zero preserves VascX before
        # any development data can justify a correction.  The fourth channel
        # is the occupancy bypass and starts in favour of the frozen path.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if not residual_logit:
            nn.init.constant_(self.output.bias[3], 2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder1(features)
        x2 = self.encoder2(F.max_pool2d(x1, 2))
        z = self.bottleneck(F.max_pool2d(x2, 2))
        z = self.decoder2(torch.cat([F.interpolate(z, x2.shape[-2:], mode="bilinear", align_corners=False), x2], 1))
        z = self.decoder1(torch.cat([F.interpolate(z, x1.shape[-2:], mode="bilinear", align_corners=False), x1], 1))
        return self.output(z)


class ConvNormAct(nn.Module):
    """Compact convolution block shared by the candidate occupancy heads."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CandidateOccupancy(nn.Module):
    """Predeclared trainable occupancy backbones for the fair comparison."""
    def __init__(self, name: str = "highres"):
        super().__init__()
        self.name = name
        in_channels = 6 if name in {"dualstream", "direction"} else 3
        self.stem = ConvNormAct(in_channels, 32)
        self.enc1 = ConvNormAct(32, 64, stride=2)
        self.enc2 = ConvNormAct(64, 128, stride=2)
        self.enc3 = ConvNormAct(128, 192, stride=2)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(192, 192, 1), nn.Sigmoid()
        ) if name == "global_local" else None
        self.dec2 = ConvNormAct(192 + 128, 128)
        self.dec1 = ConvNormAct(128 + 64, 64)
        self.dec0 = ConvNormAct(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(
        self,
        normalized: torch.Tensor,
        rgb: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.name == "dualstream":
            green = normalized[:, 1:2]
            blur = F.avg_pool2d(green, kernel_size=9, stride=1, padding=4)
            enhanced = (green - blur + 0.5).clamp(-3, 3)
            x = torch.cat([normalized[:, :3], enhanced.repeat(1, 3, 1, 1)], 1)
        elif self.name == "direction":
            if direction is None:
                direction = normalized.new_zeros((normalized.shape[0], 3, normalized.shape[2], normalized.shape[3]))
            x = torch.cat([normalized[:, :3], direction], 1)
        else:
            x = normalized
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        z = self.enc3(s2)
        if self.global_gate is not None:
            z = z * (1.0 + self.global_gate(z))
        z = self.dec2(torch.cat([F.interpolate(z, s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        z = self.dec1(torch.cat([F.interpolate(z, s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        z = self.dec0(torch.cat([F.interpolate(z, s0.shape[-2:], mode="bilinear", align_corners=False), s0], 1))
        return self.head(z)


class BaselineOccupancy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)

    def forward(self, normalized: torch.Tensor, rgb=None, direction=None) -> torch.Tensor:
        return self.net(normalized)


def build_occupancy_backbone(name: str) -> nn.Module:
    if name == "baseline":
        return BaselineOccupancy()
    return CandidateOccupancy(name)


class BiRAPV2(nn.Module):
    def __init__(
        self,
        occupancy_checkpoint: Path | None,
        mass_checkpoint: Path | None,
        route_center: float,
        route_temperature: float,
        residual_logit: bool = False,
        residual_scale: float = 0.1,
        backbone: str = "baseline",
    ):
        super().__init__()
        self.backbone = backbone
        self.occupancy = build_occupancy_backbone(backbone)
        if occupancy_checkpoint is not None and backbone == "baseline":
            checkpoint = torch.load(occupancy_checkpoint, map_location="cpu")
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.occupancy.net.load_state_dict(state)
        self.residual_logit = residual_logit
        self.residual_scale = residual_scale
        self.identity_router = IdentityRouter(residual_logit=residual_logit)
        self.mass_corrector = Corrector()
        self.mass_available = mass_checkpoint is not None
        if mass_checkpoint is not None:
            checkpoint = torch.load(mass_checkpoint, map_location="cpu")
            self.mass_corrector.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        for parameter in self.mass_corrector.parameters():
            parameter.requires_grad_(False)
        self.mass_corrector.eval()
        self.route_center = route_center
        self.route_temperature = route_temperature

    def train(self, mode: bool = True):
        super().train(mode)
        self.mass_corrector.eval()
        return self

    def forward(self, batch: dict, route_enabled: bool = True, identity_enabled: bool = True) -> dict:
        rgb = batch["rgb"]
        normalized = batch["normalized"]
        base_logits = batch["base_logits"]
        direction = batch["direction"]
        quality = batch["quality"]
        learned_occupancy = self.occupancy(normalized, rgb, direction).sigmoid()
        base_probability = base_logits.softmax(1)
        base_entropy = -(base_probability * torch.log(base_probability.clamp_min(1e-8))).sum(1, keepdim=True) / math.log(4)
        base_occupancy = 1 - base_probability[:, :1]
        scaled_logits = base_logits.clamp(-12, 12) / 6.0
        identity_features = torch.cat(
            [rgb, scaled_logits, base_probability, base_entropy, direction, learned_occupancy], 1
        )
        identity_output = self.identity_router(identity_features)
        base_conditional = base_probability[:, 1:4]
        base_conditional = base_conditional / base_conditional.sum(1, keepdim=True).clamp_min(1e-6)
        conditional_logits = torch.log(base_conditional.clamp_min(1e-6))
        if identity_enabled:
            correction_scale = self.residual_scale if self.residual_logit else 1.0
            conditional_logits = conditional_logits + correction_scale * identity_output[:, :3]
        conditional = conditional_logits.softmax(1)
        bypass = identity_output[:, 3:4].sigmoid()
        identity_mask = batch["has_identity"][:, None, None, None]
        if self.residual_logit:
            base_occupancy_logit = torch.logit(base_occupancy.clamp(1e-5, 1 - 1e-5))
            residual_delta = self.residual_scale * identity_output[:, 3:4]
            routed_occupancy = (base_occupancy_logit + residual_delta).sigmoid()
            # FIVES has occupancy labels but no A/V identity logits; it must
            # never inherit the neutral VascX background prior.
            fused_occupancy = identity_mask * routed_occupancy + (1 - identity_mask) * learned_occupancy
        else:
            residual_delta = None
            routed_occupancy = bypass * base_occupancy + (1 - bypass) * learned_occupancy
            fused_occupancy = identity_mask * routed_occupancy + (1 - identity_mask) * learned_occupancy
        joint = torch.cat([1 - fused_occupancy, fused_occupancy * conditional], 1)

        mass_features = torch.cat([rgb, scaled_logits, base_entropy, direction], 1)
        with torch.no_grad():
            mass = self.mass_corrector(mass_features, base_logits).softmax(1) if self.mass_available else base_probability

        occupancy_entropy = -(
            learned_occupancy * torch.log(learned_occupancy.clamp_min(1e-8))
            + (1 - learned_occupancy) * torch.log((1 - learned_occupancy).clamp_min(1e-8))
        ) / math.log(2)
        identity_entropy = -(conditional * torch.log(conditional.clamp_min(1e-8))).sum(1, keepdim=True) / math.log(3)
        pixel_certainty = 1 - (base_entropy + occupancy_entropy + identity_entropy) / 3
        fixed_quality = quality.mean(1, keepdim=True)[:, :, None, None]
        route_score = 0.55 * pixel_certainty.mean((2, 3), keepdim=True) + 0.45 * fixed_quality
        route = torch.sigmoid((route_score - self.route_center) / self.route_temperature)
        final = route * joint + (1 - route) * mass if route_enabled else joint
        final = final / final.sum(1, keepdim=True).clamp_min(1e-8)
        return {
            "final": final,
            "joint": joint,
            "mass": mass,
            "learned_occupancy": learned_occupancy,
            "fused_occupancy": fused_occupancy,
            "conditional": conditional,
            "bypass": bypass,
            "route": route,
            "base_entropy": base_entropy,
            "base_probability": base_probability,
            "residual_delta": residual_delta,
        }


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return -F.max_pool2d(-image, (3, 3), stride=1, padding=1)
    p1 = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, (3, 3), stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_skeleton = soft_skeleton(probability)
    target_skeleton = soft_skeleton(target)
    precision = (prediction_skeleton * target).sum((1, 2, 3)) / (prediction_skeleton.sum((1, 2, 3)) + 1)
    sensitivity = (target_skeleton * probability).sum((1, 2, 3)) / (target_skeleton.sum((1, 2, 3)) + 1)
    return 1 - (2 * precision * sensitivity / (precision + sensitivity + 1e-6)).mean()


def soft_binary_dice(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum((1, 2, 3))
    return 1 - ((2 * intersection + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


@dataclass
class LossWeights:
    occupancy_bce: float = 1.0
    occupancy_dice: float = 1.0
    cldice: float = 0.35
    identity: float = 1.0
    av_dice: float = 0.5
    direction: float = 0.10
    branch: float = 0.15
    vessel_mass: float = 0.05
    av_mass: float = 0.20
    calibration: float = 0.10
    bypass: float = 0.01
    anchor: float = 0.0


def compute_loss(
    outputs: dict,
    batch: dict,
    weights: LossWeights,
    topology_enabled: bool,
    phenotype_enabled: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    vessel_target = batch["vessel"]
    learned_occupancy = outputs["learned_occupancy"]
    occupancy_bce = F.binary_cross_entropy(learned_occupancy, vessel_target)
    occupancy_dice = soft_binary_dice(learned_occupancy, vessel_target)
    # Vessel continuity is a core occupancy objective, not an identity-only
    # ablation; keep it active from the first experiment onward.
    cldice = soft_cldice(learned_occupancy, vessel_target)

    final = outputs["final"]
    target = batch["target"]
    has_identity = batch["has_identity"] > 0.5
    valid_identity = (target != -100) & has_identity[:, None, None]
    if valid_identity.any():
        identity = F.nll_loss(torch.log(final.clamp_min(1e-8)), target, ignore_index=-100)
        artery_target = ((target == 1) | (target == 3)).float()[:, None]
        vein_target = ((target == 2) | (target == 3)).float()[:, None]
        valid = valid_identity.float()[:, None]
        artery_probability = (final[:, 1] + final[:, 3])[:, None] * valid
        vein_probability = (final[:, 2] + final[:, 3])[:, None] * valid
        av_dice = 0.5 * (
            soft_binary_dice(artery_probability, artery_target * valid)
            + soft_binary_dice(vein_probability, vein_target * valid)
        )
        if topology_enabled:
            tx, ty, vesselness = batch["direction"][:, 0], batch["direction"][:, 1], batch["direction"][:, 2]
            conditional_entropy = -(outputs["conditional"] * torch.log(outputs["conditional"].clamp_min(1e-8))).sum(1) / math.log(3)
            direction_weight = vesselness * (1 - conditional_entropy) * valid_identity.float() * (target != 3)
            direction_loss = final.new_zeros(())
            for field in (artery_probability[:, 0], vein_probability[:, 0]):
                gx = F.pad((field[:, :, 2:] - field[:, :, :-2]) * 0.5, (1, 1, 0, 0))
                gy = F.pad((field[:, 2:, :] - field[:, :-2, :]) * 0.5, (0, 0, 1, 1))
                direction_loss = direction_loss + (
                    direction_weight * (tx * gx + ty * gy).abs()
                ).sum() / (direction_weight.sum() + 1)
            branch = 0.5 * (
                soft_cldice(artery_probability, artery_target * valid)
                + soft_cldice(vein_probability, vein_target * valid)
            )
        else:
            direction_loss = final.new_zeros(())
            branch = final.new_zeros(())

        valid_sum = valid.sum((1, 2, 3)).clamp_min(1)
        predicted_vessel_fraction = ((1 - final[:, :1]) * valid).sum((1, 2, 3)) / valid_sum
        target_vessel_fraction = (vessel_target * valid).sum((1, 2, 3)) / valid_sum
        vessel_mass = F.smooth_l1_loss(
            torch.log(predicted_vessel_fraction + 1e-4),
            torch.log(target_vessel_fraction + 1e-4),
        ) if phenotype_enabled else final.new_zeros(())
        predicted_artery = (artery_probability * valid).sum((1, 2, 3))
        predicted_vein = (vein_probability * valid).sum((1, 2, 3))
        target_artery = (artery_target * valid).sum((1, 2, 3))
        target_vein = (vein_target * valid).sum((1, 2, 3))
        av_mass = F.smooth_l1_loss(
            torch.log((predicted_artery + 1) / (predicted_vein + 1)),
            torch.log((target_artery + 1) / (target_vein + 1)),
        ) if phenotype_enabled else final.new_zeros(())
        one_hot = F.one_hot(target.clamp_min(0), 4).permute(0, 3, 1, 2).float()
        calibration = (((final - one_hot) ** 2) * valid).sum() / (valid.sum() * 4 + 1)
    else:
        zero = final.new_zeros(())
        identity = av_dice = direction_loss = branch = vessel_mass = av_mass = calibration = zero

    # Preserve the frozen VascX path where its identity distribution is
    # confident. The regularizer is evaluated only for RAV samples because
    # FIVES has no VascX identity logits.
    identity_sample = has_identity.float()[:, None, None, None]
    confident_base = (1 - outputs["base_entropy"]).detach()
    if outputs.get("residual_delta") is not None:
        bypass = (
            outputs["residual_delta"].abs() * confident_base * identity_sample
        ).sum() / ((confident_base * identity_sample).sum() + 1)
    else:
        bypass = (
            (1 - outputs["bypass"]) * confident_base * identity_sample
        ).sum() / ((confident_base * identity_sample).sum() + 1)
    anchor = (
        ((outputs["final"] - outputs["base_probability"].detach()) ** 2)
        * confident_base * identity_sample
    ).sum() / ((confident_base * identity_sample).sum() * outputs["final"].shape[1] + 1)
    total = (
        weights.occupancy_bce * occupancy_bce
        + weights.occupancy_dice * occupancy_dice
        + weights.cldice * cldice
        + weights.identity * identity
        + weights.av_dice * av_dice
        + weights.direction * direction_loss
        + weights.branch * branch
        + weights.vessel_mass * vessel_mass
        + weights.av_mass * av_mass
        + weights.calibration * calibration
        + weights.bypass * bypass
        + weights.anchor * anchor
    )
    parts = {
        "occupancy_bce": float(occupancy_bce.detach()),
        "occupancy_dice": float(occupancy_dice.detach()),
        "cldice": float(cldice.detach()),
        "identity": float(identity.detach()),
        "av_dice": float(av_dice.detach()),
        "direction": float(direction_loss.detach()),
        "branch": float(branch.detach()),
        "vessel_mass": float(vessel_mass.detach()),
        "av_mass": float(av_mass.detach()),
        "calibration": float(calibration.detach()),
        "bypass": float(bypass.detach()),
        "anchor": float(anchor.detach()),
    }
    return total, parts


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = prediction.sum() + target.sum()
    return float(2 * (prediction & target).sum() / denominator) if denominator else float("nan")


def case_metrics(probability: np.ndarray, target: np.ndarray, vessel_target: np.ndarray) -> dict:
    prediction = probability.argmax(0)
    vessel_prediction = prediction > 0
    result = {"vessel_dice": binary_dice(vessel_prediction, vessel_target > 0.5)}
    if np.all(target == -100):
        return result
    valid = target != -100
    pa = np.isin(prediction, (1, 3)) & valid
    pv = np.isin(prediction, (2, 3)) & valid
    ta = np.isin(target, (1, 3)) & valid
    tv = np.isin(target, (2, 3)) & valid
    typed = np.isin(target, (1, 2))
    artery_dice = binary_dice(pa, ta)
    vein_dice = binary_dice(pv, tv)
    result.update({
        "artery_dice": artery_dice,
        "vein_dice": vein_dice,
        "mean_av_dice": float(np.nanmean([artery_dice, vein_dice])),
        "type_accuracy": float((prediction[typed] == target[typed]).mean()),
        "av_ratio_log_error": float(abs(
            np.log((pa.sum() + 1) / (pv.sum() + 1))
            - np.log((ta.sum() + 1) / (tv.sum() + 1))
        )),
        "brier": float(np.mean(
            (probability[:, valid]
             - np.moveaxis(np.eye(4)[target[valid]], -1, 0)) ** 2
        )),
    })
    return result


@torch.inference_mode()
def evaluate(
    model: BiRAPV2,
    loader: DataLoader,
    device: torch.device,
    route_enabled: bool,
    identity_enabled: bool,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        ids, sources = batch.pop("id"), batch.pop("source")
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch, route_enabled=route_enabled, identity_enabled=identity_enabled)
        for index, identifier in enumerate(ids):
            if sources[index] == "fives":
                # FIVES supplies occupancy supervision only.  Its development
                # Dice must therefore audit the occupancy head, not an A/V
                # readout whose frozen VascX logits are intentionally absent.
                occupancy = outputs["learned_occupancy"][index].cpu().numpy()
                probability = np.concatenate(
                    [1 - occupancy, occupancy, np.zeros_like(occupancy), np.zeros_like(occupancy)],
                    axis=0,
                )
            else:
                probability = outputs["final"][index].cpu().numpy()
            row = {
                "id": identifier,
                "source": sources[index],
                "route": float(outputs["route"][index].mean().cpu()),
                "bypass": float(outputs["bypass"][index].mean().cpu()),
                **case_metrics(
                    probability,
                    batch["target"][index].cpu().numpy(),
                    batch["vessel"][index, 0].cpu().numpy(),
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> dict:
    result = {"n": int(frame.id.nunique())}
    for column in (
        "vessel_dice", "artery_dice", "vein_dice", "mean_av_dice",
        "type_accuracy", "av_ratio_log_error", "brier", "route", "bypass",
    ):
        if column in frame and frame[column].notna().any():
            result[column] = float(frame[column].mean())
    return result


def make_records(args: argparse.Namespace) -> tuple[dict[str, list[dict]], dict]:
    metadata = pd.read_csv(args.rav_metadata, dtype={"id": str})
    holdout_ids = metadata.loc[metadata.device == args.holdout_device, "id"].tolist()
    development_ids = metadata.loc[metadata.device != args.holdout_device, "id"].tolist()
    random.Random(args.seed).shuffle(development_ids)
    rav_validation_ids = development_ids[: args.rav_validation]
    rav_training_ids = development_ids[args.rav_validation :]

    def rav_records(identifiers: list[str]) -> list[dict]:
        records = []
        for identifier in identifiers:
            image = args.rav / "preprocessed_rgb" / f"RAV_{identifier}.png"
            mask = args.rav_ground_truth / f"RAV_{identifier}.png"
            logits = args.rav / "logits" / f"RAV_{identifier}.npy"
            if not all(path.is_file() for path in (image, mask, logits)):
                raise FileNotFoundError(f"Incomplete RAV V2 case {identifier}: {image}, {mask}, {logits}")
            records.append({"id": f"RAV_{identifier}", "source": "rav", "image": image, "mask": mask, "logits": logits})
        return records

    train_images, train_masks, test_images, test_masks = locate_fives(args.fives)
    fives_pairs = paired_files(train_images, train_masks)
    random.Random(args.seed).shuffle(fives_pairs)
    fives_validation_pairs = fives_pairs[: args.fives_validation]
    fives_training_pairs = fives_pairs[args.fives_validation :]
    # Official FIVES test remains untouched. Paths are recorded only for audit.
    official_test_pairs = paired_files(test_images, test_masks)

    def fives_records(pairs: list[tuple[Path, Path]]) -> list[dict]:
        return [
            {"id": f"FIVES_{image.stem}", "source": "fives", "image": image, "mask": mask, "logits": None}
            for image, mask in pairs
        ]

    records = {
        "rav_training": rav_records(rav_training_ids),
        "rav_validation": rav_records(rav_validation_ids),
        "rav_holdout": rav_records(holdout_ids),
        "fives_training": fives_records(fives_training_pairs),
        "fives_validation": fives_records(fives_validation_pairs),
    }
    split = {
        "seed": args.seed,
        "holdout_device": args.holdout_device,
        "rav_training": rav_training_ids,
        "rav_validation": rav_validation_ids,
        "rav_holdout": holdout_ids,
        "fives_training": [image.name for image, _ in fives_training_pairs],
        "fives_validation": [image.name for image, _ in fives_validation_pairs],
        "fives_official_test_count": len(official_test_pairs),
        "fives_official_test_used": False,
        "fundus_avseg_used": False,
        "cimt_used": False,
        "mbrset_outcomes_used": False,
    }
    return records, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rav", type=Path, default=Path("artifacts/v2/rav_vascx"))
    parser.add_argument("--rav-ground-truth", type=Path, default=Path("restored_assets/RAV/data/external/RAV/dataset/av_masks"))
    parser.add_argument("--rav-metadata", type=Path, default=Path("restored_assets/RAV/data/external/RAV/metadata.csv"))
    parser.add_argument("--fives", type=Path, default=Path("restored_assets/FIVES"))
    parser.add_argument("--occupancy-checkpoint", type=Path, default=Path("restored_assets/models/baseline/artifacts/fives_unet_resnet34/best.pt"))
    parser.add_argument("--mass-checkpoint", type=Path, default=Path("restored_assets/models/baseline/artifacts/udct_av_mass_calibrated/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "occupancy", "joint", "topology", "phenotype", "full",
            "residual_joint", "residual_topology", "residual_phenotype", "residual_full",
        ),
        default="full",
    )
    parser.add_argument(
        "--backbone",
        choices=("baseline", "highres", "global_local", "dualstream", "direction"),
        default="baseline",
        help="Occupancy backbone candidate; shared identity/router architecture and loss stay unchanged.",
    )
    parser.add_argument("--holdout-device", default=ANALOG)
    parser.add_argument("--rav-validation", type=int, default=30)
    parser.add_argument("--fives-validation", type=int, default=120)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=768)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=(384, 512, 640, 768))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--route-center", type=float, default=0.55)
    parser.add_argument("--route-temperature", type=float, default=0.15)
    parser.add_argument("--residual-logit", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--managed-run", action="store_true",
                        help="Save full optimizer/RNG checkpoints at every completed epoch.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Checkpoint/resume verification only.")
    args = parser.parse_args()

    forbid_target_paths((args.rav, args.rav_ground_truth, args.rav_metadata, args.fives, args.output))
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and not args.managed_run:
        raise ValueError("--resume requires --managed-run")
    if args.managed_run and any((args.output / name).exists() for name in ("history.json", "last.pt", "best.pt")) and not args.resume:
        raise ValueError("Existing history found. Use --resume; never overwrite an old run.")
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    if args.managed_run:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    records, split = make_records(args)
    if args.limit_train is not None:
        records["rav_training"] = records["rav_training"][: args.limit_train]
        records["fives_training"] = records["fives_training"][: args.limit_train]
    if args.limit_validation is not None:
        records["rav_validation"] = records["rav_validation"][: args.limit_validation]
        records["fives_validation"] = records["fives_validation"][: args.limit_validation]
        records["rav_holdout"] = records["rav_holdout"][: args.limit_validation]
    split_path = args.output / "split.json"
    if args.resume and json.loads(split_path.read_text()) != split:
        raise ValueError("Resume split mismatch")
    atomic_json(split_path, split)

    train_rav = RetinalDevelopmentDataset(records["rav_training"], True, args.train_size, tuple(args.crop_sizes))
    train_fives = RetinalDevelopmentDataset(records["fives_training"], True, args.train_size, tuple(args.crop_sizes))
    train_dataset = ConcatDataset([train_rav, train_fives])
    rav_weight = 0.5 / max(len(train_rav), 1)
    fives_weight = 0.5 / max(len(train_fives), 1)
    sampler = WeightedRandomSampler(
        [rav_weight] * len(train_rav) + [fives_weight] * len(train_fives),
        num_samples=2 * max(len(train_rav), len(train_fives)), replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=False,
    )
    rav_validation_loader = DataLoader(
        RetinalDevelopmentDataset(records["rav_validation"], False, args.eval_size, tuple(args.crop_sizes)),
        batch_size=1, num_workers=args.workers,
    )
    fives_validation_loader = DataLoader(
        RetinalDevelopmentDataset(records["fives_validation"], False, args.eval_size, tuple(args.crop_sizes)),
        batch_size=1, num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        RetinalDevelopmentDataset(records["rav_holdout"], False, args.eval_size, tuple(args.crop_sizes)),
        batch_size=1, num_workers=args.workers,
    )

    device = runtime_device()
    model = BiRAPV2(
        args.occupancy_checkpoint if args.occupancy_checkpoint.is_file() else None,
        args.mass_checkpoint if args.mass_checkpoint.is_file() else None,
        args.route_center, args.route_temperature,
        residual_logit=(args.residual_logit or args.variant.startswith("residual_")),
        residual_scale=args.residual_scale,
        backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )
    residual_variant = args.variant.startswith("residual_")
    base_variant = args.variant.replace("residual_", "")
    topology_enabled = base_variant in {"topology", "phenotype", "full"}
    phenotype_enabled = base_variant in {"phenotype", "full"}
    route_enabled = base_variant == "full"
    identity_enabled = base_variant != "occupancy"
    weights = LossWeights()
    if not topology_enabled:
        # Centreline continuity belongs to the high-resolution occupancy
        # objective in every ablation. Direction and within-branch identity
        # consistency enter only at the topology stage.
        weights.direction = weights.branch = 0.0
    if not phenotype_enabled:
        weights.vessel_mass = weights.av_mass = 0.0
    if base_variant == "occupancy":
        # Keep the first ablation genuinely occupancy only.
        weights.identity = weights.av_dice = weights.calibration = weights.bypass = 0.0
    weights.anchor = args.anchor_weight if residual_variant else 0.0
    history, best_score = [], -np.inf
    first_epoch, best_payload, accumulated_seconds = 1, None, 0.0
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k not in {"resume", "stop_after_epoch", "output"}}
    configuration["crop_sizes"] = list(configuration["crop_sizes"])
    def save_last(completed_epoch, seconds):
        atomic_checkpoint(args.output / "last.pt", {
            "format_version": 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": rng_state(),
            "epoch": completed_epoch, "history": history, "best_score": best_score,
            "best_payload": best_payload, "configuration": configuration,
            "split": split, "training_seconds": seconds,
        })
    if args.resume:
        checkpoint = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["configuration"] != configuration or checkpoint["split"] != split:
            raise ValueError("Resume configuration or split mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_score = checkpoint["history"], checkpoint["best_score"]
        best_payload = checkpoint["best_payload"]
        first_epoch = checkpoint["epoch"] + 1
        accumulated_seconds = checkpoint["training_seconds"]
        restore_rng(checkpoint["rng"])
        atomic_json(args.output / "history.json", history)
        if best_payload is not None:
            atomic_checkpoint(args.output / "best.pt", best_payload)
        print(json.dumps({"event": "resumed", "completed_epoch": first_epoch - 1}), flush=True)
    elif args.managed_run:
        save_last(0, 0.0)
        atomic_json(args.output / "configuration.json", configuration)
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        losses, part_rows = [], []
        epoch_start = time.monotonic()
        for step, batch in enumerate(train_loader, 1):
            if stopping:
                progress(args.output, phase="paused", completed_epoch=epoch - 1,
                         partial_epoch=epoch, partial_steps=step - 1,
                         resume_note="Resume from last complete epoch; partial epoch is replayed.")
                raise SystemExit(75)
            batch.pop("id"); batch.pop("source")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, route_enabled=route_enabled, identity_enabled=identity_enabled)
            loss, parts = compute_loss(
                outputs, batch, weights,
                topology_enabled=topology_enabled,
                phenotype_enabled=phenotype_enabled,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite training loss in epoch {epoch}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            part_rows.append(parts)
            if args.managed_run and (step == 1 or step % 10 == 0):
                progress(args.output, phase="training", completed_epoch=epoch - 1,
                         epoch=epoch, step=step, steps_per_epoch=len(train_loader),
                         train_loss=float(loss.detach()), elapsed_seconds=time.monotonic() - started)
        if args.managed_run:
            progress(args.output, phase="validation", completed_epoch=epoch - 1, epoch=epoch)
        rav_frame = evaluate(model, rav_validation_loader, device, route_enabled, identity_enabled)
        fives_frame = evaluate(model, fives_validation_loader, device, route_enabled, identity_enabled)
        rav_summary, fives_summary = aggregate(rav_frame), aggregate(fives_frame)
        rav_score = rav_summary.get("mean_av_dice", rav_summary["vessel_dice"])
        ratio_penalty = rav_summary.get("av_ratio_log_error", 0.0)
        selection_score = rav_score + 0.25 * fives_summary["vessel_dice"] - 0.10 * ratio_penalty
        if not math.isfinite(selection_score):
            raise FloatingPointError("Nonfinite validation selection score")
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": float(selection_score),
            "rav_validation": rav_summary,
            "fives_validation": fives_summary,
            "loss_parts": pd.DataFrame(part_rows).mean().to_dict(),
            "epoch_seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        atomic_json(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
        if selection_score > best_score:
            best_score = selection_score
            best_payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "selection_score": float(selection_score),
                "variant": args.variant,
                "backbone": args.backbone,
                "residual_logit": bool(args.residual_logit or residual_variant),
                "residual_scale": args.residual_scale,
                "anchor_weight": float(weights.anchor),
                "route_center": args.route_center,
                "route_temperature": args.route_temperature,
            }
            # CPU copy prevents later optimizer steps mutating the best snapshot.
            from backbone_runtime import cpu_copy
            best_payload = cpu_copy(best_payload)
            atomic_checkpoint(args.output / "best.pt", best_payload)
        if args.managed_run:
            save_last(epoch, accumulated_seconds + time.monotonic() - started)
            progress(args.output, phase="epoch_complete", completed_epoch=epoch,
                     epoch_seconds=row["epoch_seconds"])
            if stopping or (args.stop_after_epoch is not None and epoch >= args.stop_after_epoch):
                progress(args.output, phase="paused", completed_epoch=epoch)
                raise SystemExit(75)

    checkpoint = torch.load(args.output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    rav_frame = evaluate(model, rav_validation_loader, device, route_enabled, identity_enabled)
    fives_frame = evaluate(model, fives_validation_loader, device, route_enabled, identity_enabled)
    rav_frame.to_csv(args.output / "rav_validation_metrics.csv", index=False)
    fives_frame.to_csv(args.output / "fives_validation_metrics.csv", index=False)
    result = {
        "variant": args.variant,
        "backbone": args.backbone,
        "completed_epochs": len(history),
        "seed": args.seed,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "occupancy_parameters": sum(p.numel() for p in model.occupancy.parameters()),
        "training_seconds": accumulated_seconds + time.monotonic() - started,
        "best_epoch": int(checkpoint["epoch"]),
        "best_selection_score": float(checkpoint["selection_score"]),
        "rav_validation": aggregate(rav_frame),
        "fives_validation": aggregate(fives_frame),
        "development_boundaries": {
            "fundus_avseg_used": False,
            "cimt_used": False,
            "mbrset_outcomes_used": False,
            "fives_official_test_used": False,
            "rav_device_holdout_used_for_selection": False,
        },
        "configuration": {
            "train_size": args.train_size,
            "eval_size": args.eval_size,
            "crop_sizes": args.crop_sizes,
            "route_center": args.route_center,
            "route_temperature": args.route_temperature,
            "residual_logit": bool(args.residual_logit or residual_variant),
            "residual_scale": args.residual_scale,
            "anchor_weight": float(weights.anchor),
            "backbone": args.backbone,
            "loss_weights": asdict(weights),
            "occupancy_checkpoint_sha256": sha256(args.occupancy_checkpoint) if args.occupancy_checkpoint.is_file() else None,
            "mass_checkpoint_sha256": sha256(args.mass_checkpoint) if args.mass_checkpoint.is_file() else None,
        },
    }
    if not args.skip_holdout:
        holdout_frame = evaluate(model, holdout_loader, device, route_enabled, identity_enabled)
        holdout_frame.to_csv(args.output / "rav_device_holdout_metrics.csv", index=False)
        result["rav_device_holdout_retrospective"] = aggregate(holdout_frame)
        result["rav_device_holdout_warning"] = (
            "This seven-image domain has been examined in previous pilots and is retrospective, not new external confirmation."
        )
    atomic_json(args.output / "summary.json", result)
    if args.managed_run:
        progress(args.output, phase="complete", completed_epoch=len(history))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
