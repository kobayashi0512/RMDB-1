#!/usr/bin/env python3
"""Train anatomy-normalized, artery/vein-aware BiMFO V2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from train_bimfo_v1 import device, evaluate, seed_everything
except ModuleNotFoundError:
    from scripts.train_bimfo_v1 import device, evaluate, seed_everything


BASE_QUALITY_COLUMNS = [
    "fov_coverage", "sharpness", "illumination_std", "vessel_density",
    "vessel_signal_q95", "structure_confidence", "uncertain_fraction",
]
QUALITY_COLUMNS = BASE_QUALITY_COLUMNS + [
    "landmark_confidence", "av_density", "av_balance",
]


class AnatomyCache(Dataset):
    def __init__(
        self, patients, vessel_maps, anatomy_labels, global_features,
        quality, landmarks, quality_mean, quality_std,
    ):
        self.patients = patients.reset_index(drop=True)
        self.vessel_maps = vessel_maps
        self.anatomy_labels = anatomy_labels
        self.global_features = global_features
        self.quality_frame = quality.set_index(["cache_index", "eye"])
        self.landmarks = landmarks.set_index(["cache_index", "eye"])
        raw_quality = []
        self.landmark_rows = []
        for row in self.patients.itertuples():
            eye_quality, eye_landmarks = [], []
            for eye in ("left", "right"):
                eye_quality.append(
                    self.quality_frame.loc[(row.cache_index, eye), QUALITY_COLUMNS]
                    .to_numpy(np.float32)
                )
                eye_landmarks.append(self.landmarks.loc[(row.cache_index, eye)])
            raw_quality.append(eye_quality)
            self.landmark_rows.append(eye_landmarks)
        raw = np.asarray(raw_quality, dtype=np.float32)
        self.quality = np.nan_to_num((raw - quality_mean) / quality_std).astype(np.float32)
        signal = raw[:, :, QUALITY_COLUMNS.index("vessel_signal_q95")]
        density = raw[:, :, QUALITY_COLUMNS.index("vessel_density")]
        self.usable = ((signal >= .5) & (density >= .05)).astype(np.float32)
        size = vessel_maps.shape[-1]
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        self.grid_x, self.grid_y = xx, yy

    def __len__(self):
        return len(self.patients)

    def eye_tensor(self, cache_index: int, eye_index: int, landmark) -> np.ndarray:
        probability = np.asarray(
            self.vessel_maps[cache_index, eye_index], dtype=np.float32
        )
        labels = np.asarray(
            self.anatomy_labels[cache_index, eye_index, 0], dtype=np.uint8
        )
        disc = np.asarray(
            self.anatomy_labels[cache_index, eye_index, 1], dtype=np.float32
        )
        vascular = [
            probability,
            (labels == 1).astype(np.float32),
            (labels == 2).astype(np.float32),
            (labels == 3).astype(np.float32),
        ]
        diameter = max(float(landmark.disc_diameter), 1.0)
        confidence = float(landmark.landmark_confidence)
        x_relative = np.clip(
            (self.grid_x - float(landmark.disc_x)) / (4 * diameter), -1, 1
        ) * confidence
        y_relative = np.clip(
            (self.grid_y - float(landmark.disc_y)) / (4 * diameter), -1, 1
        ) * confidence
        radius_relative = np.clip(
            np.hypot(
                self.grid_x - float(landmark.disc_x),
                self.grid_y - float(landmark.disc_y),
            ) / (4 * diameter),
            0, 1,
        ) * confidence
        sigma = max(0.25 * diameter, 2.0)
        fovea = np.exp(-(
            (self.grid_x - float(landmark.fovea_x)) ** 2
            + (self.grid_y - float(landmark.fovea_y)) ** 2
        ) / (2 * sigma * sigma)).astype(np.float32)
        landmark_map = np.full_like(probability, confidence)
        return np.stack([
            *vascular, disc, fovea, x_relative, y_relative,
            radius_relative, landmark_map,
        ]).astype(np.float32)

    def __getitem__(self, index):
        row = self.patients.iloc[index]
        cache_index = int(row.cache_index)
        maps = np.stack([
            self.eye_tensor(cache_index, eye, self.landmark_rows[index][eye])
            for eye in range(2)
        ])
        return {
            "maps": torch.from_numpy(maps),
            "global": torch.from_numpy(np.asarray(
                self.global_features[cache_index], dtype=np.float32
            )),
            "quality": torch.from_numpy(self.quality[index]),
            "usable": torch.from_numpy(self.usable[index]),
            "delta": torch.tensor(row.delta_cimt_mm, dtype=torch.float32),
            "expected": torch.tensor(row.expected_cimt_mm, dtype=torch.float32),
            "cimt": torch.tensor(row.cimt_mm, dtype=torch.float32),
            "risk": torch.tensor(row.accelerated_cimt, dtype=torch.float32),
            "age": torch.tensor(row.age, dtype=torch.float32),
            "id": str(row.patient_id),
        }


class TypeAwareAnatomyField(nn.Module):
    """Fixed multiscale directional fields, separately for vessel classes."""
    def __init__(self, directions: int = 8):
        super().__init__()
        angles = torch.arange(directions) * torch.pi / directions
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        sobel_y = sobel_x.t()
        bank = torch.stack([
            torch.cos(angle) * sobel_x + torch.sin(angle) * sobel_y
            for angle in angles
        ]).unsqueeze(1)
        self.register_buffer("bank", bank)

    def forward(self, maps):
        typed = maps[:, :4]
        responses = []
        for channel in range(4):
            directional = F.conv2d(typed[:, channel:channel + 1], self.bank, padding=1).abs()
            responses.extend([
                F.avg_pool2d(directional, kernel, 1, kernel // 2)
                for kernel in (3, 9, 21)
            ])
        return torch.cat([typed, *responses, maps[:, 4:]], dim=1)


class SpatialPyramid(nn.Module):
    def forward(self, feature):
        return torch.cat([
            F.adaptive_avg_pool2d(feature, size).flatten(1)
            for size in (1, 2, 4)
        ], dim=1)


class BiMFOV2(nn.Module):
    def __init__(self, global_features=384, quality_features=len(QUALITY_COLUMNS)):
        super().__init__()
        self.field = TypeAwareAnatomyField()
        self.encoder = nn.Sequential(
            nn.Conv2d(106, 56, 5, 2, 2), nn.GroupNorm(8, 56), nn.GELU(),
            nn.Conv2d(56, 88, 3, 2, 1), nn.GroupNorm(8, 88), nn.GELU(),
            nn.Conv2d(88, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.pyramid = SpatialPyramid()
        self.eye_projector = nn.Sequential(
            nn.Linear(96 * 21, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.15),
            nn.Linear(256, 192), nn.GELU(),
        )
        self.global_projector = nn.Sequential(
            nn.LayerNorm(global_features), nn.Linear(global_features, 192),
            nn.GELU(), nn.Dropout(.1),
        )
        self.eye_fusion = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 192), nn.GELU(),
        )
        self.quality_gate = nn.Sequential(
            nn.Linear(quality_features, 20), nn.GELU(), nn.Linear(20, 1),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 256), nn.GELU(), nn.Dropout(.2),
            nn.Linear(256, 160), nn.GELU(),
        )
        self.delta = nn.Linear(160, 1)
        self.risk = nn.Linear(160, 1)

    def forward(self, maps, global_features, quality, usable=None):
        fields = [
            self.eye_projector(self.pyramid(self.encoder(self.field(maps[:, eye]))))
            for eye in range(2)
        ]
        eyes = [
            self.eye_fusion(torch.cat([
                fields[eye], self.global_projector(global_features[:, eye])
            ], dim=1))
            for eye in range(2)
        ]
        weights = F.softplus(self.quality_gate(quality).squeeze(-1)) + .1
        if usable is not None:
            weights = weights * usable
        common = (
            eyes[0] * weights[:, 0:1] + eyes[1] * weights[:, 1:2]
        ) / weights.sum(1, keepdim=True).clamp_min(1e-6)
        asymmetry = (eyes[0] - eyes[1]).abs()
        hidden = self.fuse(torch.cat([common, asymmetry], dim=1))
        return self.delta(hidden).squeeze(1), self.risk(hidden).squeeze(1), weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/bimfo_v1_1_cache"))
    parser.add_argument("--anatomy-cache", type=Path, default=Path("artifacts/bimfo_v2_anatomy_cache"))
    parser.add_argument(
        "--patients", type=Path, default=None,
        help="Optional fold manifest. Must retain global cache_index and contain split/train-val labels.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bimfo_v2_pilot5"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_device = device()

    patients_path = args.patients or (args.cache / "patients.csv")
    patients = pd.read_csv(patients_path, dtype={"patient_id": str})
    if "cache_index" not in patients:
        patients["cache_index"] = np.arange(len(patients))
    if patients.cache_index.duplicated().any():
        raise ValueError("Each patient manifest row must have a unique global cache_index.")
    quality = pd.read_csv(args.cache / "image_quality.csv", dtype={"patient_id": str})
    landmarks = pd.read_csv(args.anatomy_cache / "anatomy.csv", dtype={"patient_id": str})
    quality = quality.merge(
        landmarks[[
            "cache_index", "eye", "landmark_confidence",
            "artery_fraction", "vein_fraction", "crossing_fraction",
        ]],
        on=["cache_index", "eye"], how="left", validate="one_to_one",
    )
    quality["av_density"] = (
        quality.artery_fraction + quality.vein_fraction + quality.crossing_fraction
    )
    quality["av_balance"] = (
        (quality.artery_fraction - quality.vein_fraction)
        / (quality.artery_fraction + quality.vein_fraction).clip(lower=1e-6)
    )
    if quality[QUALITY_COLUMNS].isna().any().any():
        raise ValueError("V2 quality/anatomy merge contains missing values.")

    vessel_maps = np.load(args.cache / "vessel_probability_maps.npy", mmap_mode="r")
    anatomy_labels = np.load(args.anatomy_cache / "anatomy_labels.npy", mmap_mode="r")
    global_features = np.load(args.cache / "global_features.npy", mmap_mode="r")
    if not (
        vessel_maps.shape[:2] == anatomy_labels.shape[:2] == global_features.shape[:2]
    ):
        raise ValueError("V2 cache patient/eye axes are inconsistent.")

    train = patients.loc[patients.split.eq(args.train_split)].copy()
    val = patients.loc[patients.split.eq(args.val_split)].copy()
    if not len(train) or not len(val):
        raise ValueError("The requested train/validation split has no patients.")
    train_indices = set(train.cache_index)
    q_train = quality[quality.cache_index.isin(train_indices)][QUALITY_COLUMNS].to_numpy(np.float32)
    quality_mean = q_train.mean(axis=0).reshape(1, 1, -1)
    quality_std = np.maximum(q_train.std(axis=0), 1e-6).reshape(1, 1, -1)
    target_mean = float(train.delta_cimt_mm.mean())
    target_std = float(train.delta_cimt_mm.std())
    threshold_z = (
        float(train.delta_cimt_mm.quantile(.75)) - target_mean
    ) / target_std
    initial_train_cache = AnatomyCache(
        train, vessel_maps, anatomy_labels, global_features,
        quality, landmarks, quality_mean, quality_std,
    )
    keep = initial_train_cache.usable.sum(axis=1) > 0
    train = train.loc[keep].copy()
    train_loader = DataLoader(AnatomyCache(
        train, vessel_maps, anatomy_labels, global_features,
        quality, landmarks, quality_mean, quality_std,
    ), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(AnatomyCache(
        val, vessel_maps, anatomy_labels, global_features,
        quality, landmarks, quality_mean, quality_std,
    ), batch_size=args.batch_size, num_workers=0)

    model = BiMFOV2(global_features=global_features.shape[2]).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    positive = float(train.accelerated_cimt.sum())
    negative = float(len(train) - positive)
    positive_weight = torch.tensor(negative / positive, device=run_device)
    history, best_mae, best_auroc = [], float("inf"), -float("inf")
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {key: 0. for key in [
            "regression_loss", "classification_loss", "concordance_loss",
            "ranking_loss", "consistency_loss",
        ]}
        batches = 0
        for batch in train_loader:
            standardized = (batch["delta"].to(run_device) - target_mean) / target_std
            prediction, risk_logit, _ = model(
                batch["maps"].to(run_device), batch["global"].to(run_device),
                batch["quality"].to(run_device), batch["usable"].to(run_device),
            )
            regression = F.smooth_l1_loss(prediction, standardized, beta=.5)
            pred_centered = prediction - prediction.mean()
            target_centered = standardized - standardized.mean()
            concordance = 1 - (
                2 * (pred_centered * target_centered).mean()
                / (
                    pred_centered.square().mean() + target_centered.square().mean()
                    + (prediction.mean() - standardized.mean()).square() + 1e-6
                )
            )
            target_difference = standardized[:, None] - standardized[None, :]
            prediction_difference = prediction[:, None] - prediction[None, :]
            pair_mask = target_difference.abs() >= .25
            ranking = (
                F.softplus(
                    -target_difference.sign()[pair_mask] * prediction_difference[pair_mask]
                ).mean() if pair_mask.any() else prediction.new_zeros(())
            )
            classification = F.binary_cross_entropy_with_logits(
                risk_logit, batch["risk"].to(run_device), pos_weight=positive_weight,
            )
            implied_risk = torch.sigmoid((prediction - threshold_z) / .5)
            consistency = F.mse_loss(risk_logit.sigmoid(), implied_risk.detach())
            loss = (
                regression + .20 * concordance + .10 * ranking
                + .25 * classification + .05 * consistency
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            for key, value in [
                ("regression_loss", regression),
                ("classification_loss", classification),
                ("concordance_loss", concordance),
                ("ranking_loss", ranking),
                ("consistency_loss", consistency),
            ]:
                running[key] += float(value.detach())
            batches += 1

        metrics, predictions = evaluate(
            model, val_loader, run_device, target_mean, target_std
        )
        record = {
            "epoch": epoch,
            **{key: value / batches for key, value in running.items()},
            **metrics,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if epoch >= 5 and metrics["prediction_variance_ratio"] < .02:
            raise RuntimeError("BiMFO V2 mean-prediction collapse detected.")
        if metrics["mae_delta_mm"] < best_mae:
            best_mae = metrics["mae_delta_mm"]
            torch.save(model.state_dict(), args.output / "best_regression.pt")
            predictions.to_csv(args.output / "validation_predictions_regression.csv", index=False)
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            torch.save(model.state_dict(), args.output / "best_classification.pt")
            predictions.to_csv(args.output / "validation_predictions_classification.csv", index=False)

    (args.output / "config.json").write_text(json.dumps({
        **vars(args), "quality_columns": QUALITY_COLUMNS,
        "map_channels": [
            "binary_probability", "artery", "vein", "crossing", "disc",
            "fovea_heatmap", "disc_relative_x", "disc_relative_y",
            "disc_relative_radius", "landmark_confidence",
        ],
        "disc_failure_policy": "confidence_gated_no_fabricated_mask",
        "test_set_used": False,
        "comparison_target": "bimfo_v1_1_antcollapse_pilot5_same_seed_five_epochs",
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
