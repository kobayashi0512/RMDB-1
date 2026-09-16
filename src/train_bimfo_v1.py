#!/usr/bin/env python3
"""Train collapse-resistant BiMFO V1.1 on canonical bilateral vessel fields."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

QUALITY_COLUMNS = [
    "fov_coverage", "sharpness", "illumination_std", "vessel_density",
    "vessel_signal_q95", "structure_confidence", "uncertain_fraction",
]


def seed_everything(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Cache(Dataset):
    def __init__(self, patients, maps, global_features, quality, quality_mean, quality_std):
        self.patients = patients.reset_index(drop=True)
        self.maps = maps
        self.global_features = global_features
        blocks = []
        for eye in ("left", "right"):
            eye_quality = (
                quality.query("eye == @eye").set_index("cache_index")
                .reindex(self.patients.cache_index)[QUALITY_COLUMNS]
            )
            blocks.append(eye_quality.to_numpy(np.float32))
        raw = np.stack(blocks, axis=1)
        self.quality = np.nan_to_num((raw - quality_mean) / quality_std).astype(np.float32)
        # Predeclared image-level QC: values below both limits correspond to
        # segmentation-empty, clinically unreadable images on manual audit.
        signal = raw[:, :, QUALITY_COLUMNS.index("vessel_signal_q95")]
        density = raw[:, :, QUALITY_COLUMNS.index("vessel_density")]
        self.usable = ((signal >= .5) & (density >= .05)).astype(np.float32)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        row = self.patients.iloc[index]
        return {
            "maps": torch.from_numpy(np.asarray(self.maps[int(row.cache_index)], dtype=np.float32)),
            "global": torch.from_numpy(np.asarray(self.global_features[int(row.cache_index)], dtype=np.float32)),
            "quality": torch.from_numpy(self.quality[index]),
            "usable": torch.from_numpy(self.usable[index]),
            "delta": torch.tensor(row.delta_cimt_mm, dtype=torch.float32),
            "expected": torch.tensor(row.expected_cimt_mm, dtype=torch.float32),
            "cimt": torch.tensor(row.cimt_mm, dtype=torch.float32),
            "risk": torch.tensor(row.accelerated_cimt, dtype=torch.float32),
            "age": torch.tensor(row.age, dtype=torch.float32),
            "id": str(row.patient_id),
        }


class TubeField(nn.Module):
    """Fixed differentiable multiscale directional vessel operator."""
    def __init__(self, directions=12):
        super().__init__()
        angles = torch.arange(directions) * torch.pi / directions
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        sobel_y = sobel_x.t()
        bank = torch.stack([
            torch.cos(angle) * sobel_x + torch.sin(angle) * sobel_y
            for angle in angles
        ]).unsqueeze(1)
        self.register_buffer("bank", bank)

    def forward(self, vessel_probability):
        directional = F.conv2d(vessel_probability, self.bank, padding=1).abs()
        multiscale = [
            F.avg_pool2d(directional, kernel, 1, kernel // 2)
            for kernel in (3, 7, 15, 31)
        ]
        # Preserve density/calibre information that disappears under derivatives.
        return torch.cat([vessel_probability, *multiscale], dim=1)


class SpatialPyramid(nn.Module):
    """Retain coarse anatomy instead of collapsing every field to one global mean."""
    def forward(self, feature):
        return torch.cat([
            F.adaptive_avg_pool2d(feature, size).flatten(1)
            for size in (1, 2, 4)
        ], dim=1)


class BiMFO(nn.Module):
    def __init__(self, global_features=384, quality_features=len(QUALITY_COLUMNS)):
        super().__init__()
        self.field = TubeField()
        self.encoder = nn.Sequential(
            nn.Conv2d(49, 48, 5, 2, 2), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 80, 3, 2, 1), nn.GroupNorm(8, 80), nn.GELU(),
            nn.Conv2d(80, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),
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
            nn.Linear(quality_features, 16), nn.GELU(), nn.Linear(16, 1),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 256), nn.GELU(), nn.Dropout(.2),
            nn.Linear(256, 160), nn.GELU(),
        )
        self.delta = nn.Linear(160, 1)
        self.risk = nn.Linear(160, 1)

    def forward(self, maps, global_features, quality, usable=None):
        fields = [
            self.eye_projector(self.pyramid(self.encoder(self.field(maps[:, eye:eye + 1]))))
            for eye in range(2)
        ]
        eyes = [
            self.eye_fusion(torch.cat([
                fields[eye], self.global_projector(global_features[:, eye])
            ], dim=1))
            for eye in range(2)
        ]
        weights = F.softplus(self.quality_gate(quality).squeeze(-1)) + 0.1
        if usable is not None:
            weights = weights * usable
        common = (
            eyes[0] * weights[:, 0:1] + eyes[1] * weights[:, 1:2]
        ) / weights.sum(1, keepdim=True).clamp_min(1e-6)
        asymmetry = (eyes[0] - eyes[1]).abs()
        hidden = self.fuse(torch.cat([common, asymmetry], dim=1))
        return self.delta(hidden).squeeze(1), self.risk(hidden).squeeze(1), weights


@torch.inference_mode()
def evaluate(model, loader, run_device, target_mean, target_std):
    model.eval()
    risks, probabilities, deltas, predictions, expected, cimt, identifiers, weights = [], [], [], [], [], [], [], []
    for batch in loader:
        pred, risk_logit, gate = model(
            batch["maps"].to(run_device), batch["global"].to(run_device),
            batch["quality"].to(run_device),
            batch["usable"].to(run_device),
        )
        risks.extend(batch["risk"].numpy())
        probabilities.extend(risk_logit.sigmoid().cpu().numpy())
        deltas.extend(batch["delta"].numpy())
        predictions.extend(pred.cpu().numpy() * target_std + target_mean)
        expected.extend(batch["expected"].numpy())
        cimt.extend(batch["cimt"].numpy())
        identifiers.extend(batch["id"])
        weights.extend(gate.cpu().numpy())
    y, probability, target, prediction = map(
        np.asarray, (risks, probabilities, deltas, predictions)
    )
    expected, cimt = np.asarray(expected), np.asarray(cimt)
    cimt_prediction = expected + prediction
    usable_patient = np.asarray(weights).sum(axis=1) > 0
    metrics = {
        "n": len(y),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "mae_delta_mm": float(mean_absolute_error(target, prediction)),
        "rmse_delta_mm": float(mean_squared_error(target, prediction) ** .5),
        "r2_delta": float(r2_score(target, prediction)),
        "mae_cimt_mm": float(mean_absolute_error(cimt, cimt_prediction)),
        "r2_cimt": float(r2_score(cimt, cimt_prediction)),
        "clinical_expected_mae_cimt_mm": float(mean_absolute_error(cimt, expected)),
        "clinical_expected_r2_cimt": float(r2_score(cimt, expected)),
        "prediction_delta_std_mm": float(prediction.std()),
        "risk_probability_std": float(probability.std()),
        "true_delta_std_mm": float(target.std()),
        "prediction_variance_ratio": float(prediction.std() / max(target.std(), 1e-8)),
        "qc_usable_n": int(usable_patient.sum()),
        "qc_coverage": float(usable_patient.mean()),
    }
    if usable_patient.any() and len(np.unique(y[usable_patient])) == 2:
        metrics.update({
            "qc_usable_auroc": float(roc_auc_score(y[usable_patient], probability[usable_patient])),
            "qc_usable_auprc": float(average_precision_score(y[usable_patient], probability[usable_patient])),
            "qc_usable_mae_delta_mm": float(mean_absolute_error(target[usable_patient], prediction[usable_patient])),
            "qc_usable_r2_delta": float(r2_score(target[usable_patient], prediction[usable_patient])),
        })
    output = pd.DataFrame({
        "patient_id": identifiers, "accelerated_cimt": y,
        "probability": probability, "delta_cimt_mm": target,
        "prediction_delta_mm": prediction,
        "expected_cimt_mm": expected, "cimt_mm": cimt,
        "prediction_cimt_mm": cimt_prediction,
        "left_gate": np.asarray(weights)[:, 0], "right_gate": np.asarray(weights)[:, 1],
        "qc_usable": usable_patient.astype(int),
    })
    return metrics, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("artifacts/bimfo_v1_1_cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bimfo_v1_1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_device = device()

    patients = pd.read_csv(args.cache / "patients.csv", dtype={"patient_id": str})
    patients["cache_index"] = np.arange(len(patients))
    quality = pd.read_csv(args.cache / "image_quality.csv")
    missing = sorted(set(QUALITY_COLUMNS) - set(quality.columns))
    if missing:
        raise ValueError(f"Cache lacks V1.1 quality fields: {missing}")
    maps = np.load(args.cache / "vessel_probability_maps.npy", mmap_mode="r")
    global_path = args.cache / "global_features.npy"
    if not global_path.exists():
        raise FileNotFoundError(
            f"{global_path} is required; run extract_bimfo_global_features.py first."
        )
    global_features = np.load(global_path, mmap_mode="r")
    if global_features.shape[:2] != maps.shape[:2]:
        raise ValueError("Global features and vessel maps have inconsistent patient/eye axes.")
    train = patients.query("split == 'train'")
    val = patients.query("split == 'val'")
    train_indices = set(train.cache_index)
    q_train = quality[quality.cache_index.isin(train_indices)][QUALITY_COLUMNS].to_numpy(np.float32)
    quality_mean = q_train.mean(axis=0).reshape(1, 1, -1)
    quality_std = np.maximum(q_train.std(axis=0), 1e-6).reshape(1, 1, -1)
    target_mean = float(train.delta_cimt_mm.mean())
    target_std = float(train.delta_cimt_mm.std())
    cache_metadata = json.loads((args.cache / "metadata.json").read_text())
    threshold_z = (
        float(cache_metadata["accelerated_threshold_train_q75_mm"]) - target_mean
    ) / target_std
    train_cache = Cache(train, maps, global_features, quality, quality_mean, quality_std)
    keep = train_cache.usable.sum(axis=1) > 0
    train = train.loc[keep].copy()
    train_loader = DataLoader(
        Cache(train, maps, global_features, quality, quality_mean, quality_std),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Cache(val, maps, global_features, quality, quality_mean, quality_std),
        batch_size=args.batch_size, num_workers=0,
    )

    model = BiMFO(global_features=global_features.shape[2]).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3, min_lr=1e-6,
    )
    positive = float(train.accelerated_cimt.sum())
    negative = float(len(train) - positive)
    positive_weight = torch.tensor(negative / positive, device=run_device)
    history, best_mae, best_auroc, stale = [], float("inf"), -float("inf"), 0
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {
            "regression_loss": 0., "classification_loss": 0.,
            "concordance_loss": 0., "ranking_loss": 0., "consistency_loss": 0.,
        }
        batches = 0
        for batch in train_loader:
            standardized_target = (batch["delta"].to(run_device) - target_mean) / target_std
            prediction, risk_logit, _ = model(
                batch["maps"].to(run_device), batch["global"].to(run_device),
                batch["quality"].to(run_device),
                batch["usable"].to(run_device),
            )
            regression = F.smooth_l1_loss(prediction, standardized_target, beta=.5)
            pred_centered = prediction - prediction.mean()
            target_centered = standardized_target - standardized_target.mean()
            concordance = 1 - (
                2 * (pred_centered * target_centered).mean()
                / (
                    pred_centered.square().mean()
                    + target_centered.square().mean()
                    + (prediction.mean() - standardized_target.mean()).square()
                    + 1e-6
                )
            )
            target_difference = standardized_target[:, None] - standardized_target[None, :]
            prediction_difference = prediction[:, None] - prediction[None, :]
            pair_mask = target_difference.abs() >= .25
            ranking = (
                F.softplus(-target_difference.sign()[pair_mask] * prediction_difference[pair_mask]).mean()
                if pair_mask.any() else prediction.new_zeros(())
            )
            classification = F.binary_cross_entropy_with_logits(
                risk_logit, batch["risk"].to(run_device), pos_weight=positive_weight
            )
            implied_risk = torch.sigmoid((prediction - threshold_z) / .5)
            consistency = F.mse_loss(risk_logit.sigmoid(), implied_risk.detach())
            # Delta-CIMT is already age/sex residualized with OOF predictions.
            # A second noisy minibatch covariance penalty suppressed all signal.
            loss = (
                regression + .20 * concordance + .10 * ranking
                + .25 * classification + .05 * consistency
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running["regression_loss"] += float(regression.detach())
            running["classification_loss"] += float(classification.detach())
            running["concordance_loss"] += float(concordance.detach())
            running["ranking_loss"] += float(ranking.detach())
            running["consistency_loss"] += float(consistency.detach())
            batches += 1

        metrics, predictions = evaluate(
            model, val_loader, run_device, target_mean, target_std
        )
        scheduler.step(metrics["mae_delta_mm"])
        record = {
            "epoch": epoch,
            **{key: value / batches for key, value in running.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
        if epoch >= 5 and metrics["prediction_variance_ratio"] < .02:
            raise RuntimeError(
                "Mean-prediction collapse detected: validation prediction SD is "
                "below 2% of target SD after five epochs."
            )

        if metrics["mae_delta_mm"] < best_mae - 1e-5:
            best_mae, stale = metrics["mae_delta_mm"], 0
            torch.save({
                "state_dict": model.state_dict(), "delta_mean": target_mean,
                "delta_std": target_std, "quality_mean": quality_mean,
                "quality_std": quality_std,
            }, args.output / "best_regression.pt")
            predictions.to_csv(args.output / "validation_predictions_regression.csv", index=False)
        else:
            stale += 1
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            torch.save({
                "state_dict": model.state_dict(), "delta_mean": target_mean,
                "delta_std": target_std, "quality_mean": quality_mean,
                "quality_std": quality_std,
            }, args.output / "best_classification.pt")
            predictions.to_csv(args.output / "validation_predictions_classification.csv", index=False)
        if stale >= args.patience:
            print(json.dumps({"early_stopping_epoch": epoch, "best_mae": best_mae}))
            break

    (args.output / "config.json").write_text(json.dumps({
        **vars(args), "quality_columns": QUALITY_COLUMNS,
        "positive_weight": float(positive_weight.cpu()), "selection_primary": "validation_mae",
        "accelerated_threshold_z": float(threshold_z),
        "anti_collapse_losses": "smooth_l1_plus_CCC_plus_pairwise_ranking",
        "qc_rule": "vessel_signal_q95>=0.5_and_vessel_density>=0.05_at_least_one_eye",
        "excluded_train_patients_both_eyes_unusable": int((~keep).sum()),
        "classification_checkpoint": "separate_validation_auroc_checkpoint",
        "uncertainty_head": "disabled_until_noncollapsed_mean_model",
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
