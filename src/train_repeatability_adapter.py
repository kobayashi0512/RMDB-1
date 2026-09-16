#!/usr/bin/env python3
"""Train an outcome-blind paired-view reliability adapter on mBRSET features.

The adapter is deliberately small and deterministic at inference.  It learns
to reduce same-eye nuisance variation while preserving the marginal feature
scale and a negative-pair separation margin, preventing a constant-output
solution.  Training uses only the patient/eye/view grouping and never reads
clinical outcomes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


FEATURES = [
    "joint_vessel_mass", "joint_artery_mass", "joint_vein_mass",
    "joint_av_ratio", "joint_vessel_density", "joint_skeleton_density",
    "joint_branch_density", "joint_component_count", "joint_identity_entropy",
]


class Adapter(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension + 4, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(), nn.Linear(32, dimension),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat([x, quality], dim=-1))
        return x + 0.25 * torch.tanh(residual)


def standardize(frame: pd.DataFrame, features: list[str], fit_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = frame.loc[frame.patient_id.isin(fit_ids), features].to_numpy(float)
    mean = np.nanmedian(fit, axis=0)
    scale = np.nanpercentile(fit, 75, axis=0) - np.nanpercentile(fit, 25, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    values = frame[features].to_numpy(float)
    values = np.where(np.isfinite(values), values, mean[None])
    return (values - mean[None]) / scale[None], mean, scale


def groups(frame: pd.DataFrame, indices: np.ndarray) -> list[np.ndarray]:
    subset = frame.iloc[indices]
    return [group.index.to_numpy() for _, group in subset.groupby(["patient_id", "eye"], sort=False) if len(group) >= 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--force-cpu", action="store_true", help="Use CPU for the small feature adapter to avoid per-step accelerator sync overhead.")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    frame = pd.read_csv(args.features, dtype={"patient_id": str, "split": str, "eye": str})
    split = pd.read_csv(args.split, dtype={"patient_id": str, "split": str})
    if any(feature not in frame for feature in FEATURES):
        missing = [feature for feature in FEATURES if feature not in frame]
        raise KeyError(f"Missing features: {missing}")
    frame = frame.merge(split[["patient_id", "split"]].drop_duplicates(), on="patient_id", suffixes=("", "_split"))
    if frame["split"].ne(frame["split_split"]).any():
        raise AssertionError("Feature and split manifests disagree.")
    frame.drop(columns=["split_split"], inplace=True)
    train_patients = split.loc[split.split.eq("train"), "patient_id"].to_numpy(str)
    x, mean, scale = standardize(frame, FEATURES, train_patients)
    quality_cols = ["occupancy_mean", "occupancy_support"]
    quality = frame[quality_cols].to_numpy(float)
    quality = np.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
    quality = np.concatenate([quality, np.zeros((len(frame), 2), dtype=float)], axis=1)
    train_idx = np.flatnonzero(frame.split.eq("train").to_numpy())
    train_groups = groups(frame, train_idx)
    if len(train_groups) < 2:
        raise ValueError("Not enough repeated eyes for paired training.")
    device = torch.device("cpu" if args.force_cpu else ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    model = Adapter(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    q_tensor = torch.tensor(quality, dtype=torch.float32, device=device)
    history = []
    baseline_distances = []
    for _ in range(min(1000, len(train_groups) * 2)):
        a, b = random.sample(train_groups, 2)
        baseline_distances.append(float(np.linalg.norm(x[a].mean(0) - x[b].mean(0))))
    margin = float(np.quantile(baseline_distances, 0.25)) if baseline_distances else 1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_groups)
        pair_losses = []
        for eye_group in train_groups:
            a_idx, b_idx = random.sample(list(eye_group), 2)
            negative = random.choice(train_groups)
            n_idx = random.choice(list(negative))
            za = model(x_tensor[a_idx:a_idx + 1], q_tensor[a_idx:a_idx + 1])
            zb = model(x_tensor[b_idx:b_idx + 1], q_tensor[b_idx:b_idx + 1])
            zn = model(x_tensor[n_idx:n_idx + 1], q_tensor[n_idx:n_idx + 1])
            consistency = F.smooth_l1_loss(za, zb)
            negative_separation = F.relu(margin - torch.linalg.vector_norm(za - zn))
            anchor = 0.10 * (F.mse_loss(za, x_tensor[a_idx:a_idx + 1]) + F.mse_loss(zb, x_tensor[b_idx:b_idx + 1]))
            loss = consistency + 0.30 * negative_separation + anchor
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            pair_losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, "loss": float(np.mean(pair_losses)), "margin": margin})
    model.eval()
    with torch.inference_mode():
        adapted = model(x_tensor, q_tensor).cpu().numpy()
    # Restore original units and write only derived features plus provenance.
    output = frame[["id", "patient_id", "eye", "view", "split"]].copy()
    for index, feature in enumerate(FEATURES):
        output[f"adapted_{feature[6:]}"] = adapted[:, index] * scale[index] + mean[index]
    output["adapter_residual_norm"] = np.linalg.norm(adapted - x, axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "adapted_features.csv", index=False)
    torch.save({"model": model.state_dict(), "features": FEATURES, "mean": mean, "scale": scale, "margin": margin}, args.output / "adapter.pt")
    report = {
        "images": int(len(output)), "patients": int(output.patient_id.nunique()),
        "train_repeat_eyes": int(len(train_groups)), "features": FEATURES,
        "epochs": args.epochs, "seed": args.seed, "outcome_blind": True,
        "clinical_outcomes_used": False, "cimt_used": False,
        "history": history,
    }
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
