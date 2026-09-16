#!/usr/bin/env python3
"""Independent PyTorch baselines for bilateral fundus CIMT prediction.

The script deliberately keeps the fusion rule simple. It is the comparator for
later topology- and prototype-aware models, not their implementation.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from reporting import choose_threshold, write_report


@dataclass
class RunConfig:
    mode: str
    backbone: str
    image_size: int
    seed: int
    epochs: int
    learning_rate: float
    regression_weight: float
    freeze_encoder_epochs: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BilateralFundusDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row.left_path) as source:
            left = self.transform(source.convert("RGB"))
        with Image.open(row.right_path) as source:
            right = self.transform(source.convert("RGB"))
        return {
            "left": left,
            "right": right,
            # Age is already normalized by the dataset provider; retain its absolute
            # position rather than normalizing age and sex within each individual.
            "clinical": torch.tensor([row.age_normalized, row.sex], dtype=torch.float32),
            "label": torch.tensor(row.label, dtype=torch.long),
            "cimt": torch.tensor(row.cimt_mm, dtype=torch.float32),
            "patient_id": str(row.patient_id),
            "sex": torch.tensor(row.sex, dtype=torch.long),
            "age": torch.tensor(row.age, dtype=torch.float32),
            "clinical_logit": torch.tensor(row.get("clinical_logit", 0.0), dtype=torch.float32),
            "clinical_cimt": torch.tensor(row.get("clinical_cimt", 0.0), dtype=torch.float32),
        }


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return value.view_as(value)
    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.strength * gradient, None


class CIMTBaseline(nn.Module):
    """Shared-eye encoder and deliberately transparent late fusion."""
    def __init__(self, backbone: str, mode: str, image_size: int, pretrained: bool = True) -> None:
        super().__init__()
        self.mode = mode
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=image_size)
        feature_dim = self.encoder.num_features
        if mode == "single":
            fusion_dim = feature_dim
        else:
            # Mean = bilateral commonality; absolute difference = asymmetry.
            fusion_dim = feature_dim * 2
        if mode == "multimodal":
            self.clinical = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Dropout(0.10))
            fusion_dim += 32
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.25)
        )
        self.classifier = nn.Linear(256, 2)
        self.regressor = nn.Linear(256, 1)
        if mode.startswith("clinical_residual"):
            self.residual_classifier = nn.Linear(256, 1)
            self.residual_regressor = nn.Linear(256, 1)
            nn.init.zeros_(self.residual_classifier.weight); nn.init.zeros_(self.residual_classifier.bias)
            nn.init.zeros_(self.residual_regressor.weight); nn.init.zeros_(self.residual_regressor.bias)
        if mode == "clinical_residual_debiased":
            self.age_adversary = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def forward(self, left: Tensor, right: Tensor, clinical: Tensor, clinical_logit: Tensor | None = None, clinical_cimt: Tensor | None = None, adversarial_strength: float = 0.) -> tuple[Tensor, ...]:
        left_feature, right_feature = self.encode(left), self.encode(right)
        if self.mode == "single":
            feature = (left_feature + right_feature) / 2
        else:
            feature = torch.cat([(left_feature + right_feature) / 2, (left_feature - right_feature).abs()], dim=1)
        if self.mode == "multimodal":
            feature = torch.cat([feature, self.clinical(clinical)], dim=1)
        feature = self.fusion(feature)
        if self.mode.startswith("clinical_residual"):
            logit = clinical_logit + self.residual_classifier(feature).squeeze(1)
            outputs = (torch.stack([torch.zeros_like(logit), logit], dim=1), clinical_cimt + self.residual_regressor(feature).squeeze(1))
            if self.mode == "clinical_residual_debiased":
                return (*outputs, self.age_adversary(GradientReverse.apply(feature, adversarial_strength)).squeeze(1))
            return outputs
        return self.classifier(feature), self.regressor(feature).squeeze(1)


def make_transforms(size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train = transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(0.12, 0.12, 0.08, 0.03), transforms.ToTensor(), normalize,
    ])
    evaluate = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), normalize])
    return train, evaluate


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, run_device: torch.device, cimt_mean: float, cimt_std: float) -> tuple[dict, dict]:
    model.eval()
    labels, probabilities, targets, predictions, patient_ids, ages, sexes = [], [], [], [], [], [], []
    for batch in loader:
        outputs = model(batch["left"].to(run_device), batch["right"].to(run_device), batch["clinical"].to(run_device), batch["clinical_logit"].to(run_device), batch["clinical_cimt"].to(run_device))
        logits, scaled_prediction = outputs[:2]
        labels.extend(batch["label"].numpy())
        probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().numpy())
        targets.extend(batch["cimt"].numpy())
        predictions.extend((scaled_prediction.cpu().numpy() * cimt_std) + cimt_mean)
        patient_ids.extend(batch["patient_id"])
        ages.extend(batch["age"].numpy())
        sexes.extend(batch["sex"].numpy())
    y, score, target, prediction = map(np.asarray, (labels, probabilities, targets, predictions))
    report = {
        "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
        "mae_mm": float(mean_absolute_error(target, prediction)),
        "rmse_mm": float(mean_squared_error(target, prediction) ** 0.5), "r2": float(r2_score(target, prediction)),
    }
    raw = pd.DataFrame({"patient_id": patient_ids, "age": ages, "sex": sexes, "label": y, "probability": score, "cimt_mm": target, "prediction_mm": prediction})
    raw["age_band"] = pd.cut(raw.age, bins=[-np.inf, 39, 49, 59, np.inf], labels=["<40", "40-49", "50-59", "≥60"])
    return report, raw


def train_one_epoch(model, loader, optimizer, run_device, class_weights, cimt_mean, cimt_std, regression_weight, age_adversarial_weight=0., max_batches=None):
    model.train()
    for step, batch in enumerate(tqdm(loader, leave=False), start=1):
        left, right, clinical = (batch[k].to(run_device) for k in ("left", "right", "clinical"))
        label = batch["label"].to(run_device)
        cimt = (batch["cimt"].to(run_device) - cimt_mean) / cimt_std
        outputs = model(left, right, clinical, batch["clinical_logit"].to(run_device), batch["clinical_cimt"].to(run_device), age_adversarial_weight)
        logits, prediction = outputs[:2]
        loss = F.cross_entropy(logits, label, weight=class_weights) + regression_weight * F.smooth_l1_loss(prediction, cimt)
        if len(outputs) == 3:
            loss = loss + age_adversarial_weight * F.mse_loss(outputs[2], clinical[:, 0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if max_batches and step >= max_batches:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--mode", choices=["single", "bilateral", "multimodal", "clinical_residual", "clinical_residual_debiased"], default="bilateral")
    parser.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--regression-weight", type=float, default=0.30)
    parser.add_argument("--age-adversarial-weight", type=float, default=0.05)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0, help="Freeze the pretrained encoder for early screening epochs.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Development-only cap; never use for reported experiments.")
    parser.add_argument("--no-pretrained", action="store_true", help="Smoke-test architecture without downloading backbone weights.")
    parser.add_argument("--evaluate-test", action="store_true", help="Use only after selecting the model on validation.")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_device = device()
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str})
    train, val, test = (frame.query("split == @split") for split in ("train", "val", "test"))
    cimt_mean, cimt_std = train.cimt_mm.mean(), train.cimt_mm.std()
    if args.mode.startswith("clinical_residual"):
        clinical_features = ["age", "sex"]
        scaler = StandardScaler().fit(train[clinical_features])
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed).fit(scaler.transform(train[clinical_features]), train.label)
        regressor = Ridge(alpha=1.0).fit(scaler.transform(train[clinical_features]), train.cimt_mm)
        for subset in (train, val, test):
            features = scaler.transform(subset[clinical_features])
            subset["clinical_logit"] = classifier.decision_function(features)
            subset["clinical_cimt"] = (regressor.predict(features) - cimt_mean) / cimt_std
    train_tf, eval_tf = make_transforms(args.image_size)
    train_loader = DataLoader(BilateralFundusDataset(train, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    val_loader = DataLoader(BilateralFundusDataset(val, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    test_loader = DataLoader(BilateralFundusDataset(test, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    model = CIMTBaseline(args.backbone, args.mode, args.image_size, pretrained=not args.no_pretrained).to(run_device)
    class_counts = train.label.value_counts().reindex([0, 1]).to_numpy()
    class_weights = torch.tensor(class_counts.sum() / (2 * class_counts), dtype=torch.float32, device=run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    artifact_dir = Path("artifacts") / f"{args.mode}_{args.backbone.replace('.', '_')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -float("inf")
    history = []
    config = RunConfig(args.mode, args.backbone, args.image_size, args.seed, args.epochs, args.learning_rate, args.regression_weight, args.freeze_encoder_epochs)
    (artifact_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        encoder_is_frozen = epoch <= args.freeze_encoder_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = not encoder_is_frozen
        train_one_epoch(model, train_loader, optimizer, run_device, class_weights, cimt_mean, cimt_std, args.regression_weight, args.age_adversarial_weight if args.mode == "clinical_residual_debiased" else 0., args.max_train_batches)
        val_metrics, _ = evaluate(model, val_loader, run_device, cimt_mean, cimt_std)
        history.append({"epoch": epoch, "encoder_frozen": encoder_is_frozen, **val_metrics})
        (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch:03d} encoder_frozen={encoder_is_frozen} validation=" + json.dumps(val_metrics))
        if val_metrics["auroc"] > best_auc:
            best_auc = val_metrics["auroc"]
            torch.save({"state_dict": model.state_dict(), "cimt_mean": cimt_mean, "cimt_std": cimt_std, "config": asdict(config)}, artifact_dir / "best.pt")
            (artifact_dir / "best_validation.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    checkpoint = torch.load(artifact_dir / "best.pt", map_location=run_device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    _, validation_raw = evaluate(model, val_loader, run_device, cimt_mean, cimt_std)
    validation_report = write_report(validation_raw, artifact_dir / "validation_report", bootstrap_repetitions=args.bootstrap_repetitions)
    if args.evaluate_test:
        _, raw = evaluate(model, test_loader, run_device, cimt_mean, cimt_std)
        threshold = choose_threshold(validation_raw.label.to_numpy(), validation_raw.probability.to_numpy())
        report = write_report(raw, artifact_dir / "test_report", threshold, args.bootstrap_repetitions)
        print("test=" + json.dumps(report["metrics"]))


if __name__ == "__main__":
    main()
