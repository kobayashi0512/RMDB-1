#!/usr/bin/env python3
"""Fixed oracle decomposition for the already inspected Fundus-AVSeg set.

No model fitting or threshold selection is performed. The two oracle
replacements are diagnostics: (i) ground-truth vessel support with predicted
identity, and (ii) predicted support with ground-truth identity. They are not
claimed to be additive causal contributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.ndimage import zoom


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data/raw/Fundus_AVSeg/Fundus-AVSeg/images"
ANN = ROOT / "data/raw/Fundus_AVSeg/Fundus-AVSeg/annotation"
LOGITS = ROOT / "data/fundus_avseg_vascx/logits"
META = ROOT / "results/birap_v2_fundus_avseg/per_image_metrics_with_metadata.csv"
OUT = ROOT / "artifacts/fundus_avseg_error_attribution"
SIZE = 768
BOOTSTRAP_REPETITIONS = 3000
BOOTSTRAP_SEED = 20260905


def labels(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[..., 0] > 0, rgb[..., 1] > 0, rgb[..., 2] > 0
    target = np.zeros(red.shape, np.int64)
    target[red & ~blue] = 1
    target[blue & ~red] = 2
    target[red & blue] = 3
    target[green] = -100
    return target


def resize_label(a: np.ndarray) -> np.ndarray:
    # Preserve the ignore code (-100); casting to uint8 would turn it into
    # 156 and incorrectly include unlabeled pixels in AV Dice.
    return cv2.resize(a.astype(np.int64), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def resize_logits(a: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    # Cached logits are H x W x 4. Bilinear interpolation preserves the
    # frozen continuous output before the fixed argmax operation.
    h, w = original_hw
    if a.shape[:2] != (h, w):
        a = np.stack([cv2.resize(a[..., k].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for k in range(4)], axis=-1)
    out = np.stack([cv2.resize(a[..., k].astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for k in range(4)], axis=-1)
    return out.astype(np.float32)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    den = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / den) if den else float("nan")


def av_scores(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    valid = target != -100
    artery_p = np.isin(pred, [1, 3]) & valid
    vein_p = np.isin(pred, [2, 3]) & valid
    artery_t = np.isin(target, [1, 3]) & valid
    vein_t = np.isin(target, [2, 3]) & valid
    a, v = dice(artery_p, artery_t), dice(vein_p, vein_t)
    return a, v, float(np.nanmean([a, v]))


def paired_bootstrap(values: np.ndarray, repetitions: int = BOOTSTRAP_REPETITIONS,
                     seed: int = BOOTSTRAP_SEED) -> dict[str, float | list[float]]:
    """Deterministic paired bootstrap CI for a per-image diagnostic delta."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[index].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def main() -> None:
    metadata = META.parent / "per_image_metrics_with_metadata.csv"
    strata = pd.read_csv(metadata)[["id", "disease type", "image quality"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for image in sorted(IMG.glob("*.png")):
        ident = image.stem
        rgb = np.asarray(Image.open(image).convert("RGB"))
        target_rgb = np.asarray(Image.open(ANN / image.name).convert("RGB"))
        if target_rgb.shape[:2] != rgb.shape[:2]:
            target_rgb = cv2.resize(target_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        target = labels(target_rgb)
        target = resize_label(target)
        target_vessel_rgb = resize_label((target_rgb.max(-1) > 0).astype(np.uint8)).astype(bool)
        logits = resize_logits(np.load(LOGITS / f"{ident}.npy"), rgb.shape[:2])
        prob = np.exp(logits - logits.max(-1, keepdims=True))
        prob /= np.maximum(prob.sum(-1, keepdims=True), 1e-8)
        pred = prob.argmax(-1).astype(np.int64)
        valid = target != -100
        target_vessel = target_vessel_rgb
        pred_vessel = pred > 0
        # Predicted identity conditional on oracle vessel support.
        conditional = prob[..., 1:]
        identity_on_gt_support = conditional.argmax(-1).astype(np.int64) + 1
        identity_oracle_pred = np.where(target_vessel, identity_on_gt_support, 0)
        # Ground-truth identity on predicted support.
        support_oracle_pred = np.where(pred_vessel, target, 0)
        av_actual = av_scores(pred, target)
        av_identity = av_scores(identity_oracle_pred, target)
        av_support = av_scores(support_oracle_pred, target)
        # Match the locked evaluator: vessel Dice uses the full occupancy mask;
        # the ignore code is only applied to artery/vein metrics.
        base_vessel = dice(pred_vessel, target_vessel)
        row = {
            "id": ident,
            "vessel_dice": base_vessel,
            "actual_mean_av_dice": av_actual[2],
            "identity_oracle_mean_av_dice": av_identity[2],
            "support_oracle_mean_av_dice": av_support[2],
            "identity_oracle_gain": av_identity[2] - av_actual[2],
            "support_oracle_gain": av_support[2] - av_actual[2],
            "artery_dice_actual": av_actual[0],
            "vein_dice_actual": av_actual[1],
            "artery_dice_identity_oracle": av_identity[0],
            "vein_dice_identity_oracle": av_identity[1],
            "artery_dice_support_oracle": av_support[0],
            "vein_dice_support_oracle": av_support[1],
        }
        match = strata[strata.id.eq(ident)]
        if len(match):
            row["disease type"] = str(match.iloc[0]["disease type"])
            row["image quality"] = str(match.iloc[0]["image quality"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "oracle_decomposition_per_image.csv", index=False)
    metrics = ["vessel_dice", "actual_mean_av_dice", "identity_oracle_mean_av_dice", "support_oracle_mean_av_dice", "identity_oracle_gain", "support_oracle_gain"]
    # Compact table for manuscript or supplement assembly.
    strata_rows: list[dict[str, object]] = [{
        "stratum": "overall", "level": "all", "n": int(len(frame)),
        **{m: float(frame[m].mean()) for m in metrics}
    }]
    for column in ("disease type", "image quality"):
        for level, group in frame.groupby(column, sort=True):
            strata_rows.append({
                "stratum": column, "level": str(level), "n": int(len(group)),
                **{m: float(group[m].mean()) for m in metrics}
            })
    pd.DataFrame(strata_rows).to_csv(OUT / "oracle_decomposition_strata.csv", index=False)
    report: dict[str, object] = {
        "status": "superseded_due_to_missing_annotation_crop_transform",
        "correction": "../fundus_coordinate_alignment_audit/summary.json",
        "protocol": "fixed_oracle_error_decomposition",
        "images": int(len(frame)),
        "target_labels_used_for_selection": False,
        "cimt_used": False,
        "outcome_driven_tuning": False,
        "note": "Oracle gains are diagnostic and non-additive; interacting support, identity, boundary and domain errors prevent causal percentage attribution.",
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "identity_oracle_gain": paired_bootstrap(frame["identity_oracle_gain"].to_numpy()),
            "support_oracle_gain": paired_bootstrap(frame["support_oracle_gain"].to_numpy()),
        },
        "overall": {m: float(frame[m].mean()) for m in metrics},
        "strata": {},
    }
    for column in ("disease type", "image quality"):
        report["strata"][column] = {
            str(level): {"n": int(len(g)), **{m: float(g[m].mean()) for m in metrics}}
            for level, g in frame.groupby(column, sort=True)
        }
    (OUT / "oracle_decomposition.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
