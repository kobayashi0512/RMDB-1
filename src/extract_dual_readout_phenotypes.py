#!/usr/bin/env python3
"""Extract locked, label-blind dual-readout retinal vascular phenotypes.

This program deliberately never reads age, sex, CIMT, or the binary outcome.
It operates on one manifest split, writes one atomic JSON per eye for safe
resumption, and only then creates eye- and patient-level tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from train_udct_av_pilot import Corrector, hessian_field, softmax_np


EPSILON = 1e-8


def normalized_binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, EPSILON, 1 - EPSILON)
    return -(
        probability * np.log(probability)
        + (1 - probability) * np.log(1 - probability)
    ) / np.log(2)


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def safe_fraction(mask: np.ndarray, denominator: np.ndarray) -> float:
    total = int(denominator.sum())
    return float((mask & denominator).sum() / total) if total else float("nan")


def log_ratio(artery_mass: float, vein_mass: float) -> float:
    return float(np.log((artery_mass + 1.0) / (vein_mass + 1.0)))


def anatomy_masks(prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "vessel": prediction > 0,
        "artery": np.isin(prediction, [1, 3]),
        "vein": np.isin(prediction, [2, 3]),
        "crossing": prediction == 3,
    }


def topology(mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    mask = mask & valid
    valid_pixels = max(int(valid.sum()), 1)
    skeleton = skeletonize(mask)
    skeleton_pixels = int(skeleton.sum())
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
    ) - skeleton.astype(np.uint8)
    endpoints = skeleton & (neighbor_count == 1)
    branch_pixels = skeleton & (neighbor_count >= 3)
    branch_clusters = ndimage.label(branch_pixels)[1]
    components = ndimage.label(skeleton)[1]
    return {
        "skeleton_length_density": float(skeleton_pixels / valid_pixels),
        "endpoint_density": float(endpoints.sum() / valid_pixels),
        "branch_cluster_density": float(branch_clusters / valid_pixels),
        "component_density": float(components / valid_pixels),
    }


def hard_phenotypes(
    prediction: np.ndarray,
    valid: np.ndarray,
    zone_b: np.ndarray | None,
    prefix: str,
) -> dict[str, float]:
    masks = anatomy_masks(prediction)
    result: dict[str, float] = {}
    for name, mask in masks.items():
        result[f"{prefix}_{name}_fraction"] = safe_fraction(mask, valid)
    result[f"{prefix}_av_log_area_ratio"] = log_ratio(
        float((masks["artery"] & valid).sum()),
        float((masks["vein"] & valid).sum()),
    )
    vessel_topology = topology(masks["vessel"], valid)
    for name, value in vessel_topology.items():
        result[f"{prefix}_vessel_{name}"] = value
    for kind in ("artery", "vein"):
        result[f"{prefix}_{kind}_skeleton_length_density"] = topology(
            masks[kind], valid
        )["skeleton_length_density"]
    for kind in ("vessel", "artery", "vein"):
        key = f"{prefix}_zone_b_{kind}"
        if zone_b is None:
            result[f"{key}_fraction"] = float("nan")
            result[f"{key}_skeleton_length_density"] = float("nan")
        else:
            result[f"{key}_fraction"] = safe_fraction(masks[kind], zone_b)
            result[f"{key}_skeleton_length_density"] = topology(
                masks[kind], zone_b
            )["skeleton_length_density"]
    return result


def build_zone_b(
    disc_path: Path, size: int, valid: np.ndarray
) -> tuple[np.ndarray | None, dict[str, float | bool | int]]:
    disc = np.asarray(
        Image.open(disc_path).resize((size, size), Image.Resampling.NEAREST)
    ) > 0
    area = int(disc.sum())
    if area == 0:
        return None, {
            "disc_detected": False,
            "disc_radius_px": float("nan"),
            "zone_b_valid_pixels": 0,
        }
    coordinates = np.argwhere(disc)
    center_y, center_x = coordinates.mean(0)
    radius = float(np.sqrt(area / np.pi))
    yy, xx = np.ogrid[:size, :size]
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    zone = (distance >= 2 * radius) & (distance < 3 * radius) & valid
    if not zone.any():
        return None, {
            "disc_detected": False,
            "disc_radius_px": radius,
            "zone_b_valid_pixels": 0,
        }
    return zone, {
        "disc_detected": True,
        "disc_radius_px": radius,
        "zone_b_valid_pixels": int(zone.sum()),
    }


def load_features(rgb_path: Path, logits_path: Path, size: int):
    rgb = np.asarray(
        Image.open(rgb_path).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        )
    )
    logits = np.load(logits_path).astype(np.float32)
    logits = cv2.resize(logits, (size, size), interpolation=cv2.INTER_LINEAR)
    probability = softmax_np(logits)
    entropy = -(
        probability * np.log(np.maximum(probability, EPSILON))
    ).sum(-1) / np.log(4)
    direction = hessian_field(rgb)
    image = np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)
    base = np.moveaxis(np.clip(logits, -12, 12) / 6.0, -1, 0)
    features = np.concatenate(
        [image, base, entropy[None].astype(np.float32), direction], axis=0
    )
    return (
        torch.from_numpy(features),
        torch.from_numpy(logits).permute(2, 0, 1),
        np.moveaxis(probability, -1, 0),
    )


def eye_phenotypes(
    identifier: str,
    patient_id: str,
    eye: str,
    split: str,
    corrected_probability: np.ndarray,
    vascx_probability: np.ndarray,
    adapted_occupancy: np.ndarray,
    disc_path: Path,
    size: int,
    gate_center: float,
    gate_temperature: float,
    threshold: float,
) -> dict[str, object]:
    valid = np.ones((size, size), dtype=bool)
    zone_b, disc_information = build_zone_b(disc_path, size, valid)

    vascx_entropy = -(
        vascx_probability
        * np.log(np.maximum(vascx_probability, EPSILON))
    ).sum(0) / np.log(4)
    confidence = 1 - vascx_entropy
    gate = 1 / (
        1 + np.exp(-(confidence - gate_center) / gate_temperature)
    )
    fused_occupancy = (
        gate * (1 - vascx_probability[0])
        + (1 - gate) * adapted_occupancy
    )
    mass_prediction = corrected_probability.argmax(0)
    vascx_prediction = vascx_probability.argmax(0)
    identity = corrected_probability[1:4].argmax(0) + 1
    factorized_prediction = np.where(
        fused_occupancy >= threshold, identity, 0
    )

    artery_probability = corrected_probability[1] + corrected_probability[3]
    vein_probability = corrected_probability[2] + corrected_probability[3]
    vessel_probability = 1 - corrected_probability[0]
    soft_support = vessel_probability >= 0.5
    av_total = corrected_probability[1] + corrected_probability[2]
    artery_identity = corrected_probability[1] / np.maximum(av_total, EPSILON)
    identity_valid = soft_support & (av_total > 1e-4)

    row: dict[str, object] = {
        "id": identifier,
        "patient_id": str(patient_id),
        "eye": eye,
        "split": split,
        **disc_information,
        "soft_support_pixels": int(soft_support.sum()),
        "soft_vessel_mass_fraction": safe_mean(vessel_probability, valid),
        "soft_artery_mass_fraction": safe_mean(artery_probability, valid),
        "soft_vein_mass_fraction": safe_mean(vein_probability, valid),
        "soft_av_log_mass_ratio": log_ratio(
            float(artery_probability[valid].sum()),
            float(vein_probability[valid].sum()),
        ),
        "occupancy_entropy": safe_mean(
            normalized_binary_entropy(fused_occupancy), valid
        ),
        "identity_entropy": safe_mean(
            normalized_binary_entropy(artery_identity), identity_valid
        ),
        "mean_gate": safe_mean(gate, valid),
    }
    for kind, probability in {
        "vessel": vessel_probability,
        "artery": artery_probability,
        "vein": vein_probability,
    }.items():
        row[f"soft_zone_b_{kind}_mass_fraction"] = (
            safe_mean(probability, zone_b)
            if zone_b is not None else float("nan")
        )
    if zone_b is None:
        row["soft_zone_b_av_log_mass_ratio"] = float("nan")
    else:
        row["soft_zone_b_av_log_mass_ratio"] = log_ratio(
            float(artery_probability[zone_b].sum()),
            float(vein_probability[zone_b].sum()),
        )

    row.update(hard_phenotypes(
        mass_prediction, valid, zone_b, "mass"
    ))
    row.update(hard_phenotypes(
        factorized_prediction, valid, zone_b, "factorized"
    ))
    mass_masks = anatomy_masks(mass_prediction)
    factorized_masks = anatomy_masks(factorized_prediction)
    vascx_masks = anatomy_masks(vascx_prediction)
    union_vessel = mass_masks["vessel"] | factorized_masks["vessel"]
    common_vessel = mass_masks["vessel"] & factorized_masks["vessel"]
    base_mass_common_vessel = (
        mass_masks["vessel"] & vascx_masks["vessel"]
    )
    row["vessel_occupancy_disagreement"] = safe_fraction(
        mass_masks["vessel"] ^ factorized_masks["vessel"], union_vessel
    )
    row["base_to_mass_identity_disagreement"] = safe_fraction(
        (mass_prediction != vascx_prediction) & base_mass_common_vessel,
        base_mass_common_vessel,
    )
    row["av_log_ratio_absolute_disagreement"] = abs(
        float(row["mass_av_log_area_ratio"])
        - float(row["factorized_av_log_area_ratio"])
    )
    return row


def atomic_json(destination: Path, content: dict[str, object]) -> None:
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(content, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(destination)


def combine_outputs(
    parts: Path, output: Path, expected_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(parts.glob("*.json"))
        if path.stem in expected_ids
    ]
    eyes = pd.DataFrame(rows).sort_values(["patient_id", "eye", "id"])
    eyes.to_csv(output / "per_eye.csv", index=False)
    metadata = ["id", "patient_id", "eye", "split"]
    numeric = [
        column for column in eyes.columns
        if column not in metadata
        and column != "disc_detected"
        and pd.api.types.is_numeric_dtype(eyes[column])
    ]
    patient_rows = []
    for patient_id, group in eyes.groupby("patient_id", sort=True):
        record: dict[str, object] = {
            "patient_id": str(patient_id),
            "split": group["split"].iloc[0],
            "eyes_present": int(len(group)),
            "both_eyes_present": bool(group["eye"].nunique() == 2),
            "both_discs_detected": bool(
                group["eye"].nunique() == 2 and group["disc_detected"].all()
            ),
        }
        for column in numeric:
            values = group[column].dropna().to_numpy(dtype=float)
            record[f"{column}__bilateral_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            left = group.loc[group["eye"].eq("left"), column].dropna()
            right = group.loc[group["eye"].eq("right"), column].dropna()
            record[f"{column}__absolute_difference"] = (
                float(abs(left.mean() - right.mean()))
                if len(left) and len(right) else float("nan")
            )
            record[f"{column}__within_eye_std"] = float(np.nanmean([
                left.std(ddof=0) if len(left) > 1 else np.nan,
                right.std(ddof=0) if len(right) > 1 else np.nan,
            ])) if len(left) > 1 or len(right) > 1 else float("nan")
        patient_rows.append(record)
    patients = pd.DataFrame(patient_rows).sort_values("patient_id")
    patients.to_csv(output / "per_patient.csv", index=False)
    return eyes, patients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument(
        "--eye-manifest", type=Path,
        help="Optional generic manifest with id/patient_id/eye columns.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--rgb", type=Path,
        default=Path("artifacts/vascx_anatomy/preprocessed_rgb"),
    )
    parser.add_argument(
        "--disc", type=Path, default=Path("artifacts/vascx_anatomy/disc")
    )
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--occupancy", type=Path, required=True)
    parser.add_argument(
        "--udct", type=Path,
        default=Path("artifacts/udct_av_mass_calibrated/best.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gate-center", type=float, default=.35)
    parser.add_argument("--gate-temperature", type=float, default=.20)
    parser.add_argument("--threshold", type=float, default=.525)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.eye_manifest is not None:
        manifest = pd.read_csv(
            args.eye_manifest,
            usecols=["id", "patient_id", "eye"],
            dtype={"id": str, "patient_id": str},
        )
        eye_records = [
            (row.id, str(row.patient_id), row.eye)
            for row in manifest.itertuples(index=False)
        ]
    else:
        manifest = pd.read_csv(
            args.manifest,
            usecols=["patient_id", "split"],
            dtype={"patient_id": str},
        )
        manifest = manifest.loc[manifest["split"].eq(args.split)].copy()
        eye_records = []
        for record in manifest.itertuples(index=False):
            for suffix, eye in (("L", "left"), ("R", "right")):
                identifier = f"{record.patient_id}_{suffix}"
                eye_records.append((identifier, str(record.patient_id), eye))
    expected_ids = {item[0] for item in eye_records}
    missing = {
        kind: [
            identifier for identifier, _, _ in eye_records
            if not (directory / f"{identifier}.{extension}").is_file()
        ]
        for kind, directory, extension in (
            ("rgb", args.rgb, "png"),
            ("disc", args.disc, "png"),
            ("logits", args.logits, "npy"),
            ("occupancy", args.occupancy, "npy"),
        )
    }
    if args.require_complete and any(missing.values()):
        counts = {name: len(values) for name, values in missing.items()}
        raise FileNotFoundError(f"Incomplete inputs: {counts}")
    available = [
        item for item in eye_records
        if all(item[0] not in values for values in missing.values())
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    parts = args.output / "per_eye_parts"
    parts.mkdir(exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = Corrector().to(device)
    checkpoint = torch.load(args.udct, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pending = [
        item for item in available
        if not (parts / f"{item[0]}.json").is_file()
    ]
    completed = len(available) - len(pending)
    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch_records = pending[start:start + args.batch_size]
            loaded = [
                load_features(
                    args.rgb / f"{identifier}.png",
                    args.logits / f"{identifier}.npy",
                    args.size,
                )
                for identifier, _, _ in batch_records
            ]
            features = torch.stack([item[0] for item in loaded]).to(device)
            base_logits = torch.stack([item[1] for item in loaded]).to(device)
            corrected = model(features, base_logits).softmax(1).cpu().numpy()
            for index, (identifier, patient_id, eye) in enumerate(batch_records):
                adapted = np.load(
                    args.occupancy / f"{identifier}.npy"
                ).astype(np.float32)
                if adapted.shape != (args.size, args.size):
                    adapted = cv2.resize(
                        adapted, (args.size, args.size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                row = eye_phenotypes(
                    identifier=identifier,
                    patient_id=patient_id,
                    eye=eye,
                    split=args.split,
                    corrected_probability=corrected[index],
                    vascx_probability=loaded[index][2],
                    adapted_occupancy=adapted,
                    disc_path=args.disc / f"{identifier}.png",
                    size=args.size,
                    gate_center=args.gate_center,
                    gate_temperature=args.gate_temperature,
                    threshold=args.threshold,
                )
                atomic_json(parts / f"{identifier}.json", row)
                completed += 1
            if completed % 20 < len(batch_records):
                print(f"completed={completed}/{len(available)}", flush=True)

    eyes, patients = combine_outputs(parts, args.output, expected_ids)
    metadata = {
        "protocol": "CIMT_双读出表型锁定协议.md",
        "split": args.split,
        "expected_eyes": len(eye_records),
        "available_eyes": len(available),
        "completed_eyes": len(eyes),
        "patients": len(patients),
        "missing_inputs": {name: len(values) for name, values in missing.items()},
        "size": args.size,
        "gate_center": args.gate_center,
        "gate_temperature": args.gate_temperature,
        "threshold": args.threshold,
        "label_blind": True,
        "complete": len(eyes) == len(eye_records),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
