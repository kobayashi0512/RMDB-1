#!/usr/bin/env python3
"""Locked high-resolution paired-quality stress test for RMDB readouts.

The public HRF IQA set has 18 same-eye pairs: a defocused/poor-quality first
capture and a better repeat capture from the same Canon CR-1 camera.  This is
not a same-condition retest and not a cross-camera study.  It is analysed as a
paired acquisition-quality sensitivity test and never pooled with mBRSET.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from extract_mbrset_repeatability_features import readout_features
from final_measurement_controls import ARMS, OCC_WEIGHT, check_prob, load_occupancy, simple_readouts
from run_fundus_fair_four_arm import ROOT, WEIGHT, Corrector, corrected, sha, softmax_np
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.ensembles.predict_output import decollate_predict_full


SOURCE = ROOT / "data/raw/HRF_IQA_20260909"
IMAGES = SOURCE / "images"
ARCHIVE = SOURCE / "allQuality.zip"
OUT = ROOT / "artifacts/hrf_iqa_readout_stress_test_v1"
GATE = ROOT / "experiments/measurement_confirmation_v1/analyze_measurement_gate.py"
PATTERN = re.compile(r"^(?P<pair>[1-9]|1[0-8])_(?P<quality>bad|good)\.JPG$")
FEATURES = [
    "vessel_mass", "artery_mass", "vein_mass", "av_ratio", "vessel_density",
    "skeleton_density", "branch_density", "endpoint_density", "component_count",
    "fractal_dimension", "occupancy_entropy", "identity_entropy",
]
EXPORT_ARM = {"raw": "raw_vascx", "mass": "mass", "joint": "joint", "smooth": "smooth", "cleanup": "cleanup"}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_images() -> list[tuple[int, str, Path]]:
    records: list[tuple[int, str, Path]] = []
    for path in sorted(IMAGES.glob("*.JPG")):
        match = PATTERN.match(path.name)
        if not match:
            raise ValueError(f"Unexpected HRF IQA source filename: {path.name}")
        records.append((int(match["pair"]), match["quality"], path))
    expected = {(number, quality) for number in range(1, 19) for quality in ("bad", "good")}
    if {(number, quality) for number, quality, _ in records} != expected:
        raise ValueError("Expected exactly 18 bad/good source pairs")
    return records


def prepare() -> None:
    if not ARCHIVE.is_file() or not IMAGES.is_dir():
        raise FileNotFoundError("Download and extract the official allQuality.zip archive before preparation")
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for pair, quality, path in source_images():
        with Image.open(path) as image:
            width, height = image.size
        identifier = f"HRF_IQA_{pair:02d}_{quality.upper()}"
        rows.append({
            "image_id": identifier,
            "patient_id": f"HRF_IQA_PAIR_{pair:02d}",
            "eye": "not_released",
            "session_id": f"HRF_IQA_PAIR_{pair:02d}",
            "acquisition_id": f"{pair:02d}_{'01' if quality == 'bad' else '02'}_{quality}",
            "camera_id": "Canon_CR-1",
            "camera_configuration": "Canon_CR-1__native_resolution_varies",
            "native_resolution": f"{width}x{height}",
            "capture_time": "not_released",
            "pair_role": quality,
            "original_path": str(path.relative_to(ROOT)),
            "native_width": str(width),
            "native_height": str(height),
            "sha256": sha(path),
            "prior_project_use": "false",
            "split": "high_resolution_quality_stress_test",
        })
    rows.sort(key=lambda row: (row["patient_id"], row["acquisition_id"]))
    manifest = OUT / "manifest.csv"
    fields = list(rows[0])
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
        if previous != rows:
            raise ValueError("Existing HRF IQA manifest differs from the frozen source inventory")
    else:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    protocol = {
        "protocol_id": "RMDB-HRF-IQA-readout-stress-test-v1",
        "purpose": "High-resolution paired acquisition-quality sensitivity test of frozen readouts; not a same-condition retest, cross-camera validation, structural-reference study, or clinical study.",
        "source": {
            "url": "https://www5.cs.fau.de/research/data/fundus-images/",
            "archive": str(ARCHIVE.relative_to(ROOT)),
            "archive_sha256": sha(ARCHIVE),
            "published_design": "18 same-eye pairs from 18 subjects, Canon CR-1, first poor/defocused capture then repeat improved capture; approximately same field of view.",
            "native_images": 36,
            "pairs": 18,
            "native_resolutions": sorted({f"{row['native_width']}x{row['native_height']}" for row in rows}),
            "patient_eye_laterality": "not publicly released; pair ID is the only pairing key used",
        },
        "readouts": {
            "raw": "frozen VascX artery-vein probabilities",
            "mass": "frozen two-step mass corrector",
            "joint": "frozen occupancy-conditioned original VascX identity",
            "smooth": "fixed Gaussian probability control",
            "cleanup": "fixed small-component removal control",
        },
        "feature_operator": "same 512 probability grid and 256 skeleton grid as final_measurement_controls; source-quality labels are never supplied to a model",
        "statistics": {
            "pair": "bad first capture versus good repeat capture within the published same-eye pair",
            "report": ["ICC(1,1)", "within-eye MAD", "SEM", "Bland-Altman bias", "Bland-Altman 95% LoA"],
            "bootstrap": "18 pair IDs treated as patient clusters; 10000 paired percentile resamples; simultaneous 98.333% central intervals for three predeclared readout endpoints",
        },
        "interpretation": [
            "A small paired difference measures robustness to a quality perturbation, not physiological repeatability.",
            "This public set has one camera only and cannot support a cross-camera claim.",
            "The IQA subset has no vessel ground truth, so no structural-fidelity claim is made here.",
            "Results are a new high-resolution stress test and are not pooled with historic mBRSET or Fundus-AVSeg results.",
            "The test compares readouts on top of VascX and does not claim to replace VascX."
        ],
        "source_hashes": {
            "VascX_weight": sha(ROOT / "models/third_party/vascx_weights/av_july24.pt"),
            "mass_weight": sha(WEIGHT),
            "occupancy_weight": sha(OCC_WEIGHT),
            "runner": sha(Path(__file__)),
            "feature_operator": sha(ROOT / "src/extract_mbrset_repeatability_features.py"),
        },
        "no_training": True,
        "no_clinical_fields": True,
        "no_output_driven_tuning": True,
    }
    destination = OUT / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing HRF IQA protocol differs from this frozen runner")
    else:
        write_json(destination, protocol)
    print(f"PREPARED {len(rows)} images / 18 published same-eye pairs", flush=True)


def source_path(row: pd.Series) -> Path:
    return ROOT / str(row.original_path)


def probabilities(model: SegmentationEnsemble, corrector: Corrector, occnet: torch.nn.Module, device: torch.device, row: pd.Series) -> tuple[dict[str, np.ndarray], np.ndarray]:
    path = source_path(row)
    loader = model._make_inference_dataloader(
        {"images": [{"id": str(row.image_id), "image": str(path)}]}, preprocess=True,
        batch_size=1, num_workers=0, ignore_exceptions=False,
    )
    with torch.inference_mode():
        output = [item for batch in loader for item in decollate_predict_full(model.predict_step_full(batch))]
    if len(output) != 1:
        raise RuntimeError(f"VascX did not produce exactly one result for {row.image_id}")
    native_logits = np.asarray(output[0]["logits"], dtype=np.float32)
    canonical = np.asarray(output[0]["preprocessed_image"], dtype=np.uint8)
    if native_logits.shape != (1024, 1024, 4) or canonical.shape != (1024, 1024, 3):
        raise ValueError(f"Unexpected VascX geometry for {row.image_id}: {native_logits.shape}/{canonical.shape}")
    rgb = np.asarray(Image.fromarray(canonical).resize((512, 512), Image.Resampling.BILINEAR), dtype=np.uint8)
    logits = cv2.resize(native_logits, (512, 512), interpolation=cv2.INTER_LINEAR)
    raw = np.moveaxis(softmax_np(logits), -1, 0)
    mass = corrected(corrector, device, rgb, logits)
    normalized = (np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0) - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    with torch.inference_mode():
        occupancy = occnet(torch.from_numpy(normalized[None]).to(device)).sigmoid()[0, 0].cpu().numpy()
    occupancy = occupancy.astype(np.float16).astype(np.float32)
    identity = raw[1:] / np.maximum(raw[1:].sum(0, keepdims=True), 1e-7)
    joint = np.concatenate([1.0 - occupancy[None], occupancy[None] * identity], axis=0)
    fov = rgb.max(axis=2) > 10
    all_probs = {"raw": raw, "mass": mass, "joint": joint, **simple_readouts(raw, fov)}
    if set(all_probs) != set(ARMS):
        raise AssertionError("Readout arm mismatch")
    for value in all_probs.values():
        check_prob(value)
    return all_probs, fov


def run() -> None:
    prepare()
    manifest = pd.read_csv(OUT / "manifest.csv", dtype=str)
    case_dir = OUT / "cases"
    case_dir.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    corrector = Corrector().to(device).eval()
    corrector.load_state_dict(torch.load(WEIGHT, map_location=device)["model"])
    occnet = load_occupancy(device)
    model = SegmentationEnsemble.from_torchscript(ROOT / "models/third_party/vascx_weights/av_july24.pt").to(device).eval()
    model.config["inference"].update({"tta": False, "tile_batch_size": 2})
    started = time.monotonic()
    for number, row in manifest.iterrows():
        destination = case_dir / f"{row.image_id}.json"
        if destination.exists():
            continue
        probs, fov = probabilities(model, corrector, occnet, device, row)
        values: list[dict[str, object]] = []
        for arm, probability in probs.items():
            values.append({"image_id": row.image_id, "arm": EXPORT_ARM[arm], **readout_features(probability, fov, "")})
        record = {
            "image_id": row.image_id,
            "patient_id": row.patient_id,
            "pair_role": row.pair_role,
            "fov_pixels": int(fov.sum()),
            "probability_checks": {arm: check_prob(probability) for arm, probability in probs.items()},
            "values": values,
        }
        write_json(destination, record)
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        print(f"{number + 1}/{len(manifest)} completed; {time.monotonic() - started:.1f}s", flush=True)
    cases = [json.loads((case_dir / f"{row.image_id}.json").read_text(encoding="utf-8")) for _, row in manifest.iterrows()]
    if len(cases) != 36:
        raise AssertionError("Incomplete HRF IQA inference")
    values = [value for case in cases for value in case["values"]]
    frame = pd.DataFrame(values)
    if len(frame) != len(manifest) * len(ARMS) or frame.duplicated(["image_id", "arm"]).any():
        raise AssertionError("Invalid case/arm output cardinality")
    for feature in FEATURES:
        if not np.isfinite(frame[feature].to_numpy(float)).all():
            raise ValueError(f"Non-finite feature: {feature}")
    long_rows = [
        {"image_id": value["image_id"], "arm": value["arm"], "feature": feature, "value": value[feature]}
        for value in values for feature in FEATURES
    ]
    pd.DataFrame(long_rows).to_csv(OUT / "feature_values_long.csv", index=False)
    frame.to_csv(OUT / "per_image_features.csv", index=False)
    write_json(OUT / "run_complete.json", {
        "images": 36, "published_pairs": 18, "arm_evaluations": len(frame), "feature_values": len(long_rows),
        "all_five_arms": sorted(frame.arm.unique().tolist()) == sorted(EXPORT_ARM.values()),
        "no_training": True, "seconds": time.monotonic() - started,
    })
    print("RUN COMPLETE", flush=True)


def analyze() -> None:
    run_complete = OUT / "run_complete.json"
    if not run_complete.is_file():
        raise FileNotFoundError("Run inference before analysis")
    command = [
        sys.executable, str(GATE), "--manifest", str(OUT / "manifest.csv"),
        "--feature-values", str(OUT / "feature_values_long.csv"), "--out", str(OUT / "measurement_stress_gate.json"),
        "--bootstrap", "10000", "--seed", "20260909",
    ]
    subprocess.run(command, check=True)
    report = json.loads((OUT / "measurement_stress_gate.json").read_text(encoding="utf-8"))
    report["analysis_role"] = "paired acquisition-quality stress test, not same-condition retest reliability"
    report["prohibited_interpretations"] = [
        "No cross-camera or resolution-generalization claim.",
        "No clinical measurement-usability claim.",
        "No structural-fidelity claim because this IQA subset has no vessel ground truth.",
    ]
    write_json(OUT / "measurement_stress_gate.json", report)
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase]()
