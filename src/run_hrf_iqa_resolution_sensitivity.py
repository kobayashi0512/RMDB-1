#!/usr/bin/env python3
"""Controlled digital-resolution sensitivity experiment on unseen HRF IQA good captures.

This is explicitly a synthetic downsampling experiment: it isolates input-pixel
resolution from camera optics and acquisition conditions.  It cannot establish
cross-camera transportability.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from final_measurement_controls import OCC_WEIGHT, load_occupancy
from run_fundus_fair_four_arm import ROOT, WEIGHT, Corrector, sha
from run_hrf_iqa_readout_stress_test import ARMS, EXPORT_ARM, FEATURES, IMAGES, probabilities, source_images, write_json
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble


OUT = ROOT / "artifacts/hrf_iqa_resolution_sensitivity_v1"
GATE = ROOT / "experiments/measurement_confirmation_v1/analyze_measurement_gate.py"
LEVELS = {"native": 1.0, "75pct": 0.75, "50pct": 0.50}


def generated_path(pair: int, level: str) -> Path:
    return OUT / "derived_inputs" / f"HRF_IQA_{pair:02d}_GOOD_{level}.JPG"


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "derived_inputs").mkdir(exist_ok=True)
    originals = {(pair, quality): path for pair, quality, path in source_images()}
    rows: list[dict[str, str]] = []
    for pair in range(1, 19):
        original = originals[pair, "good"]
        with Image.open(original) as image:
            native_width, native_height = image.size
            for level, factor in LEVELS.items():
                if level == "native":
                    current = original
                    width, height = native_width, native_height
                else:
                    current = generated_path(pair, level)
                    width, height = round(native_width * factor), round(native_height * factor)
                    if not current.exists():
                        image.resize((width, height), Image.Resampling.LANCZOS).save(current, quality=100, subsampling=0, optimize=False)
                    with Image.open(current) as derived:
                        if derived.size != (width, height):
                            raise ValueError(f"Derived resolution mismatch for {current.name}")
                rows.append({
                    "image_id": f"HRF_IQA_{pair:02d}_GOOD_{level.upper()}",
                    "patient_id": f"HRF_IQA_PAIR_{pair:02d}",
                    "eye": "not_released",
                    "session_id": f"HRF_IQA_PAIR_{pair:02d}",
                    "acquisition_id": f"{pair:02d}_{level}",
                    "camera_configuration": "Canon_CR-1__digital_resolution_sensitivity",
                    "resolution_level": level,
                    "source_image_id": f"HRF_IQA_{pair:02d}_GOOD",
                    "original_path": str(current.relative_to(ROOT)),
                    "native_width": str(width), "native_height": str(height), "sha256": sha(current),
                    "split": "controlled_digital_resolution_sensitivity",
                })
    rows.sort(key=lambda item: (item["patient_id"], item["resolution_level"]))
    manifest = OUT / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            if list(csv.DictReader(handle)) != rows:
                raise ValueError("Existing resolution manifest differs from the frozen inputs")
    else:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for target in ("75pct", "50pct"):
        pair_rows: list[dict[str, str]] = []
        for pair in range(1, 19):
            for level, order in (("native", "01"), (target, "02")):
                source = next(row for row in rows if row["patient_id"] == f"HRF_IQA_PAIR_{pair:02d}" and row["resolution_level"] == level)
                pair_rows.append({
                    "image_id": source["image_id"], "patient_id": source["patient_id"], "eye": source["eye"],
                    "session_id": source["session_id"], "camera_configuration": source["camera_configuration"],
                    "capture_time": "synthetic_resolution_order", "acquisition_id": f"{pair:02d}_{order}_{level}",
                })
        with (OUT / f"manifest_native_vs_{target}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    protocol = {
        "protocol_id": "RMDB-HRF-IQA-digital-resolution-sensitivity-v1",
        "source": "18 never-before-project-analysed HRF IQA good-quality native images",
        "input_levels": LEVELS,
        "resampling": "Pillow LANCZOS; JPEG quality=100; subsampling=0; native image unchanged",
        "comparisons": ["native_vs_75pct", "native_vs_50pct"],
        "readouts": ["raw VascX", "frozen mass", "frozen joint", "fixed smooth", "fixed cleanup"],
        "statistics": "paired same source image, 18 pair IDs, 10000 bootstrap resamples; ICC/MAD/SEM/LoA all reported",
        "limits": ["synthetic digital downsampling, not a camera experiment", "no vessel reference labels", "not pooled with HRF IQA bad/good quality stress test"],
        "source_hashes": {"runner": sha(Path(__file__)), "mass_weight": sha(WEIGHT), "occupancy_weight": sha(OCC_WEIGHT)},
        "no_training": True, "no_tuning": True,
    }
    destination = OUT / "protocol.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing resolution protocol differs from frozen runner")
    if not destination.exists():
        write_json(destination, protocol)
    print("PREPARED 18 native images × 3 resolution levels", flush=True)


def run() -> None:
    prepare()
    manifest = pd.read_csv(OUT / "manifest.csv", dtype=str)
    cases = OUT / "cases"; cases.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4); cv2.setNumThreads(2)
    corrector = Corrector().to(device).eval(); corrector.load_state_dict(torch.load(WEIGHT, map_location=device)["model"])
    occupancy = load_occupancy(device)
    model = SegmentationEnsemble.from_torchscript(ROOT / "models/third_party/vascx_weights/av_july24.pt").to(device).eval()
    model.config["inference"].update({"tta": False, "tile_batch_size": 2})
    started = time.monotonic()
    for index, row in manifest.iterrows():
        path = cases / f"{row.image_id}.json"
        if path.exists():
            continue
        probs, fov = probabilities(model, corrector, occupancy, device, row)
        values = [{"image_id": row.image_id, "arm": EXPORT_ARM[arm], **readout_features(probability, fov, "")} for arm, probability in probs.items()]
        write_json(path, {"image_id": row.image_id, "patient_id": row.patient_id, "resolution_level": row.resolution_level, "fov_pixels": int(fov.sum()), "values": values})
        if device.type == "mps": torch.mps.synchronize(); torch.mps.empty_cache()
        print(f"{index + 1}/{len(manifest)} completed; {time.monotonic() - started:.1f}s", flush=True)
    values = [value for _, row in manifest.iterrows() for value in json.loads((cases / f"{row.image_id}.json").read_text(encoding="utf-8"))["values"]]
    frame = pd.DataFrame(values)
    if len(frame) != 54 * 5 or frame.duplicated(["image_id", "arm"]).any(): raise ValueError("Invalid resolution output cardinality")
    if not all(np.isfinite(frame[feature].to_numpy(float)).all() for feature in FEATURES): raise ValueError("Non-finite feature")
    long = [{"image_id": item["image_id"], "arm": item["arm"], "feature": feature, "value": item[feature]} for item in values for feature in FEATURES]
    pd.DataFrame(long).to_csv(OUT / "feature_values_long.csv", index=False)
    frame.to_csv(OUT / "per_image_features.csv", index=False)
    write_json(OUT / "run_complete.json", {"images": 54, "source_eyes": 18, "levels": list(LEVELS), "arm_evaluations": len(frame), "feature_values": len(long), "no_training": True, "seconds": time.monotonic() - started})
    print("RUN COMPLETE", flush=True)


def analyze() -> None:
    if not (OUT / "run_complete.json").is_file(): raise FileNotFoundError("Run inference before analysis")
    for target in ("75pct", "50pct"):
        output = OUT / f"gate_native_vs_{target}.json"
        subprocess.run([sys.executable, str(GATE), "--manifest", str(OUT / f"manifest_native_vs_{target}.csv"), "--feature-values", str(OUT / "feature_values_long.csv"), "--out", str(output), "--bootstrap", "10000", "--seed", "20260909"], check=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        report["analysis_role"] = "controlled digital downsampling sensitivity; not an optical, cross-camera, or clinical transportability experiment"
        write_json(output, report)
    print("ANALYSIS COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("prepare", "run", "analyze")); args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase]()

