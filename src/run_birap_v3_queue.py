#!/usr/bin/env python3
"""Run the preregistered native-resolution logit-residual BiRAP V3 queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = ("residual_joint", "residual_topology", "residual_phenotype", "residual_full")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3/experiments"))
    parser.add_argument("--rav", type=Path, default=Path("artifacts/v2/rav_vascx"))
    parser.add_argument("--rav-ground-truth", type=Path, default=Path("restored_assets/RAV/data/external/RAV/dataset/av_masks"))
    parser.add_argument("--rav-metadata", type=Path, default=Path("restored_assets/RAV/data/external/RAV/metadata.csv"))
    parser.add_argument("--fives", type=Path, default=Path("restored_assets/FIVES"))
    parser.add_argument("--occupancy-checkpoint", type=Path, default=Path("restored_assets/models/baseline/artifacts/fives_unet_resnet34/best.pt"))
    parser.add_argument("--mass-checkpoint", type=Path, default=Path("restored_assets/models/baseline/artifacts/udct_av_mass_calibrated/best.pt"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "queue_state.json"
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "development_only": True,
        "native_resolution_eval": True,
        "fundus_avseg_used": False,
        "cimt_used": False,
        "mbrset_outcomes_used": False,
        "experiments": [],
    }
    state_path.write_text(json.dumps(state, indent=2))
    for variant in VARIANTS:
        output = args.output / variant
        if (output / "summary.json").is_file():
            state["experiments"].append({"variant": variant, "status": "existing_complete"})
            state_path.write_text(json.dumps(state, indent=2))
            continue
        if (output / "negative_result.json").is_file():
            state["experiments"].append({"variant": variant, "status": "existing_negative_stopped"})
            state_path.write_text(json.dumps(state, indent=2))
            continue
        command = [
            sys.executable, "src/train_birap_v2.py",
            "--variant", variant,
            "--output", str(output),
            "--rav", str(args.rav),
            "--rav-ground-truth", str(args.rav_ground_truth),
            "--rav-metadata", str(args.rav_metadata),
            "--fives", str(args.fives),
            "--occupancy-checkpoint", str(args.occupancy_checkpoint),
            "--mass-checkpoint", str(args.mass_checkpoint),
            "--epochs", str(1 if args.smoke else args.epochs),
            "--batch-size", str(args.batch_size),
            "--train-size", str(256 if args.smoke else args.train_size),
            "--eval-size", str(256 if args.smoke else args.eval_size),
            "--seed", str(args.seed),
            "--residual-logit",
            "--residual-scale", str(args.residual_scale),
            "--anchor-weight", str(args.anchor_weight),
        ]
        if args.smoke:
            command.extend(["--limit-train", "2", "--limit-validation", "1", "--skip-holdout"])
        output.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".log")
        with log_path.open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        status = "complete" if process.returncode == 0 else "failed"
        state["experiments"].append({
            "variant": variant,
            "status": status,
            "returncode": process.returncode,
            "log": str(log_path),
        })
        state_path.write_text(json.dumps(state, indent=2))
        if process.returncode:
            # A preregistered branch that is negative or unstable is recorded
            # and stopped; independent variants continue without retuning.
            state.setdefault("failed_variants", []).append(variant)
            state_path.write_text(json.dumps(state, indent=2))
            continue
    state["status"] = "complete_with_failed_variants" if state.get("failed_variants") else "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
