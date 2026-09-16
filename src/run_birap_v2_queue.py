#!/usr/bin/env python3
"""Run the prespecified RMDB-BiRAP V2 ablation queue sequentially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = ("occupancy", "joint", "topology", "phenotype", "full")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2/experiments"))
    parser.add_argument("--rav", type=Path, default=Path("artifacts/v2/rav_vascx"))
    parser.add_argument("--fives", type=Path, default=Path("restored_assets/FIVES"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "queue_state.json"
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "development_only": True,
        "fundus_avseg_used": False,
        "experiments": [],
    }
    state_path.write_text(json.dumps(state, indent=2))
    for variant in args.variants:
        output = args.output / variant
        if (output / "summary.json").is_file():
            state["experiments"].append({"variant": variant, "status": "existing_complete"})
            state_path.write_text(json.dumps(state, indent=2))
            continue
        command = [
            sys.executable, "src/train_birap_v2.py",
            "--variant", variant,
            "--output", str(output),
            "--rav", str(args.rav),
            "--fives", str(args.fives),
            "--epochs", str(1 if args.smoke else args.epochs),
            "--batch-size", str(args.batch_size),
            "--train-size", str(args.train_size),
            "--eval-size", str(args.eval_size),
            "--seed", str(args.seed),
        ]
        if args.smoke:
            command.extend(["--limit-train", "2", "--limit-validation", "1", "--skip-holdout"])
        log_path = output.with_suffix(".log")
        output.mkdir(parents=True, exist_ok=True)
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
            state["status"] = "failed"
            state["failed_variant"] = variant
            state_path.write_text(json.dumps(state, indent=2))
            raise SystemExit(process.returncode)
    state["status"] = "complete"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
