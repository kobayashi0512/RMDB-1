#!/usr/bin/env python3
"""Outcome blind, resumable parallel queue for RMDB backbone candidates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CANDIDATES = ("baseline", "highres", "global_local", "dualstream", "direction")
SEEDS = (20260906, 20260907, 20260908)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_command(args: argparse.Namespace, candidate: str, seed: int, out: Path) -> list[str]:
    return [
        PYTHON, str(ROOT / "src" / "train_birap_v2.py"),
        "--rav", str(ROOT / "data" / "rav_vascx"),
        "--rav-ground-truth", str(ROOT / "data" / "raw" / "RAV" / "data" / "external" / "RAV" / "dataset" / "av_masks"),
        "--rav-metadata", str(ROOT / "data" / "raw" / "RAV" / "data" / "external" / "RAV" / "metadata.csv"),
        "--fives", str(ROOT / "data" / "raw" / "FIVES"),
        "--occupancy-checkpoint", str(ROOT / "models" / "baseline" / "artifacts" / "fives_unet_resnet34" / "best.pt"),
        "--mass-checkpoint", str(ROOT / "models" / "baseline" / "artifacts" / "udct_av_mass_calibrated" / "best.pt"),
        "--output", str(out),
        "--variant", "full", "--backbone", candidate,
        "--epochs", str(args.epochs), "--train-size", str(args.train_size),
        "--eval-size", str(args.eval_size), "--crop-sizes", "384", "512", "640", "768",
        "--batch-size", str(args.batch_size), "--workers", "0",
        "--rav-validation", "30", "--fives-validation", "120", "--seed", str(seed),
        "--skip-holdout",
    ]


def write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(path)


def is_complete(out: Path) -> bool:
    required = (out / "summary.json", out / "best.pt", out / "history.json", out / "rav_validation_metrics.csv", out / "fives_validation_metrics.csv")
    if not all(p.is_file() for p in required):
        return False
    try:
        summary = json.loads((out / "summary.json").read_text())
        history = json.loads((out / "history.json").read_text())
        return bool(history) and "rav_validation" in summary and "fives_validation" in summary
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=ROOT / "results" / "backbone_candidates_v1")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = p.parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    jobs = {}
    for candidate in args.candidates:
        for seed in args.seeds:
            key = f"{candidate}/seed_{seed}"
            out = args.output_root / candidate / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            jobs[key] = {
                "candidate": candidate, "seed": seed, "output": str(out),
                "status": "complete" if is_complete(out) else "queued",
                "pid": None, "started_at": None, "finished_at": None,
                "returncode": None, "log": str(out / "run.log"),
            }
    state = {"protocol": "BACKBONE_CANDIDATES_PROTOCOL_V1", "created_at": now(), "jobs": jobs}
    write_state(state_path, state)
    active: dict[str, subprocess.Popen] = {}
    while True:
        for key, proc in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            job = jobs[key]
            job["returncode"] = rc
            job["pid"] = None
            job["finished_at"] = now()
            job["status"] = "complete" if rc == 0 and is_complete(Path(job["output"])) else "failed"
            del active[key]
        for key, job in jobs.items():
            if len(active) >= args.max_parallel:
                break
            if job["status"] != "queued":
                continue
            out = Path(job["output"])
            log_handle = (out / "run.log").open("a")
            command = build_command(args, job["candidate"], int(job["seed"]), out)
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            active[key] = proc
            job["pid"] = proc.pid
            job["started_at"] = now()
            job["status"] = "running"
        state["updated_at"] = now()
        state["active_count"] = len(active)
        state["complete_count"] = sum(j["status"] == "complete" for j in jobs.values())
        state["failed_count"] = sum(j["status"] == "failed" for j in jobs.values())
        write_state(state_path, state)
        print(json.dumps({k: state[k] for k in ("updated_at", "active_count", "complete_count", "failed_count")}, ensure_ascii=False), flush=True)
        if not active and all(j["status"] in {"complete", "failed"} for j in jobs.values()):
            break
        time.sleep(args.poll_seconds)
    state["completed_at"] = now()
    write_state(state_path, state)
    print(json.dumps({"queue_state": str(state_path), "complete": state["complete_count"], "failed": state["failed_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
