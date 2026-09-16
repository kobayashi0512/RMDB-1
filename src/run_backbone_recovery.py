#!/usr/bin/env python3
"""Run an explicit checkpointed queue; a user pause prevents new workers."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from backbone_runtime import atomic_json, process_matches

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("baseline", "highres", "global_local", "dualstream", "direction")
SEEDS = (20260906, 20260907, 20260908)
DEFAULT_OUTPUT = ROOT / "results/backbone_candidates_recovery_20260907"

def now():
    return datetime.now(timezone.utc).isoformat()

def build_command(args, candidate, seed, out):
    command = [sys.executable, str(ROOT / "src/train_birap_v2.py"),
        "--rav", str(ROOT / "data/rav_vascx"),
        "--rav-ground-truth", str(ROOT / "data/rav_geometry_recovery_20260907/aligned_masks"),
        "--rav-metadata", str(ROOT / "data/raw/RAV/data/external/RAV/metadata.csv"),
        "--fives", str(ROOT / "data/raw/FIVES"),
        "--occupancy-checkpoint", str(ROOT / "models/baseline/artifacts/fives_unet_resnet34/best.pt"),
        "--mass-checkpoint", str(ROOT / "models/baseline/artifacts/udct_av_mass_calibrated/best.pt"),
        "--output", str(out), "--variant", "full", "--backbone", candidate,
        "--epochs", str(args.epochs), "--train-size", str(args.train_size),
        "--eval-size", str(args.eval_size), "--crop-sizes", "384", "512", "640", "768",
        "--batch-size", str(args.batch_size), "--workers", "0",
        "--rav-validation", "30", "--fives-validation", "120", "--seed", str(seed),
        "--skip-holdout", "--managed-run"]
    if (out / "last.pt").exists():
        command.append("--resume")
    elif (out / "history.json").exists() or (out / "best.pt").exists():
        raise RuntimeError(f"Legacy partial run has no resumable checkpoint: {out}")
    return command

def is_complete(out, epochs):
    required = ("summary.json", "best.pt", "last.pt", "history.json",
                "rav_validation_metrics.csv", "fives_validation_metrics.csv")
    if not all((out / name).is_file() for name in required):
        return False
    try:
        result = json.loads((out / "summary.json").read_text())
        history = json.loads((out / "history.json").read_text())
        return (len(history) == epochs and result["completed_epochs"] == epochs
                and [r["epoch"] for r in history] == list(range(1, epochs + 1))
                and all(math.isfinite(r["train_loss"]) and math.isfinite(r["selection_score"]) for r in history)
                and result["rav_validation"]["n"] == 30
                and result["fives_validation"]["n"] == 120)
    except (ValueError, KeyError, TypeError):
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--resume-queue", action="store_true", help="Explicit user-authorized resume after a pause.")
    args = p.parse_args()
    if args.max_parallel < 1:
        raise ValueError("Invalid concurrency")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Supervisor already active")
    state_path, pause_path = args.output_root / "queue_state.json", args.output_root / "PAUSE"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    geometry_path = ROOT / 'data/rav_geometry_recovery_20260907/audit.json'
    geometry = json.loads(geometry_path.read_text())
    if geometry.get('status') != 'passed' or geometry.get('exact_rgb_replays') != 206:
        raise ValueError('RAV geometry has not passed exact image replay audit')
    source_hashes = {name: hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest()
                     for name in ('train_birap_v2.py', 'backbone_runtime.py', 'run_backbone_recovery.py',
                                  'train_udct_av_pilot.py', 'prepare_rav_backbone_geometry.py')}
    geometry_hash = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    if previous.get('source_sha256') and previous['source_sha256'] != source_hashes:
        raise ValueError('Training sources changed. Do not mix versions within this queue.')
    if previous.get('geometry_audit_sha256') and previous['geometry_audit_sha256'] != geometry_hash:
        raise ValueError('RAV geometry audit changed. Refusing mixed data provenance.')
    if (pause_path.exists() or previous.get("queue_status") == "paused") and not args.resume_queue:
        raise SystemExit("User pause is active. Explicit --resume-queue required.")
    if args.resume_queue and pause_path.exists():
        pause_path.rename(args.output_root / f"pause_acknowledged_{time.time_ns()}.txt")
    config = {k: v for k, v in vars(args).items() if k not in {"output_root", "resume_queue", "poll_seconds", "max_parallel"}}
    if previous.get("configuration") and previous["configuration"] != config:
        raise ValueError("Queue configuration changed; refusing to mix experiments")
    for job in previous.get("jobs", {}).values():
        if process_matches(job.get("pid"), str(job["output"])):
            raise SystemExit("An existing worker is active; refusing duplicate training")
    jobs = {}
    for seed in args.seeds:
        for candidate in args.candidates:
            key = f"{candidate}/seed_{seed}"
            out = args.output_root / candidate / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            old = previous.get("jobs", {}).get(key, {})
            status = "complete" if is_complete(out, args.epochs) else "queued"
            if old.get("status") == "failed" and status != "complete":
                status = "failed"
            jobs[key] = {**old, "candidate": candidate, "seed": seed, "output": str(out),
                         "status": status, "pid": None, "log": str(out / "run.log")}
    state = {"protocol": "BACKBONE_RECOVERY_20260907", "created_at": previous.get("created_at", now()),
             "configuration": config, "supervisor_pid": os.getpid(), "jobs": jobs,
             "geometry_audit_sha256": geometry_hash, "source_sha256": source_hashes}
    active = {}
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)
    def persist(status):
        state.update(queue_status=status, updated_at=now(), active_count=len(active),
                     complete_count=sum(j["status"] == "complete" for j in jobs.values()),
                     failed_count=sum(j["status"] == "failed" for j in jobs.values()))
        atomic_json(state_path, state)
    def pause_workers():
        nonlocal stopping
        stopping = True
        for key, proc in active.items():
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 45
        for key, proc in active.items():
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            jobs[key].update(status="paused", pid=None, returncode=proc.returncode, paused_at=now())
        active.clear()
        persist("paused")
    try:
        persist("running")
        while True:
            if stopping or pause_path.exists():
                pause_workers()
                return
            for key, proc in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                job = jobs[key]
                job.update(pid=None, finished_at=now(), returncode=rc)
                job["status"] = "complete" if rc == 0 and is_complete(Path(job["output"]), args.epochs) else "paused" if rc == 75 else "failed"
                del active[key]
                if rc == 75:
                    stopping = True
            if stopping:
                pause_workers()
                return
            for key, job in jobs.items():
                if stopping or pause_path.exists() or len(active) >= args.max_parallel:
                    break
                if job["status"] != "queued":
                    continue
                command = build_command(args, job["candidate"], job["seed"], Path(job["output"]))
                with Path(job["log"]).open("a") as log:
                    log.write(json.dumps({"event": "worker_launch", "at": now(), "command": command}) + "\n")
                    log.flush()
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                active[key] = proc
                job.update(status="running", pid=proc.pid, started_at=now(), command=command)
            persist("running")
            if not active and all(j["status"] in {"complete", "failed"} for j in jobs.values()):
                state["completed_at"] = now()
                persist("complete" if not state["failed_count"] else "completed_with_failures")
                return
            deadline = time.monotonic() + args.poll_seconds
            while time.monotonic() < deadline and not stopping and not pause_path.exists():
                time.sleep(0.5)
    finally:
        if active:
            pause_workers()

if __name__ == "__main__":
    main()
