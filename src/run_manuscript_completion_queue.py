#!/usr/bin/env python3
"""Restartable sequential queue for RMDB--BiRAP manuscript completion."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-vascx/bin/python"
BASE = ROOT / "artifacts/manuscript_completion"
STATE = BASE / "queue_state.json"
QUEUE_LOG = BASE / "queue.log"
LOCK = BASE / ".queue.lock"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"created_at": now(), "updated_at": now(), "steps": {}}


def save_state(state: dict) -> None:
    state["updated_at"] = now()
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(STATE)


def steps(quick: bool) -> list[dict]:
    bootstrap = "100" if quick else "2000"
    outcome_perm = "200" if quick else "5000"
    pair_perm = "5" if quick else "250"
    script = "scripts/manuscript_completion_analyses.py"
    return [
        {"id": "01_access_audit", "command": [str(PYTHON), script, "access-audit", "--output", "artifacts/manuscript_completion/01_access_audit"],
         "marker": "artifacts/manuscript_completion/01_access_audit/access_audit.json"},
        {"id": "02_repeatability_extended", "command": [str(PYTHON), script, "repeatability", "--output", "artifacts/manuscript_completion/02_repeatability_extended", "--bootstrap-repetitions", bootstrap],
         "marker": "artifacts/manuscript_completion/02_repeatability_extended/repeatability_extended.csv"},
        {"id": "03_negative_controls", "command": [str(PYTHON), script, "negative-controls", "--output", "artifacts/manuscript_completion/03_negative_controls", "--outcome-permutations", outcome_perm, "--pairing-permutations", pair_perm],
         "marker": "artifacts/manuscript_completion/03_negative_controls/negative_control_summary.json"},
        {"id": "04_calibration_utility", "command": [str(PYTHON), script, "calibration", "--output", "artifacts/manuscript_completion/04_calibration_utility", "--bootstrap-repetitions", bootstrap],
         "marker": "artifacts/manuscript_completion/04_calibration_utility/calibration_metrics.csv"},
        {"id": "05_evidence_tables", "command": [str(PYTHON), script, "evidence-tables", "--output", "artifacts/manuscript_completion/05_evidence_tables"],
         "marker": "artifacts/manuscript_completion/05_evidence_tables/fair_model_comparison.csv"},
        {"id": "05b_external_phenotype_transport", "command": [str(PYTHON), "scripts/analyze_external_phenotype_transport.py", "--output", "artifacts/external_phenotype_transport", "--bootstrap-repetitions", bootstrap],
         "marker": "artifacts/external_phenotype_transport/metadata.json"},
        {"id": "05c_quality_overlap_transport", "command": [str(PYTHON), "scripts/analyze_quality_overlap_transport.py", "--output", "artifacts/quality_overlap_transport", "--bootstrap-repetitions", bootstrap],
         "marker": "artifacts/quality_overlap_transport/metadata.json"},
        {"id": "06_external_confirmation", "command": [str(PYTHON), script, "external-preflight", "--output", "artifacts/manuscript_completion/06_external_confirmation"],
         "marker": "artifacts/manuscript_completion/06_external_confirmation/external_preflight.json", "allow_blocked": True},
        {"id": "07_summary", "command": [str(PYTHON), script, "summary", "--output", "artifacts/manuscript_completion/07_summary"],
         "marker": "artifacts/manuscript_completion/07_summary/completion_summary.json", "always_run": True},
    ]


def run_step(item: dict, state: dict, force: bool) -> None:
    step_id = item["id"]
    marker = ROOT / item["marker"]
    current = state["steps"].get(step_id, {})
    # A marker produced by a blocked/failed/interrupted run is diagnostic output,
    # not evidence that the experiment completed.  In particular, rerun the
    # external step automatically after the missing cohort is later supplied.
    non_reusable_statuses = {"blocked_external_data", "failed", "failed_missing_marker", "running"}
    marker_is_reusable = current.get("status") not in non_reusable_statuses
    if marker.is_file() and marker.stat().st_size > 0 and marker_is_reusable and not item.get("always_run") and not force:
        reusable_status = current.get("status") if current.get("status") in {"completed", "completed_existing"} else "completed_existing"
        current.update({"status": reusable_status, "marker": item["marker"], "ended_at": current.get("ended_at", now())})
        state["steps"][step_id] = current; save_state(state)
        log(f"SKIP {step_id}: marker exists")
        return
    step_log = BASE / f"{step_id}.log"
    current = {"status": "running", "started_at": now(), "command": item["command"], "marker": item["marker"]}
    state["steps"][step_id] = current; save_state(state)
    log(f"START {step_id}")
    with step_log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(item["command"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode == 20 and item.get("allow_blocked"):
        current.update({"status": "blocked_external_data", "returncode": 20, "ended_at": now(), "log": str(step_log.relative_to(ROOT))})
        state["steps"][step_id] = current; save_state(state)
        log(f"BLOCKED {step_id}: independent external data absent; queue continues")
        return
    if result.returncode != 0:
        current.update({"status": "failed", "returncode": result.returncode, "ended_at": now(), "log": str(step_log.relative_to(ROOT))})
        state["steps"][step_id] = current; save_state(state)
        log(f"FAIL {step_id}: exit={result.returncode}; inspect {step_log}")
        raise SystemExit(result.returncode)
    if not marker.is_file() or marker.stat().st_size == 0:
        current.update({"status": "failed_missing_marker", "returncode": 2, "ended_at": now()})
        state["steps"][step_id] = current; save_state(state)
        log(f"FAIL {step_id}: expected marker missing or empty")
        raise SystemExit(2)
    current.update({"status": "completed", "returncode": 0, "ended_at": now(), "log": str(step_log.relative_to(ROOT))})
    state["steps"][step_id] = current; save_state(state)
    log(f"DONE {step_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Short smoke run with reduced bootstrap/permutation counts.")
    parser.add_argument("--force", action="store_true", help="Rerun steps even when markers exist.")
    parser.add_argument("--from-step", help="Start at this step id; prior steps are left unchanged.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(steps(args.quick), indent=2, ensure_ascii=False)); return
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} started={now()}\n".encode()); os.close(fd)
    except FileExistsError:
        raise SystemExit(f"Queue lock already exists: {LOCK}")
    state = load_state()
    try:
        active = args.from_step is None
        for item in steps(args.quick):
            if args.from_step and item["id"] == args.from_step:
                active = True
            if active:
                run_step(item, state, args.force)
        state["status"] = "completed_with_external_block" if any(x.get("status") == "blocked_external_data" for x in state["steps"].values()) else "completed"
        save_state(state)
        log(f"QUEUE {state['status']}")
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
