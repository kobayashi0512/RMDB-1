#!/usr/bin/env python3
"""Mark the RMDB backbone queue paused after its owned workers are stopped."""
from pathlib import Path
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "backbone_candidates_v1" / "queue_state.json"

if STATE.is_file():
    state = json.loads(STATE.read_text())
    for job in state.get("jobs", {}).values():
        # The pause command owns the workers, so an incomplete run that was
        # marked failed at SIGTERM is recorded as paused rather than as a
        # methodological failure. Its partial history is retained.
        if job.get("status") in {"running", "failed"} and not Path(job.get("output", ""), "summary.json").is_file():
            job["status"] = "paused"
            job["pause_reason"] = "user_requested_pause"
            job["paused_at"] = datetime.now(timezone.utc).isoformat()
            job["pid"] = None
    state["queue_status"] = "paused"
    state["paused_at"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(STATE)
    print(STATE)
else:
    print(f"queue state not found: {STATE}")
