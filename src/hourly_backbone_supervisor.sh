#!/bin/zsh
# Hourly non-invasive audit. It only inspects the queue this experiment owns.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/backbone_candidates_v1/hourly_monitor.log"
mkdir -p "$(dirname "$LOG")"
while true; do
  date -u >> "$LOG"
  "$ROOT/.venv/bin/python" "$ROOT/src/monitor_backbone_candidates.py" >> "$LOG" 2>&1 || true
  state="$ROOT/results/backbone_candidates_v1/queue_state.json"
  if [[ -f "$state" ]]; then
    complete=$("$ROOT/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s.get("complete_count",0))' "$state" 2>/dev/null || echo 0)
    failed=$("$ROOT/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s.get("failed_count",0))' "$state" 2>/dev/null || echo 0)
    if [[ "$complete" -ge 15 || "$failed" -gt 0 ]]; then
      date -u >> "$LOG"
      echo "terminal queue state: complete=$complete failed=$failed" >> "$LOG"
      break
    fi
  fi
  sleep 3600
done
