#!/usr/bin/env python3
"""Compact audit for the resumable backbone queue."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from datetime import datetime, timezone
from backbone_runtime import process_matches

def now(): return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--state',type=Path,default=Path(__file__).resolve().parents[1]/'results/backbone_candidates_recovery_20260907/queue_state.json'); p.add_argument('--json',action='store_true'); a=p.parse_args()
    state=json.loads(a.state.read_text())
    rows=[]
    for key,job in state['jobs'].items():
        pid=job.get('pid'); alive=process_matches(pid, job['output'])
        out=Path(job['output']); history=out/'history.json'; summary=out/'summary.json'
        epochs=0; finite=True
        if history.is_file():
            try:
                h=json.loads(history.read_text()); epochs=len(h)
                finite=all(math.isfinite(float(r['train_loss'])) for r in h)
            except Exception: finite=False
        detail=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        effective='interrupted' if job['status']=='running' and not alive else job['status']
        rows.append({'job':key,'status':effective,'recorded_status':job['status'],'pid':pid,'alive':alive,'epochs':epochs,'finite_loss':finite if epochs else None,'has_summary':summary.is_file(),'progress':detail})
    result={'checked_at':now(),'queue':str(a.state),'queue_status':state.get('queue_status'),'supervisor_alive':process_matches(state.get('supervisor_pid'),'run_backbone_recovery.py'),'jobs':rows,'counts':{s:sum(x['status']==s for x in rows) for s in ('queued','running','complete','failed','paused','interrupted')}}
    if a.json: print(json.dumps(result,indent=2,ensure_ascii=False))
    else:
        print(f"[{result['checked_at']}] {result['counts']}")
        for r in rows: print(f"{r['job']:28s} {r['status']:8s} pid={str(r['pid']):>7s} alive={str(r['alive']):5s} epochs={r['epochs']:>2d} finite={r['finite_loss']}")
if __name__=='__main__': main()
