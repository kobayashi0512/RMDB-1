#!/usr/bin/env python3
"""Start or pause only the checkpointed backbone recovery queue."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from backbone_runtime import atomic_json, process_matches
from run_backbone_recovery import ROOT, DEFAULT_OUTPUT

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['start','pause']); a=p.parse_args()
    DEFAULT_OUTPUT.mkdir(parents=True,exist_ok=True)
    state_path=DEFAULT_OUTPUT/'queue_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    pid=state.get('supervisor_pid')
    if a.action=='pause':
        atomic_json(DEFAULT_OUTPUT/'PAUSE',{'reason':'user_requested_pause'})
        if process_matches(pid,'run_backbone_recovery.py'):
            os.kill(pid,signal.SIGTERM)
        else:
            for job in state.get('jobs',{}).values():
                if process_matches(job.get('pid'),job['output']):
                    os.kill(job['pid'],signal.SIGTERM)
            state['queue_status']='paused'
            atomic_json(state_path,state)
        print('Pause requested; the supervisor will stop dispatch and drain its workers.')
        return
    if process_matches(pid,'run_backbone_recovery.py'):
        print(f'Supervisor already running: {pid}'); return
    for job in state.get('jobs',{}).values():
        if process_matches(job.get('pid'),job['output']):
            raise SystemExit('Existing worker found; do not duplicate it.')
    command=[sys.executable,str(ROOT/'src/run_backbone_recovery.py'),'--max-parallel','2',
             '--epochs','12','--train-size','512','--eval-size','768','--batch-size','4','--resume-queue']
    with (DEFAULT_OUTPUT/'supervisor.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
    # Prevent idle sleep while this explicitly requested local experiment runs.
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(proc.pid)],stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps({'supervisor_pid':proc.pid,'state':str(state_path)}))

if __name__=='__main__': main()
