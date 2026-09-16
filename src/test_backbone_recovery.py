"""Integration checks: epoch resume equivalence and supervisor pause semantics."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from backbone_runtime import atomic_json, process_matches
from run_backbone_recovery import ROOT

def fake_supervisor():
    import run_backbone_recovery as runner
    runner.build_command=lambda args,candidate,seed,out: [sys.executable,'-c','import time; time.sleep(90)',str(out)]
    sys.argv=[sys.argv[0],'--output-root',sys.argv[2],'--poll-seconds','1']
    runner.main()

def main():
    destination=Path(tempfile.mkdtemp(prefix='backbone_runtime_qa_',dir=ROOT/'results'))
    common=[sys.executable,str(ROOT/'src/train_birap_v2.py'),
        '--rav',str(ROOT/'data/rav_vascx'),
        '--rav-ground-truth',str(ROOT/'data/rav_geometry_recovery_20260907/aligned_masks'),
        '--rav-metadata',str(ROOT/'data/raw/RAV/data/external/RAV/metadata.csv'),
        '--fives',str(ROOT/'data/raw/FIVES'),
        '--occupancy-checkpoint',str(ROOT/'models/baseline/artifacts/fives_unet_resnet34/best.pt'),
        '--mass-checkpoint',str(ROOT/'models/baseline/artifacts/udct_av_mass_calibrated/best.pt'),
        '--backbone','highres','--variant','full','--epochs','2','--train-size','64','--eval-size','64',
        '--crop-sizes','128','--batch-size','2','--rav-validation','2','--fives-validation','2',
        '--limit-train','2','--limit-validation','2','--skip-holdout','--managed-run','--seed','20260906']
    def train(out,extra=(),expected=0):
        with (destination/(out+str(len(extra))+'.log')).open('w') as handle:
            result=subprocess.run(common+['--output',str(destination/out)]+list(extra),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        assert result.returncode==expected,(out,result.returncode,destination)
    train('continuous')
    train('interrupted',['--stop-after-epoch','1'],75)
    train('interrupted',['--resume'])
    one=torch.load(destination/'continuous/last.pt',map_location='cpu',weights_only=False)
    two=torch.load(destination/'interrupted/last.pt',map_location='cpu',weights_only=False)
    assert one['epoch']==two['epoch']==2
    difference=max((one['model'][k].float()-two['model'][k].float()).abs().max().item() for k in one['model'])
    assert difference<1e-5,difference
    assert one['split']==two['split']
    assert one['optimizer']['state'].keys()==two['optimizer']['state'].keys()
    queue=destination/'queue'
    with (destination/'fake_supervisor.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,__file__,'--fake-supervisor',str(queue)],stdout=log,stderr=subprocess.STDOUT)
    for _ in range(100):
        if (queue/'queue_state.json').exists():
            state=json.loads((queue/'queue_state.json').read_text())
            if state['active_count']==2: break
        time.sleep(.1)
    else:
        proc.terminate(); raise AssertionError('Supervisor did not launch fake workers')
    workers=[j['pid'] for j in state['jobs'].values() if j['pid']]
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=15)
    final=json.loads((queue/'queue_state.json').read_text())
    assert final['queue_status']=='paused' and final['active_count']==0
    assert sum(j['status']=='paused' for j in final['jobs'].values())==2
    assert sum(j['status']=='queued' for j in final['jobs'].values())==13
    assert not any(process_matches(pid,'time.sleep') for pid in workers)
    result={'qa_only':True,'resumed_model_max_abs_difference':difference,
            'checkpoint_epoch':two['epoch'],'pause_stopped_all_workers':True,
            'no_replacement_jobs_after_pause':True,'output':str(destination)}
    atomic_json(destination/'verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fake-supervisor': fake_supervisor()
    else: main()
