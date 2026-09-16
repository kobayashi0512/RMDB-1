"""Independent final checkpoint integrity and optimizer budget audit."""
import json
import torch
from train_controlled_residual_ablation import OUT, ARMS, SEEDS, count, model_for
from backbone_runtime import atomic_json

def main():
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            path=OUT/'runs'/arm/str(seed)/'last.pt'
            ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['epoch']==3
            assert all(torch.isfinite(t).all() for t in ck['model'].values())
            steps={int(state['step'].item()) for state in ck['optimizer']['state'].values()}
            assert steps=={270}, (arm,seed,steps)
            net=model_for(arm);net.load_state_dict(ck['model'],strict=True)
            rows.append({'arm':arm,'seed':seed,'parameters':count(net),'all_weights_finite':True,'optimizer_steps':270})
    atomic_json(OUT/'checkpoint_audit.json',{'runs':rows,'all_21_verified':True})
    print('21 checkpoints finite, strictly loadable, all optimizers at exactly 270 steps.')

if __name__=='__main__': main()
