#!/usr/bin/env python3
"""Small, fixed FIVES restoration experiment. Never opens clinical/AVSeg data.

The structure arm adds vessel-weighted reconstruction and image-gradient
preservation; it does NOT implement a topological guarantee or AV supervision.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/structure_restoration_pilot'
FIVES=ROOT/'data/raw/FIVES/train'
SPLIT=ROOT/'results/birap_v2_residual_full_512/split.json'
ARMS=('reconstruction','structure')
SEED=20260906

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));temp.replace(path)

def seed_int(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def degrade(rgb,rng,severity='train'):
    """Image-only synthetic stress; no label access and no spatial remapping."""
    f=rgb.astype(np.float32)/255;h,w=f.shape[:2]
    if severity=='moderate':
        sigma=1.3;scale=.65;noise=.012;gamma=1.3;shade=.25;jpeg=65
    else:
        sigma=float(rng.uniform(.3,1.7));scale=float(rng.uniform(.55,1));noise=float(rng.uniform(.002,.018));gamma=float(rng.uniform(.8,1.4));shade=float(rng.uniform(0,.3));jpeg=int(rng.integers(55,96))
    f=cv2.GaussianBlur(f,(0,0),sigma)
    f=cv2.resize(cv2.resize(f,(max(8,int(w*scale)),max(8,int(h*scale))),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
    yy,xx=np.mgrid[-1:1:complex(h),-1:1:complex(w)]
    angle=float(rng.uniform(0,2*np.pi));illum=1-shade*(.5+.5*(xx*np.cos(angle)+yy*np.sin(angle))/math.sqrt(2))
    f=np.clip(f,0,1)**gamma*illum[...,None]
    f+=rng.normal(0,noise,f.shape)
    encoded=cv2.imencode('.jpg',cv2.cvtColor(np.uint8(np.clip(f*255,0,255)),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,jpeg])[1]
    result=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    field=rgb.max(-1)>10;result[~field]=rgb[~field]
    return result

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())

class Restorer(nn.Module):
    def __init__(self):
        super().__init__();self.e1=block(3,16);self.e2=block(16,32);self.mid=block(32,64);self.d2=block(96,32);self.d1=block(48,16);self.head=nn.Conv2d(16,3,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));z=self.mid(F.avg_pool2d(b,2))
        z=self.d2(torch.cat([F.interpolate(z,b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        z=self.d1(torch.cat([F.interpolate(z,a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        field=(x.amax(1,keepdim=True)>.04).to(x.dtype)
        return torch.clamp(x+.25*torch.tanh(self.head(z))*field,0,1)

def gradients(x):return (x[:,:,:,1:]-x[:,:,:,:-1],x[:,:,1:,:]-x[:,:,:-1,:])

def losses(pred,target,vessel,structured):
    # Common validation criterion is deliberately identical for both arms.
    absolute=(pred-target).abs();mae=absolute.mean();gx,gy=gradients(pred);tx,ty=gradients(target)
    edge=((gx-tx).abs().mean()+(gy-ty).abs().mean())/2
    total=mae
    if structured:
        region=(absolute*vessel).sum()/(3*vessel.sum()+1e-6)
        wx=torch.maximum(vessel[:,:,:,1:],vessel[:,:,:,:-1]);wy=torch.maximum(vessel[:,:,1:,:],vessel[:,:,:-1,:])
        vgrad=((gx-tx).abs()*wx).sum()/(3*wx.sum()+1e-6)+((gy-ty).abs()*wy).sum()/(3*wy.sum()+1e-6)
        total=mae+.5*region+.2*vgrad
    return total,mae,edge

class Patches(Dataset):
    def __init__(self,ids,training):self.ids=ids;self.training=training;self.epoch=0
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        ident=self.ids[i];rng=np.random.default_rng(seed_int(f'{SEED}:{ident}:{self.epoch if self.training else -1}'))
        rgb=np.load(OUT/'cache'/f'{ident}.rgb.npy',mmap_mode='r');mask=np.load(OUT/'cache'/f'{ident}.mask.npy',mmap_mode='r')
        size=192
        # Use the same label-blind crop draws for both training arms.
        for _ in range(12):
            y=int(rng.integers(0,rgb.shape[0]-size+1));x=int(rng.integers(0,rgb.shape[1]-size+1))
            ref=np.array(rgb[y:y+size,x:x+size]);v=np.array(mask[y:y+size,x:x+size])
            if (ref.max(-1)>10).mean()>.75:break
        if self.training:
            k=int(rng.integers(0,4));ref=np.rot90(ref,k).copy();v=np.rot90(v,k).copy()
        # Undegraded examples penalize unnecessary changes through the same reconstruction loss.
        inp=ref.copy() if self.training and rng.random()<.25 else degrade(ref,rng)
        return torch.from_numpy(inp.transpose(2,0,1).copy()).float()/255,torch.from_numpy(ref.transpose(2,0,1).copy()).float()/255,torch.from_numpy(v[None].copy()).float()

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    split=json.loads(SPLIT.read_text());train=split['fives_training'];val=split['fives_validation']
    assert len(train)==480 and len(val)==120 and not(set(train)&set(val))
    protocol={'training_ids':train,'validation_ids':val,'rav_evaluation_ids':split['rav_validation'],'split_source':str(SPLIT),'split_sha256':sha(SPLIT),'seed':SEED,'epochs':20,'batch_size':8,'patch':192,'canonical_size':1024,'arms':list(ARMS),'model':'small residual U-Net; 16/32/64 channels; delta bounded by 0.25; identity initialization; background bypass','loss_reconstruction':'pixel MAE','loss_structure':'pixel MAE + 0.5 vessel-region MAE + 0.2 sum of two vessel-weighted gradient MAEs','no_AV_training_supervision':True,'not_a_topology_guarantee':True,'same_architecture_initialization_batches_and_optimizer':True,'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.0001,'cosine_final_fraction':.1},'validation_selection':'lowest MAE + 0.2 unweighted gradient MAE on fixed FIVES validation patches; same criterion for both','clean_training_fraction':.25,'reference_caution':'FIVES original images are reference targets, not certified degradation-free ground truth','degradations':'Gaussian blur sigma .3-1.7; downsample .55-1; noise .002-.018; gamma .8-1.4; spatial shading 0-.3; JPEG55-95; no geometry change','evaluation':'all 30 existing RAV validation images, undegraded and fixed moderate synthetic stress, four arms input/CLAHE/reconstruction/structure; frozen 5-member VascX, no TTA; 768 scoring; no model selection with RAV results','primary':'structure minus CLAHE mean AV Dice on moderate stress','additional_control':'structure minus matched reconstruction-only network','pilot_success_rule':'primary mean gain >=.005 and paired CI lower>0; precision loss no worse than .005; AV-log-error increase <=.005; undegraded Dice loss vs input <=.005; all heuristic feasibility guards, not clinical margins','bootstrap_unit':'image, 3000 paired resamples; two conditions kept separate; RAV patient independence not established','stop_rule':'finish all fixed comparisons then stop; negative branches are not retuned','prohibited':'No Fundus-AVSeg, CIMT, mBRSET outcomes, or FIVES official test access','novelty_caution':'structure preserving fundus enhancement already exists; this is a prototype, not a state-of-the-art comparison','related_work':['https://arxiv.org/abs/2005.05594','https://arxiv.org/abs/2302.11795']}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:write(OUT/'protocol.json',protocol)
    hashes={}
    for i,name in enumerate(train+val):
        ip=FIVES/'Original'/name;mp=FIVES/'Ground truth'/name
        assert ip.is_file() and mp.is_file()
        hashes[name]={'image':sha(ip),'mask':sha(mp)}
        cp=OUT/'cache'/f'{name}.rgb.npy';vp=OUT/'cache'/f'{name}.mask.npy'
        if not(cp.exists() and vp.exists()):
            rgb=cv2.cvtColor(cv2.imread(str(ip)),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
            assert rgb.shape[:2]==mask.shape and rgb.shape[0]==rgb.shape[1]
            np.save(cp,cv2.resize(rgb,(1024,1024),interpolation=cv2.INTER_AREA))
            np.save(vp,(cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_NEAREST)>127).astype(np.uint8))
        if (i+1)%100==0:print('prepared',i+1,'/600',flush=True)
    write(OUT/'input_sha256.json',hashes)
    return train,val

@torch.inference_mode()
def validate(models,loader,device):
    result={a:[] for a in ARMS}
    for model in models.values():model.eval()
    for inp,ref,v in loader:
        inp,ref,v=inp.to(device),ref.to(device),v.to(device)
        for a,model in models.items():
            pred=model(inp);_,mae,edge=losses(pred,ref,v,False)
            result[a].append([float(mae.cpu()),float(edge.cpu()),len(inp)])
    answer={}
    for a,rows in result.items():
        x=np.asarray(rows);mae,edge=np.average(x[:,:2],axis=0,weights=x[:,2]);answer[a]={'mae':float(mae),'edge_mae':float(edge),'selection_score':float(mae+.2*edge)}
    return answer

def train():
    torch.set_num_threads(4);cv2.setNumThreads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    train_ids,val_ids=prepare()
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    initial=Restorer();models={a:copy.deepcopy(initial).to(device) for a in ARMS}
    opts={a:torch.optim.AdamW(models[a].parameters(),lr=.0003,weight_decay=.0001) for a in ARMS}
    dataset=Patches(train_ids,True);validation=DataLoader(Patches(val_ids,False),batch_size=8,num_workers=0)
    best={a:float('inf') for a in ARMS};history=[];start_epoch=0
    if (OUT/'last.pt').exists():
        c=torch.load(OUT/'last.pt',map_location=device);start_epoch=c['epoch'];best=c['best'];history=c['history']
        for a in ARMS:models[a].load_state_dict(c['models'][a]);opts[a].load_state_dict(c['optimizers'][a])
    start=time.monotonic()
    for epoch in range(start_epoch,20):
        dataset.epoch=epoch;gen=torch.Generator().manual_seed(SEED+epoch)
        loader=DataLoader(dataset,batch_size=8,shuffle=True,generator=gen,num_workers=0)
        running={a:[] for a in ARMS};lr=.0003*(.1+.9*(1+math.cos(math.pi*epoch/19))/2)
        for a in ARMS:
            models[a].train()
            for g in opts[a].param_groups:g['lr']=lr
        for inp,ref,v in loader:
            inp,ref,v=inp.to(device),ref.to(device),v.to(device)
            for a in ARMS:
                opts[a].zero_grad(set_to_none=True);pred=models[a](inp);loss,_,_=losses(pred,ref,v,a=='structure')
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(models[a].parameters(),1.);opts[a].step();running[a].append(float(loss.detach().cpu()))
        scores=validate(models,validation,device)
        row={'epoch':epoch+1,'lr':lr,'elapsed_seconds':time.monotonic()-start,'train_loss':{a:float(np.mean(running[a])) for a in ARMS},'validation':scores}
        for a in ARMS:
            if scores[a]['selection_score']<best[a]:
                best[a]=scores[a]['selection_score'];torch.save({'model':models[a].state_dict(),'epoch':epoch+1,'criterion':best[a],'arm':a},OUT/f'{a}_best.pt')
        history.append(row);write(OUT/'history.json',history)
        torch.save({'epoch':epoch+1,'models':{a:models[a].state_dict() for a in ARMS},'optimizers':{a:opts[a].state_dict() for a in ARMS},'best':best,'history':history},OUT/'last.tmp');(OUT/'last.tmp').replace(OUT/'last.pt')
        print(json.dumps(row),flush=True)
    write(OUT/'training_complete.json',{'epochs':20,'training_images':480,'validation_images':120,'parameters_per_model':sum(p.numel() for p in initial.parameters()),'best_criterion':best,'checkpoint_sha256':{a:sha(OUT/f'{a}_best.pt') for a in ARMS}})
    print('TRAINING COMPLETE',flush=True)

@torch.inference_mode()
def restore_image(model,rgb,device):
    # Overlapping tiles limit memory. A constant image cannot create a seam by padding.
    h,w=rgb.shape[:2];tile=384;stride=256;ys=list(range(0,max(h-tile,0)+1,stride));xs=list(range(0,max(w-tile,0)+1,stride))
    if ys[-1]!=h-tile:ys.append(h-tile)
    if xs[-1]!=w-tile:xs.append(w-tile)
    total=np.zeros((h,w,3),np.float32);weight=np.zeros((h,w,1),np.float32)
    win=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.001)[...,None].astype(np.float32)
    for y in ys:
        for x in xs:
            patch=rgb[y:y+tile,x:x+tile];t=torch.from_numpy(patch.transpose(2,0,1).copy()).float()[None].to(device)/255
            out=model(t)[0].cpu().numpy().transpose(1,2,0)
            total[y:y+tile,x:x+tile]+=out*win;weight[y:y+tile,x:x+tile]+=win
    return np.uint8(np.clip(np.rint(total/weight*255),0,255))

if __name__=='__main__':train()
