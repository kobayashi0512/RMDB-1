#!/usr/bin/env python3
"""Train a FIVES vessel segmenter; official test images remain untouched by default."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

def device(): return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class Fives(Dataset):
    def __init__(self, files, image_dir, mask_dir, size, augment=False):
        self.files, self.image_dir, self.mask_dir, self.size, self.augment = files, image_dir, mask_dir, size, augment
        self.norm = transforms.Normalize((.485,.456,.406),(.229,.224,.225))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        name=self.files[i]
        image=Image.open(self.image_dir/name).convert('RGB').resize((self.size,self.size), Image.BILINEAR)
        mask=Image.open(self.mask_dir/name).convert('L').resize((self.size,self.size), Image.NEAREST)
        if self.augment and random.random() < .5: image, mask = image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return self.norm(transforms.ToTensor()(image)), (torch.from_numpy(np.array(mask, dtype=np.float32))/255.).unsqueeze(0)

def dice(logits, target):
    pred=(logits.sigmoid()>.5).float(); return ((2*(pred*target).sum((1,2,3))+1e-6)/(pred.sum((1,2,3))+target.sum((1,2,3))+1e-6)).mean()
@torch.inference_mode()
def evaluate(model, loader, dev):
    model.eval(); values=[]
    for x,y in loader: values.append(dice(model(x.to(dev)),y.to(dev)).item())
    return float(np.mean(values))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('../data/FIVES')); p.add_argument('--size',type=int,default=512); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--workers',type=int,default=2); p.add_argument('--seed',type=int,default=2026); p.add_argument('--evaluate-test',action='store_true'); p.add_argument('--evaluate-only',action='store_true'); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    image_dir=a.root/'train'/'Original'; mask_dir=a.root/'train'/'Ground truth'; files=sorted(x.name for x in image_dir.glob('*.png') if (mask_dir/x.name).is_file()); random.Random(a.seed).shuffle(files); n_val=120; train_files,val_files=files[n_val:],files[:n_val]
    dev=device(); train=DataLoader(Fives(train_files,image_dir,mask_dir,a.size,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers); val=DataLoader(Fives(val_files,image_dir,mask_dir,a.size),batch_size=a.batch_size,num_workers=a.workers)
    model=smp.Unet(encoder_name='resnet34',encoder_weights='imagenet',in_channels=3,classes=1).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); bce=nn.BCEWithLogitsLoss(); out=Path('artifacts/fives_unet_resnet34'); out.mkdir(parents=True,exist_ok=True); best=-1; history=[]
    if a.evaluate_only:
        model.load_state_dict(torch.load(out/'best.pt',map_location=dev)); test=DataLoader(Fives(sorted(x.name for x in (a.root/'test'/'Original').glob('*.png')),a.root/'test'/'Original',a.root/'test'/'Ground truth',a.size),batch_size=a.batch_size,num_workers=a.workers); (out/'official_test.json').write_text(json.dumps({'dice':evaluate(model,test,dev)},indent=2)); return
    for epoch in range(1,a.epochs+1):
        model.train()
        for x,y in tqdm(train,leave=False):
            x,y=x.to(dev),y.to(dev); logits=model(x); soft=(2*(logits.sigmoid()*y).sum((1,2,3))+1e-6)/(logits.sigmoid().sum((1,2,3))+y.sum((1,2,3))+1e-6); loss=bce(logits,y)+(1-soft.mean()); opt.zero_grad(); loss.backward(); opt.step()
        score=evaluate(model,val,dev); history.append({'epoch':epoch,'val_dice':score}); (out/'history.json').write_text(json.dumps(history,indent=2)); print(history[-1])
        if score>best: best=score; torch.save(model.state_dict(),out/'best.pt')
    if a.evaluate_test:
        model.load_state_dict(torch.load(out/'best.pt',map_location=dev)); test=DataLoader(Fives(sorted(x.name for x in (a.root/'test'/'Original').glob('*.png')),a.root/'test'/'Original',a.root/'test'/'Ground truth',a.size),batch_size=a.batch_size,num_workers=a.workers); (out/'official_test.json').write_text(json.dumps({'dice':evaluate(model,test,dev)},indent=2))
if __name__=='__main__': main()
