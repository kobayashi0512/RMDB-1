"""Report the identity/input reference error on the exact fixed validation patches."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from structure_restoration_pilot import OUT,Patches,losses,write
torch.set_num_threads(2)
p=json.loads((OUT/'protocol.json').read_text());h=json.loads((OUT/'history.json').read_text())
rows=[]
for inp,ref,v in DataLoader(Patches(p['validation_ids'],False),batch_size=8,num_workers=0):
    _,mae,edge=losses(inp,ref,v,False);rows.append([float(mae),float(edge),len(inp)])
x=np.asarray(rows);m,e=np.average(x[:,:2],axis=0,weights=x[:,2])
result={'fixed_validation_patches':120,'input_mae':float(m),'input_edge_mae':float(e),'input_selection_score':float(m+.2*e),'selected':{}}
for a in ('reconstruction','structure'):
    c=torch.load(OUT/f'{a}_best.pt',map_location='cpu');epoch=int(c['epoch'])
    result['selected'][a]={'epoch':epoch,**h[epoch-1]['validation'][a],'best_at_budget_boundary':epoch==20}
write(OUT/'validation_reference_baseline.json',result)
print(json.dumps(result,indent=2))
