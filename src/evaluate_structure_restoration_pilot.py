#!/usr/bin/env python3
"""One-shot fixed RAV stress/safety evaluation after FIVES training freezes."""
import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity
from rtnls_fundusprep.cfi_bounds import CFIBounds
from rtnls_fundusprep.transformation import Interpolation
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.ensembles.predict_output import decollate_predict_full
from structure_restoration_pilot import ROOT,OUT,Restorer,degrade,restore_image,seed_int,sha,write
from train_udct_av_pilot import target_labels,softmax_np
from train_birap_v2 import case_metrics

ARMS=('input','clahe','reconstruction','structure')
METRICS=('mean_av_dice','vessel_dice','vessel_precision','vessel_recall','thin_centerline_recall','av_ratio_log_error','brier','reference_mae','reference_psnr','reference_ssim')

def clahe(rgb,field):
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB);lab[...,0]=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(lab[...,0])
    x=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    out=np.uint8(.5*rgb.astype(np.float32)+.5*x.astype(np.float32));out[~field]=rgb[~field];return out

def interval(d,metric):
    d=np.asarray(d);rng=np.random.default_rng(20260906);samples=d[rng.integers(0,len(d),(3000,len(d)))].mean(1)
    lower=metric in ('brier','av_ratio_log_error','reference_mae')
    return {'delta':float(d.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'n':len(d),'improved':int(((d<0) if lower else (d>0)).sum())}

def summarize(protocol):
    frame=pd.DataFrame([json.loads(p.read_text()) for p in sorted((OUT/'evaluation/cases').glob('*.json'))])
    if frame.empty:return
    frame.to_csv(OUT/'evaluation/per_image.csv',index=False)
    report={'complete':len(frame)==len(protocol['rav_evaluation_ids'])*8,'n_images':len(protocol['rav_evaluation_ids']),'evaluations_done':len(frame),'exploratory_only':True,'conditions':{}}
    for condition,g in frame.groupby('condition'):
        dest={'means':{},'contrasts':{}}
        for a,x in g.groupby('arm'):dest['means'][a]={m:float(x[m].mean()) for m in METRICS}
        for a,b in [('clahe','input'),('reconstruction','input'),('structure','input'),('structure','clahe'),('structure','reconstruction')]:
            key=a+'_minus_'+b;dest['contrasts'][key]={}
            for m in METRICS:
                p=g.pivot(index='id',columns='arm',values=m)
                if a in p and b in p:
                    d=(p[a]-p[b]).dropna()
                    if len(d):dest['contrasts'][key][m]=interval(d,m)
        report['conditions'][condition]=dest
    if report['complete']:
        moderate=report['conditions']['moderate']['contrasts']['structure_minus_clahe'];clean=report['conditions']['undegraded']['contrasts']['structure_minus_input']
        gates={'dice_gain_at_least_005':moderate['mean_av_dice']['delta']>=.005,'dice_lower_CI_positive':moderate['mean_av_dice']['ci95'][0]>0,'precision_loss_within_005':moderate['vessel_precision']['delta']>=-.005,'av_error_increase_within_005':moderate['av_ratio_log_error']['delta']<=.005,'undegraded_dice_loss_within_005':clean['mean_av_dice']['delta']>=-.005}
        report['feasibility_guards']=gates;report['pilot_passed_all_guards']=all(gates.values())
    write(OUT/'evaluation/summary.json',report)

def final_report():
    s=json.loads((OUT/'evaluation/summary.json').read_text());assert s['complete']
    history=json.loads((OUT/'history.json').read_text());training=json.loads((OUT/'training_complete.json').read_text())
    selected={a:int(torch.load(OUT/f'{a}_best.pt',map_location='cpu')['epoch']) for a in ('reconstruction','structure')}
    lines=['# 血管结构约束图像恢复：固定小规模验证','',
    '状态：两种恢复网络完成 20 轮 FIVES 开发训练；RAV 30 张图像、两种条件、四种处理，共 240 个推理完成。本报告是可行性试验，不是临床验证或新算法优越性的最终证明。','',
    '## 设计与隔离','',
    '- 复用已存在的 FIVES 480/120 开发划分。原图是配对参考，不保证完全清晰；未读取 FIVES 官方 test。',
    '- 两网络架构、初始化、训练批次和优化器相同；仅结构约束损失不同。每图每轮一个 192 像素 patch，共每分支 9600 个训练 patch。',
    '- 普通模型使用重建 MAE。结构模型另加血管区域 MAE 与血管加权图像梯度误差；并非严格拓扑约束、不是 clDice，也未使用动静脉训练监督。',
    '- 输出保留原图残差旁路，单像素修正上界 0.25；25% 训练样本无人工退化。限制修正幅度不保证没有假血管，因此另作像素评价。',
    '- 检查点仅按 FIVES 固定验证 patch 的共同标准 MAE+0.2 梯度误差选择。RAV 结果不用于选择或调参。',
    '- RAV 已被既往研究查看，且冻结 VascX 与其存在开发背景关联；不能称为全新独立外部队列。未读取 Fundus-AVSeg、CIMT 或 mBRSET。',
    '- 固定中等退化：模糊σ1.3、缩小到65%再放大、噪声0.012、gamma1.3、不均匀照明0.25、JPEG65。未人工退化条件用于检查增强是否损伤现有信息。','',
    f"每模型参数：{training['parameters_per_model']}。选择轮次：{selected}。是否充分收敛需结合 history.json，20 轮只代表本轮固定预算，不保证达到性能上限。",'',
    '## 结果','',
    '|条件|处理|mean AV Dice|血管精确率|血管召回率|细中心线召回|AV比例误差↓|四类均值Brier↓|参考MAE↓|参考PSNR↑|',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    columns=('mean_av_dice','vessel_precision','vessel_recall','thin_centerline_recall','av_ratio_log_error','brier','reference_mae','reference_psnr')
    for condition,d in s['conditions'].items():
        for a in ARMS:lines.append('|'+condition+'|'+a+'|'+'|'.join(f"{d['means'][a][m]:.5f}" for m in columns)+'|')
    lines+=['','## 主对照与约束贡献','',
    '|条件|对照|Δmean AV Dice|95% CI|','|---|---|---:|---|']
    for c,d in s['conditions'].items():
        for contrast in ('structure_minus_clahe','structure_minus_reconstruction','structure_minus_input'):
            v=d['contrasts'][contrast]['mean_av_dice'];lines.append(f"|{c}|{contrast}|{v['delta']:+.5f}|[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]|")
    lines+=['','## 预先固定的可行性判断','']+[f"- {k}: {'通过' if v else '未通过'}" for k,v in s['feasibility_guards'].items()]
    lines+=['',('通过本轮全部可行性守则，但仍需独立验证及标准恢复模型对照。' if s['pilot_passed_all_guards'] else '未通过全部守则。本轮停止该分支，不在 RAV 或 Fundus-AVSeg 上继续调参找正结果。'),'','## 解读边界','',
    '人工退化可检验已知信息被损坏后的恢复，但不能证明能恢复真实拍摄时丢失的信息。重建更好与分割更好是不同终点。普通重建网络是同架构消融对照，不是 Cofe-Net 等成熟方法的替代复现；本轮不能声称优于这些方法。','',
    '置信区间是图像级配对 bootstrap 3000 次，未作多重校正；RAV 的患者独立性不能由文件名保证。两种退化条件不当作独立患者合并。PSNR 对零误差采用 1e-12 MSE 下限，120dB 是数值上限而非估计值。','',
    'CLAHE 配置固定，来源于之前探索；此处禁止把0.55错误坐标结果用作基线。所有 RAV 评分使用原始图像裁剪的同一几何变换，绿色未知类别只忽略身份、保留血管支持，红蓝交叠属于交叉点。','',
    '已有相关工作：[Cofe-Net](https://arxiv.org/abs/2005.05594)、[MAGE-Net](https://arxiv.org/abs/2302.11795)。本原型没有首次提出结构保持增强的资格声明。','',
    '复现文件：protocol.json、history.json、training_complete.json、evaluation/per_image.csv、evaluation/summary.json、integrity.json。图像与概率保存在 evaluation/inputs、evaluation/predictions、evaluation/probabilities。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--training-pid',type=int);args=p.parse_args()
    started=time.monotonic()
    while not(OUT/'training_complete.json').exists():
        if args.training_pid:
            try:os.kill(args.training_pid,0)
            except ProcessLookupError:raise RuntimeError('Training process stopped without completion; inspect training log')
        if time.monotonic()-started>6*3600:raise TimeoutError('Training wait exceeded six hours')
        time.sleep(5)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    protocol=json.loads((OUT/'protocol.json').read_text());frozen=json.loads((OUT/'training_complete.json').read_text())
    for a in ('reconstruction','structure'):assert sha(OUT/f'{a}_best.pt')==frozen['checkpoint_sha256'][a]
    for d in ('cases','inputs','predictions','probabilities'):(OUT/'evaluation'/d).mkdir(parents=True,exist_ok=True)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    restorers={a:Restorer().to(device).eval() for a in ('reconstruction','structure')}
    for a in restorers:restorers[a].load_state_dict(torch.load(OUT/f'{a}_best.pt',map_location=device)['model'])
    model=SegmentationEnsemble.from_torchscript(ROOT/'models/third_party/vascx_weights/av_july24.pt').to(device).eval();model.config['inference'].update({'tta':False,'tile_batch_size':2})
    cache=ROOT/'data/rav_vascx';raw=ROOT/'data/raw/RAV/data/external/RAV/dataset';bounds_table=pd.read_csv(cache/'bounds_latest_batch.csv').set_index('id')
    geometry_records=[];start=time.monotonic()
    for i,ident in enumerate(protocol['rav_evaluation_ids']):
        name='RAV_'+ident;rgb=np.asarray(Image.open(raw/'rgb'/f'{name}.png').convert('RGB'))
        b=ast.literal_eval(re.sub(r'np\.float64\(([^()]*)\)',r'\1',bounds_table.loc[name,'bounds']))
        transform,bounds=CFIBounds.from_dict(rgb,b).crop(1024);reference=transform.warp(rgb,(1024,1024))
        assert np.array_equal(reference,np.asarray(Image.open(cache/'preprocessed_rgb'/f'{name}.png'))),name
        mask=np.asarray(Image.open(raw/'av_masks'/f'{name}.png').convert('RGB'));assert mask.shape==rgb.shape
        ann=cv2.resize(transform.warp(mask,(1024,1024),mode=Interpolation.NEAREST),(768,768),interpolation=cv2.INTER_NEAREST)
        target=target_labels(ann);vessel=ann.max(-1)>0;thin=skeletonize(vessel)&(distance_transform_edt(vessel)<=2)
        geometry_records.append({'id':ident,'reference_sha256':sha(raw/'rgb'/f'{name}.png'),'annotation_sha256':sha(raw/'av_masks'/f'{name}.png'),'exact_cached_rgb_reconstruction':True})
        for condition in ('undegraded','moderate'):
            input_rgb=reference.copy() if condition=='undegraded' else degrade(reference,np.random.default_rng(seed_int('eval:'+ident)),'moderate')
            for a in ARMS:
                key=f'{ident}__{condition}__{a}';resultpath=OUT/'evaluation/cases'/f'{key}.json'
                if resultpath.exists():continue
                if a=='input':enhanced=input_rgb
                elif a=='clahe':enhanced=clahe(input_rgb,bounds.make_binary_mask(.01))
                else:enhanced=restore_image(restorers[a],input_rgb,device)
                ip=OUT/'evaluation/inputs'/f'{key}.png';cp=OUT/'evaluation/inputs'/f'{key}__ce.png'
                Image.fromarray(enhanced).save(ip);Image.fromarray(CFIBounds.from_dict(enhanced,bounds.to_dict()).contrast_enhanced_5).save(cp)
                loader=model._make_inference_dataloader({'images':[{'id':key,'image':str(ip),'contrast_enhanced':str(cp)}]},preprocess=False,batch_size=1,num_workers=0,ignore_exceptions=False)
                with torch.inference_mode():out=[x for batch in loader for x in decollate_predict_full(model.predict_step_full(batch))]
                assert len(out)==1;logits=np.asarray(out[0]['logits'],np.float32);assert logits.shape==(1024,1024,4) and np.isfinite(logits).all()
                prob=np.moveaxis(softmax_np(cv2.resize(logits,(768,768))),-1,0);pred=prob.argmax(0)>0
                m=case_metrics(prob,target,vessel);tp=(pred&vessel).sum()
                m.update(vessel_precision=float(tp/max(pred.sum(),1)),vessel_recall=float(tp/max(vessel.sum(),1)),thin_centerline_recall=float((pred&thin).sum()/max(thin.sum(),1)))
                error=(enhanced.astype(np.float32)-reference.astype(np.float32))/255
                m.update(reference_mae=float(np.abs(error).mean()),reference_psnr=float(-10*np.log10(max(float((error**2).mean()),1e-12))),reference_ssim=float(structural_similarity(reference,enhanced,channel_axis=-1,data_range=255)))
                assert all(np.isfinite(v) for v in m.values())
                np.save(OUT/'evaluation/probabilities'/f'{key}.npy',prob)
                palette=np.array([[0,0,0],[255,0,0],[0,0,255],[255,0,255]],np.uint8);Image.fromarray(palette[prob.argmax(0)]).save(OUT/'evaluation/predictions'/f'{key}.png')
                write(resultpath,{'id':ident,'condition':condition,'arm':a,**m})
                if device.type=='mps':torch.mps.empty_cache()
        print(f'evaluated {i+1}/30 images, eight conditions each, {time.monotonic()-start:.1f}s',flush=True)
        summarize(protocol)
    paths=list((OUT/'evaluation/probabilities').glob('*.npy'));assert len(paths)==240
    for path in paths:
        a=np.load(path);assert np.isfinite(a).all() and a.shape==(4,768,768) and np.allclose(a.sum(0),1,atol=2e-6)
    write(OUT/'integrity.json',{'outputs':240,'finite_probability_audit':True,'all_cached_RGB_exact':True,'images':geometry_records,'training_checkpoint_hashes':frozen['checkpoint_sha256'],'source_hashes':{p.name:sha(p) for p in (Path(__file__),ROOT/'src/structure_restoration_pilot.py')}})
    final_report();write(OUT/'complete.json',{'training':True,'evaluation':True,'audit':True});print('PIPELINE COMPLETE',flush=True)

if __name__=='__main__':main()
