"""Independent arithmetic audit and all-endpoint report for final controls."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, label, convolve, distance_transform_edt
from skimage.morphology import skeletonize
from final_measurement_controls import OUT,ROOT,OLD,ARMS,FEATURES,STRUCTURE,PAIRS,METRICS,FCACHE,WEIGHT,geometry,load_occupancy,retention,check_prob,rows_for_feature,verify_sources,estimate
from audit_report_three_readout import direct,LABELS
from evaluate_fundus_avseg_external import load_features
from train_udct_av_pilot import Corrector
from extract_mbrset_repeatability_features import readout_features
from run_fundus_fair_four_arm import sha
from three_readout_repeatability import write,CACHE as MCACHE

NAMES={'raw':'原始VascX','mass':'Mass UDCT','joint':'占据条件读出','smooth':'固定概率平滑','cleanup':'固定小连通域移除'}


def manual_structure(truth,fov):
    small=np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST),bool)
    sk=skeletonize(small);neigh=convolve(sk.astype(int),np.ones((3,3),int),mode='constant')-sk
    lab,n=label(small,structure=np.ones((3,3),int));sizes=np.bincount(lab.ravel())[1:]
    scales=np.array([2,4,8,16,32,64]);boxes=[]
    for s in scales:
        occupied=set(map(tuple,np.argwhere(sk)//s));boxes.append(max(len(occupied),1))
    fd=np.polyfit(np.log(1/scales),np.log(boxes),1)[0] if sk.sum()>=20 else np.nan
    return {'vessel_density':truth.sum()/fov.sum(),'skeleton_density':sk.sum()/65536,
            'branch_density':(sk&(neigh>=3)).sum()/65536,'endpoint_density':(sk&(neigh==1)).sum()/65536,
            'component_count':int((sizes>=8).sum()),'fractal_dimension':fd}


def audit():
    protocol=verify_sources();f=pd.read_csv(OUT/'fundus_per_image.csv');r=pd.read_csv(OUT/'mbrset_per_image.csv',dtype={'patient_id':str})
    assert len(f)==500 and set(f.id)==set(protocol['fundus_ids']) and not f.duplicated(['id','arm']).any()
    assert len(r)==1960 and set(r.id)==set(protocol['mbrset_ids']) and r.id.nunique()==1960
    assert set(r.columns)==set(['id','patient_id','eye','view','split']+[a+'_'+k for a in ARMS for k in FEATURES])
    assert r.groupby('patient_id').split.nunique().max()==1
    original=pd.read_csv(OLD/'per_image.csv',dtype={'patient_id':str}).set_index('id');merged=r.set_index('id')
    for k in ['patient_id','eye','view','split']:
        assert (merged[k].astype(str)==original.loc[merged.index,k].astype(str)).all()
    for a in ['raw','mass','joint']:
        for k in FEATURES:np.testing.assert_allclose(merged[a+'_'+k],original.loc[merged.index,a+'_'+k],atol=1e-12)
    est=pd.read_csv(OUT/'repeat_estimates.csv').set_index(['split','feature','arm','metric']);maximum=0.;bootchecks=0
    for split,g in r.groupby('split'):
        for feature in FEATURES:
            rows=rows_for_feature(g,feature);slow=direct([v[2] for v in rows])
            for met,val in slow.items():
                for j,a in enumerate(ARMS):
                    err=abs(val[j]-est.loc[(split,feature,a,met),'estimate']);maximum=max(maximum,err);assert err<1e-9
            # Verify patient-cluster copies against explicit concatenation.
            pids,ix=np.unique([v[0] for v in rows],return_inverse=True);rng=np.random.default_rng(19)
            n=np.array([len(v[2]) for v in rows],float);mu=np.stack([v[2].mean(0) for v in rows]);ss=np.stack([((v[2]-v[2].mean(0))**2).sum(0) for v in rows]);mad=np.stack([abs(v[2]-v[2].mean(0)).mean(0) for v in rows]);pair=np.stack([abs(v[2][:,None]-v[2][None,:])[np.triu_indices(len(v[2]),1)].mean(0) for v in rows])
            for _ in range(3):
                selected=rng.integers(0,len(pids),len(pids));w=np.bincount(selected,minlength=len(pids))[ix]
                explicit=[row[2] for j in selected for row in rows if row[0]==pids[j]]
                fast=estimate(w[None].astype(float),n,mu,ss,mad,pair);truth=direct(explicit)
                for met in fast:np.testing.assert_allclose(fast[met][0],truth[met],atol=1e-9,rtol=1e-9)
                bootchecks+=1
    print('Independent repeatability formulas passed',flush=True)
    # Independent mean, contrast and percentile-bootstrap arithmetic for reference metrics.
    points=pd.read_csv(OUT/'reference_estimates.csv').set_index(['metric','arm']);contrasts=pd.read_csv(OUT/'reference_contrasts.csv').set_index(['metric','contrast'])
    idx=np.random.default_rng(protocol['bootstrap']['seed']).integers(0,100,(5000,100));cierror=0.
    for met in METRICS+['abs_error_'+k for k in STRUCTURE]+['error_'+k for k in STRUCTURE]:
        v=f.pivot(index='id',columns='arm',values=met)[ARMS].to_numpy()
        for j,a in enumerate(ARMS):np.testing.assert_allclose(v[:,j].mean(),points.loc[(met,a),'estimate'],atol=1e-10)
        for a,b in PAIRS:
            delta=v[:,ARMS.index(a)]-v[:,ARMS.index(b)];ci=np.quantile(delta[idx].mean(1),[.025,.975]);row=contrasts.loc[(met,a+'_minus_'+b)]
            np.testing.assert_allclose([delta.mean(),*ci],[row.delta,row.ci_low,row.ci_high],atol=1e-10)
            cierror=max(cierror,float(abs(ci-np.array([row.ci_low,row.ci_high])).max()))
    table=pd.read_csv(FCACHE/'bounds_latest_batch.csv').set_index('id');referr=0.
    for ident in protocol['fundus_ids']:
        rgb,_,ann,_=geometry(ident,table)
        rgb=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR));fov=rgb.max(2)>10
        truth=(cv2.resize(ann,(512,512),interpolation=cv2.INTER_NEAREST).max(2)>0)&fov
        manual=manual_structure(truth,fov);sub=f[f.id.eq(ident)]
        for k,v in manual.items():
            err=float(abs(sub['reference_'+k]-v).max());referr=max(referr,err);assert err<1e-9,(ident,k,err)
            np.testing.assert_allclose(sub['error_'+k],sub['value_'+k]-v,atol=1e-10)
            np.testing.assert_allclose(sub['abs_error_'+k],abs(sub['error_'+k]),atol=1e-10)
    print('All100 manual-mask structural references independently reproduced',flush=True)
    # Replay six cases with original Mass loader and independent simple-control implementations.
    torch.set_num_threads(4);cv2.setNumThreads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    massnet=Corrector().to(device).eval();massnet.load_state_dict(torch.load(WEIGHT,map_location=device)['model']);occnet=load_occupancy(device)
    replayerr=0.;hashes=0
    for i in np.linspace(0,99,6,dtype=int):
        ident=protocol['fundus_ids'][i];native,_,aligned,_=geometry(ident,table)
        rgb=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR));fov=rgb.max(2)>10;truth=(cv2.resize(aligned,(512,512),interpolation=cv2.INTER_NEAREST).max(2)>0)&fov
        x,z,p=load_features(FCACHE/'preprocessed_rgb'/f'{ident}.png',FCACHE/'logits'/f'{ident}.npy',512)
        ox=torch.from_numpy(np.moveaxis(rgb.astype(np.float32)/255,-1,0)[None]).to(device)
        ox=(ox-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        with torch.inference_mode():mass=massnet(x[None].to(device),z[None].to(device))[0].softmax(0).cpu().numpy();o=occnet(ox).sigmoid()[0,0].cpu().numpy().astype(np.float16).astype(np.float32)
        q=p[1:]/np.maximum(p[1:].sum(0,keepdims=True),1e-7);joint=np.concatenate([1-o[None],o[None]*q],0)
        smooth=np.stack([gaussian_filter(channel,1,mode='reflect',truncate=4) for channel in p]);smooth/=smooth.sum(0,keepdims=True)
        mask=(p[1:].sum(0)>=.5)&fov;count,lab=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        sizes=np.bincount(lab.ravel(),minlength=count);remove=(lab>0)&(sizes[lab]<32);clean=p.copy();clean[:,remove]=0;clean[0,remove]=1
        for a,prob in zip(ARMS,[p,mass,joint,smooth,clean]):
            check_prob(prob);feat=readout_features(prob,fov,'');mask=(np.clip(prob,1e-7,1)[1:].sum(0)>=.5)&fov
            vals=retention(mask,truth);vals['predicted_support_fraction']=mask.sum()/fov.sum()
            vals.update({'value_'+k:feat[k] for k in STRUCTURE});row=f[f.id.eq(ident)&f.arm.eq(a)].iloc[0]
            for k,v in vals.items():
                err=abs(v-row[k]);replayerr=max(replayerr,err);assert err<1e-6,(ident,a,k,err)
        record=json.loads((OUT/'fundus_cases'/f'{ident}.json').read_text())
        for name,h in record['input_sha256'].items():assert sha(ROOT/name)==h;hashes+=1
    manifest=pd.read_csv(OUT/'manifest.csv',dtype=str).set_index('id')
    repeated_control_error=0.
    for i in np.linspace(0,len(r)-1,6,dtype=int):
        row=r.iloc[i];record=json.loads((OUT/'mbrset_cases'/f'{row.id}.json').read_text())
        for name,h in record['audit']['input_sha256'].items():assert sha(ROOT/name)==h;hashes+=1
        key=Path(manifest.loc[row.id,'file']).stem
        rgb=np.asarray(Image.open(MCACHE/'preprocessed_rgb'/f'{key}.png').convert('RGB').resize((512,512),Image.Resampling.BILINEAR));fov=rgb.max(2)>10
        _,_,p=load_features(MCACHE/'preprocessed_rgb'/f'{key}.png',MCACHE/'logits'/f'{key}.npy',512)
        smooth=np.stack([gaussian_filter(channel,1,mode='reflect',truncate=4) for channel in p]);smooth/=smooth.sum(0,keepdims=True)
        count,lab=cv2.connectedComponents(((p[1:].sum(0)>=.5)&fov).astype(np.uint8),connectivity=8)
        sizes=np.bincount(lab.ravel(),minlength=count);remove=(lab>0)&(sizes[lab]<32);clean=p.copy();clean[:,remove]=0;clean[0,remove]=1
        for a,prob in [('smooth',smooth),('cleanup',clean)]:
            for k,v in readout_features(prob,fov,a+'_').items():
                err=abs(v-row[k]);repeated_control_error=max(repeated_control_error,err);assert err<1e-6,(row.id,k,err)
    fundus_integrity=[json.loads((OUT/'fundus_cases'/f'{ident}.json').read_text()) for ident in protocol['fundus_ids']]
    repeat_integrity=[json.loads((OUT/'mbrset_cases'/f'{ident}.json').read_text())['audit'] for ident in protocol['mbrset_ids']]
    assert all(x['geometry_exact'] for x in fundus_integrity+repeat_integrity)
    assert max(x['probability_sum_error'] for x in fundus_integrity+repeat_integrity)<2e-6
    audit={'complete':True,'reference_images':100,'repeat_images':1960,'repeat_point_checks':len(est),
           'max_direct_repeat_error':maximum,'explicit_patient_bootstrap_checks':bootchecks,
           'reference_bootstrap_independent_recalculation':True,'max_reference_ci_error':cierror,
           'all100_independent_reference_checks':True,'max_reference_feature_error':referr,
           'original_loader_and_independent_control_replays':6,'max_replay_metric_error':replayerr,'input_hash_checks':hashes,
           'independent_repeated_image_control_replays':6,'max_repeated_control_error':repeated_control_error,
           'all2060_image_geometry_checks_passed':True,
           'original3_readouts_unchanged':True,'no_clinical_fields':True,'CIMT_used':False}
    write(OUT/'audit.json',audit)
    return audit


def report(audit):
    ref=pd.read_csv(OUT/'reference_estimates.csv').set_index(['metric','arm'])
    rd=pd.read_csv(OUT/'reference_contrasts.csv').set_index(['metric','contrast'])
    rep=pd.read_csv(OUT/'repeat_estimates.csv').set_index(['split','feature','arm','metric'])
    delta=pd.read_csv(OUT/'repeat_contrasts.csv').set_index(['split','feature','contrast','metric'])
    lines=['# 最后一轮测量对照完整报告','',
      '本轮仅回答：读出是否更接近人工掩模参考、是否更可重复、是否通过删除结构换取稳定。未训练新模型，未读取临床结局或CIMT。全部比较为探索性，未校正多重比较；不将本轮结果追认为预设临床终点。','',
      '## 1 固定方案','',
      '五组：原始VascX、原权重Mass UDCT、原始VascX条件身份×冻结FIVES占据概率、固定高斯概率平滑、固定小连通域移除。joint不含Mass，也不是联合训练网络。',
      '平滑逐通道采用sigma=1像素、reflect边界、truncate=4，再归一化。去小连通域在原始血管边际概率≥0.5掩模中移除面积<32像素的8连通分量，移除处赋背景概率1，其余不变。两个参数均在本轮结果产生前固定，未优化，不能代表所有简单方法中的最优方法。',
      '统一512概率网格、256骨架网格、图像RGBmax>10视野。五组结构掩模均由血管边际概率≥0.5产生。该端点不是旧版四类argmax下的mean A/V Dice，也不与768网格数值混合。',
      'Fundus-AVSeg的100张图用于参考评价：图像和标注使用同一图像几何变换；所有非黑标注，包括未知动静脉身份的白色血管，均计入二值血管。参考是掩模计算值，不是人工逐一确认的解剖分叉点。',
      'mBRSET既有开发/评价划分共1960张图用于重复性：开发472重复眼、253患者；评价475重复眼、255患者。单次拍摄眼不参与重复性。所有12项按五组共同有限值比较，不把本轮通过数替换旧稿九选四。',
      '参考区间使用5000次图像配对bootstrap；重复性区间使用10000次患者聚类配对bootstrap，同时保留患者双眼与全部视图。两套数据回答不同问题，不能构造同一患者的准确性与重复性关联。','',
      '## 2 全部结构参考误差','',
      '下表为平均绝对误差，越小越好。不同表型单位不同，不求跨指标总平均。骨架、分支和端点密度均以256×256全网格像素数为分母；分支为邻居数≥3的骨架像素，端点为邻居数=1的骨架像素。','',
      '|指标|'+'|'.join(NAMES[a] for a in ARMS)+'|','|---|'+'---:|'*5]
    for k in STRUCTURE:lines.append('|'+LABELS[k]+'|'+'|'.join(f'{ref.loc[("abs_error_"+k,a),"estimate"]:.6f}' for a in ARMS)+'|')
    lines+=['','## 3 全部结构保留指标','',
      'Dice、精确率、召回率、细血管中心线召回率和clDice越高越好；预测支持面积只描述输出规模，不是越大越好。细血管采用512网格中标注骨架半径≤2像素的固定定义，不是临床口径。','',
      '|指标|'+'|'.join(NAMES[a] for a in ARMS)+'|','|---|'+'---:|'*5]
    for k in METRICS:lines.append('|'+k+'|'+'|'.join(f'{ref.loc[(k,a),"estimate"]:.6f}' for a in ARMS)+'|')
    lines+=['','## 4 全部表型评价划分重复性','',
      '每格为ICC / 同眼MAD。MAD是对自身眼均值的平均绝对偏差，不是相对人工真值的误差。ICC增高与MAD下降不必一致。','',
      '|表型|'+'|'.join(NAMES[a] for a in ARMS)+'|','|---|'+'---:|'*5]
    for k in FEATURES:
        cells=[f'{rep.loc[("test",k,a,"icc"),"estimate"]:.4f} / {rep.loc[("test",k,a,"within_eye_mad"),"estimate"]:.6f}' for a in ARMS]
        lines.append('|'+LABELS[k]+'|'+'|'.join(cells)+'|')
    for a in ['mass','joint']:
        lines+=['',f'## {5 if a=="mass" else 6} {NAMES[a]}与原始及简单对照的完整配对结果','',
          '参考误差差值为新方法减比较组，负值有利；ΔICC正值有利；MAD下降百分比为100×(1−新MAD/比较组MAD)，正值有利。未见显著差异不等于等效或无损，不使用事后非劣效界值。']
        for b in ['raw','smooth','cleanup']:
            contrast=a+'_minus_'+b
            lines+=['',f'### 对照 {NAMES[b]}','',
              '|参考终点|差值|95% CI|','|---|---:|---|']
            for met in ['abs_error_'+k for k in STRUCTURE]+['vessel_precision','vessel_recall','thin_centerline_recall','cldice']:
                d=rd.loc[(met,contrast)];lines.append(f'|{met}|{d.delta:+.6f}|[{d.ci_low:+.6f}, {d.ci_high:+.6f}]|')
            lines+=['','|重复性表型|ΔICC [95% CI]|MAD下降% [95% CI]|','|---|---|---|']
            for k in FEATURES:
                d=delta.loc[('test',k,contrast,'icc')];e=delta.loc[('test',k,contrast,'MAD_reduction_percent')]
                lines.append(f'|{LABELS[k]}|{d.delta:+.4f} [{d.ci_low:+.4f}, {d.ci_high:+.4f}]|{e.delta:+.2f} [{e.ci_low:+.2f}, {e.ci_high:+.2f}]|')
    lines+=['','## 7 解释边界与停止规则','',
      '- 本轮结果用于界定已有读出适用范围，不能按表现重选表型、改变阈值或训练下一版本。',
      '- 结构误差下降、重复性提高及细节保留必须分开报告。若仅稳定性改善，不宣称测量准确性或临床有效性。',
      '- Fundus已经反复查看，本轮不是新的独立确认；mBRSET原图为224px公开镜像，升采样不产生高分辨率证据。',
      '- 既有三组特征被逐图原始分支复算核对后复用，两个简单组新计算；未将旧softmax/缩放口径混入。',
      '- 全部结果公开，不用三个较好的表型代替全部12项。局部正结果不抵消A/V比例等不利结果。',
      '- 本轮两个简单对照没有经过参数优化。即使某读出优于它们，也不等于优于所有去噪方法；平滑可能损坏原本很细的血管，因此必须同时展示原始VascX。',
      '- 既有mass_operating_point_diagnosis实验指出工作点变化可解释相当部分默认Dice改善。本轮结构参考误差结果不能推翻该证据，也不能将不同阈值定义和网格的分数拼成算法提升曲线。',
      '- 本轮完成后停止增益搜索。论文按读出与测量质量这一主线收口，不承诺新网络全面优越。','',
      '## 8 完整数据与复现','',
      'protocol.json记录运行前方案；fundus_per_image.csv和mbrset_per_image.csv为逐图数据；reference_estimates.csv、reference_contrasts.csv为参考评价；repeat_estimates.csv、repeat_contrasts.csv为全部开发及评价重复性；audit.json记录独立复算。','',
      '```sh','.venv/bin/python src/test_final_measurement_controls.py','.venv/bin/python src/final_measurement_controls.py prepare','.venv/bin/python src/final_measurement_controls.py fundus','.venv/bin/python src/final_measurement_controls.py repeats','.venv/bin/python src/final_measurement_controls.py analyze','.venv/bin/python src/audit_report_final_controls.py','```','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    write(OUT/'source_sha256.json',{p.name:sha(p) for p in [ROOT/'src/final_measurement_controls.py',ROOT/'src/test_final_measurement_controls.py',Path(__file__)]})
    write(OUT/'complete.json',{'extraction':True,'analysis':True,'independent_audit':True,'report':True,'new_training':False,'CIMT_used':False})
    print('FINAL CONTROLS COMPLETE',flush=True)


if __name__=='__main__':report(audit())
