"""Independent repeatability audit and transparent all-feature handoff."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run_fundus_fair_four_arm import ROOT,WEIGHT,sha
from evaluate_fundus_avseg_external import load_features
from train_udct_av_pilot import Corrector
from extract_mbrset_repeatability_features import readout_features
from three_readout_repeatability import OUT,ARMS,FEATURES,CACHE,OCC,estimate,write
from PIL import Image

LABELS={'vessel_mass':'血管概率质量','artery_mass':'动脉概率质量','vein_mass':'静脉概率质量',
        'av_ratio':'动静脉概率质量比','vessel_density':'血管面积密度','skeleton_density':'骨架像素密度',
        'branch_density':'分支像素密度','endpoint_density':'端点像素密度','component_count':'连通分量数',
        'fractal_dimension':'盒计数分形维数','occupancy_entropy':'占据熵','identity_entropy':'条件身份熵'}


def group_arrays(g,feature):
    cols=[a+'_'+feature for a in ARMS]
    sub=g[['patient_id','eye',*cols]].replace([np.inf,-np.inf],np.nan).dropna()
    return [(pid,eye,x[cols].to_numpy(float)) for (pid,eye),x in sub.groupby(['patient_id','eye']) if len(x)>=2]


def direct(groups):
    ns=np.array([len(x) for x in groups],float)
    mus=np.stack([x.mean(0) for x in groups]);n=ns.sum();k=len(groups)
    grand=np.concatenate(groups).mean(0)
    B=(ns[:,None]*(mus-grand)**2).sum(0)/(k-1)
    W=sum(((x-x.mean(0))**2).sum(0) for x in groups)/(n-k)
    ek=(n-(ns**2).sum()/n)/(k-1)
    return {'icc':(B-W)/(B+(ek-1)*W),'within_eye_mad':np.stack([abs(x-x.mean(0)).mean(0) for x in groups]).mean(0),
            'mean_pairwise_abs_diff':np.stack([abs(x[:,None]-x[None,:])[np.triu_indices(len(x),1)].mean(0) for x in groups]).mean(0),
            'within_eye_sd':np.sqrt(W),'between_eye_sd':np.sqrt(np.maximum((B-W)/ek,0)),
            'between_ms':B,'within_ms':W}


def audit():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(WEIGHT)==protocol['mass_weight_sha256']
    assert sha(ROOT/'models/baseline/artifacts/fives_unet_resnet34/best.pt')==protocol['occupancy_weight_sha256']
    assert sha(ROOT/'src/extract_mbrset_repeatability_features.py')==protocol['phenotype_source_sha256']
    manifest=pd.read_csv(OUT/'manifest.csv',dtype=str)
    f=pd.read_csv(OUT/'per_image.csv',dtype={'id':str,'patient_id':str})
    assert len(f)==1960 and f.id.nunique()==1960 and set(f.id)==set(manifest.id)
    assert set(f.columns)==set(['id','patient_id','eye','view','split']+[a+'_'+v for a in ARMS for v in FEATURES])
    assert f.groupby('patient_id').split.nunique().max()==1
    source=pd.read_csv(ROOT/'artifacts/mbrset_repeatability_split/per_image.csv',usecols=['id','patient_id','eye','view','split'],dtype=str)
    pairs=f[['id','patient_id','eye','view','split']].astype(str).merge(source,on='id',suffixes=('_new','_old'),validate='one_to_one')
    for c in ['patient_id','eye','view','split']:assert (pairs[c+'_new']==pairs[c+'_old']).all()
    est=pd.read_csv(OUT/'estimates.csv').set_index(['split','feature','arm','metric'])
    maxerr=0.;replicas=0
    for split,g in f.groupby('split'):
        for feature in FEATURES:
            groups=group_arrays(g,feature)
            vals=direct([r[2] for r in groups])
            for m,v in vals.items():
                for j,a in enumerate(ARMS):
                    err=abs(v[j]-est.loc[(split,feature,a,m),'estimate']);maxerr=max(maxerr,err)
                    assert err<1e-9,(split,feature,a,m,err)
            # Explicitly duplicate selected patients, not individual eyes, for
            # ten bootstrap replicates and compare to sufficient-statistic code.
            pids=sorted({r[0] for r in groups});patient_index={p:i for i,p in enumerate(pids)}
            ns=np.array([len(r[2]) for r in groups],float)
            means=np.stack([r[2].mean(0) for r in groups]);ss=np.stack([((r[2]-r[2].mean(0))**2).sum(0) for r in groups])
            mad=np.stack([abs(r[2]-r[2].mean(0)).mean(0) for r in groups])
            pair=np.stack([abs(r[2][:,None]-r[2][None,:])[np.triu_indices(len(r[2]),1)].mean(0) for r in groups])
            rng=np.random.default_rng(200)
            for _ in range(10):
                selected=rng.choice(pids,len(pids),replace=True)
                copies=[r[2] for p in selected for r in groups if r[0]==p]
                counts={p:int((selected==p).sum()) for p in pids}
                weights=np.array([[counts[r[0]] for r in groups]],float)
                fast=estimate(weights,ns,means,ss,mad,pair);slow=direct(copies)
                for m in fast:np.testing.assert_allclose(fast[m][0],slow[m],atol=1e-9,rtol=1e-9)
                replicas+=1
            print('audited stats',split,feature,flush=True)
    # Independent feature-loader replay through the original Mass evaluator.
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(4);net=Corrector().to(device).eval();net.load_state_dict(torch.load(WEIGHT,map_location=device)['model'])
    feature_error=0.;input_hash_checks=0
    indexed=f.set_index('id')
    for i in np.linspace(0,len(manifest)-1,6,dtype=int):
        r=manifest.iloc[i];key=Path(r.file).stem
        record=json.loads((OUT/'cases'/f'{r.id}.json').read_text())
        for path,h in record['audit']['input_sha256'].items():
            assert sha(ROOT/path)==h;input_hash_checks+=1
        x,z,base=load_features(CACHE/'preprocessed_rgb'/f'{key}.png',CACHE/'logits'/f'{key}.npy',512)
        with torch.inference_mode():mass=net(x[None].to(device),z[None].to(device))[0].softmax(0).cpu().numpy()
        o=np.load(OCC/f'{key}.npy').astype(np.float32)
        cond=base[1:]/np.maximum(base[1:].sum(0,keepdims=True),1e-7)
        joint=np.concatenate([1-o[None],o[None]*cond],0)
        rgb=np.asarray(Image.open(CACHE/'preprocessed_rgb'/f'{key}.png').convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
        for a,p in zip(ARMS,[base,mass,joint]):
            row=readout_features(p,rgb.max(2)>10,a+'_')
            for k,v in row.items():
                old=indexed.loc[r.id,k]
                if np.isnan(v) and pd.isna(old):continue
                err=abs(v-old);feature_error=max(feature_error,err);assert err<1e-6,(r.id,k,err)
    points=pd.read_csv(OUT/'estimates.csv');contrasts=pd.read_csv(OUT/'paired_patient_contrasts.csv')
    assert len(points)==2*12*3*7 and len(contrasts)==2*12*3*8
    assert np.isfinite(points[['estimate','ci_low','ci_high']]).all().all()
    assert np.isfinite(contrasts[['delta','ci_low','ci_high']]).all().all()
    audit={'complete':True,'unique_images':1960,'same_three_arm_rows':True,'patient_split_overlap':False,
           'independent_point_metric_checks':len(points),'max_independent_point_error':maxerr,
           'explicit_patient_bootstrap_replicates_checked':replicas,'bootstrap_intervals':'computed from10000 patient resamples; explicit replication tests verify the engine, not an independent rerun of every interval',
           'original_loader_replay_images':6,'max_replayed_feature_error':feature_error,'sample_input_hash_checks':input_hash_checks,
           'all_probability_integrity_checks':'see extraction_complete.json; extraction verifies all5880 outputs',
           'clinical_columns_in_feature_table':[],'CIMT_used':False}
    write(OUT/'independent_audit.json',audit)
    return f,points,contrasts


def report(f,points,contrasts):
    p=json.loads((OUT/'protocol.json').read_text());meta=json.loads((OUT/'analysis_complete.json').read_text())
    est=points.set_index(['split','feature','arm','metric'])
    con=contrasts.set_index(['split','feature','contrast','metric'])
    def e(v,a,m):return est.loc[('test',v,a,m),'estimate']
    def c(v,a,m):return con.loc[('test',v,a+'_minus_raw',m)]
    joint_support=[];mass_support=[];mass_adverse=[]
    for feat in FEATURES:
        for arm,out in [('mass',mass_support),('joint',joint_support)]:
            if c(feat,arm,'icc').ci_low>0 and c(feat,arm,'within_eye_mad').ci_high<0:out.append(feat)
        if c(feat,'mass','icc').ci_high<0 or c(feat,'mass','within_eye_mad').ci_low>0:mass_adverse.append(feat)
    # Trace the numerical change caused by the OLD probability-resize convention.
    legacy=pd.read_csv(ROOT/'artifacts/mbrset_repeatability_features/per_image.csv',usecols=lambda k:k=='id' or k.startswith(('raw_','joint_')))
    bridge=f.merge(legacy,on='id',suffixes=('_new','_legacy'),validate='one_to_one')
    rows=[]
    for a in ['raw','joint']:
        for feat in FEATURES:
            col=a+'_'+feat;d=bridge[col+'_new']-bridge[col+'_legacy']
            rows.append({'arm':a,'feature':feat,'n_complete':int(d.notna().sum()),'mean_abs_numerical_difference':float(d.abs().mean()),'max_abs_numerical_difference':float(d.abs().max())})
    pd.DataFrame(rows).to_csv(OUT/'legacy_resampling_bridge.csv',index=False)
    lines=['# 三路血管表型重复性对照：完整结果','',
           '状态：完成。使用既有开发/评价划分1960张图像，三路统一重算；无新训练，无临床结局调参。ICC是重复性指标，不是分割Dice。','',
           '## 1. 这次比较究竟是什么','',
           '- raw：冻结VascX原始概率读出。',
           '- mass：同一VascX输入，经过原有冻结Mass UDCT两步修正器。',
           '- joint：冻结占据概率O乘以原始VascX的条件动静脉身份；没有Mass修正，也不是新训练的联合网络。',
           '- 三路都使用同一图像、视野掩模、512网格、阈值0.5和同一组12个表型算子；骨架及其派生指标统一在256网格计算。',
           '- 旧表先softmax再缩放概率；本轮统一先缩放logits再softmax，遵循冻结Mass原实现。三路全部重算，旧ICC不可与新Mass直接拼表。legacy_resampling_bridge.csv保留数值桥接。','',
           '## 2. 总体判断','',
           '以下“支持”只是探索性95%区间同时满足ICC差值>0、同眼MAD差值<0，不是预先验证的临床通过标准，也未做多重比较校正。全部12项均报告，不能据此重新筛选模型或特征。','',
           'Mass同时改善两项的表型：'+('、'.join(LABELS[x] for x in mass_support) if mass_support else '无')+'。',
           '占据条件读出同时改善两项的表型：'+('、'.join(LABELS[x] for x in joint_support) if joint_support else '无')+'。',
           'Mass至少一项出现不利区间的表型：'+('、'.join(LABELS[x] for x in mass_adverse) if mass_adverse else '无')+'。','',
           '**没有表型真值，因此“重复得更一致”仍不是“测得更准确”。不得把占据读出的收益归给Mass，或把它们合并成整个RMDB–BiRAP已全面优于基线。**','',
           '## 3. 样本与统计口径','',
           '|划分|图像|患者|重复眼|剔除的单次拍摄眼|','|---|---:|---:|---:|---:|']
    for name,d in meta['splits'].items():lines.append(f"|{name}|{d['images']}|{d['patients']}|{d['repeated_eyes']}|{d['singleton_eyes_excluded']}|")
    lines+=['',
            '每项表型使用三路共同有限的图像；只有至少两次测量的眼进入统计。ICC采用单向随机效应ICC(1,1)及不等重复次数的有效k。MAD是各次测量相对自身眼均值的绝对偏差，再跨眼平均；两次拍摄时它等于成对绝对差的一半。不是人工真值误差。',
            '10000次患者级bootstrap在每次重采样中一起保留患者双眼和全部视图。差值使用同一批重采样，给出百分位95%区间。开发与评价划分均已被过去研究查看，本轮不是新独立确认。训练集不重新提取、不用于本轮选择。','',
            '## 4. 评价划分：全部12个表型','',
            '|表型|raw ICC|Mass ICC|joint ICC|raw同眼MAD|Mass同眼MAD|joint同眼MAD|','|---|---:|---:|---:|---:|---:|---:|']
    for v in FEATURES:
        lines.append('|'+LABELS[v]+'|'+'|'.join(f'{e(v,a,m):.6f}' for m in ['icc','within_eye_mad'] for a in ARMS)+'|')
    for a,label in [('mass','Mass UDCT'),('joint','占据条件读出')]:
        lines+=['',f'## {5 if a=="mass" else 6}. {label}相对raw：患者配对差','',
                'MAD下降百分比为100×(1−新MAD/原MAD)，正值有利；跨不同表型不可直接比较原始单位的MAD。','',
                '|表型|ΔICC|95% CI|MAD下降%|95% CI|','|---|---:|---|---:|---|']
        for v in FEATURES:
            d=c(v,a,'icc');r=c(v,a,'MAD_reduction_percent')
            lines.append(f"|{LABELS[v]}|{d.delta:+.6f}|[{d.ci_low:+.6f}, {d.ci_high:+.6f}]|{r.delta:+.2f}|[{r.ci_low:+.2f}, {r.ci_high:+.2f}]|")
    lines+=['','## 7. 防止把ICC上升误解为噪声下降','',
            'ICC取决于眼间差异相对眼内波动的大小。下面同时展示眼内标准差与眼间标准差，不能把眼间差异全部解释成生物学差异。完整均方、标准差及区间见estimates.csv。','',
            '|表型|raw眼内SD|Mass眼内SD|joint眼内SD|raw眼间SD|Mass眼间SD|joint眼间SD|','|---|---:|---:|---:|---:|---:|---:|']
    for v in FEATURES:lines.append('|'+LABELS[v]+'|'+'|'.join(f'{e(v,a,m):.6f}' for m in ['within_eye_sd','between_eye_sd'] for a in ARMS)+'|')
    lines+=['','## 8. 指标定义与证据边界','',
            '- 概率质量为视野内平均概率，不是血管口径。交叉类同时贡献动脉和静脉质量。',
            '- 骨架、分支及端点密度的分母均为整个256×256网格的像素数。分支像素指具有至少3个骨架邻居的骨架像素，不是去重后的解剖分叉点；端点为恰有1个骨架邻居的像素。连通分量数只统计256网格上面积≥8像素的分量。',
            '- joint在代数上保留原始条件身份；其条件身份熵基本不变不应解释成算法新收益。实现沿用统一的小概率裁剪。',
            '- 原图来自224px公开镜像。512/256处理不产生新的解剖细节，也不构成高分辨率验证。',
            '- 本轮12个全局表型不等同于旧稿的9个候选表型，不能用本轮通过数量替换“四个可靠表型”等旧结论。',
            '- 没有使用新阈值搜索、临床结局或CIMT。两项指标改善也可能来自平滑、标尺变化或偏差，不足以证明真实表型准确性或临床增量。',
            '- 保留所有负结果，不将表现较好的一个表型改成事后主终点。','',
            '## 9. 输出与复现','',
            '- protocol.json、manifest.csv：固定口径及不含临床字段的本轮清单。',
            '- per_image.csv：三路完整逐图表型；cases/中保存来源哈希与几何重放审计。',
            '- estimates.csv：分划分、分表型、分读出的点估计与患者聚类区间。',
            '- paired_patient_contrasts.csv：全部配对差与区间，包括Mass和joint直接比较。',
            '- occupancy_replay.json：预先固定六张图上用冻结占据模型复核缓存。',
            '- independent_audit.json：独立ANOVA/MAD复算、显式复制患者的bootstrap检查、原始Mass加载器特征重放。','',
            '```sh',
            '.venv/bin/python src/test_three_readout_repeatability.py',
            '.venv/bin/python src/three_readout_repeatability.py prepare',
            '.venv/bin/python src/three_readout_repeatability.py extract',
            '.venv/bin/python src/three_readout_repeatability.py analyze',
            '.venv/bin/python src/audit_report_three_readout.py','```','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    write(OUT/'interpretation.json',{'mass_both_exploratory_intervals_favorable':mass_support,
                                    'joint_both_exploratory_intervals_favorable':joint_support,
                                    'mass_at_least_one_adverse_interval':mass_adverse,'not_a_confirmatory_success_rule':True})
    write(OUT/'source_sha256.json',{p.name:sha(p) for p in [ROOT/'src/three_readout_repeatability.py',ROOT/'src/test_three_readout_repeatability.py',Path(__file__)]})
    write(OUT/'complete.json',{'extraction':True,'patient_cluster_analysis':True,'audit':True,'report':True,'new_training':False})
    print(json.dumps(json.loads((OUT/'interpretation.json').read_text()),ensure_ascii=False,indent=2))

if __name__=='__main__':report(*audit())
