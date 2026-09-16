"""Data-grounded Chinese handoff and editable scientific diagnostic plots."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from diagnose_mass_operating_point import OUT, ROOT, ARMS, GRID, sha

NAMES={'vascx':'VascX','mass':'Mass UDCT','clahe':'CLAHE + VascX','clahe_mass':'CLAHE + Mass'}
COLORS=['#596269','#188C85','#0F4D92','#BE8835']
SKILL=Path('/Users/tongyue/.codex/skills/engineering-figure-banana')


def main():
    s=json.loads((OUT/'summary.json').read_text())
    pts=pd.read_csv(OUT/'operating_points_per_image.csv')
    ranks=pd.read_csv(OUT/'ranking_per_image.csv')
    curves={a:np.stack([np.load(p)['precision_envelope'] for p in sorted((OUT/'curves').glob(f'*__{a}.npz'))]).mean(0) for a in ARMS}
    ids=json.loads((OUT/'protocol.json').read_text())['fundus_ids']
    assert all(len(list((OUT/'curves').glob(f'*__{a}.npz')))==100 for a in ARMS)
    plot_rows=[{'recall':float(r),**{a:float(curves[a][j]) for a in ARMS}} for j,r in enumerate(GRID)]
    pd.DataFrame(plot_rows).to_csv(OUT/'macro_PR_display.csv',index=False)
    labels=[NAMES[a] for a in ARMS]
    common={'type':'trend','x':GRID.tolist(),'y_series':[curves[a].tolist() for a in ARMS],
            'labels':labels,'colors':COLORS,'xlabel':'Vessel recall','ylabel':'Interpolated precision',
            'legend':False,'line_width':2.2}
    targets=[.85,.90,.95]
    def match_panel(metric,title,ylabel):
        return {'type':'trend','title':title,'x':targets,
                'y_series':[[s['operating_point_means'][a+f'/oracle_P{p:.2f}'][metric] for p in targets] for a in ARMS],
                'labels':labels,'colors':COLORS,'xlabel':'Required precision (per image)',
                'ylabel':ylabel,'xticks':targets,'legend':False,'line_width':2.2}
    spec={'suptitle':'Operating point diagnosis: exploratory Fundus-AVSeg (n = 100)',
          'style':{'font_size':13,'axes_linewidth':1.2,'font_family':['Arial','DejaVu Sans']},
          'layout':{'nrows':2,'ncols':2,'figsize':[13,10],'tight_layout_pad':2.0},
          'panels':[{**common,'title':'A  Full PR envelope','xlim':[0,1],'ylim':[0,1.01],'legend':True,'legend_loc':'lower left'},
                    {**common,'title':'B  High-recall detail','xlim':[.65,.95],'ylim':[.55,1.]},
                    match_panel('thin_recall','C  Thin-vessel recall (oracle)','Thin centerline recall'),
                    match_panel('cldice','D  Skeleton overlap (oracle)','clDice')]}
    (OUT/'plot_spec.json').write_text(json.dumps(spec,indent=2))
    subprocess.run([sys.executable,str(SKILL/'scripts/plot_publication_figure.py'),str(OUT/'plot_spec.json'),
                    '--out-path',str(OUT/'diagnostic_curves'),'--formats','png','pdf','svg','--dpi','180'],check=True)
    dev=pd.read_csv(OUT/'development_bias_grid.csv').groupby('bias').mean_av_dice.mean()
    dev_spec={'style':spec['style'],'layout':{'nrows':1,'ncols':1,'figsize':[7,4.5]},
              'panels':[{'type':'trend','title':'RAV development only: freeze one global bias',
                         'x':dev.index.tolist(),'y_series':[dev.tolist()],
                         'labels':['VascX + shared foreground bias'],'colors':['#596269'],
                         'xlabel':'Shared foreground logit bias','ylabel':'Mean A/V Dice','legend':False}]}
    (OUT/'development_plot_spec.json').write_text(json.dumps(dev_spec,indent=2))
    subprocess.run([sys.executable,str(SKILL/'scripts/plot_publication_figure.py'),str(OUT/'development_plot_spec.json'),
                    '--out-path',str(OUT/'development_bias_curve'),'--formats','png','svg','--dpi','180'],check=True)
    if '--plots-only' in sys.argv:
        return
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['complete']
    c=s['contrasts'];m=s['operating_point_means']
    pd.DataFrame([{'contrast':name,'endpoint':metric,'delta':d['delta'],
                   'ci_low':d['ci95'][0] if d['ci95'] else None,
                   'ci_high':d['ci95'][1] if d['ci95'] else None,'n':d['n']}
                  for name,metrics in c.items() for metric,d in metrics.items()]).to_csv(OUT/'paired_contrasts.csv',index=False)
    primary=c['mass_minus_vascx']['decision_ap']
    if primary['ci95'][1]<0:
        verdict='Mass UDCT 的血管决策排序 AP 低于原始 VascX。默认阈值 Dice 的小幅改善不能据此解释为整体排序能力提升。'
    elif primary['ci95'][0]>0:
        verdict='Mass UDCT 的血管决策排序 AP 高于原始 VascX，因此效果不完全等价于给全部血管类别统一加一个常数。但幅度、误检代价和外部可靠性仍须单独判断，不能据此宣称重大算法突破。'
    else:
        verdict='血管决策排序 AP 的差值区间跨零，尚不能确认 Mass UDCT 带来了超出简单常数偏置的排序收益。区间跨零不是两者等效的证明。'
    residual=c['mass_default_minus_vascx_RAV_frozen_bias']['mean_av_dice']
    matched=c['mass_minus_vascx']['oracle_P0.90/vessel_recall']
    decision=[f"简单RAV偏置已把Fundus默认Dice从{m['vascx/default']['mean_av_dice']:.6f}提高到{m['vascx/RAV_frozen_bias']['mean_av_dice']:.6f}；Mass UDCT为{m['mass/default']['mean_av_dice']:.6f}。Mass在该简单对照之外只剩Δ={residual['delta']:+.6f}，95% CI [{residual['ci95'][0]:+.6f}, {residual['ci95'][1]:+.6f}]。这不是说Mass与偏置完全等价，而是说明不能忽略工作点因素。",
              f"在逐图精确率至少90%的描述性前沿上，Mass相对VascX的召回差为{matched['delta']:+.6f}，95% CI [{matched['ci95'][0]:+.6f}, {matched['ci95'][1]:+.6f}]。默认阈值下的细血管召回优势必须结合此误检控制结果解释。"]
    if primary['ci95'][1]<0 and matched['ci95'][1]<0:
        decision+=['当前结果不支持将该冻结修正器作为“显著提高血管识别能力”的核心创新。按约定停止这条增益搜索分支，不继续在Fundus调阈值或修改损失寻找正结果。若继续论文，应将其定位为测量与评价研究，保留该方法作为透明消融，而不是优越的新分割网络。',
                   '也不能隐藏局部正结果：Mass在90%和95%精确率点的AV像素比例误差较小，但这是次要、探索性、工作点相关的差异；85%点未体现稳定优势。它不足以替代失败的主终点或证明临床用途。']
    def line_delta(name,metric):
        d=c[name][metric]
        if d['ci95'] is None:return f'|{name}|{metric}|不可估计|n=0|'
        return f"|{name}|{metric}|{d['delta']:+.6f}|[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]|"
    lines=['# RMDB：排序改进还是工作点变化？一次固定诊断','',
           '状态：已完成。30 张 RAV 开发图像确定简单偏置；100 张 Fundus-AVSeg 图像、4 个既有方法完成全阈值排序与工作点分析。没有新训练，没有读取 CIMT 或临床结局。','',
           '## 1. 结论','',verdict,'',*sum(([x,''] for x in decision),[]),
           '**本次不挑选新的 Fundus 最优部署阈值，也不把描述性的 oracle 阈值当成可部署算法。所有结果为探索性。**','',
           '## 2. 冻结的简单对照','',
           f"只在已有 RAV 开发划分上，以 mean A/V Dice 选择统一前景 logit 偏置。网格为 −3 至 +3、步长 0.1，冻结 b={s['frozen_baseline']['bias']:.1f}。开发集默认 Dice={s['frozen_baseline']['development_default_av_dice']:.6f}；加偏置后={s['frozen_baseline']['development_mean_av_dice']:.6f}。",'',
           '这相当于同时给动脉、静脉、交叉类别加相同偏置，只改变前景与背景的取舍，不改变前景内部类别的大小关系。没有拟合空间模型，也不是新网络。RAV 已用于既往开发，不是独立验证。','',
           '## 3. 冻结规则的实际比较（统一768评分网格；不代表已获临床部署验证）','',
           '|规则|mean AV Dice|精确率|召回率|细血管召回率|clDice|AV比例对数误差↓|',
           '|---|---:|---:|---:|---:|---:|---:|']
    columns=['mean_av_dice','vessel_precision','vessel_recall','thin_recall','cldice','av_ratio_log_error']
    for a,point,label in [('vascx','default','VascX 默认'),('vascx','RAV_frozen_bias','VascX＋RAV冻结偏置'),
                          ('mass','default','Mass UDCT 默认'),('clahe','default','固定 CLAHE＋VascX'),('clahe_mass','default','固定 CLAHE＋Mass UDCT')]:
        lines.append('|'+label+'|'+'|'.join(f'{m[a+"/"+point][v]:.6f}' for v in columns)+'|')
    lines+=['','**不可用0.55作此处基线。原有0.55→0.8左右包含坐标修复，绝不是算法提高；512与768尺度的增量分别报告。**','',
            '## 4. 不依赖单一阈值的排序','',
            '|方法|决策排序 AP（主终点）|血管边际概率 AP|平均动静脉边际概率 AP|','|---|---:|---:|---:|']
    for a in ARMS:
        r=s['ranking_means'][a]
        lines.append('|'+NAMES[a]+'|'+'|'.join(f'{r[k]:.6f}' for k in ['decision_ap','vessel_marginal_ap','mean_av_marginal_ap'])+'|')
    lines+=['','主终点对每张图像使用 score=log(max(PA,PV,PC))−log(Pbackground)，求精确的非插值 average precision，再跨图平均。统一前景常数偏置只给 score 加常数，不改变排序，因此不能改变其 AP。AP 仍依赖标注口径与患病/像素分布，不等于临床判别性能。','',
            '血管边际概率为1−Pbackground；动静脉边际概率分别为PA+PC、PV+PC。这两个辅助 AP 与主终点不可混称。','',
            '## 5. 精确率至少90%时的描述性比较','',
            '在每张图像全部经验阈值中选取精确率≥90%条件下召回率最高的点。四组都采用同样规则。此处消耗评价标注，只是 PR 前沿诊断；每张图像阈值不同，绝不能拿去宣称部署成绩。相同精确率控制的是假发现比例，不等于相同假阳性像素绝对数。','',
            '|方法|实际精确率|召回率|细血管召回率|clDice|AV比例对数误差↓|不可达到图像数|','|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        r=m[a+'/oracle_P0.90']
        lines.append('|'+NAMES[a]+'|'+'|'.join(f'{r[k]:.6f}' for k in columns[1:])+f"|{s['oracle_unattainable'][a+'/P0.90']}|")
    lines+=['','85%与95%精确率的全部结果保存在逐图表与summary.json中，不从三个点里挑一个重新定义主终点。','',
            '## 6. 关键配对差及95%区间','',
            '5000次图像级配对bootstrap，差值为前者减后者；AV比例误差负值有利。区间未校正多重比较，且未包含先前模型筛选与RAV阈值选择的不确定性。','',
            '|比较|终点|均值差|95% CI|','|---|---|---:|---|']
    for k in ['decision_ap','vessel_marginal_ap','mean_av_marginal_ap','default/mean_av_dice',
              'oracle_P0.90/vessel_recall','oracle_P0.90/thin_recall','oracle_P0.90/cldice','oracle_P0.90/av_ratio_log_error']:
        lines.append(line_delta('mass_minus_vascx',k))
    for k in columns:
        lines.append(line_delta('mass_default_minus_vascx_RAV_frozen_bias',k))
    for k in ['decision_ap','default/mean_av_dice','oracle_P0.90/thin_recall','oracle_P0.90/cldice']:
        lines.append(line_delta('clahe_mass_minus_clahe',k))
    lines+=['','## 7. 各指标能说明什么，不能说明什么','',
            '- 细血管定义固定为768网格上标注骨架距离变换半径≤2像素；不等同于临床按微米测量的小血管。',
            '- clDice 衡量骨架与血管区域的相互覆盖，能辅助检查连接结构，但不保证正确的拓扑，也不能替代人工核查每条分支。',
            '- AV比例误差是硬分割的动静脉像素数比值对数误差，不是临床管径AVR。交叉点计入动静脉两者，未知身份像素不计入身份评分。',
            '- 统一常数偏置只是一个明确、低复杂度的替代解释；击败它不等于击败所有后处理方法。没击败它也不能自动证明完全等效。',
            '- 简单基线的偏置在RAV选择，目标域可能具有不同工作点；它在Fundus的结果必须结合阈值无关排序与oracle前沿解读。',
            '- 未声称提升临床结局预测、泛化到新医院或证明重大新算法。没有新增模型调参或预处理参数搜索。','',
            '## 8. 图注','',
            'Diagnostic curves：A为逐图插值精确率包络的宏平均，B为高召回区域放大。图形仅在1001个召回网格点展示，主表AP由全部经验阈值精确计算，不是插值曲线下面积。C、D是在每张图像精确率分别至少85%、90%、95%的oracle点计算细血管召回和clDice后求宏平均。所有四组均展示，不隐去组合失败结果。','',
            '## 9. 完整性与复现','',
            f"独立检查：{audit['probability_outputs']}份概率、{audit['operating_point_mask_checks']}个工作点。独立AP最大绝对误差{audit['max_independent_AP_error']:.3g}；逐掩模指标最大误差{audit['max_independent_metric_error']:.3g}。所有原图/标注哈希与前一轮一致，主终点bootstrap已复算。",'',
            '- protocol.json：在本次评价前固定的分析口径。',
            '- frozen_baseline.json、development_bias_grid.csv：RAV选择记录与冻结偏置。',
            '- ranking_per_image.csv、operating_points_per_image.csv、summary.json：全部逐图数据和配对差。',
            '- diagnostic_curves.svg/pdf/png：由真实数值渲染，可编辑矢量图。',
            '- independent_audit.json：独立计数、排序AP、最佳可达召回与哈希验证。',
            '- src/diagnose_mass_operating_point.py、test_mass_operating_point.py、audit_mass_operating_point.py、report_mass_operating_point.py：全流程源码。','',
            '复现命令（项目目录内）：','```sh',
            '.venv/bin/python src/test_mass_operating_point.py',
            '.venv/bin/python src/diagnose_mass_operating_point.py prepare',
            '.venv/bin/python src/diagnose_mass_operating_point.py develop',
            '.venv/bin/python src/diagnose_mass_operating_point.py evaluate',
            '.venv/bin/python src/audit_mass_operating_point.py',
            '.venv/bin/python src/report_mass_operating_point.py','```','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    (OUT/'output_sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['output_sha256.json','complete.json']},indent=2))
    (OUT/'complete.json').write_text(json.dumps({'diagnosis':True,'independent_audit':True,'report_and_plots':True,
                                               'training_performed':False},indent=2))
    print(verdict)


if __name__=='__main__':main()
