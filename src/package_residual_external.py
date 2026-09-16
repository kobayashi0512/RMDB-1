"""Package audited external measurements; checkpoints remain in the previous bundle."""
import zipfile
from pathlib import Path
from evaluate_residual_ablation_external import OUT,TRAIN
import final_measurement_controls as fc

def main():
    for name in ['audit.json','联合评估报告.md','fundus_analysis_complete.json','repeat_analysis_complete.json']:
        assert (OUT/name).exists(),name
    target=OUT/'RMDB_固定权重外部联合评估.zip'
    scripts=['evaluate_residual_ablation_external.py','analyze_residual_ablation_external.py',
             'audit_residual_external.py','screen_residual_external_overlap.py','report_residual_external.py',
             'package_residual_external.py','record_residual_external_runtime.py']
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(OUT))
        for name in scripts:z.write(fc.ROOT/'src'/name,Path('复现脚本')/name)
        z.write(TRAIN/'training_protocol.json','权重来源/training_protocol.json')
        z.write(TRAIN/'checkpoint_audit.json','权重来源/checkpoint_audit.json')
        z.writestr('阅读说明.txt','本包保存本轮全部逐图测量、统计结果、协议、审计和入口脚本。21 个权重保留于上一轮残差消融结果目录和权重包，不在此重复复制。复现仍依赖原实验项目的数据、缓存、辅助源码和 Python 环境。本包不是脱离原项目独立运行的数据全集。\n')
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(target)

if __name__=='__main__':main()
