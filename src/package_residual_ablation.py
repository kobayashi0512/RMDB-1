"""Bundle completed outputs and weights; original raw data remain in the project."""
import json
import zipfile
from pathlib import Path
from prepare_residual_ablation import ROOT, OUT

def main():
    for name in ['training_complete.json','final_audit.json','checkpoint_audit.json','input_and_compute_audit.json']:
        assert (OUT/name).exists(),name
    frozen=ROOT/'artifacts/frozen_residual_steps_20260914_v1'
    assert (frozen/'complete.json').exists()
    target=OUT/'RMDB_残差消融_阶段结果与权重.zip'
    scripts=['prepare_residual_ablation.py','train_controlled_residual_ablation.py',
             'test_controlled_residual_ablation.py','report_controlled_residual_ablation.py',
             'audit_residual_ablation_inputs.py','audit_residual_ablation_checkpoints.py',
             'evaluate_frozen_residual_steps.py','train_udct_av_pilot.py','backbone_runtime.py',
             'package_residual_ablation.py','audit_residual_ablation_statistics.py']
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for folder,prefix in [(OUT,'受控重训练'),(frozen,'原权重迭代检查')]:
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix!='.zip': z.write(p,Path(prefix)/p.relative_to(folder))
        for name in scripts: z.write(ROOT/'src'/name,Path('复现脚本')/name)
        z.writestr('阅读说明.txt','本包保存本轮结果、协议、21 个新训练权重及优化器、对齐标注和复现脚本。原始 RAV/Fundus 图像、VascX 缓存及其余项目依赖仍位于原实验项目，不在此包重复收录，因此不是可脱离原项目独立运行的数据全集。旧论文和旧主模型未被覆盖。\n')
    with zipfile.ZipFile(target) as z: assert z.testzip() is None
    print(str(target));print(target.stat().st_size)

if __name__=='__main__':main()
