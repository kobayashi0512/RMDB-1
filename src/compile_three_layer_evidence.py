#!/usr/bin/env python3
"""Compile frozen measurement, transfer, and exploratory clinical evidence."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "three_layer_evidence"

def load(rel: str, default=None):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else default

def row(layer, dataset, evidence, n, metric, value, ci="", status=""):
    return {"layer": layer, "dataset": dataset, "evidence": evidence,
            "n": n, "metric": metric, "value": value, "ci95": ci,
            "status": status}

def main():
    rows = []
    afio = load("artifacts/afio_mendeley_audit/audit.json", {})
    aph = load("artifacts/afio_mendeley/phenotypes/metadata.json", {})
    rows.extend([
        row("phenotype_accuracy", "AFIO/Mendeley", "public annotation inventory", afio.get("images", ""), "images", afio.get("images", ""), status="measurement-only; outcome-blind"),
        row("phenotype_accuracy", "AFIO/Mendeley", "paired vessel masks", afio.get("images", ""), "paired_vessel_masks", afio.get("paired_vessel_masks", ""), status="complete"),
        row("phenotype_accuracy", "AFIO/Mendeley", "paired A/V masks", afio.get("images", ""), "paired_av_masks", afio.get("paired_av_masks", ""), status="one source image missing; not patched"),
        row("phenotype_accuracy", "AFIO/Mendeley", "frozen RMDB-BiRAP extraction", aph.get("patients", ""), "completed_eyes", aph.get("completed_eyes", ""), status="label-blind; complete"),
    ])
    for ds, d in load("artifacts/external_transfer/transfer_audit.json", {}).get("datasets", {}).items():
        rows.append(row("phenotype_accuracy", ds, "frozen vessel transfer", d.get("images", ""), "vessel_dice_mean", d.get("vessel_dice_mean", ""), status="label-blind transfer"))
        rows.append(row("phenotype_accuracy", ds, "frozen vessel transfer", d.get("images", ""), "vessel_iou_mean", d.get("vessel_iou_mean", ""), status="label-blind transfer"))
    for p in ("artifacts/structure_migration/fives_vessel_metrics.json", "artifacts/structure_migration/stare_vessel_metrics.json"):
        d = load(p, {})
        rows.append(row("cross_domain_robustness", d.get("dataset", ""), "frozen VascX vessel migration", d.get("images_scored", ""), "vessel_dice_mean", d.get("vessel_dice_mean", ""), status="negative/weak; no retuning"))
    rows.append(row("cross_domain_robustness", "FIVES", "frozen RMDB-BiRAP phenotype extraction", 200, "completed_eyes", 200, status="label-blind; complete"))
    rows.append(row("cross_domain_robustness", "STARE", "public AH labels", 20, "annotated_images", 20, status="20 AH labels; one label per image subset; weak migration"))
    rows.append(row("cross_domain_robustness", "DRIVE", "official access status", "", "download", "not run", status="official page login-restricted; no mirror bypass"))
    # Outcome-blind all-cohort transport stress test. This extends the
    # measurement-domain layer with AFIO and pairwise domain-separability CIs;
    # it never reads a clinical outcome or a CIMT validation/test row.
    transport = ROOT / "artifacts/four_cohort_transport/pairwise_domain_transport.csv"
    if transport.exists():
        import pandas as pd
        transport_df = pd.read_csv(transport)
        for _, t in transport_df.iterrows():
            rows.append(row(
                "cross_domain_robustness",
                f"{t['left']} vs {t['right']}",
                "outcome-blind pairwise transport stress test",
                int(t["n_eyes"]),
                "domain_AUROC",
                float(t["domain_auroc"]),
                ci=f"{float(t['ci95_low']):.6f}–{float(t['ci95_high']):.6f}",
                status="measurement-domain separability; not clinical validation",
            ))
    clinical = [
        ("mBRSET hypertension strict-disc", "artifacts/mbrset_external/cv_hypertension_reliable_core_strict_disc/metrics.json"),
        ("mBRSET hypertension no-DR", "artifacts/mbrset_external/cv_hypertension_reliable_core_no_dr/metrics.json"),
        ("mBRSET vascular disease", "artifacts/mbrset_external/cv_vascular_disease/metrics.json"),
        ("mBRSET diabetic foot", "artifacts/mbrset_external/cv_diabetic_foot/metrics.json"),
        ("mBRSET acute myocardial infarction", "artifacts/mbrset_external/cv_acute_myocardial_infarction/metrics.json"),
    ]
    keys = ("R0_clinical", "R2_clinical_lowdim_vascular", "R3_full_lowdim_vascular", "M0_clinical", "M1_clinical_ocular", "M2_clinical_vascular", "M3_full_vascular", "M4_full_plus_uncertainty", "M5_full_plus_uncertainty_zoneb")
    for name, p in clinical:
        d = load(p, {})
        for key in keys:
            m = d.get("overall", {}).get(key)
            if not m: continue
            for metric in ("auroc", "auprc", "brier"):
                rows.append(row("clinical_association", name, key, d.get("patients", ""), metric.upper(), m.get(metric, ""), status="exploratory internal patient-level CV"))
    perm = load("artifacts/mbrset_external/permutation_negative_control/metrics.json", {})
    for key, m in perm.get("summary", {}).items():
        rows.append(row("clinical_association", "mBRSET permutation negative control", key, perm.get("patients", ""), "observed_AUROC", m.get("observed_auroc", ""), status=f"200 permutations; p={m.get('empirical_p_one_sided', '')}"))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"title": "RMDB-BiRAP three-layer evidence chain", "scope": "AFIO/Mendeley measurement; FIVES/DRIVE/STARE migration; BRSET/mBRSET exploratory clinical association", "cimt_fixed_validation_or_test_used": False, "rows": rows, "interpretation": {"layer_1": "AFIO establishes outcome-blind structural annotation compatibility; RAV and Fundus-AVSeg provide transfer scores.", "layer_2": "FIVES and STARE expose domain-shift degradation of the frozen structure reader; DRIVE remains access-blocked.", "layer_3": "mBRSET associations are patient-level exploratory internal CV only; no independent high-resolution clinical confirmation."}}
    (OUT / "evidence_summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "evidence_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "README.md").write_text("# Three-layer evidence chain\n\nRead-only synthesis of frozen artifacts. No CIMT fixed validation/test cohort was read or used for model selection.\n\n1. **Phenotype accuracy:** AFIO/Mendeley outcome-blind annotation audit and frozen RMDB–BiRAP extraction; RAV/Fundus-AVSeg transfer scores.\n2. **Cross-domain robustness:** FIVES and STARE frozen VascX migration, negative/weak scores retained; DRIVE is explicitly blocked by official access requirements.\n3. **Clinical association:** BRSET/mBRSET patient-level exploratory CV and permutation negative control; not independent clinical validation.\n")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))

if __name__ == "__main__":
    main()
