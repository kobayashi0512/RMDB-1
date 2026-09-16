#!/usr/bin/env python3
"""Build the first Word manuscript draft from locked RMDB outputs.

The document is deliberately a journal-style scientific draft: it uses the
VOCT-inspired claim-first Results and Discussion arrangement, but all text and
results are specific to RMDB. Numerical statements are parsed from local
locked JSON outputs rather than typed manually.
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscript_draft"
OUT = OUT_DIR / "RMDB_initial_manuscript_draft.docx"
FIG_DIR = ROOT / "figures/rmdb_age_mirage/output"

RAW = json.loads((ROOT / "artifacts/cimt_dual_readout_development/outer_cv/metrics.json").read_text())
E7 = json.loads((ROOT / "artifacts/cimt_dual_readout_development/normative_e7/metrics.json").read_text())
E8 = json.loads((ROOT / "artifacts/cimt_dual_readout_development/e8_double_orthogonal/metrics.json").read_text())
E11 = json.loads((ROOT / "artifacts/e11_dual_head_full/metrics.json").read_text())
MHTN = json.loads((ROOT / "artifacts/mbrset_external/cv_hypertension/metrics.json").read_text())
MHTN_NODR = json.loads((ROOT / "artifacts/mbrset_external/cv_hypertension_no_dr/metrics.json").read_text())
MVD = json.loads((ROOT / "artifacts/mbrset_external/cv_vascular_disease/metrics.json").read_text())
MDF = json.loads((ROOT / "artifacts/mbrset_external/cv_diabetic_foot/metrics.json").read_text())
MAMI = json.loads((ROOT / "artifacts/mbrset_external/cv_acute_myocardial_infarction/metrics.json").read_text())
ACCESS = json.loads((ROOT / "artifacts/manuscript_completion/01_access_audit/access_audit.json").read_text())


NAVY = "16324F"
TEAL = "168C83"
PURPLE = "5E477F"
GRAY = "68737D"
LIGHT_BLUE = "EAF0F6"
LIGHT_PURPLE = "F1ECF8"
LIGHT_TEAL = "E6F5F1"
WHITE = "FFFFFF"


def set_run_font(run, size=None, bold=None, italic=None, color=None, name="Arial"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa):
    tcPr = cell._tc.get_or_add_tcPr()
    tcW = tcPr.find(qn("w:tcW"))
    if tcW is None:
        tcW = OxmlElement("w:tcW")
        tcPr.append(tcW)
    tcW.set(qn("w:w"), str(width_dxa))
    tcW.set(qn("w:type"), "dxa")


def fixed_table(table, widths_dxa):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tblPr = table._tbl.tblPr
    tblW = tblPr.first_child_found_in("w:tblW")
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(sum(widths_dxa)))
    tblW.set(qn("w:type"), "dxa")
    tbl_layout = tblPr.first_child_found_in("w:tblLayout")
    if tbl_layout is None:
        tbl_layout = OxmlElement("w:tblLayout")
        tblPr.append(tbl_layout)
    tbl_layout.set(qn("w:type"), "fixed")
    tbl_ind = tblPr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tblPr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    set_run_font(run, size=8.5, color=GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def add_header_footer(doc):
    section = doc.sections[0]
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("RMDB | Initial manuscript draft")
    set_run_font(r, size=8.5, color=GRAY)
    footer = section.footer
    p = footer.paragraphs[0]
    add_page_number(p)


def set_keep_with_next(p):
    pPr = p._p.get_or_add_pPr()
    keep = OxmlElement("w:keepNext")
    pPr.append(keep)


def add_para(doc, text="", *, bold_lead=None, italic=False, color=None, align=None, after=7, before=0, size=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.18
    if align is not None:
        p.alignment = align
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, size=size or 10.8, bold=True, color=color)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r, size=size or 10.8, italic=italic, color=color)
    else:
        r = p.add_run(text)
        set_run_font(r, size=size or 10.8, italic=italic, color=color)
    return p


def add_heading(doc, text, level):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    set_keep_with_next(p)
    return p


def add_callout(doc, label, text, fill=LIGHT_PURPLE, ink=PURPLE):
    table = doc.add_table(rows=1, cols=1)
    fixed_table(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(label + " ")
    set_run_font(r, size=10.4, bold=True, color=ink)
    r = p.add_run(text)
    set_run_font(r, size=10.4, color=ink)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_table(doc, title, headers, rows, widths, note=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    set_run_font(r, size=10.4, bold=True, color=NAVY)
    table = doc.add_table(rows=1, cols=len(headers))
    fixed_table(table, widths)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell_shading(cell, LIGHT_BLUE)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(text))
        set_run_font(r, size=8.2, bold=True, color=NAVY)
    for ix, row in enumerate(rows):
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            if ix % 2 == 1:
                set_cell_shading(cell, "F8FAFC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if isinstance(text, str) and len(text) > 14 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(text))
            set_run_font(r, size=7.9, color="243B53")
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(7)
        r = p.add_run(note)
        set_run_font(r, size=8.5, italic=True, color=GRAY)
    return table


def add_figure(doc, image_name, caption, width=6.25):
    path = FIG_DIR / image_name
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.paragraph_format.space_before = Pt(0)
    c.paragraph_format.space_after = Pt(9)
    r = c.add_run(caption)
    set_run_font(r, size=8.7, italic=True, color="243B53")


def clean_caption(path):
    """Flatten the source Markdown caption into Word-ready plain text."""
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return " ".join(lines).replace("**", "")


def add_placeholder_figure(doc, label, description):
    table = doc.add_table(rows=1, cols=1)
    fixed_table(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F7F8FA")
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(label + "\n")
    set_run_font(r, size=12, bold=True, color=GRAY)
    r = p.add_run(description)
    set_run_font(r, size=9.5, italic=True, color=GRAY)


def setup_styles(doc):
    sec = doc.sections[0]
    sec.top_margin = Inches(0.78)
    sec.bottom_margin = Inches(0.72)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    sec.header_distance = Inches(0.32)
    sec.footer_distance = Inches(0.32)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    normal.font.size = Pt(10.8)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.18
    style_specs = {
        "Heading 1": (15.5, NAVY, 18, 6),
        "Heading 2": (12.5, NAVY, 12, 4),
        "Heading 3": (11.0, TEAL, 9, 3),
    }
    for name, (size, color, before, after) in style_specs.items():
        s = doc.styles[name]
        s.font.name = "Arial"
        s._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        s._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor.from_string(color)
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
    cap = doc.styles["Caption"]
    cap.font.name = "Arial"
    cap.font.size = Pt(8.7)


def fmt_ci(x):
    return f"{x[0]:+.4f} to {x[1]:+.4f}"


def m(metric, key):
    return metric["overall"][key]


def add_title_page(doc):
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Beyond the Age Mirage")
    set_run_font(r, size=25, bold=True, color=NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("A Patient-Level Audit of Retinal Microvascular Phenotypes for Age-Discordant Carotid Intima–Media Thickness")
    set_run_font(r, size=15.5, bold=False, color="243B53")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Initial scientific manuscript draft | Original Research")
    set_run_font(r, size=10.5, italic=True, color=GRAY)
    add_callout(doc, "DRAFT STATUS.", "Author names, affiliations, ethics approval wording, funding, acknowledgements and journal-formatted references remain to be inserted. All numerical results in this draft were parsed from locked local analysis artifacts.", LIGHT_BLUE, NAVY)
    add_callout(doc, "EVIDENCE BOUNDARY.", "This manuscript reports internal cIMT development, label-blind external anatomical transport and mBRSET mechanism-proxy analyses. It does not claim independent external clinical cIMT confirmation.", LIGHT_PURPLE, PURPLE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(60)
    r = p.add_run("Short running title: Retinal phenotypes beyond the age mirage")
    set_run_font(r, size=10, color=GRAY)
    doc.add_page_break()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    setup_styles(doc)
    add_header_footer(doc)
    add_title_page(doc)

    e0 = m(RAW, "E0_clinical")
    e4 = m(RAW, "E4_dual_readout")
    e4pair = RAW["paired_bootstrap"]["E4_dual_readout__minus__E0_clinical"]
    e7d = E7["overall"]["E7D_mass_z"]
    e7d_ci = E7["paired_vs_normative"]["E7D_mass_z"]["accelerated_auroc"]["ci95"]
    e8_rows = {x["experiment"]: x for x in E8["summary"]}
    e8b = e8_rows["E8B_double_orthogonal"]
    e8a = e8_rows["E8A_raw_mass"]
    e8b_mae_ci = E8["ensemble_mae_comparisons"]["E8B_double_orthogonal__minus__E8A_raw_mass"]["ci95"]
    e11 = {x["experiment"]: x for x in E11["summary"]}
    e11_ci = E11["bootstrap"]["fused_minus_mean_mae_mm"]["ci95"]

    # Structured abstract
    add_heading(doc, "Abstract", 1)
    abstract = (
        "Background: Retinal imaging offers a scalable window into systemic vascular biology, but high apparent performance for carotid intima–media thickness (cIMT) may reflect shared age information rather than retinal signal beyond age. "
        "Objective: To audit this age-proxy risk and develop a reliability-aware, bilateral retinal microvascular phenotype for age-discordant cIMT. "
        "Methods: In 2,603 participants (5,206 eyes) from the China-Fundus-CIMT development cohort, we compared raw cIMT models with age- and sex-conditioned deviation models using patient-level nested cross-validation. The proposed Reliability-aware Multi-readout Bilateral Descriptor (RMDB) retained complementary artery-vein identity and vessel-occupancy evidence, soft mass and structural topology, optic-disc-referenced Zone B measurements, bilateral mean and discordance, and quality provenance. We then performed repeated-seed reliability analyses, frozen-prediction falsification controls, label-blind external anatomical transport, and mBRSET mechanism-proxy analyses. "
        f"Results: For the conventional raw endpoint, age and sex alone achieved AUROC {e0['auroc']:.3f}; the full dual-readout model achieved {e4['auroc']:.3f} (paired difference {e4pair['auroc']['mean_delta']:+.4f}, 95% CI {fmt_ci(e4pair['auroc']['ci95'])}). After conditioning cIMT on age and sex, the best locked mass-z phenotype achieved accelerated-cIMT AUROC {e7d['accelerated_auroc']:.3f} (95% CI {e7d_ci[0]:.3f} to {e7d_ci[1]:.3f}). Double orthogonalization produced a small repeated-seed MAE improvement versus raw mass ({E8['ensemble_mae_comparisons']['E8B_double_orthogonal__minus__E8A_raw_mass']['point']:.6f} mm; 95% CI {e8b_mae_ci[0]:.6f} to {e8b_mae_ci[1]:.6f}). "
        "Interpretation: Strong discrimination for raw cIMT did not establish retinal information beyond age and sex. RMDB provides an auditable measurement framework and bounded reliability gains, not independent clinical confirmation."
    )
    add_para(doc, abstract, after=5)
    add_para(doc, "Keywords: retinal imaging; carotid intima–media thickness; vascular ageing; artery-vein phenotyping; uncertainty; bilateral aggregation; domain transport.", bold_lead="Keywords:", after=12, size=10.2)
    add_callout(doc, "CORE CLAIM.", "The central contribution is not a new opaque prediction backbone. It is a patient-level measurement audit that separates age-linked raw cIMT prediction from age-discordant vascular ageing, while making image quality, bilateral discordance and model disagreement observable.", LIGHT_TEAL, TEAL)

    # Introduction
    add_heading(doc, "1. Introduction", 1)
    add_heading(doc, "1.1 The clinical opportunity", 2)
    add_para(doc, "Carotid intima–media thickness is a widely used ultrasound marker of arterial-wall phenotype and subclinical vascular change. Its acquisition nevertheless requires dedicated equipment and trained operators, limiting repeated measurement in ophthalmic screening and low-resource settings. Retinal photography offers an attractive complementary window because it is non-invasive, low cost and already deployed at scale in eye-care workflows. This motivation has encouraged efforts to infer systemic vascular status from retinal appearance and retinal vascular measurements.")
    add_heading(doc, "1.2 The hidden problem: the age mirage", 2)
    add_para(doc, "The usual image-to-cIMT framing contains a hidden ambiguity. Age and sex influence cIMT, and age also changes retinal appearance and microvascular morphology. A model that discriminates high versus low raw cIMT can therefore obtain a strong score by exploiting age-linked information without establishing a retinal signal specific to vascular ageing. The clinically sharper question is not simply whether cIMT is high; it is whether cIMT is higher or lower than expected for an individual's age and sex.")
    add_heading(doc, "1.3 Retinal vascular features are a measurement chain", 2)
    add_para(doc, "Retinal vascular phenotypes are not direct facts extracted from an image. They are outputs of a chain that includes image preprocessing, vessel-existence probability, artery-vein identity, optic-disc localization, geometric feature calculation, bilateral aggregation and quality handling. If these intermediate uncertainties are discarded, apparent feature richness can conflate biology with segmentation error, image quality, eye-level discordance or domain shift.")
    add_heading(doc, "1.4 Objective and contributions", 2)
    add_para(doc, "We asked whether retinal microvascular evidence remains informative after the age-proxy problem is made explicit, and whether it can be represented in a way that is auditable at the patient level. Our contributions are fourfold. First, we separate conventional raw-cIMT discrimination from age- and sex-conditioned cIMT deviation. Second, we introduce RMDB, a reliability-aware dual-readout phenotype that preserves probabilistic vessel mass and structural topology. Third, we retain bilateral mean state, inter-eye discordance and quality provenance instead of collapsing both eyes to a single average. Fourth, we distinguish internal outcome development, external anatomical transport and mechanism-proxy evidence, rather than presenting every external image dataset as clinical validation.")

    # Methods
    add_heading(doc, "2. Materials and Methods", 1)
    add_heading(doc, "2.1 Study design and evidence roles", 2)
    add_para(doc, "The primary analysis was a development-only, patient-level study in the China-Fundus-CIMT cohort. All locked RMDB development analyses used the manifest split=train and did not evaluate fixed validation or test rows. External datasets were assigned narrower roles: RAV and Fundus-AVSeg were used for outcome-blind anatomical transport and failure-domain audit; mBRSET was used for patient-level mechanism-proxy analyses. These roles are not interchangeable.")
    add_table(doc, "Table 1 | Cohorts, labels and evidentiary roles.",
              ["Cohort", "Participants / images", "Available label", "Role", "Supported inference"],
              [
                  ["China-Fundus-CIMT", "2,603 / 5,206 eyes", "cIMT; age; sex", "Internal outcome development", "Patient-level association and endpoint audit"],
                  ["RAV", "External images", "Anatomical references", "Label-blind transport", "Anatomical phenotype availability; not cIMT prediction"],
                  ["Fundus-AVSeg", "External images", "Anatomical references", "Stress-test transport", "Failure-domain and phenotype-shift audit"],
                  ["mBRSET (224 px mirror)", "1,276 for hypertension", "Systemic proxy labels", "Mechanism proxy", "Exploratory systemic-label association; not independent cIMT confirmation"],
              ], [1680, 1560, 1580, 1720, 2820],
              note="All external analyses were performed without using them to tune the RMDB representation. Exact external sample completeness and transport diagnostics are reported in Table 4 and the corresponding figure.")
    add_heading(doc, "2.2 Endpoint construction", 2)
    add_para(doc, "We prespecified three related but non-equivalent tasks. Task A was the conventional raw cIMT level task, including a high-versus-low cIMT discrimination endpoint. Task B modeled cIMT deviation after accounting for age and sex. Task C was an auxiliary accelerated vascular-ageing classification derived from the age-conditioned deviation. Performance measures from Task A and Task B/C were never interpreted as directly interchangeable because they answer different questions.")
    add_heading(doc, "2.3 RMDB: reliability-aware multi-readout bilateral descriptor", 2)
    add_heading(doc, "2.3.1 Complementary soft mass and artery-vein identity", 3)
    add_para(doc, "For every eye, stored VascX logits were converted to artery-vein identity probabilities, while a separately exported occupancy model produced vessel-existence probabilities. Soft vessel, artery and vein mass fractions, soft artery-vein ratios, support-pixel quantities and probability entropy were retained. Disagreement between identity and occupancy evidence was encoded rather than treated as a hidden preprocessing defect.")
    add_heading(doc, "2.3.2 Structural topology and anatomical reference", 3)
    add_para(doc, "Structural readouts quantified vessel fraction, artery and vein fractions, skeleton-length density, endpoints, branch clusters, connected components, crossings and artery-vein log ratios. Optic-disc localization provided disc radius and Zone B reference measurements. Disc failure or fallback was retained as a provenance condition; it did not become a disease label and did not silently generate a Zone B value.")
    add_heading(doc, "2.3.3 Bilateral aggregation and quality provenance", 3)
    add_para(doc, "Patient-level phenotypes retained bilateral mean and absolute left-right difference for each eye-level measurement. Quality-related variables included disc status, Zone B availability, support pixels, occupancy entropy, identity entropy and model-gate summaries. This design distinguishes a common bilateral state from inter-eye discordance and measurement uncertainty.")
    add_heading(doc, "2.4 Evaluation and falsification", 2)
    add_para(doc, "The internal raw-endpoint analysis used five outer folds and three inner folds at the patient level. We reported out-of-fold AUROC, AUPRC, Brier score, cIMT MAE and R² as applicable, with paired bootstrap comparisons wherever models shared out-of-fold predictions. Repeated-seed analyses used five seeds and 25 outer folds. Additional reliability work included nested-fusion uncertainty analysis, coverage summaries, frozen-prediction label permutation and random-eye-pairing controls. We describe an effect as an improvement only when its paired 95% confidence interval excludes zero.")
    add_heading(doc, "2.5 External transport and mechanism-proxy analyses", 2)
    add_para(doc, "External phenotype transport was outcome-blind and used the internal training phenotype artifact only as the reference distribution. The transport protocol explicitly states that it is a measurement-domain audit, not external clinical-outcome validation. mBRSET analyses used patient-level nested cross-validation for systemic hypertension and a no-diabetic-retinopathy sensitivity subset, followed by low-event exploratory outcomes (vascular disease, diabetic foot and acute myocardial infarction). The 224 px mBRSET mirror is reported only as a mechanism-proxy setting.")
    add_heading(doc, "2.6 Reporting integrity", 2)
    add_para(doc, "An access audit determined that an early clinical-baseline test artifact exists. It was not used for RMDB feature development, model selection or locked dual-readout analyses, but the cIMT test split cannot be described as pristine or unopened. The present manuscript therefore treats all reported cIMT outcomes as internal development evidence and does not use that test artifact as confirmation.")

    # Results & discussion
    add_heading(doc, "3. Results and Discussion", 1)
    add_heading(doc, "3.1 Raw cIMT discrimination does not establish retinal signal beyond age", 2)
    add_para(doc, f"The first result was a reversal of the conventional interpretation. In the raw high-versus-low cIMT task, the clinical age-and-sex model (E0) achieved AUROC {e0['auroc']:.3f}, AUPRC {e0['auprc']:.3f} and Brier score {e0['brier']:.3f}. The full dual-readout model (E4) achieved AUROC {e4['auroc']:.3f}, AUPRC {e4['auprc']:.3f} and Brier score {e4['brier']:.3f}. The paired E4-minus-E0 AUROC difference was {e4pair['auroc']['mean_delta']:+.4f} (95% CI {fmt_ci(e4pair['auroc']['ci95'])}); its MAE difference was {e4pair['mae_mm']['mean_delta']:+.6f} mm (95% CI {fmt_ci(e4pair['mae_mm']['ci95'])}). Thus, in this setting, more retinal vascular detail did not provide evidence of incremental raw-endpoint discrimination beyond the age-and-sex baseline.")
    add_figure(doc, "Figure1_age_mirage.png", clean_caption(ROOT / "figures/rmdb_age_mirage/Figure1_caption.md"), width=6.25)
    add_figure(doc, "Figure2_performance_reversal.png", clean_caption(ROOT / "figures/rmdb_age_mirage/Figure2_caption.md"), width=6.25)
    add_table(doc, "Table 2 | Internal endpoint audit. Metrics are compared only within an endpoint.",
              ["Task", "Feature set", "AUROC", "AUPRC", "Brier", "cIMT MAE (mm)"],
              [
                  ["Raw cIMT", "E0 clinical", f"{e0['auroc']:.3f}", f"{e0['auprc']:.3f}", f"{e0['brier']:.3f}", f"{e0['mae_mm']:.3f}"],
                  ["Raw cIMT", "E1 legacy VascX", f"{m(RAW, 'E1_legacy_vascx')['auroc']:.3f}", f"{m(RAW, 'E1_legacy_vascx')['auprc']:.3f}", f"{m(RAW, 'E1_legacy_vascx')['brier']:.3f}", f"{m(RAW, 'E1_legacy_vascx')['mae_mm']:.3f}"],
                  ["Raw cIMT", "E2 mass", f"{m(RAW, 'E2_mass_udct')['auroc']:.3f}", f"{m(RAW, 'E2_mass_udct')['auprc']:.3f}", f"{m(RAW, 'E2_mass_udct')['brier']:.3f}", f"{m(RAW, 'E2_mass_udct')['mae_mm']:.3f}"],
                  ["Raw cIMT", "E3 structure", f"{m(RAW, 'E3_factorized')['auroc']:.3f}", f"{m(RAW, 'E3_factorized')['auprc']:.3f}", f"{m(RAW, 'E3_factorized')['brier']:.3f}", f"{m(RAW, 'E3_factorized')['mae_mm']:.3f}"],
                  ["Raw cIMT", "E4 dual readout", f"{e4['auroc']:.3f}", f"{e4['auprc']:.3f}", f"{e4['brier']:.3f}", f"{e4['mae_mm']:.3f}"],
                  ["Age-conditioned", "E7D mass-z", f"{e7d['accelerated_auroc']:.3f}", f"{e7d['accelerated_auprc']:.3f}", f"{e7d['accelerated_brier']:.3f}", f"{e7d['cimt_mae_mm']:.3f}"],
              ], [1350, 2460, 1080, 1080, 1080, 2310],
              note="The first five rows answer the conventional raw-cIMT question; the final row answers the age-conditioned accelerated-cIMT question. AUROC values therefore should not be interpreted as a direct contest across task blocks.")

    add_heading(doc, "3.2 Age-discordant cIMT leaves a weak but testable residual signal", 2)
    add_para(doc, f"Once cIMT was conditioned on age and sex, discrimination was substantially more demanding. The E7D mass-z phenotype achieved accelerated-cIMT AUROC {e7d['accelerated_auroc']:.3f}, AUPRC {e7d['accelerated_auprc']:.3f} and Brier score {e7d['accelerated_brier']:.3f}; its stored bootstrap interval for AUROC was {e7d_ci[0]:.3f} to {e7d_ci[1]:.3f}. Repeated-seed E8B double orthogonalization achieved cIMT MAE {e8b['mae_mm_mean']:.5f} ± {e8b['mae_mm_std']:.5f} mm and AUROC {e8b['auroc_mean']:.3f} ± {e8b['auroc_std']:.3f}. Relative to E8A raw mass, the paired MAE difference was {E8['ensemble_mae_comparisons']['E8B_double_orthogonal__minus__E8A_raw_mass']['point']:.6f} mm (95% CI {e8b_mae_ci[0]:.6f} to {e8b_mae_ci[1]:.6f}). This is a small reliability improvement, not evidence of a large clinical discrimination gain.")
    add_callout(doc, "INTERPRETATION.", "The weak age-conditioned signal is an informative result: it shows why raw cIMT accuracy cannot be used as a proxy for identifying accelerated vascular ageing.", LIGHT_PURPLE, PURPLE)

    add_heading(doc, "3.3 RMDB makes phenotype reliability visible", 2)
    add_para(doc, "Figure 3 makes the unit of analysis explicit. A single participant contributes two distinct retinal images, two complementary probabilistic readouts per eye, an optic-disc-referenced anatomical coordinate system and retained quality metadata. Rather than averaging away left-right discrepancy or converting every intermediate failure into a missing value, RMDB carries the bilateral mean, absolute difference, entropy and disc/Zone B availability forward to the patient-level representation.")
    add_figure(doc, "Figure3_rmdb_real_image_pipeline.png", clean_caption(ROOT / "figures/rmdb_age_mirage/Figure3_caption.md"), width=6.25)

    add_heading(doc, "3.4 Uncertainty-aware aggregation yields bounded reliability gains", 2)
    add_para(doc, f"The reliability analyses were directionally consistent but bounded. Across 25 outer-fold fits, E8B improved or matched its reference in {E8['fold_direction']['E8B_double_orthogonal']['folds_improved']} of {E8['fold_direction']['E8B_double_orthogonal']['folds_total']} folds. In the dual-estimator analysis, nested fusion (H2) achieved MAE {e11['H2_nested_fusion']['mae_mm_mean']:.5f} mm versus {e11['H0_mean']['mae_mm_mean']:.5f} mm for the mean head. The fused-minus-mean paired MAE difference was {E11['bootstrap']['fused_minus_mean_mae_mm']['point']:.6f} mm (95% CI {e11_ci[0]:.6f} to {e11_ci[1]:.6f}). However, the corresponding RMSE interval crossed zero ({E11['bootstrap']['fused_minus_mean_rmse_mm']['ci95'][0]:+.6f} to {E11['bootstrap']['fused_minus_mean_rmse_mm']['ci95'][1]:+.6f} mm). We therefore report a bounded MAE improvement rather than a universal error reduction.")
    add_table(doc, "Table 3 | Reliability and uncertainty analyses.",
              ["Analysis", "Reference", "Result", "Paired 95% CI", "Interpretation"],
              [
                  ["E8B vs E8A", "Raw mass", "Δ MAE −0.000186 mm", "−0.000307 to −0.000064", "Small, supported MAE gain"],
                  ["H2 vs H0", "Mean head", "Δ MAE −0.001381 mm", "−0.001859 to −0.000898", "Supported MAE gain"],
                  ["H2 vs H0", "Mean head", "Δ RMSE +0.000361 mm", "−0.000140 to +0.000861", "No claim of RMSE gain"],
                  ["E8B fold direction", "25 outer folds", "24/25 improved", "Not applicable", "Directionally consistent, effect remains small"],
              ], [1540, 1460, 1970, 2250, 2140],
              note="All claims are limited to analyses whose stored paired bootstrap interval excludes zero. Repeatability, selective coverage and negative controls are reserved for Figure 4 and supplementary materials in the next draft iteration.")
    add_placeholder_figure(doc, "Figure 4 placeholder | Reliability is measurable, and its gains are bounded.", "To be populated from E8/E11 repeated-seed, coverage, repeatability and negative-control outputs. The planned figure will report only effects supported by paired confidence intervals.")

    add_heading(doc, "3.5 Cross-domain transport is a phenotype audit, not external cIMT validation", 2)
    add_para(doc, "The external transport protocol was deliberately outcome-blind. RAV and Fundus-AVSeg were used to inspect whether the measurement chain could produce finite, anatomically interpretable phenotypes outside the internal imaging domain and to identify failure domains. Quality-overlap transport sensitivity used disc status, disc radius, Zone B validity, occupancy entropy, identity entropy and mean gate as explicit covariates. These analyses can support a claim about phenotype availability, shift and failure detection; they cannot support an external clinical cIMT prediction claim because neither cohort contains the corresponding cIMT outcome confirmation.")
    add_placeholder_figure(doc, "Figure 5 placeholder | Cross-dataset transport is a phenotype audit, not external cIMT validation.", "Will show real RAV and Fundus-AVSeg images, local annotations/overlays, finite-output and quality-overlap diagnostics, and phenotype drift. No clinical-effect performance axis will be included.")
    add_placeholder_figure(doc, "Figure 6 placeholder | Failure cases are data, not exclusions.", "Will show real examples of disc fallback, low overlap, inter-eye discordance and repeat-view instability, alongside the retained quality metadata and handling decision.")
    add_table(doc, "Table 4 | External evidence boundary.",
              ["Dataset", "Analysis", "What it supports", "What it does not support"],
              [
                  ["RAV", "Label-blind anatomical transport", "Availability and anatomical consistency of phenotypes", "External cIMT outcome validation"],
                  ["Fundus-AVSeg", "Failure-domain stress test", "Domain shift and failure-mode characterization", "Clinical risk prediction"],
                  ["mBRSET", "Patient-level mechanism proxy", "Association heterogeneity for available systemic labels", "High-resolution or independent cIMT confirmation"],
              ], [1650, 2080, 2830, 2800],
              note="This table is deliberately restrictive: it records the maximum defensible inference for each external data source.")

    add_heading(doc, "3.6 Mechanism-proxy results do not justify a universal systemic-risk claim", 2)
    h0, h1, h2, h3 = (MHTN["overall"][x] for x in ["M0_clinical", "M1_clinical_ocular", "M2_clinical_vascular", "M3_full"])
    h1pair = MHTN["paired_bootstrap"]["M1_clinical_ocular__minus__M0_clinical"]
    h3pair = MHTN["paired_bootstrap"]["M3_full__minus__M0_clinical"]
    add_para(doc, f"In mBRSET hypertension analysis ({MHTN['patients']} participants; {MHTN['positive']} positive labels), clinical-only M0 achieved AUROC {h0['auroc']:.4f}. Adding ocular severity variables (M1) achieved AUROC {h1['auroc']:.4f}; the paired AUROC interval crossed zero ({fmt_ci(h1pair['auroc']['ci95'])}), whereas the Brier difference was {h1pair['brier']['mean_delta']:+.4f} (95% CI {fmt_ci(h1pair['brier']['ci95'])}). The vascular-only augmented model M2 and full model M3 did not demonstrate stable discrimination benefit; M3 minus M0 AUROC was {h3pair['auroc']['mean_delta']:+.4f} (95% CI {fmt_ci(h3pair['auroc']['ci95'])}). In the no-diabetic-retinopathy subset ({MHTN_NODR['patients']} participants; {MHTN_NODR['positive']} positive labels), M0 and M1 AUROC were {MHTN_NODR['overall']['M0_clinical']['auroc']:.4f} and {MHTN_NODR['overall']['M1_clinical_ocular']['auroc']:.4f}, respectively; the M1-minus-M0 AUROC interval again crossed zero. These results argue against a universal claim that the vascular feature block incrementally predicts every systemic proxy endpoint.")
    add_para(doc, f"The three secondary outcomes were explicitly exploratory because of lower event counts: vascular disease ({MVD['positive']} events), diabetic foot ({MDF['positive']} events) and acute myocardial infarction ({MAMI['positive']} events). Their heterogeneous model rankings are useful as a stress test of mechanism claims, but they should not be used as confirmatory evidence.")
    add_table(doc, "Table 5 | mBRSET mechanism-proxy analyses.",
              ["Outcome / sample", "Model", "AUROC", "AUPRC", "Brier", "Interpretation"],
              [
                  [f"Hypertension\n{MHTN['patients']} / {MHTN['positive']}", "M0 clinical", f"{h0['auroc']:.4f}", f"{h0['auprc']:.4f}", f"{h0['brier']:.4f}", "Reference"],
                  [f"Hypertension\n{MHTN['patients']} / {MHTN['positive']}", "M1 clinical + ocular", f"{h1['auroc']:.4f}", f"{h1['auprc']:.4f}", f"{h1['brier']:.4f}", "AUROC CI crosses 0 vs M0; Brier improves"],
                  [f"Hypertension\n{MHTN['patients']} / {MHTN['positive']}", "M2 clinical + vascular", f"{h2['auroc']:.4f}", f"{h2['auprc']:.4f}", f"{h2['brier']:.4f}", "No stable incremental benefit"],
                  [f"Hypertension\n{MHTN['patients']} / {MHTN['positive']}", "M3 full", f"{h3['auroc']:.4f}", f"{h3['auprc']:.4f}", f"{h3['brier']:.4f}", "AUROC lower than M0"],
                  [f"No DR hypertension\n{MHTN_NODR['patients']} / {MHTN_NODR['positive']}", "M0 / M1", f"{MHTN_NODR['overall']['M0_clinical']['auroc']:.4f} / {MHTN_NODR['overall']['M1_clinical_ocular']['auroc']:.4f}", "—", "—", "Sensitivity; AUROC CI crosses 0"],
                  ["Secondary proxy outcomes", "Vascular disease / foot / AMI", "—", "—", "—", f"Exploratory; {MVD['positive']}/{MDF['positive']}/{MAMI['positive']} events"],
              ], [1860, 1700, 960, 960, 960, 2920],
              note="mBRSET is a 224 px external mechanism-proxy setting. It is not an independent high-resolution cIMT cohort and is not described as confirmatory clinical validation.")
    add_placeholder_figure(doc, "Figure 7 placeholder | Evidence hierarchy and mechanism-proxy analyses.", "Will distinguish internal cIMT development, external anatomical transport and mBRSET mechanism-proxy analyses; the final panel will state the remaining need for independent high-resolution cIMT confirmation.")

    # Conclusion, limitations and declarations
    add_heading(doc, "4. Conclusion", 1)
    add_para(doc, "This study reframes retinal-systemic vascular modelling as an endpoint and measurement problem before it is a feature-count problem. The raw cIMT result shows that strong discrimination can be obtained from age-linked information and therefore cannot, by itself, establish a retinal biomarker of accelerated vascular ageing. RMDB provides a transparent alternative: complementary soft and structural readouts, anatomy-aware variables, bilateral discordance and quality provenance are retained at the patient level. The observed reliability gains were small but measurable, and the external analyses identify what the current evidence can and cannot support. Independent high-resolution cIMT confirmation is the necessary next step before clinical claims about systemic vascular screening can be made.")
    add_heading(doc, "Limitations", 1)
    add_para(doc, "First, all cIMT results are internal development analyses, and the access audit establishes that an early clinical-baseline test artifact exists; the test split cannot be presented as pristine confirmation. Second, age conditioning reduces the target to a substantially harder residual problem and may attenuate both clinically meaningful and nuisance variation. Third, RAV and Fundus-AVSeg provide anatomical, not cIMT-outcome, evidence. Fourth, mBRSET is a lower-resolution, proxy-label cohort with heterogeneous and partly low-event outcomes. Finally, the current RMDB feature set is a measurement representation; it does not identify a causal retinal mechanism for arterial-wall thickening.")
    add_heading(doc, "Data and code availability", 1)
    add_para(doc, "Analysis scripts, locked local artifacts and figure-generation scripts are organized within the project workspace. Public-data access and redistribution terms, as well as a de-identified sharing statement for internal data, must be finalized with the target journal and relevant institutional approvals.")
    add_heading(doc, "Ethics, funding, and author contributions", 1)
    add_para(doc, "[To be completed by the study team: institutional review board approval/waiver, consent language, funding sources, conflict-of-interest statement, author contributions and acknowledgements.]")
    add_heading(doc, "References", 1)
    add_para(doc, "Reference-manager insertion pending. The submission version should include verified citations for cIMT clinical interpretation, retinal microvascular epidemiology, retinal imaging and systemic vascular risk, artery-vein segmentation methodology, calibration/decision analysis and all source datasets. Citation numbering is intentionally not frozen until a target journal is selected.", italic=True, color=GRAY)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
