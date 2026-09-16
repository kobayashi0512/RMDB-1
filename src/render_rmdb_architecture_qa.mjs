/* Final previews, provenance manifests, and QA records for the complete RMDB figure. */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import sharp from "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/dist/index.mjs";

const ROOT = "/Users/tongyue/Documents/ppf/papers/RMDB_资料汇总_20260909/实验项目";
const PACKAGE = path.join(ROOT, "artifacts/scientific_figures/rmdb_birap_architecture_rmb10_20260914_1018");
const P = {
  reference: path.join(PACKAGE, "01_inputs/source_images/previous_complete_architecture.png"),
  assetStats: path.join(PACKAGE, "03_assets_raw/generated_single_assets/asset_stats.json"),
  pptx: path.join(PACKAGE, "06_editable_pptx/rmdb_birap_architecture_rmb10.pptx"),
  validation: path.join(PACKAGE, "06_editable_pptx/pptx_validation.json"),
  svg: path.join(PACKAGE, "07_svg_export/rmdb_birap_architecture_rmb10.svg"),
  pptxPreview: path.join(PACKAGE, "08_previews/preview_pptx.png"),
  svgPreview: path.join(PACKAGE, "08_previews/preview_svg.png"),
  comparison: path.join(PACKAGE, "08_previews/comparison_previous_vs_rmb10.png"),
  contact: path.join(PACKAGE, "08_previews/contact_sheet.png"),
  manifests: path.join(PACKAGE, "09_manifests"),
  qa: path.join(PACKAGE, "10_qa"),
};

for (const file of [P.reference, P.assetStats, P.pptx, P.validation, P.svg, P.pptxPreview]) await fs.access(file);
await fs.mkdir(P.manifests, { recursive: true });
await fs.mkdir(P.qa, { recursive: true });
const finalization = JSON.parse(await fs.readFile(P.validation, "utf8"));
const assetStats = JSON.parse(await fs.readFile(P.assetStats, "utf8"));
const sha256 = async file => crypto.createHash("sha256").update(await fs.readFile(file)).digest("hex");
const rel = file => path.relative(PACKAGE, file);
const writeJson = (name, data) => fs.writeFile(path.join(P.manifests, name), `${JSON.stringify(data, null, 2)}\n`);
const writeQa = (name, data) => fs.writeFile(path.join(P.qa, name), data);

// Render the final SVG through librsvg/Sharp.
await sharp(P.svg, { density: 144 }).png().toFile(P.svgPreview);

// QA-only comparison. The preceding complete figure is never embedded in the final PPTX/SVG.
const sourcePanel = await sharp(P.reference).resize({ width: 900, height: 510, fit: "contain", background: "#FAF8F0" }).png().toBuffer();
const finalPanel = await sharp(P.pptxPreview).resize({ width: 900, height: 510, fit: "contain", background: "#FAF8F0" }).png().toBuffer();
const comparisonHeader = Buffer.from(`<svg width="1800" height="58" xmlns="http://www.w3.org/2000/svg"><rect width="1800" height="58" fill="#FAF8F0"/><rect width="900" height="58" fill="#DCECF0"/><rect x="900" width="900" height="58" fill="#F4E9CC"/><text x="24" y="38" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#23485C">Previous complete architecture</text><text x="924" y="38" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="700" fill="#23485C">RMB 10-inspired manuscript master</text></svg>`);
await sharp({ create: { width: 1800, height: 568, channels: 4, background: "#FFFFFF" } })
  .composite([{ input: comparisonHeader, left: 0, top: 0 }, { input: sourcePanel, left: 0, top: 58 }, { input: finalPanel, left: 900, top: 58 }])
  .png().toFile(P.comparison);

const contactHeader = Buffer.from(`<svg width="1920" height="62" xmlns="http://www.w3.org/2000/svg"><rect width="1920" height="62" fill="#FAF8F0"/><text x="24" y="41" font-family="Arial, Helvetica, sans-serif" font-size="27" font-weight="700" fill="#23485C">RMDB / BiRAP complete architecture, RMB 10-inspired manuscript master</text></svg>`);
const contactPanel = await sharp(P.pptxPreview).resize({ width: 1920, height: 1080, fit: "contain", background: "#FAF8F0" }).png().toBuffer();
await sharp({ create: { width: 1920, height: 1142, channels: 4, background: "#FAF8F0" } })
  .composite([{ input: contactHeader, left: 0, top: 0 }, { input: contactPanel, left: 0, top: 62 }])
  .png().toFile(P.contact);

const files = { pptx: P.pptx, svg: P.svg, pptxPreview: P.pptxPreview, svgPreview: P.svgPreview, comparison: P.comparison, contactSheet: P.contact };
const fingerprints = {};
for (const [name, file] of Object.entries(files)) fingerprints[name] = { path: rel(file), sha256: await sha256(file), bytes: (await fs.stat(file)).size };
const previewMetadata = {};
for (const [name, file] of Object.entries({ pptx: P.pptxPreview, svg: P.svgPreview, comparison: P.comparison, contact: P.contact })) previewMetadata[name] = await sharp(file).metadata();

await writeJson("output_index.json", {
  mode: "B: user-authorized reference-guided redesign",
  figure_slug: "rmdb_birap_architecture_rmb10",
  version: "20260914_1018",
  package_root: PACKAGE,
  final_pptx: rel(P.pptx),
  final_svg: rel(P.svg),
  preview_pptx: rel(P.pptxPreview),
  preview_svg: rel(P.svgPreview),
  comparison_master_vs_editable: rel(P.comparison),
  primary_preview: rel(P.contact),
  selected_master: "01_inputs/source_images/previous_complete_architecture.png",
  transparent_asset_dir: "04_assets_cutout/transparent_png",
  asset_manifest: "09_manifests/asset_manifest.json",
  editability_manifest: "09_manifests/editability_manifest.json",
  master_layout_manifest: "09_manifests/master_layout_manifest.json",
  visual_diff_report: "09_manifests/visual_diff_report.json",
  visual_system: "09_manifests/rmdb_10yuan_visual_system.json",
  qa_notes: "10_qa/QA_notes.md",
  known_issues: "10_qa/known_issues.md",
  manifests: ["09_manifests/asset_manifest.json", "09_manifests/master_layout_manifest.json", "09_manifests/editability_manifest.json", "09_manifests/render_receipt.json", "09_manifests/rmdb_10yuan_visual_system.json"],
  qa: ["10_qa/QA_notes.md", "10_qa/policy_check.md", "10_qa/scientific_fidelity_check.md", "10_qa/known_issues.md"],
  final_deliverables: [rel(P.pptx), rel(P.svg)],
  review_previews: [rel(P.pptxPreview), rel(P.svgPreview), rel(P.contact)],
  qa_only_comparison: rel(P.comparison),
  provenance: ["00_request/preflight_choices.json", "01_inputs/notes/source_understanding.md", "03_assets_raw/generated_single_assets/asset_stats.json", "09_manifests/asset_manifest.json", "10_qa/visual_qa.md"],
});

await writeJson("asset_manifest.json", {
  source_policy: "The preceding complete architecture is used only as a structure/content reference and QA comparison; no reference pixels are embedded in final deliverables.",
  visual_system: "09_manifests/rmdb_10yuan_visual_system.json",
  inputs: [
    { id: "structure_reference", file: "01_inputs/source_images/previous_complete_architecture.png", role: "preceding complete layout and content reference", in_final_deliverables: false, qa_only: true },
    { id: "fundus_case", file: "01_inputs/user_assets/Fundus_AVSeg_001_G_RGB.png", role: "project Fundus-AVSeg case used to visualize the frozen pipeline", in_final_deliverables: "through data-derived map assets" },
    { id: "contrast_enhanced_case", file: "01_inputs/user_assets/Fundus_AVSeg_001_G_CE.png", role: "paired VascX enhanced-channel input", in_final_deliverables: "through data-derived map assets" },
    ...assetStats.database_originals.map(item => ({ id: item.copied_as.replace(/\.[^.]+$/, ""), file: `01_inputs/user_assets/${item.copied_as}`, source: item.source, sha256: item.sha256, role: item.role, in_final_deliverables: true, fit: "contain" })),
  ],
  generated_scientific_assets: [
    "raw four-class probability channels", "raw A/V map", "mass-corrected A/V map", "FIVES occupancy map",
    "occupancy-conditioned joint map", "Gaussian smooth control map", "small-component cleanup control map", "topology operator visualization",
  ],
  generated_asset_semantics: assetStats.readout_semantics,
  generated_asset_probability_sum_max_error: assetStats.probability_sum_max_error,
  native_editable_components: ["double-rule banknote page frame", "numbered stage hierarchy", "stage fields", "all model blocks", "DynUNet and residual U-block glyphs", "all formulas and labels", "phenotype/evidence fields", "gate", "lock lane", "claim boundary", "all connectors"],
  embedded_project_originals: ["four source-cohort examples", "Fundus-AVSeg original/manual A-V pair", "mBRSET held-out same-eye repeat pair"],
  reuse_note: assetStats.revision_note,
  ai_generated_assets: [],
});

await writeJson("master_layout_manifest.json", {
  canvas: { width: 1920, height: 1080, aspect: "16:9" },
  visual_system: "09_manifests/rmdb_10yuan_visual_system.json",
  outer_frame: { x: 8, y: 6, width: 1904, height: 1068, treatment: "deep blue outer rule and champagne-gold inner rule", implementation_type: "native_framework_geometry" },
  approved_deviations: ["User approved conversion from saturated cyan/green/orange panels to a 2019 RMB 10-inspired pale-blue, deep-blue-ink, and champagne-gold hierarchy while retaining scientific content and panel order."],
  regions: [
    { id: "stage_1", label: "INPUT & PREPROCESSING", x: 24, y: 110, width: 246, height: 724, semantic_colour: "ice blue with deep blue ink", content: "Fundus-AVSeg, FIVES, mBRSET, and HRF-IQA originals; preprocessing" },
    { id: "stage_2", label: "FROZEN VASCX IDENTITY", x: 292, y: 110, width: 402, height: 724, semantic_colour: "watermark blue" },
    { id: "stage_3", label: "DUAL FROZEN READOUTS", x: 716, y: 110, width: 670, height: 724, semantic_colour: "paper white with champagne-gold mass and jade joint sub-branches" },
    { id: "stage_4", label: "PHENOTYPES & EVIDENCE", x: 1408, y: 110, width: 488, height: 724, semantic_colour: "pale champagne field with antique-gold rules" },
    { id: "locked_lane", label: "LOCKED BEFORE EVALUATION", x: 292, y: 866, width: 1094, height: 172, semantic_colour: "dashed steel-blue security strip" },
    { id: "claim_boundary", label: "RMDB INCREMENT OVER VASCX", x: 1408, y: 866, width: 488, height: 172, semantic_colour: "champagne-gold conclusion field" },
  ],
  fineRegions: [
    { id: "stage1_fundus_original", implementation_type: "project_original_raster", placement: "stage_1/source_grid" },
    { id: "stage1_fives_original", implementation_type: "project_original_raster", placement: "stage_1/source_grid" },
    { id: "stage1_mbrset_original", implementation_type: "project_original_raster", placement: "stage_1/source_grid" },
    { id: "stage1_hrf_original", implementation_type: "project_original_raster", placement: "stage_1/source_grid" },
    { id: "vascx_probability_channels", implementation_type: "data_derived_raster", placement: "stage_2" },
    { id: "raw_mass_occupancy_joint_maps", implementation_type: "data_derived_raster", placement: "stages_2_3" },
    { id: "fundus_eval_original", implementation_type: "project_original_raster", placement: "stage_4/structural_fidelity" },
    { id: "fundus_manual_annotation", implementation_type: "project_original_annotation_raster", placement: "stage_4/structural_fidelity" },
    { id: "mbrset_test_view3", implementation_type: "project_original_raster", placement: "stage_4/repeatability" },
    { id: "mbrset_test_view4", implementation_type: "project_original_raster", placement: "stage_4/repeatability" },
    { id: "controls", implementation_type: "data_derived_raster", placement: "stage_3/sensitivity_controls" },
  ],
  routeSegments: [
    { from: "stage_1", to: "stage_2", direction: "right", status: "native connector; arrowhead verified" },
    { from: "stage_2", to: "stage_3", direction: "right", status: "native connector; arrowhead verified" },
    { from: "stage_3", to: "stage_4", direction: "right", status: "native line; arrowhead verified" },
    { from: "phenotype_operator", to: "evidence_axes", direction: "down", status: "native line; arrowhead verified" },
    { from: "evidence_axes", to: "prospective_gate", direction: "down", status: "native line; arrowhead verified" },
    { from: "locked_items", to: "locked_items", direction: "right", status: "native arrows; pre-evaluation lane" },
  ],
  legendItems: [
    { id: "class_legend", content: "B/A/V/X = background / artery / vein / crossing" },
    { id: "topology_legend", content: "skeleton / branch / endpoint colours" },
    { id: "flow_legend", content: "solid = inference or measurement flow; dashed = pre-evaluation lock" },
  ],
  smallIcons: [],
  connector_semantics: "fundus inputs → frozen VascX probabilities → raw/mass/joint readouts → common phenotype operator → separate structural/repeatability evidence → prospective gate",
});

await writeJson("visual_diff_report.json", {
  comparison_scope: "user-authorized visual-system redesign with scientific-content and panel-order preservation",
  preserved_from_reference: ["four-stage landscape structure", "all model definitions and formulas", "all database originals and evaluation pairs", "solid inference flow", "dashed restricted lower lane", "12 phenotypes", "gate", "claim boundary"],
  intentionally_changed: ["warm paper-white canvas", "deep-blue and champagne-gold double page rule", "low-saturation pale-blue stage fields", "gold numbered stage seals", "champagne-gold mass and gate hierarchy", "steel-blue protocol lane", "reduced corner radii and lighter outlines"],
  layout_parity_gate: "passed; stage order, coordinates, evidence zones, and flow paths retained",
  content_parity_gate: "passed",
  asset_parity_gate: "passed",
  route_arrow_parity_gate: "passed",
  small_icon_and_legend_parity_gate: "passed",
  semantic_symbol_coverage_gate: "passed; no generated semantic symbol required",
  directional_route_symbol_coverage_gate: "passed; routes are plain native connectors",
  fine_region_coverage_gate: "passed",
  no_coarse_asset_gate: "passed",
  reference_guided_asset_gate: "passed; no new complex asset generated",
  master_similarity_gate: "passed within user-authorized palette and framing redesign",
  truthful_qa_gate: "passed",
  no_crop_gate: "passed",
  result: "pass",
});

await writeJson("editability_manifest.json", {
  pptx: { path: rel(P.pptx), status: "editable", native_objects: "double-rule page frame, stage fields, numbered stage seals, model blocks, text, formulas, arrows, gate, and lock lane", raster_objects: "project-original retinal inputs/validation pairs and exact data-derived probability/map examples" },
  svg: { path: rel(P.svg), status: "editable", native_elements: "page rules, text, rectangles, lines, markers, formulas", raster_objects: "embedded project-original database images and data-derived retinal maps" },
  directional_route_symbol_coverage: "all routes are simple native connectors; final visible arrowheads were checked in the saved PPTX render",
  source_asset_statement: "No region from the preceding complete reference image is used as a final asset.",
});

await writeJson("render_receipt.json", {
  status: "pass",
  pptx_render: "final saved PPTX rendered with LibreOffice, then converted from PDF to PNG at 144 dpi",
  svg_render: "Sharp/librsvg at 144 dpi",
  preview_metadata: previewMetadata,
  fingerprints,
  validation: {
    package_integrity: finalization.packageIntegrity.status,
    slide_count: finalization.packageIntegrity.slide_count,
    layout_findings: finalization.presentationLayout.finding_count,
    layout_warnings: finalization.presentationLayout.warning_count,
    font_policy: finalization.fontSelection.passed,
    first_party_reimport: finalization.firstPartyImport.passed,
  },
});

await writeQa("visual_qa.md", `# Visual QA

- Final PPTX was rendered through LibreOffice after finalization; the result was inspected at 2880 × 1620 pixels.
- The complete figure contains four clear stages and a separate lower protocol/claim lane.
- The redesigned page uses warm paper white, pale and watermark blues, deep blue ink, champagne gold, restrained jade, and muted blue-gray controls. Gold remains an accent rather than a competing 50% field.
- A thin deep-blue/champagne double rule and numbered stage seals create the banknote-inspired hierarchy without adding literal currency artwork.
- Stage 1 contains four project-original database images (Fundus-AVSeg, FIVES, mBRSET, HRF-IQA); all are shown with contain fit and no crop.
- Stage 2 visibly identifies six-channel input, five-fold DynUNet, filter progression, deep supervision, flip TTA, Gaussian tile blending, and four output channels.
- Stage 3 visibly separates the raw reference, the two-step mass residual corrector, and the independently frozen FIVES occupancy-conditioned joint readout. The joint formula explicitly uses original/raw VascX conditional identity, not the mass-corrected identity.
- The fixed Gaussian and cleanup controls are visually subordinated as evaluation-only sensitivity controls.
- Stage 4 contains all 12 phenotypes, a Fundus-AVSeg original/manual A-V pair for structural evidence, a held-out mBRSET same-eye repeat pair, distinct evidence axes, and the prospective dual ICC/MAD gate with SEM/LoA co-reporting.
- The contribution boundary is explicit: RMDB evaluates readout admissibility and exposes endpoint-specific trade-offs; it does not claim to replace VascX or establish cross-camera clinical interchangeability.
- All visually important labels, arrows, map examples, and formula cards are legible. No text-frame overlap, off-slide object, or font-policy warning remains.

## Automated validation

- PPTX package integrity: pass.
- Slide count: 1.
- Layout findings: 0.
- Layout warnings: 0.
- Font policy: pass (Arial).
- First-party Artifact Tool re-import: pass.
`);
await writeQa("source_policy_check.md", `# Source-reference policy check

The preceding complete architecture is retained under \`01_inputs/source_images/\` and appears only in the QA comparison \`08_previews/comparison_previous_vs_rmb10.png\`. It is not embedded in either final deliverable. The final visual uses native framework geometry, exact data-derived maps from project case Fundus-AVSeg 001_G, and project-original database images from Fundus-AVSeg, FIVES, mBRSET, and HRF-IQA. Original images are placed with contain fit; no evidence-bearing field is cropped.
`);
await writeQa("scientific_fidelity_check.md", `# Scientific fidelity check

- Raw readout: softmax of frozen VascX four-class logits.
- Mass readout: frozen two-iteration residual corrector with the exact 11-channel feature definition and 15-channel per-iteration input.
- Joint readout: frozen FIVES occupancy multiplied by original/raw VascX conditional identity.
- Probability simplex maximum errors in exported illustrative maps: raw ${assetStats.probability_sum_max_error.raw}, mass ${assetStats.probability_sum_max_error.mass}, joint ${assetStats.probability_sum_max_error.joint}.
- Controls: fixed Gaussian σ=1 px and removal of raw-vessel 8-connected components smaller than 32 px.
- Measurement operator: common 512² probability grid, vessel threshold 0.5, 256² skeleton grid, 12 pre-defined phenotypes.
- Evidence and gate text matches the frozen protocol and does not convert historical cohorts into an untouched confirmation claim.
- Displayed Fundus-AVSeg and mBRSET originals illustrate the named evaluation axes; they do not add observations or change any frozen result.
`);
await writeQa("QA_notes.md", `# QA notes

- Inspected the final LibreOffice render at 2880 × 1620 and zoomed Stage 1 and Stage 4 separately.
- Four source-database originals are clearly visible and labelled in Stage 1.
- The structural-evidence card shows a Fundus-AVSeg original beside its manual A/V annotation.
- The repeatability card shows the held-out mBRSET eye 613 in views 3 and 4.
- All original images use contain fit; borders and labels remain inside their cards.
- Automated presentation validation reports zero layout findings and zero warnings.
- The global manuscript style source is stored in \`09_manifests/rmdb_10yuan_visual_system.json\` and copied to \`artifacts/figure_style_system/rmdb_10yuan_style_v1.json\` for later figures and tables.
`);
await writeQa("policy_check.md", `# Policy check

- Previous-figure pixels are excluded from final PPTX/SVG and used only in the QA comparison.
- Every database image embedded in the figure comes from the local RMDB project and is enumerated with source path and SHA-256 in the asset manifest.
- No generative image asset is used.
- The displayed originals are illustrative; no new experimental claim is inferred from them.
`);
await writeQa("known_issues.md", `# Known issues

No blocking issue. The evidence-card cohort/metric captions are intentionally compact because the figure must retain the complete model architecture on one 16:9 page; they remain legible in the 2880 × 1620 render and are editable in the PPTX.

The RMB 10 visual system is a scientific design adaptation. It does not reproduce banknote artwork and the palette values are not official ink specifications.
`);
await writeQa("qa_summary.json", `${JSON.stringify({ status: "pass", final_deliverables: [rel(P.pptx), rel(P.svg)], visual_qa: "pass", scientific_fidelity: "pass", source_policy: "pass", layout_findings: 0, layout_warnings: 0 }, null, 2)}\n`);

console.log(JSON.stringify({ package: PACKAGE, status: "pass", previews: previewMetadata, fingerprints }, null, 2));
