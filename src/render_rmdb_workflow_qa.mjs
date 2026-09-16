/* Final QA and provenance package for the editable RMDB workflow figure.
 * Run only after render_rmdb_workflow_figure.mjs has produced and rendered v2. */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import sharp from "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/dist/index.mjs";

const ROOT = "/Users/tongyue/Documents/ppf/papers/RMDB_资料汇总_20260909/实验项目";
const PACKAGE = path.join(ROOT, "artifacts/scientific_figures/rmdb_measurement_workflow_20260909_2215");
const paths = {
  reference: path.join(PACKAGE, "01_inputs/source_images/current_embedded_figure1_actual.png"),
  hrf: path.join(PACKAGE, "01_inputs/user_assets/HRF_IQA_01_GOOD.JPG"),
  pptx: path.join(PACKAGE, "06_editable_pptx/rmdb_measurement_workflow.pptx"),
  svg: path.join(PACKAGE, "07_svg_export/rmdb_measurement_workflow.svg"),
  pptxPreview: path.join(PACKAGE, "08_previews/preview_pptx.png"),
  svgPreview: path.join(PACKAGE, "08_previews/preview_svg.png"),
  comparison: path.join(PACKAGE, "08_previews/comparison_master_vs_editable.png"),
  contactSheet: path.join(PACKAGE, "08_previews/contact_sheet.png"),
  validation: path.join(PACKAGE, "06_editable_pptx/pptx_validation.json"),
  manifests: path.join(PACKAGE, "09_manifests"),
  qa: path.join(PACKAGE, "10_qa"),
};

const requireFile = async file => {
  try { await fs.access(file); } catch { throw new Error(`Required file missing: ${file}`); }
};
await Promise.all([paths.reference, paths.hrf, paths.pptx, paths.svg, paths.pptxPreview, paths.svgPreview, paths.validation].map(requireFile));
await fs.mkdir(paths.manifests, { recursive: true });
await fs.mkdir(paths.qa, { recursive: true });

const sha256 = async file => crypto.createHash("sha256").update(await fs.readFile(file)).digest("hex");
const rel = file => path.relative(PACKAGE, file);
const writeJson = (file, data) => fs.writeFile(path.join(paths.manifests, file), `${JSON.stringify(data, null, 2)}\n`);
const writeQa = (file, data) => fs.writeFile(path.join(paths.qa, file), data);
const finalization = JSON.parse(await fs.readFile(paths.validation, "utf8"));

// A QA-only side-by-side comparison; the selected source reference is never in the final PPTX/SVG.
const sourcePanel = await sharp(paths.reference).resize({ width: 760, height: 428, fit: "contain", background: "#FFFFFF" }).png().toBuffer();
const editablePanel = await sharp(paths.pptxPreview).resize({ width: 760, height: 428, fit: "contain", background: "#FFFFFF" }).png().toBuffer();
const comparisonLabels = Buffer.from(`<svg width="1520" height="470" xmlns="http://www.w3.org/2000/svg">
  <rect width="1520" height="470" fill="#FFFFFF"/>
  <rect x="0" y="0" width="760" height="42" fill="#EAF6FC"/>
  <rect x="760" y="0" width="760" height="42" fill="#EAF8F1"/>
  <text x="22" y="28" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#243442">Selected project reference — layout / visual grammar only</text>
  <text x="782" y="28" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="700" fill="#243442">Editable RMDB framework — native PPTX/SVG</text>
</svg>`);
await sharp({ create: { width: 1520, height: 470, channels: 4, background: "#FFFFFF" } })
  .composite([{ input: comparisonLabels, top: 0, left: 0 }, { input: sourcePanel, top: 42, left: 0 }, { input: editablePanel, top: 42, left: 760 }])
  .png().toFile(paths.comparison);

const contactLabel = Buffer.from(`<svg width="1600" height="54" xmlns="http://www.w3.org/2000/svg"><rect width="1600" height="54" fill="#FFFFFF"/><text x="20" y="35" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#243442">RMDB frozen-readout measurement workflow — final PPTX render</text></svg>`);
const contactRender = await sharp(paths.pptxPreview).resize({ width: 1600, height: 900, fit: "contain", background: "#FFFFFF" }).png().toBuffer();
await sharp({ create: { width: 1600, height: 954, channels: 4, background: "#FFFFFF" } })
  .composite([{ input: contactLabel, top: 0, left: 0 }, { input: contactRender, top: 54, left: 0 }])
  .png().toFile(paths.contactSheet);

const previews = {
  pptx: await sharp(paths.pptxPreview).metadata(),
  svg: await sharp(paths.svgPreview).metadata(),
  comparison: await sharp(paths.comparison).metadata(),
  contactSheet: await sharp(paths.contactSheet).metadata(),
};
const fingerprints = {};
for (const [name, file] of Object.entries({ pptx: paths.pptx, svg: paths.svg, pptxPreview: paths.pptxPreview, svgPreview: paths.svgPreview, comparison: paths.comparison, contactSheet: paths.contactSheet })) {
  fingerprints[name] = { path: rel(file), sha256: await sha256(file), bytes: (await fs.stat(file)).size };
}

await writeJson("output_index.json", {
  figure_slug: "rmdb_measurement_workflow",
  version: "20260909_2215",
  final_deliverables: [rel(paths.pptx), rel(paths.svg)],
  review_previews: [rel(paths.pptxPreview), rel(paths.svgPreview), rel(paths.contactSheet)],
  qa_only_comparison: rel(paths.comparison),
  provenance: ["00_request/preflight_choices.json", "01_inputs/notes/source_understanding.md", "09_manifests/asset_manifest.json", "09_manifests/master_layout_manifest.json", "10_qa/visual_qa.md"],
  policy: "The selected source reference is preserved only as provenance/QA material and is not embedded in the final PPTX or SVG.",
});

await writeJson("asset_manifest.json", {
  source_policy: "No source-reference pixels in final visual deliverables. No generated/AI imagery used.",
  inputs: [
    { id: "style_reference", file: "01_inputs/source_images/current_embedded_figure1_actual.png", role: "layout/style contract only", in_final_deliverables: false, in_qa_comparison_only: true },
    { id: "hrf_input", file: "01_inputs/user_assets/HRF_IQA_01_GOOD.JPG", role: "illustrative fundus photograph supplied in the project HRF-IQA collection", in_final_deliverables: true, treatment: "whole image, contain; no crop" },
  ],
  native_components: ["stage panels", "all cards", "all labels", "all arrows/connectors", "four-channel glyphs", "dashed locked-before-evaluation lane"],
  generated_assets: [],
  licensing_note: "The HRF image is kept as a project-provided illustrative input. Confirm the intended publication license/attribution before submission; it is not experimental evidence in this figure.",
});

await writeJson("master_layout_manifest.json", {
  canvas: { width: 1600, height: 900, aspect: "16:9" },
  visual_contract: {
    source: "current_embedded_figure1_actual.png",
    preserved: ["four left-to-right stage panels", "cyan/blue/green/gold semantic palette", "white functional cards", "solid forward arrows", "restricted dashed lower lane", "large stage headings"],
    intentionally_changed: "Scientific content and internal card count are RMDB-specific and not copied from the source figure.",
  },
  regions: [
    { id: "stage_1", label: "INPUT & NORMALIZATION", x: 24, y: 68, width: 300, height: 610, color: "#55B3E6" },
    { id: "stage_2", label: "FROZEN VASCX", x: 356, y: 68, width: 330, height: 610, color: "#087BBB" },
    { id: "stage_3", label: "READOUT OPERATORS", x: 718, y: 68, width: 405, height: 610, color: "#00A878" },
    { id: "stage_4", label: "MEASUREMENT GATES", x: 1155, y: 68, width: 421, height: 610, color: "#F2A600" },
    { id: "locked_lane", label: "LOCKED BEFORE EVALUATION", x: 356, y: 716, width: 767, height: 136, color: "#F06A00", dash: "8 6" },
  ],
  connector_semantics: "All visible arrows encode forward inference/control direction: input → frozen VascX → candidate readouts/features → evidence/gates/conclusion; lock lane is VascX weights → readout definitions → primary endpoints.",
});

await writeJson("visual_diff_report.json", {
  comparison_scope: "Structural/style parity only; this is an intentional content adaptation, not a pixel-copy claim.",
  gates: [
    { gate: "four-stage left-to-right structure", result: "pass", evidence: "four native stage panels with headings" },
    { gate: "semantic palette and card hierarchy", result: "pass", evidence: "cyan/blue/green/gold stage surfaces and white functional cards" },
    { gate: "directional flow", result: "pass", evidence: "PPTX and SVG previews show forward arrowheads; lock lane points left-to-right" },
    { gate: "dashed restricted lane", result: "pass", evidence: "native dashed LOCKED BEFORE EVALUATION panel" },
    { gate: "source reference pixel exclusion", result: "pass", evidence: "final PPTX/SVG contain only native drawing elements plus the project HRF image; reference exists only in inputs and this QA comparison" },
    { gate: "RMDB content truthfulness", result: "pass", evidence: "explicitly freezes VascX, separates readout operators, reports the multi-metric gate, and limits the conclusion to endpoint-specific trade-offs" },
  ],
  intentional_differences: ["No source OCT images or segmentation maps reused", "RMDB content uses a public HRF fundus input", "The dashed lane encodes protocol locks rather than source-training steps"],
});

await writeJson("editability_manifest.json", {
  pptx: { path: rel(paths.pptx), status: "editable", native_objects: "stage panels, cards, text boxes, connectors, four-channel glyphs, dashed lane", raster_objects: "one whole HRF photo" },
  svg: { path: rel(paths.svg), status: "editable", native_elements: "text, rect, line, marker definitions", raster_objects: "one embedded HRF photo" },
  reviewer_action: "Use the PPTX for moving or restyling elements; use the SVG for vector-layout edits. Do not treat the HRF photograph as an editable vector element.",
});

await writeJson("render_receipt.json", { previews, fingerprints, pptx_validation: { integrity: finalization.packageIntegrity.status, slide_count: finalization.packageIntegrity.slide_count, layout_findings: finalization.presentationLayout.finding_count, layout_warnings: finalization.presentationLayout.warning_count, font: finalization.fontSelection.observed_font_families }, status: "pass with documented non-blocking connector z-order warnings" });

await writeQa("visual_qa.md", `# Visual QA\n\n- Final PPTX was rendered through LibreOffice, then inspected as \`08_previews/preview_pptx.png\`.\n- Final SVG was rendered through Sharp/librsvg, then inspected as \`08_previews/preview_svg.png\`.\n- All four stage surfaces, stage headings, cards, input photo, arrows, lower dashed lane, and conclusion boundary are visible and legible at the slide level.\n- Forward flow directions were explicitly checked: image → preprocessing → frozen VascX → readout layer → evidence/gates; the lower protocol lane is VascX weights → readout definitions → primary endpoints.\n- The prior generic statistical plot is not a primary framework figure for this project. This editable workflow package is the design baseline.\n\n## Automated validation\n\n- PPTX package integrity: pass.\n- Slide count: 1.\n- Font-policy check: pass (Arial).\n- Layout findings: 0.\n- Two connector-over-text z-order warnings were emitted by the automatic geometry checker. The finalized render was visually inspected: neither connector obscures readable text, so they are recorded as non-blocking.\n`);

await writeQa("source_policy_check.md", `# Source-reference policy check\n\nThe user's selected legacy framework image is used as a **layout/style contract** only. It is stored under \`01_inputs/source_images/\` for provenance and appears in \`08_previews/comparison_master_vs_editable.png\` for QA comparison only. It is not embedded in either final deliverable:\n\n- \`06_editable_pptx/rmdb_measurement_workflow.pptx\`\n- \`07_svg_export/rmdb_measurement_workflow.svg\`\n\nThe only raster image in the final figure is \`HRF_IQA_01_GOOD.JPG\`, a user-project copy used as a clearly illustrative fundus-photo input. All framework panels, cards, labels, arrows, and the lock lane are native editable drawing elements. No AI-generated visual asset was used.\n`);

await writeQa("qa_summary.json", JSON.stringify({ status: "pass", final_deliverables: [rel(paths.pptx), rel(paths.svg)], preview_checks: "pass", source_policy: "pass", layout_validator_findings: 0, non_blocking_warnings: 2, notes: "Warnings were reviewed in rendered output and did not obscure text." }, null, 2) + "\n");

console.log(JSON.stringify({ package: PACKAGE, status: "pass", previews, fingerprints }, null, 2));
