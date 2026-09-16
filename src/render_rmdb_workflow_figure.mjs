/* Build an editable RMDB workflow figure from the user's selected style reference.
 * The reference controls layout and visual grammar only. No reference pixels are
 * inserted into the deliverables. */
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "/Users/tongyue/Documents/ppf/papers/RMDB_资料汇总_20260909/实验项目";
const SKILL_DIR = "/Users/tongyue/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const RUNTIME_PYTHON = "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const PACKAGE = path.join(ROOT, "artifacts/scientific_figures/rmdb_measurement_workflow_20260909_2215");
const BUILD = path.join(PACKAGE, "_archive/intermediate_build");
const FINAL_PPTX = path.join(PACKAGE, "06_editable_pptx/rmdb_measurement_workflow.pptx");
const FINAL_SVG = path.join(PACKAGE, "07_svg_export/rmdb_measurement_workflow.svg");
const STYLE_REFERENCE = "/Users/tongyue/Documents/ppf/papers/current_embedded_figure1_actual.png";
const HRF_SAMPLE = path.join(ROOT, "data/raw/HRF_IQA_20260909/images/1_good.JPG");

const C = {
  ink: "#243442", muted: "#6B7A8F", white: "#FFFFFF", line: "#AAB7C4",
  blue: "#55B3E6", blueDark: "#087BBB", blueFill: "#EAF6FC",
  green: "#00A878", greenFill: "#EAF8F1", gold: "#F2A600", goldFill: "#FFF4D8",
  orange: "#F06A00", grayFill: "#F4F6F8", magenta: "#D96FA6",
  red: "#D94444", vein: "#2B79C2",
};

if (await fs.stat(PACKAGE).catch(() => null)) throw new Error(`Refusing to overwrite package: ${PACKAGE}`);

for (const folder of [
  "00_request", "01_inputs/source_images", "01_inputs/user_assets", "01_inputs/notes",
  "02_image2_master/master_raw", "02_image2_master/master_selected", "02_image2_master/rejected_or_buggy",
  "03_assets_raw/generated_single_assets", "04_assets_cutout/transparent_png", "04_assets_cutout/contact_sheets",
  "05_assets_vector/svg_wrappers", "05_assets_vector/vectorized_candidates", "05_assets_vector/vectorization_comparison",
  "06_editable_pptx", "07_svg_export", "08_previews", "09_manifests", "10_qa", "_archive/intermediate_build",
]) await fs.mkdir(path.join(PACKAGE, folder), { recursive: true });

const write = (relative, text) => fs.writeFile(path.join(PACKAGE, relative), text, "utf8");
const json = (relative, value) => write(relative, `${JSON.stringify(value, null, 2)}\n`);
const copy = (source, relative) => fs.copyFile(source, path.join(PACKAGE, relative));
const sha256 = async source => {
  const crypto = await import("node:crypto");
  return crypto.createHash("sha256").update(await fs.readFile(source)).digest("hex");
};

await Promise.all([
  copy(STYLE_REFERENCE, "01_inputs/source_images/current_embedded_figure1_actual.png"),
  copy(HRF_SAMPLE, "01_inputs/user_assets/HRF_IQA_01_GOOD.JPG"),
]);

await write("00_request/user_prompt.md", "用户要求：参照项目中既有论文的高质量框架图，先建立 RMDB 的框架图视觉基准；不要沿用通用统计图风格。\n");
await json("00_request/preflight_choices.json", {
  mode: "B: image-to-editable reference-guided reconstruction",
  purpose: "manuscript pre-writing figure package",
  figure_type: "method/workflow diagram",
  language: "English",
  palette: "custom, matched to selected reference: blue, green, gold stage panels",
  target_standard: "Nature-like manuscript figure using the user's selected project reference",
  editability: "native editable PPTX/SVG framework; one movable public HRF input photograph",
  output_root: PACKAGE,
  figure_slug: "rmdb_measurement_workflow",
  source_constraint: "the reference image is a layout/style contract only; no source-reference pixels may enter final visual assets",
  approved_deviation: "content and internal card count are changed to truthfully represent RMDB; the four-stage layout, color hierarchy, solid-flow arrows, dashed lower lane, cards, and output logic are preserved",
});
await json("00_request/mode.json", { mode: "B", selected_reference: STYLE_REFERENCE });
await write("01_inputs/notes/source_understanding.md", `# Selected reference understanding

Reference: \`current_embedded_figure1_actual.png\`.

- Four left-to-right stage panels with large headings above each panel.
- Semantic palette: cyan input, blue shared model, green method core, gold output/gates.
- Each stage uses a large pale rounded panel containing a small number of white functional cards.
- Solid horizontal arrows show inference flow. A dashed lower lane isolates a restricted training/control pathway.
- The source image contains a real OCT image and segmentation maps. Those are not reused. RMDB instead uses one public HRF fundus image supplied by the project as an illustrative input with no crop.
- The RMDB figure preserves the visual grammar but changes all scientific content to the frozen VascX/readout/measurement-gate workflow.
`);

const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const font = resolvePresentationFont({ fontFamily: "Arial" });
const presentation = Presentation.create({ slideSize: { width: 1600, height: 900 } });
const slide = presentation.slides.add();
slide.background.fill = C.white;

const shape = (geometry, position, fill, line = { style: "solid", fill: C.line, width: 1 }, radius = undefined) => {
  const item = slide.shapes.add({ geometry, position, fill, line, ...(radius ? { borderRadius: radius } : {}) });
  return item;
};
const text = (value, position, options = {}) => {
  const item = shape("textbox", position, "none", { fill: "none", width: 0 });
  item.text = value;
  item.text.style = {
    typeface: font, fontSize: options.size ?? 18, bold: options.bold ?? false,
    color: options.color ?? C.ink, autoFit: "shrinkText",
    ...(options.italic ? { italic: true } : {}),
  };
  return item;
};
const card = (label, position, accent, opts = {}) => {
  const item = shape("roundRect", position, opts.fill ?? C.white, { style: "solid", fill: accent, width: opts.width ?? 2 }, 14);
  item.text = label;
  item.text.style = { typeface: font, fontSize: opts.size ?? 17, bold: opts.bold ?? false, color: opts.color ?? C.ink, autoFit: "shrinkText" };
  return item;
};
const flow = (from, to, color = C.ink, dashed = false, sides = ["right", "left"]) => {
  const connector = slide.shapes.connect(from, to, {
    kind: "straight", fromSide: sides[0], toSide: sides[1],
    line: { style: dashed ? "dashed" : "solid", fill: color, width: dashed ? 2 : 2.5 },
    // In the saved PPTX connector schema, PowerPoint renders this endpoint as
    // the forward arrowhead.  (Using `head` reverses the visible flow.)
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
  connector.bringToFront();
  return connector;
};

// Stage titles and surfaces follow the selected source's four-column grammar.
const stages = [
  { x: 24, w: 300, heading: "INPUT & NORMALIZATION", fill: C.blueFill, stroke: C.blue, label: "STAGE 1" },
  { x: 356, w: 330, heading: "FROZEN VASCX", fill: "#EAF4FB", stroke: C.blueDark, label: "STAGE 2" },
  { x: 718, w: 405, heading: "READOUT OPERATORS", fill: C.greenFill, stroke: C.green, label: "STAGE 3" },
  { x: 1155, w: 421, heading: "MEASUREMENT GATES", fill: C.goldFill, stroke: C.gold, label: "STAGE 4" },
];
for (const stage of stages) {
  text(stage.heading, { left: stage.x, top: 22, width: stage.w, height: 32 }, { size: 21, bold: true });
  shape("roundRect", { left: stage.x, top: 68, width: stage.w, height: 610 }, stage.fill, { style: "solid", fill: stage.stroke, width: 2.2 }, 22);
  text(stage.label, { left: stage.x + 22, top: 94, width: stage.w - 44, height: 30 }, { size: 20, bold: true, color: stage.stroke });
}

// Stage 1: a public project input image, rendered whole (contain), never cropped.
const sampleBytes = await fs.readFile(HRF_SAMPLE);
slide.images.add({ blob: sampleBytes, contentType: "image/jpeg", alt: "Public HRF IQA fundus input used as illustrative workflow example", fit: "contain", geometry: "roundRect", borderRadius: 12, position: { left: 62, top: 145, width: 224, height: 180 } });
const sampleFrame = shape("roundRect", { left: 62, top: 145, width: 224, height: 180 }, "none", { style: "solid", fill: C.ink, width: 2 }, 12);
const preprocess = card("VascX preprocessing\ncanonical 1024 px", { left: 60, top: 380, width: 228, height: 96 }, C.blue, { size: 18, bold: true });
text("Fundus photograph", { left: 70, top: 330, width: 208, height: 28 }, { size: 16, bold: true, color: C.ink });
const onlyImage = card("Inference: image only", { left: 62, top: 545, width: 224, height: 46 }, C.gold, { size: 16, bold: true, fill: "#FFF9EB" });
flow(sampleFrame, preprocess, C.blue, false, ["bottom", "top"]);

// Stage 2: VascX stays visibly frozen, with a native probability tensor glyph.
const vascx = card("Frozen VascX\nA/V probability map", { left: 390, top: 210, width: 260, height: 108 }, C.blueDark, { size: 20, bold: true });
for (let i = 0; i < 4; i += 1) {
  shape("roundRect", { left: 423 + i * 43, top: 334, width: 31, height: 31 }, [C.grayFill, C.red, C.vein, C.magenta][i], { style: "solid", fill: C.ink, width: 1 }, 5);
}
text("four output channels", { left: 412, top: 373, width: 216, height: 24 }, { size: 14, color: C.muted });
const raw = card("Raw VascX readout", { left: 401, top: 440, width: 238, height: 60 }, C.blueDark, { size: 18, bold: true });
const frozen = card("No VascX fine-tuning", { left: 402, top: 545, width: 236, height: 48 }, C.line, { size: 16, fill: C.grayFill, color: C.muted });
flow(preprocess, vascx, C.blueDark);
flow(vascx, raw, C.blueDark, false, ["bottom", "top"]);

// Stage 3: the readout layer, using native editable cards rather than image text.
const readouts = card("Candidate readouts", { left: 752, top: 148, width: 230, height: 58 }, C.green, { size: 19, bold: true });
const rawRow = card("raw VascX", { left: 768, top: 234, width: 200, height: 42 }, C.blueDark, { size: 16, bold: true });
const massRow = card("mass correction", { left: 768, top: 288, width: 200, height: 42 }, C.orange, { size: 16, bold: true });
const jointRow = card("joint occupancy + A/V identity", { left: 768, top: 342, width: 200, height: 42 }, C.green, { size: 15, bold: true });
const controlsRow = card("fixed smooth / cleanup controls", { left: 768, top: 396, width: 200, height: 42 }, C.magenta, { size: 14, bold: true });
const features = card("Feature readouts\nbranch density\nendpoint density\noccupancy entropy", { left: 986, top: 246, width: 120, height: 198 }, C.green, { size: 13, bold: true, fill: "#F7FFFB" });
flow(raw, readouts, C.green);
flow(readouts, rawRow, C.green, false, ["bottom", "top"]);
flow(rawRow, features, C.green);

// Stage 4: paired gate and claim boundary intentionally show evidence, not a universal win.
const cohorts = card("Evidence strata\nHRF quality stress\nDigital resolution sensitivity\nHistorical mBRSET control", { left: 1193, top: 144, width: 346, height: 136 }, C.gold, { size: 17, bold: true });
const gate = card("Paired gate\nICC + MAD + SEM + LoA\npatient/pair bootstrap", { left: 1193, top: 335, width: 346, height: 104 }, C.gold, { size: 18, bold: true });
const allowed = card("Permitted conclusion\nExposes endpoint-specific trade-offs", { left: 1193, top: 495, width: 346, height: 82 }, C.green, { size: 18, bold: true, fill: "#F7FFFB" });
const boundary = card("No VascX replacement\nNo cross-camera clinical claim", { left: 1193, top: 598, width: 346, height: 48 }, C.line, { size: 15, fill: C.grayFill, color: C.muted });
flow(features, cohorts, C.green);
flow(cohorts, gate, C.gold, false, ["bottom", "top"]);
flow(gate, allowed, C.gold, false, ["bottom", "top"]);

// Lower dashed lane preserves the reference's restricted pathway, here used for locks rather than training.
const lockLane = shape("roundRect", { left: 356, top: 716, width: 767, height: 136 }, "#FFFCF5", { style: "dashed", fill: C.orange, width: 2 }, 16);
text("LOCKED BEFORE EVALUATION", { left: 382, top: 738, width: 300, height: 26 }, { size: 17, bold: true, color: C.orange });
const weights = card("VascX weights", { left: 394, top: 782, width: 166, height: 46 }, C.orange, { size: 16, fill: C.white });
const definitions = card("Readout definitions", { left: 653, top: 782, width: 186, height: 46 }, C.orange, { size: 16, fill: C.white });
const endpoints = card("Primary endpoints", { left: 930, top: 782, width: 155, height: 46 }, C.orange, { size: 16, fill: C.white });
flow(weights, definitions, C.orange);
flow(definitions, endpoints, C.orange);
text("Solid: inference flow   Dashed: pre-evaluation lock", { left: 1185, top: 858, width: 370, height: 24 }, { size: 14, color: C.muted });
slide.speakerNotes.textFrame.setText("Source content: frozen RMDB evidence ledger and protocol. Input photograph: public HRF IQA allQuality data. Style/layout reference: user's current_embedded_figure1_actual.png; no reference pixels used in final figure.");

// Native SVG mirror. The SVG contains editable text, boxes, and arrows; only the project HRF photo is embedded raster data.
const b64 = sampleBytes.toString("base64");
const esc = value => String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const svgParts = [
  `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1600" height="900" viewBox="0 0 1600 900">`,
  `<defs><marker id="arrowInk" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="${C.ink}"/></marker><marker id="arrowBlue" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="${C.blueDark}"/></marker><marker id="arrowGreen" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="${C.green}"/></marker><marker id="arrowGold" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="${C.gold}"/></marker><marker id="arrowOrange" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="${C.orange}"/></marker></defs><rect width="1600" height="900" fill="#fff"/>`,
];
const sRect = (x, y, w, h, fill, stroke, radius = 14, dash = "") => svgParts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`);
const sText = (value, x, y, size = 18, color = C.ink, bold = false, anchor = "start") => { const lines = String(value).split("\n"); svgParts.push(`<text x="${x}" y="${y}" fill="${color}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" font-weight="${bold ? 700 : 400}" text-anchor="${anchor}">${lines.map((line, i) => `<tspan x="${x}" dy="${i ? size * 1.22 : 0}">${esc(line)}</tspan>`).join("")}</text>`); };
const sArrow = (x1, y1, x2, y2, color, marker, dash = "") => svgParts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2.5" marker-end="url(#${marker})" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`);
for (const stage of stages) { sText(stage.heading, stage.x, 47, 21, C.ink, true); sRect(stage.x, 68, stage.w, 610, stage.fill, stage.stroke, 22); sText(stage.label, stage.x + 22, 116, 20, stage.stroke, true); }
svgParts.push(`<image x="62" y="145" width="224" height="180" preserveAspectRatio="xMidYMid meet" xlink:href="data:image/jpeg;base64,${b64}"/><rect x="62" y="145" width="224" height="180" rx="12" fill="none" stroke="${C.ink}" stroke-width="2"/>`);
sText("Fundus photograph", 70, 353, 16, C.ink, true); sRect(60, 380, 228, 96, C.white, C.blue); sText("VascX preprocessing\ncanonical 1024 px", 174, 417, 18, C.ink, true, "middle"); sRect(62, 545, 224, 46, "#FFF9EB", C.gold); sText("Inference: image only", 174, 574, 16, C.ink, true, "middle"); sArrow(174, 325, 174, 380, C.blue, "arrowBlue");
sRect(390, 210, 260, 108, C.white, C.blueDark); sText("Frozen VascX\nA/V probability map", 520, 248, 20, C.ink, true, "middle"); [C.grayFill, C.red, C.vein, C.magenta].forEach((fill, i) => sRect(423 + i * 43, 334, 31, 31, fill, C.ink, 5)); sText("four output channels", 520, 387, 14, C.muted, false, "middle"); sRect(401, 440, 238, 60, C.white, C.blueDark); sText("Raw VascX readout", 520, 477, 18, C.ink, true, "middle"); sRect(402, 545, 236, 48, C.grayFill, C.line); sText("No VascX fine-tuning", 520, 575, 16, C.muted, false, "middle"); sArrow(288, 428, 390, 264, C.blueDark, "arrowBlue"); sArrow(520, 318, 520, 440, C.blueDark, "arrowBlue");
sRect(752, 148, 230, 58, C.white, C.green); sText("Candidate readouts", 867, 184, 19, C.ink, true, "middle"); [["raw VascX",C.blueDark,16],["mass correction",C.orange,16],["joint occupancy + A/V identity",C.green,15],["fixed smooth / cleanup controls",C.magenta,14]].forEach(([label, stroke, size], i) => { sRect(768, 234 + i * 54, 200, 42, C.white, stroke); sText(label, 868, 260 + i * 54, size, C.ink, true, "middle"); }); sRect(986, 246, 120, 198, "#F7FFFB", C.green); sText("Feature readouts\nbranch density\nendpoint density\noccupancy entropy", 1046, 281, 13, C.ink, true, "middle"); sArrow(639, 470, 752, 177, C.green, "arrowGreen"); sArrow(867, 206, 867, 234, C.green, "arrowGreen"); sArrow(968, 255, 986, 345, C.green, "arrowGreen");
sRect(1193, 144, 346, 136, C.white, C.gold); sText("Evidence strata\nHRF quality stress\nDigital resolution sensitivity\nHistorical mBRSET control", 1217, 179, 17, C.ink, true); sRect(1193, 335, 346, 104, C.white, C.gold); sText("Paired gate\nICC + MAD + SEM + LoA\npatient/pair bootstrap", 1217, 368, 18, C.ink, true); sRect(1193, 495, 346, 82, "#F7FFFB", C.green); sText("Permitted conclusion\nExposes endpoint-specific trade-offs", 1217, 529, 18, C.ink, true); sRect(1193, 598, 346, 48, C.grayFill, C.line); sText("No VascX replacement\nNo cross-camera clinical claim", 1217, 619, 15, C.muted, false); sArrow(1094, 345, 1193, 212, C.green, "arrowGreen"); sArrow(1366, 280, 1366, 335, C.gold, "arrowGold"); sArrow(1366, 439, 1366, 495, C.gold, "arrowGold");
sRect(356, 716, 767, 136, "#FFFCF5", C.orange, 16, "8 6"); sText("LOCKED BEFORE EVALUATION", 382, 759, 17, C.orange, true); [[394,166,"VascX weights"],[653,186,"Readout definitions"],[930,155,"Primary endpoints"]].forEach(([x,w,label])=>{sRect(x,782,w,46,C.white,C.orange);sText(label,x+w/2,811,16,C.ink,false,"middle")}); sArrow(560,805,653,805,C.orange,"arrowOrange");sArrow(839,805,930,805,C.orange,"arrowOrange");sText("Solid: inference flow   Dashed: pre-evaluation lock",1185,875,14,C.muted,false);
svgParts.push("</svg>");
await fs.writeFile(FINAL_SVG, svgParts.join(""), "utf8");

const candidate = path.join(BUILD, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const requirements = { explicitTotalSlideCount: 1, requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [] };
const finalization = await finalizePresentation({
  ...requirements,
  workspaceDir: ROOT,
  candidatePath: candidate,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "15240000,8572500", "--validate-bullet-geometry", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, "finalization.validation.json"),
});
await json("06_editable_pptx/pptx_validation.json", finalization);

const quickPreview = await presentation.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(path.join(PACKAGE, "08_previews/preview_pptx_fallback.png"), new Uint8Array(await quickPreview.arrayBuffer()));
await write("07_svg_export/svg_render_notes.md", "The SVG is native text, rectangle, and arrow geometry plus one embedded public HRF input photograph. It contains no source-reference pixels.\n");
console.log(JSON.stringify({ package: PACKAGE, pptx: FINAL_PPTX, svg: FINAL_SVG, font, validation: finalization }, null, 2));
