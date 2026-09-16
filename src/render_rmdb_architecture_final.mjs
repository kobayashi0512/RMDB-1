/* Build the complete, editable RMDB/BiRAP architecture figure in the
 * user-approved 2019 RMB 10-inspired pale-blue and champagne-gold system. */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "/Users/tongyue/Documents/ppf/papers/RMDB_资料汇总_20260909/实验项目";
const SKILL_DIR = "/Users/tongyue/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const RUNTIME_PYTHON = "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const PACKAGE = path.join(ROOT, "artifacts/scientific_figures/rmdb_birap_architecture_final_20260914_v5");
const PREVIOUS_PACKAGE = path.join(ROOT, "artifacts/scientific_figures/rmdb_birap_architecture_rmb10_20260914_1018");
const BUILD = path.join(PACKAGE, "_archive/intermediate_build");
const FINAL_PPTX = path.join(PACKAGE, "06_editable_pptx/RMDB_Architecture_Final_v6.pptx");
const FINAL_SVG = path.join(PACKAGE, "07_svg_export/RMDB_Architecture_Final.svg");
const STYLE_REFERENCE = path.join(PREVIOUS_PACKAGE, "08_previews/preview_pptx.png");
const ASSET_DIR = path.join(PACKAGE, "03_assets_raw/generated_single_assets");
const USER_ASSET_DIR = path.join(PACKAGE, "01_inputs/user_assets");

const C = {
  paper: "#FAF8F0", ink: "#23485C", ink2: "#315D70", muted: "#6E838C", white: "#FFFFFF", line: "#AFC4CB",
  blue: "#B9D5DE", blueDark: "#3F7892", blueFill: "#EDF5F6", blueFill2: "#DCECF0",
  green: "#57958F", greenDark: "#3E746F", greenFill: "#E7F0EE", mint: "#F3F7F5",
  gold: "#D2B468", goldDark: "#9A7B34", goldFill: "#F4E9CC", orange: "#A8873B", orangeFill: "#F8F1E1",
  magenta: "#8FA8B1", magentaFill: "#EEF3F4", grayFill: "#F3F5F2", grayBlue: "#E8EFF1",
  red: "#A85D55", vein: "#3F7892", crossing: "#8C7285", teal: "#57958F", navy: "#183948",
};

if (await fs.stat(FINAL_PPTX).catch(() => null)) throw new Error(`Refusing to overwrite final figure: ${FINAL_PPTX}`);
for (const folder of [
  "00_request", "01_inputs/source_images", "01_inputs/user_assets", "01_inputs/notes",
  "02_image2_master/master_raw", "02_image2_master/master_selected", "02_image2_master/rejected_or_buggy",
  "03_assets_raw/generated_single_assets", "04_assets_cutout/transparent_png", "04_assets_cutout/contact_sheets",
  "05_assets_vector/svg_wrappers", "05_assets_vector/vectorized_candidates", "05_assets_vector/vectorization_comparison",
  "06_editable_pptx", "07_svg_export", "08_previews", "09_manifests", "10_qa", "_archive/intermediate_build",
]) await fs.mkdir(path.join(PACKAGE, folder), { recursive: true });

const write = (relative, value) => fs.writeFile(path.join(PACKAGE, relative), value, "utf8");
const json = (relative, value) => write(relative, `${JSON.stringify(value, null, 2)}\n`);
await write("00_request/user_prompt.md", "用户确认以 2019 年版第五套人民币 10 元纸币的浅蓝、蓝灰和香槟金视觉体系为全文图表母版，并要求在保留完整模型、数据库原图和医学语义色的前提下重新优化 RMDB/BiRAP 总框架图。\n");
await json("00_request/preflight_choices.json", {
  mode: "B: user-authorized reference-guided redesign",
  purpose: "manuscript method/architecture figure before article drafting",
  figure_type: "model architecture plus measurement-evidence workflow",
  language: "English",
  palette: "custom RMB 10-inspired paper white, ice blue, steel blue, deep blue ink, champagne gold, restrained jade, and preserved clinical semantic colours",
  target_standard: "Nature-like biomedical AI architecture figure with a restrained 2019 RMB 10 visual language",
  editability: "native editable PPTX and SVG; data-derived map examples remain movable raster assets",
  output_root: PACKAGE,
  figure_slug: "rmdb_birap_architecture_rmb10",
  selected_reference: STYLE_REFERENCE,
  source_constraint: "the preceding complete RMDB figure is a structure/content reference only; no reference pixels are embedded in final deliverables",
  approved_deviation: "Restore the supplied detailed blue/gold mother layout. Scientific corrections only: sample counts, ratio, grid order, shared residual weights, exploratory-vs-future gate. Preserve all original image assets and four-column hierarchy.",
});
await json("00_request/mode.json", { mode: "B", reason: "user approved a visual-system redesign of the preceding complete editable figure" });
await write("01_inputs/notes/source_understanding.md", `# Source and scientific understanding

## Visual contract

- Four left-to-right stages, database originals, model-detail blocks, solid inference arrows, and a lower protocol/claim lane are preserved from the preceding complete version.
- The visual system uses warm paper white, ice blue, water blue, steel blue, deep blue ink, champagne gold, and restrained jade. Gold marks hierarchy and key decisions rather than occupying equal area with blue.
- The large stage surfaces use thin banknote-like double-line hierarchy and low-saturation fills. Dense decorative guilloche artwork is intentionally omitted because it would reduce scientific legibility.
- Each stage must contain an actual model or output object rather than generic prose.
- The preceding complete figure is stored for provenance/QA only and is not embedded in final outputs.

## Scientific contract

- Frozen VascX is a five-fold DynUNet ensemble receiving preprocessed RGB plus contrast-enhanced RGB (six channels), using 512-pixel sliding windows with 50% overlap, Gaussian blending, and flip TTA. It outputs background, artery, vein, and crossing logits/probabilities.
- The mass readout is a frozen two-iteration residual corrector. Its 11-channel features are RGB(3), clipped/scaled VascX logits(4), normalized four-class entropy(1), and Hessian tangent/vesselness(3). Each iteration concatenates current four-class probabilities, producing 15 input channels to a 24→48→96→48→24→4 U-block.
- The occupancy-conditioned joint readout uses a separately frozen FIVES U-Net/ResNet34 sigmoid occupancy probability and the ORIGINAL raw VascX conditional identity. It is post-hoc and not jointly trained.
- The fixed Gaussian smooth and small-component cleanup arms are evaluation controls, not core models.
- All readouts feed the same 12-feature measurement operator and are judged on separate structural-fidelity and same-eye-repeatability evidence axes.
`);

// The model-derived maps are unchanged from the immediately preceding validated
// version, so they are copied byte-for-byte. Only project-original database
// images and their layout are new in this revision.
const sha256 = async file => crypto.createHash("sha256").update(await fs.readFile(file)).digest("hex");
await fs.copyFile(STYLE_REFERENCE, path.join(PACKAGE, "01_inputs/source_images/previous_complete_architecture.png"));
for (const name of ["Fundus_AVSeg_001_G_RGB.png", "Fundus_AVSeg_001_G_CE.png"]) {
  await fs.copyFile(path.join(PREVIOUS_PACKAGE, "01_inputs/user_assets", name), path.join(USER_ASSET_DIR, name));
}
const originalSources = {
  "Fundus_AVSeg_001_G_original.png": path.join(ROOT, "data/raw/Fundus_AVSeg/Fundus-AVSeg/images/001_G.png"),
  "FIVES_train_100_A_original.png": path.join(ROOT, "data/raw/FIVES/train/Original/100_A.png"),
  "mBRSET_9_3_original.jpg": path.join(ROOT, "data/raw/mBRSET_Kaggle/mBRSET Data/0/9.3.jpg"),
  "HRF_IQA_1_good_original.JPG": path.join(ROOT, "data/raw/HRF_IQA_20260909/images/1_good.JPG"),
  "Fundus_AVSeg_002_N_original.png": path.join(ROOT, "data/raw/Fundus_AVSeg/Fundus-AVSeg/images/002_N.png"),
  "Fundus_AVSeg_002_N_manual_AV.png": path.join(ROOT, "data/raw/Fundus_AVSeg/Fundus-AVSeg/annotation/002_N.png"),
  "mBRSET_test_613_3_original.jpg": path.join(ROOT, "data/raw/mBRSET_Kaggle/mBRSET Data/0/613.3.jpg"),
  "mBRSET_test_613_4_original.jpg": path.join(ROOT, "data/raw/mBRSET_Kaggle/mBRSET Data/0/613.4.jpg"),
};
for (const [name, source] of Object.entries(originalSources)) await fs.copyFile(source, path.join(USER_ASSET_DIR, name));

const requiredAssets = [
  "fundus_rgb_512.png", "fundus_ce_512.png", "raw_probability_background.png", "raw_probability_artery.png",
  "raw_probability_vein.png", "raw_probability_crossing.png", "raw_av_map.png", "mass_av_map.png",
  "occupancy_probability.png", "joint_av_map.png", "smooth_control_map.png", "cleanup_control_map.png",
  "joint_topology_map.png", "asset_stats.json",
];
for (const name of requiredAssets) {
  await fs.copyFile(
    path.join(PREVIOUS_PACKAGE, "03_assets_raw/generated_single_assets", name),
    path.join(ASSET_DIR, name),
  );
}
const reusedStatsPath = path.join(ASSET_DIR, "asset_stats.json");
const reusedStats = JSON.parse(await fs.readFile(reusedStatsPath, "utf8"));
reusedStats.revision_note = "Model-derived assets and project originals copied byte-for-byte from validated package rmdb_birap_architecture_20260910_111506; only the user-authorized RMB 10-inspired palette, panel hierarchy, and visual layout changed.";
reusedStats.database_originals = await Promise.all(Object.entries(originalSources).map(async ([name, source]) => ({
  copied_as: name,
  source: path.relative(ROOT, source),
  sha256: await sha256(source),
  role: name.includes("002_N") || name.includes("613_") ? "evaluation-evidence example" : "input-domain example",
})));
await fs.writeFile(reusedStatsPath, `${JSON.stringify(reusedStats, null, 2)}\n`, "utf8");
for (const name of requiredAssets) await fs.access(path.join(ASSET_DIR, name));
const requiredOriginals = [
  "Fundus_AVSeg_001_G_original.png", "FIVES_train_100_A_original.png", "mBRSET_9_3_original.jpg",
  "HRF_IQA_1_good_original.JPG", "Fundus_AVSeg_002_N_original.png", "Fundus_AVSeg_002_N_manual_AV.png",
  "mBRSET_test_613_3_original.jpg", "mBRSET_test_613_4_original.jpg",
];
for (const name of requiredOriginals) await fs.access(path.join(USER_ASSET_DIR, name));

const visualSystem = {
  name: "RMDB RMB 10-inspired scientific figure system",
  version: "1.0",
  basis: "2019 fifth-series RMB 10 colour hierarchy translated for scientific figures; not an official banknote colour specification",
  colour_ratio: { paper_and_pale_blue: "60%", steel_and_deep_blue: "25%", champagne_gold: "10%", jade_and_alert_accents: "5% maximum" },
  palette: {
    paper_white: C.paper, ice_blue: C.blueFill, watermark_blue: C.blueFill2, mist_blue: C.blue,
    steel_blue: C.blueDark, deep_ink_blue: C.ink, champagne_gold: C.gold, pale_gold: C.goldFill,
    antique_gold: C.goldDark, jade: C.green, control_blue_gray: C.magenta, warning_red: C.red,
  },
  scientific_series: {
    raw_vascx: C.blueDark,
    mass_readout: C.gold,
    joint_readout: C.green,
    sensitivity_controls: C.magenta,
    failed_gate_or_warning: C.red,
    artery_semantic: "#A85D55",
    vein_semantic: "#3F7892",
    crossing_semantic: "#8C7285",
  },
  chart_rules: {
    canvas: C.paper,
    axes_and_titles: C.ink,
    gridlines: C.blueFill2,
    table_header: C.blueFill2,
    highlighted_table_row: C.goldFill,
    panel_outline: C.blueDark,
    key_result_outline: C.goldDark,
    note: "Retain natural retinal-image colours and clinical artery/vein/crossing semantics. Use line style or marker shape in addition to colour when series compete.",
  },
};
await json("09_manifests/rmdb_10yuan_visual_system.json", visualSystem);
await write("01_inputs/notes/global_figure_style_guide.md", `# RMDB RMB 10-inspired global figure style

This package is the master for later manuscript figures. Use warm paper white and pale blue as the dominant field, deep blue ink for text and primary flow, champagne gold for hierarchy or key decisions, jade for the joint readout, blue-gray for controls, and muted red only for failed gates or clinical artery semantics.

Do not recolour fundus photographs or probability maps. Preserve artery, vein, and crossing semantics. Use thin rules, generous pale fields, and restrained gold accents rather than equal blue/gold blocks.
`);
const stableStyleDir = path.join(ROOT, "artifacts/figure_style_system");
await fs.mkdir(stableStyleDir, { recursive: true });
await fs.writeFile(path.join(stableStyleDir, "rmdb_10yuan_style_v1.json"), `${JSON.stringify(visualSystem, null, 2)}\n`, "utf8");
await fs.writeFile(path.join(stableStyleDir, "README.md"), "# RMDB figure style system\n\n`rmdb_10yuan_style_v1.json` is the frozen visual-system source for the manuscript's remaining figures and tables.\n", "utf8");

// The layout/source manifests are frozen before slide assembly and then enriched by final QA.
await json("09_manifests/master_layout_manifest.json", {
  canvas: { width: 1920, height: 1080, aspect: "16:9" },
  visual_system: "09_manifests/rmdb_10yuan_visual_system.json",
  frame: { x: 8, y: 6, width: 1904, height: 1068, implementation_type: "native_framework_geometry", role: "thin deep-blue and champagne-gold double rule" },
  approved_deviations: ["User authorized redesign of the previous cyan/green/orange panel styling into a 2019 RMB 10-inspired light-blue, deep-blue-ink, and champagne-gold hierarchy."],
  regions: [
    { id: "stage_1", x: 24, y: 110, width: 246, height: 724, role: "four source-database originals plus preprocessing" },
    { id: "stage_2", x: 292, y: 110, width: 402, height: 724, role: "frozen VascX identity backbone and four probabilities" },
    { id: "stage_3", x: 716, y: 110, width: 670, height: 724, role: "mass and occupancy-conditioned frozen readouts" },
    { id: "stage_4", x: 1408, y: 110, width: 488, height: 724, role: "phenotypes, paired original-image evidence, prospective gate" },
    { id: "locked_lane", x: 292, y: 866, width: 1094, height: 172, role: "pre-evaluation lock" },
    { id: "claim_boundary", x: 1408, y: 866, width: 488, height: 172, role: "increment and claim boundary" },
  ],
  fineRegions: [
    { id: "source_fundus", implementation_type: "project_original_raster", placement: "stage_1" },
    { id: "source_fives", implementation_type: "project_original_raster", placement: "stage_1" },
    { id: "source_mbrset", implementation_type: "project_original_raster", placement: "stage_1" },
    { id: "source_hrf", implementation_type: "project_original_raster", placement: "stage_1" },
    { id: "fundus_eval_pair", implementation_type: "project_original_and_manual_annotation", placement: "stage_4_structural" },
    { id: "mbrset_repeat_pair", implementation_type: "two_project_original_rasters", placement: "stage_4_repeatability" },
    { id: "vascx_probability_channels", implementation_type: "data_derived_raster", placement: "stage_2" },
    { id: "readout_maps", implementation_type: "data_derived_raster", placement: "stage_3" },
    { id: "model_cards_and_formulas", implementation_type: "native_editable", placement: "stages_2_3" },
  ],
  routeSegments: [
    { from: "stage_1", to: "stage_2", direction: "right", status: "native connector" },
    { from: "stage_2", to: "stage_3", direction: "right", status: "native connector" },
    { from: "stage_3", to: "stage_4", direction: "right", status: "native arrow" },
    { from: "phenotypes", to: "evidence", direction: "down", status: "native arrow" },
    { from: "evidence", to: "gate", direction: "down", status: "native arrow" },
  ],
  legendItems: ["B/A/V/X classes", "topology colours", "solid inference flow", "dashed pre-evaluation lock"],
  smallIcons: [],
});
await json("09_manifests/asset_manifest.json", {
  status: "prebuild-frozen",
  policy: "database originals are embedded with contain fit; the preceding complete figure is QA-only and no reference pixels are embedded",
  visual_system: "09_manifests/rmdb_10yuan_visual_system.json",
  original_assets: requiredOriginals.map(name => `01_inputs/user_assets/${name}`),
  generated_assets: requiredAssets.filter(name => name !== "asset_stats.json").map(name => `03_assets_raw/generated_single_assets/${name}`),
});

const { resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const font = resolvePresentationFont({ fontFamily: "Arial" });
const presentation = Presentation.create({ slideSize: { width: 1920, height: 1080 } });
const slide = presentation.slides.add();
slide.background.fill = C.paper;

const svg = [
  `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1920" height="1080" viewBox="0 0 1920 1080">`,
  `<defs>`,
  ...Object.entries({ Ink: C.ink, Blue: C.blueDark, Green: C.green, Gold: C.gold, Orange: C.orange, Magenta: C.magenta }).map(([name, colour]) => `<marker id="arrow${name}" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="${colour}"/></marker>`),
  `</defs><rect width="1920" height="1080" fill="${C.paper}"/>`,
];
const esc = value => String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const markerFor = colour => colour === C.blueDark ? "Blue" : colour === C.green ? "Green" : colour === C.gold ? "Gold" : colour === C.orange ? "Orange" : colour === C.magenta ? "Magenta" : "Ink";

const rect = (x, y, w, h, fill, stroke = C.line, radius = 9, width = 1.4, dash = "") => {
  const item = slide.shapes.add({
    geometry: "roundRect", position: { left: x, top: y, width: w, height: h }, fill,
    line: { style: dash ? "dashed" : "solid", fill: stroke, width }, borderRadius: radius,
  });
  svg.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`);
  return item;
};
const line = (x1, y1, x2, y2, colour = C.ink, width = 2, arrow = false, dash = "") => {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const item = slide.shapes.add({
    geometry: "line", position: {
      left: Math.min(x1, x2), top: Math.min(y1, y2),
      width: Math.abs(dx), height: Math.abs(dy),
      horizontalFlip: dx < 0, verticalFlip: dy < 0,
    },
    line: { style: dash ? "dashed" : "solid", fill: colour, width },
  });
  if (arrow) {
    const vertical=Math.abs(dy)>Math.abs(dx);
    slide.shapes.add({geometry:vertical?(dy>0?"downArrow":"upArrow"):(dx>0?"rightArrow":"leftArrow"),position:{left:vertical?x2-4:(dx>0?x2-8:x2),top:vertical?(dy>0?y2-8:y2):y2-4,width:8,height:8},fill:colour,line:{fill:colour,width:0}});
  }
  svg.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${colour}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ""} ${arrow ? `marker-end="url(#arrow${markerFor(colour)})"` : ""}/>`);
  return item;
};
const text = (value, x, y, w, h, options = {}) => {
  options.size=Math.max(options.size??16,11.5);
  const item = slide.shapes.add({ geometry: "textbox", position: { left: x, top: y, width: w, height: h }, fill: "none", line: { fill: "none", width: 0 } });
  item.text = value;
  item.text.style = {
    typeface: font, fontSize: options.size ?? 16, bold: options.bold ?? false, color: options.color ?? C.ink,
    autoFit: "none", insets:{left:0,right:0,top:0,bottom:0},alignment:options.align??"left",verticalAlignment:"top", ...(options.italic ? { italic: true } : {}),
  };
  const anchor = options.align === "center" ? "middle" : options.align === "right" ? "end" : "start";
  const tx = anchor === "middle" ? x + w / 2 : anchor === "end" ? x + w : x;
  const lines = String(value).split("\n");
  svg.push(`<text x="${tx}" y="${y + (options.size ?? 16)}" fill="${options.color ?? C.ink}" font-family="Arial, Helvetica, sans-serif" font-size="${options.size ?? 16}" font-weight="${options.bold ? 700 : 400}" font-style="${options.italic ? "italic" : "normal"}" text-anchor="${anchor}">${lines.map((row, index) => `<tspan x="${tx}" dy="${index ? (options.size ?? 16) * (options.leading ?? 1.18) : 0}">${esc(row)}</tspan>`).join("")}</text>`);
  return item;
};
const labelCard = (value, x, y, w, h, stroke, options = {}) => {
  const item = rect(x, y, w, h, options.fill ?? C.white, stroke, options.radius ?? 7, options.width ?? 1.4, options.dash ?? "");
  item.text = value;
  item.text.style = { typeface: font, fontSize: options.size ?? 15, bold: options.bold ?? false, color: options.color ?? C.ink, autoFit: "none",insets:{left:10,right:10,top:6,bottom:6},verticalAlignment:"middle",alignment:options.align??"left" };
  if(w<60) item.text.style={insets:{left:0,right:0,top:0,bottom:0},alignment:"center",wrap:"none"};
  const lines = String(value).split("\n");
  svg.push(`<text x="${x + w / 2}" y="${y + h / 2 - ((lines.length - 1) * (options.size ?? 15) * .55)}" fill="${options.color ?? C.ink}" font-family="Arial, Helvetica, sans-serif" font-size="${options.size ?? 15}" font-weight="${options.bold ? 700 : 400}" text-anchor="middle">${lines.map((row, index) => `<tspan x="${x + w / 2}" dy="${index ? (options.size ?? 15) * 1.12 : 0}">${esc(row)}</tspan>`).join("")}</text>`);
  return item;
};
const fileImageBox = async (file, x, y, w, h, alt, stroke = C.ink, radius = 9) => {
  const bytes = await fs.readFile(file);
  const extension = path.extname(file).toLowerCase();
  const contentType = extension === ".jpg" || extension === ".jpeg" ? "image/jpeg" : "image/png";
  slide.images.add({ blob: bytes, contentType, alt, fit: "contain", geometry: "roundRect", borderRadius: radius, position: { left: x, top: y, width: w, height: h } });
  rect(x, y, w, h, "none", stroke, radius, 1.4);
  svg.push(`<image x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet" xlink:href="data:${contentType};base64,${bytes.toString("base64")}"/>`);
  svg.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="none" stroke="${stroke}" stroke-width="1.4"/>`);
};
const imageBox = async (name, x, y, w, h, alt, stroke = C.ink, radius = 9) => fileImageBox(path.join(ASSET_DIR, name), x, y, w, h, alt, stroke, radius);
const originalImageBox = async (name, x, y, w, h, alt, stroke = C.ink, radius = 9) => fileImageBox(path.join(USER_ASSET_DIR, name), x, y, w, h, alt, stroke, radius);
const flow = (from, to, colour = C.ink, sides = ["right", "left"], dashed = false) => {
  const connector = slide.shapes.connect(from, to, {
    kind: "straight", fromSide: sides[0], toSide: sides[1],
    line: { style: dashed ? "dashed" : "solid", fill: colour, width: dashed ? 2 : 2.4 },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
  connector.bringToFront();
  return connector;
};

// Banknote-inspired page frame and title hierarchy. These are native framework
// rules rather than generated decoration, so the scientific content stays clean.
rect(8, 6, 1904, 1068, C.paper, C.blueDark, 20, 1.5);
rect(15, 13, 1890, 1054, "none", C.gold, 16, 0.9);
line(26, 71, 1894, 71, C.gold, 1.2);
line(26, 75, 1894, 75, C.blue, 1.0);
text("RMDB / BiRAP MEASUREMENT ARCHITECTURE", 28, 14, 800, 31, { size: 23, bold: true, color: C.ink });
text("Frozen identity backbone, post-hoc readouts, and endpoint-specific evidence gates", 28, 44, 1040, 22, { size: 14.5, color: C.muted });
labelCard("MODEL-CENTRED, OUTCOME-BLIND", 1542, 24, 344, 31, C.gold, { size: 11.8, bold: true, fill: C.blueFill, color: C.ink2, radius: 4, width: 1.1 });

const stages = [
  { x: 24, w: 246, heading: "INPUT & PREPROCESSING", fill: C.blueFill, stroke: C.blueDark, label: "01", kicker: "SOURCE IMAGES" },
  { x: 292, w: 402, heading: "FROZEN VASCX IDENTITY", fill: C.blueFill2, stroke: C.blueDark, label: "02", kicker: "IDENTITY BACKBONE" },
  { x: 716, w: 670, heading: "DUAL FROZEN READOUTS", fill: C.mint, stroke: C.blueDark, label: "03", kicker: "MEASUREMENT READOUTS" },
  { x: 1408, w: 488, heading: "PHENOTYPES & EVIDENCE", fill: "#F7F1E3", stroke: C.goldDark, label: "04", kicker: "EVIDENCE GATES" },
];
for (const stage of stages) {
  text(stage.heading, stage.x + 4, 82, stage.w - 8, 24, { size: 16.5, bold: true, color: C.ink });
  rect(stage.x, 110, stage.w, 724, stage.fill, stage.stroke, 14, 1.6);
  labelCard(stage.label, stage.x + 18, 119, 42, 28, C.gold, { size: 14, bold: true, fill: C.goldFill, color: C.goldDark, radius: 14, width: 1.1 });
  text(stage.kicker, stage.x + 69, 125, stage.w - 88, 18, { size: 11, bold: true, color: C.ink2 });
}

// Stage 1: actual source-database originals and invariant preprocessing.
await originalImageBox("Fundus_AVSeg_001_G_original.png", 48, 170, 88, 76, "Original Fundus-AVSeg image 001_G", C.blueDark, 7);
await originalImageBox("FIVES_train_100_A_original.png", 158, 170, 88, 76, "Original FIVES training image 100_A", C.green, 7);
text("Fundus-AVSeg", 43, 249, 98, 18, { size: 10.5, bold: true, align: "center", color: C.blueDark });
text("FIVES", 158, 249, 88, 18, { size: 10.5, bold: true, align: "center", color: C.greenDark });
await originalImageBox("mBRSET_9_3_original.jpg", 48, 275, 88, 76, "Original mBRSET same-eye development image 9.3", C.gold, 7);
await originalImageBox("HRF_IQA_1_good_original.JPG", 158, 275, 88, 76, "Original HRF-IQA high-resolution image 1_good", C.orange, 7);
text("mBRSET", 48, 354, 88, 18, { size: 10.5, bold: true, align: "center", color: C.gold });
text("HRF-IQA", 158, 354, 88, 18, { size: 10.5, bold: true, align: "center", color: C.orange });
const preproc = labelCard("FOV crop: 1024²\nRGB + enhanced RGB\n3 + 3 input channels", 48, 386, 198, 68, C.blue, { size: 13, bold: true, fill: C.white });
const tiling = labelCard("Sliding-window inference\n512² · 50% overlap · Gaussian blend", 48, 477, 198, 72, C.blueDark, { size: 13, bold: true, fill: C.white });
const fov = labelCard("Common FOV · max(RGB) > 10", 48, 575, 198, 54, C.blue, { size: 13.5, bold: true, fill: C.white });
labelCard("IMAGE-ONLY INFERENCE\nNo clinical outcome inputs", 48, 674, 198, 76, C.goldDark, { size: 13.5, bold: true, fill: C.goldFill });
text("Sources have separate roles:\ndevelopment, benchmark, repeats,\nand supplementary quality stress", 48, 766, 198, 51, {size:11.5,color:C.muted});
line(147, 373, 147, 386, C.blueDark, 2.2, true);
line(147, 454, 147, 477, C.blueDark, 2.2, true);
line(147, 549, 147, 575, C.blueDark, 2.2, true);

// Stage 2: VascX details and exact four-channel output.
const vascxHeader = labelCard("VascX A/V branch · 5-fold ensemble", 318, 156, 350, 62, C.blueDark, { size: 19, bold: true, fill: C.white });
text("Five frozen DynUNets; ensemble-mean logits cached", 328, 221, 330, 30, { size: 13, color: C.muted, align: "center" });
const unetFrame = rect(318, 257, 350, 190, C.white, C.blueDark, 12, 1.7);
text("DYNUNET WITH DEEP SUPERVISION", 334, 269, 318, 20, { size: 13, bold: true, color: C.blueDark, align: "center" });
text("8 encoder scales: 32 / 64 / 96 / 128\n192 / 256 / 384 / 512 channels", 334, 297, 318, 29, { size: 12.5, bold: true, color: C.ink, align: "center" });
const unetNodes = Array.from({length:15},(_,i)=>[335+i*21.5,350+Math.min(i,14-i)*7]);
for(let i=0;i<7;i++) line(unetNodes[i][0]+16,unetNodes[i][1]+8,unetNodes[14-i][0],unetNodes[14-i][1]+8,C.line,0.8,false,"3 3");
for (let i = 0; i < unetNodes.length; i += 1) {
  const [x, y] = unetNodes[i];
  rect(x, y, 16, 16, i <= 7 ? C.blueFill : C.mint, i <= 7 ? C.blueDark : C.greenDark, 4, 1.2);
  if (i < unetNodes.length - 1) line(x + 16, y + 8, unetNodes[i + 1][0], unetNodes[i + 1][1] + 8, i < 7 ? C.blueDark : C.greenDark, 1.5);
}
text("encoder ↓", 337, 330, 80, 18, { size: 10.5, bold: true, color: C.blueDark });
text("512-channel bottleneck", 422, 416, 145, 15, { size: 11.5, bold: true, color: C.ink, align: "center" });
text("decoder + skips", 548, 330, 108, 18, { size: 11.5, bold: true, color: C.greenDark, align: "right" });
text("3×3 kernels / seven 2× downsamplings / 4 logits", 334, 432, 318, 15, { size: 11.5, color: C.muted, align: "center" });
labelCard("Frozen cached outputs", 318, 461, 166, 35, C.blueDark, { size: 12, bold: true, fill: C.grayFill, radius: 17 });
labelCard("Gaussian tile blend", 502, 461, 166, 35, C.blueDark, { size: 12, bold: true, fill: C.grayFill, radius: 17 });

const channelNames = ["B", "A", "V", "X"];
const channelFiles = ["raw_probability_background.png", "raw_probability_artery.png", "raw_probability_vein.png", "raw_probability_crossing.png"];
const channelStrokes = [C.line, C.red, C.vein, C.crossing];
for (let i = 0; i < 4; i += 1) {
  await imageBox(channelFiles[i], 319 + i * 87, 524, 72, 72, `${channelNames[i]} probability channel from frozen VascX`, channelStrokes[i], 7);
  text(channelNames[i], 319 + i * 87, 599, 72, 18, { size: 12, bold: true, align: "center", color: channelStrokes[i] === C.line ? C.ink : channelStrokes[i] });
}
const rawReadout = labelCard("Resize cached logits to 512² first\np_raw = softmax(z₅₁₂)\nbackground / artery / vein / crossing", 318, 632, 350, 68, C.blueDark, { size: 14, bold: true, fill: C.white });
await imageBox("raw_av_map.png", 318, 716, 86, 86, "Hard visualization of the raw VascX four-class probability readout", C.blueDark);
text("raw map", 410, 736, 80, 20, { size: 13, bold: true, color: C.blueDark });
text("pretrained: HRF · LESAV · LEUVEN · RITE · RS · IOSTAR\nfrozen; no outcome-guided fine-tuning", 410, 752, 235, 52, { size: 11.5, color: C.muted });

// Stage 3: reference fork, mass branch, occupancy-conditioned branch, and controls.
const fork = labelCard("Cached z and p_raw supply parallel readouts; raw remains the comparator", 741, 156, 620, 54, C.blueDark, { size: 16, bold: true, fill: C.white });

rect(738, 224, 626, 239, C.orangeFill, C.orange, 14, 1.8);
text("A · MASS-PRESERVING READOUT", 757, 236, 360, 23, { size: 16, bold: true, color: C.orange });
text("Frozen corrector developed on RAV", 1000, 252, 340, 17, { size: 12, color: C.muted, align: "right" });
const massFeat = labelCard("11-channel fixed features\nRGB: 3\nclipped/scaled logits z: 4\nnormalized H₄(p): 1\nHessian cosθ, sinθ, vesselness: 3", 757, 272, 190, 142, C.orange, { size: 13, bold: true, fill: C.white });
const massNet = rect(964, 272, 222, 142, C.white, C.orange, 11, 1.6);
text("SHARED U-BLOCK × 2 STEPS", 978, 283, 194, 19, { size: 12, bold: true, color: C.orange, align: "center" });
const massBlocks = [[974, 317, "15"], [1004, 330, "24"], [1034, 343, "48"], [1064, 356, "96"], [1094, 343, "48"], [1124, 330, "24"], [1154, 317, "4"]];
for (const [x, y, value] of massBlocks) {
  rect(x, y, 25, 23, C.orangeFill, C.orange, 5, 1.1);
  text(value, x, y + 4, 25, 15, { size: 11.5, bold: true, align: "center", color: C.orange });
}
for (let i = 0; i < massBlocks.length - 1; i += 1) line(massBlocks[i][0] + 25, massBlocks[i][1] + 12, massBlocks[i + 1][0], massBlocks[i + 1][1] + 12, C.orange, 1.3);
text("3×3 Conv / GroupNorm / SiLU\nSkip fusion; 1×1 output Δz (4)", 975, 385, 200, 29, { size: 11.5, color: C.muted, align: "center" });
await imageBox("mass_av_map.png", 1205, 276, 126, 126, "Mass-preserving frozen readout visualization", C.orange);
text("p_mass", 1205, 406, 126, 18, { size: 12, bold: true, color: C.orange, align: "center" });
text("[features, pᵗ] = 15 ch     zᵗ⁺¹ = zᵗ + Δzᵗ,  t=0,1     p_mass = softmax(z²)", 757, 429, 574, 22, { size: 13, bold: true, color: C.ink, align: "center" });
line(947, 343, 964, 343, C.orange, 1.8, true);
line(1186, 343, 1205, 343, C.orange, 1.8, true);

rect(738, 479, 626, 244, C.mint, C.green, 14, 1.8);
text("B · OCCUPANCY-CONDITIONED JOINT READOUT", 757, 491, 430, 23, { size: 16, bold: true, color: C.greenDark });
text("RGB occupancy + original raw identity", 1067, 514, 275, 15, { size: 11.5, color: C.muted, align: "right" });
const occNet = labelCard("FIVES occupancy estimator\nU-Net · ResNet34 encoder\nRGB 3 → sigmoid o(u)\nBCE + soft-Dice training", 757, 529, 178, 126, C.green, { size: 13, bold: true, fill: C.white });
await imageBox("occupancy_probability.png", 950, 532, 120, 120, "Frozen FIVES occupancy probability", C.green);
text("occupancy o(u)", 950, 655, 120, 18, { size: 11.5, bold: true, color: C.greenDark, align: "center" });
const qCard = labelCard("Original/raw identity\ns = p_A + p_V + p_X\nqₖ = p_raw,k / max(s, ε)\nk ∈ {A,V,X}, ε = 10⁻⁷", 1086, 529, 151, 91, C.blueDark, { size: 11.5, bold: true, fill: C.white });
const jointFormula = labelCard("p_joint = [1−o,\no·q_A, o·q_V, o·q_X]", 1086, 631, 151, 57, C.green, { size: 12.5, bold: true, fill: C.white });
await imageBox("joint_av_map.png", 1248, 532, 91, 91, "Occupancy-conditioned joint A/V map", C.green);
text("p_joint", 1248, 626, 91, 18, { size: 11.5, bold: true, color: C.greenDark, align: "center" });
text("POST HOC · NO JOINT TRAINING · JOINT NEVER USES MASS-CORRECTED IDENTITY", 757, 694, 582, 18, { size: 11.5, bold: true, color: C.greenDark, align: "center" });
line(935, 592, 950, 592, C.green, 1.8, true);
line(1070, 640, 1086, 640, C.green, 1.8, true);
line(1162, 620, 1162, 631, C.green, 1.8, true);
line(1237,659,1347,659,C.green,1.8);
line(1347,659,1347,577,C.green,1.8);
line(1347,577,1339,577,C.green,1.8,true);

rect(738, 741, 626, 70, C.magentaFill, C.magenta, 12, 1.5, "6 4");
text("FIXED RAW-READOUT CONTROLS", 754, 752, 300, 18, { size: 13, bold: true, color: C.blueDark });
text("Same 512² grid and phenotype operators",754,778,310,18,{size:12,color:C.muted});
await imageBox("smooth_control_map.png", 1077, 749, 52, 52, "Fixed Gaussian smooth control", C.magenta, 6);
text("Gaussian σ=1 px", 1135, 755, 103, 18, { size: 11.5, bold: true, color: C.ink });
await imageBox("cleanup_control_map.png", 1243, 749, 52, 52, "Fixed component-cleanup control", C.magenta, 6);
text("<32 px\ncleanup", 1301, 755, 58, 34, { size: 11.5, bold: true, color: C.ink });

// Stage 4: one common operator, two evidence axes, prospective gate.
rect(1430, 156, 444, 316, C.white, C.goldDark, 9, 1.5);
text("COMMON PHENOTYPE OPERATOR", 1449, 169, 406, 23, { size: 16, bold: true, color: C.goldDark });
labelCard("s = p_A + p_V + p_X\nmask = [s ≥ 0.5] ∩ FOV  (512²)\nskeleton / topology on 256²", 1449, 205, 267, 82, C.green, { size: 13, bold: true, fill: C.mint });
text("A/V = meanΩ(p_A+p_X) /\nmax(meanΩ(p_V+p_X), ε); no log",1449,295,267,39,{size:12,bold:true,color:C.ink});
await imageBox("joint_topology_map.png", 1730, 199, 124, 124, "Topology visualization: skeleton, branch points, and endpoints", C.green);
text("skeleton", 1730, 326, 53, 15, { size: 10, bold: true, color: C.teal });
text("branch", 1785, 326, 43, 15, { size: 10, bold: true, color: C.crossing });
text("end", 1830, 326, 25, 15, { size: 10, bold: true, color: C.gold });
rect(1449, 348, 190, 105, C.blueFill, C.blueDark, 10, 1.3);
text("PROBABILITY (6)\nVessel / artery / vein mass\nA/V mass ratio\nOccupancy entropy\nIdentity entropy (A/V/X)", 1461, 358, 166, 88, { size: 12.5, bold: true, color: C.ink });
rect(1653, 348, 201, 105, C.mint, C.green, 10, 1.3);
text("GEOMETRY / TOPOLOGY (6)\nVessel / skeleton density\nBranch / endpoint density\nComponent count\nFractal dimension", 1665, 358, 177, 88, { size: 12, bold: true, color: C.ink });

rect(1430, 491, 444, 200, C.white, C.goldDark, 9, 1.5);
text("SEPARATE EVIDENCE AXES", 1449, 503, 406, 22, { size: 16, bold: true, color: C.goldDark });
rect(1449, 537, 193, 143, C.blueFill, C.blueDark, 10, 1.3);
text("STRUCTURAL FIDELITY", 1460, 546, 171, 17, { size: 11.3, bold: true, color: C.blueDark, align: "center" });
await originalImageBox("Fundus_AVSeg_002_N_original.png", 1461, 567, 74, 51, "Fundus-AVSeg original image 002_N used to illustrate structural-fidelity evaluation", C.blueDark, 6);
await originalImageBox("Fundus_AVSeg_002_N_manual_AV.png", 1543, 567, 86, 51, "Fundus-AVSeg manual artery-vein annotation for image 002_N", C.crossing, 6);
text("image", 1461, 620, 74, 13, { size: 8.5, bold: true, color: C.blueDark, align: "center" });
text("manual A/V", 1543, 620, 86, 13, { size: 8.5, bold: true, color: C.crossing, align: "center" });
text("100 images; 5,000 resamples\nDice / clDice / thin recall\nGeometry errors", 1456, 633, 182, 39, { size: 11.5, bold: true, color: C.ink, align: "center", leading: 1.05 });
rect(1656, 537, 198, 143, C.mint, C.green, 10, 1.3);
text("SAME-EYE REPEATABILITY", 1668, 546, 174, 17, { size: 11.1, bold: true, color: C.greenDark, align: "center" });
await originalImageBox("mBRSET_test_613_3_original.jpg", 1668, 567, 79, 51, "mBRSET held-out test eye 613 view 3", C.green, 6);
await originalImageBox("mBRSET_test_613_4_original.jpg", 1755, 567, 87, 51, "mBRSET held-out test eye 613 view 4", C.green, 6);
text("view 3", 1668, 620, 79, 13, { size: 8.5, bold: true, color: C.greenDark, align: "center" });
text("view 4", 1755, 620, 87, 13, { size: 8.5, bold: true, color: C.greenDark, align: "center" });
text("475 eyes / 255 patients\n950 images; 10,000 resamples\nICC / MAD / SEM / 95% LoA", 1662, 633, 186, 39, { size: 11.5, bold: true, color: C.ink, align: "center", leading: 1.05 });

rect(1430, 711, 444, 100, C.goldFill, C.goldDark, 9, 1.6);
text("PROSPECTIVE ADMISSIBILITY GATE · FUTURE UNTOUCHED COHORT", 1447, 721, 410, 18, { size: 12, bold: true, color: C.goldDark, align: "center" });
text("LCI₉₅[ICC(1,1)] ≥ 0.60  AND\nUCI₉₅[MAD_arm / MAD_raw] ≤ 1.00", 1447, 746, 410, 35, { size: 14, bold: true, color: C.ink, align: "center" });
text("Clinical SEM/LoA tolerances and confirmation remain open", 1447, 786, 410, 18, { size: 11.5, color: C.muted, align: "center" });

// Main directional flow in the whitespace between stages.
line(246,513,281,513,C.blueDark,2.4);
line(281,513,281,187,C.blueDark,2.4);
line(281,187,318,187,C.blueDark,2.4,true);
line(668,666,705,666,C.blueDark,2.4);
line(705,666,705,183,C.blueDark,2.4);
line(705,183,741,183,C.blueDark,2.4,true);
line(749,210,749,224,C.blueDark,1.8,true);
line(741,197,727,197,C.blueDark,1.8);
line(727,197,727,507,C.blueDark,1.8);
line(727,507,738,507,C.blueDark,1.8,true);
line(1364,343,1375,343,C.orange,1.8);
line(1364,601,1375,601,C.green,1.8);
line(1375,343,1375,601,C.blueDark,1.8);
line(1375,466,1419,466,C.blueDark,2.4);
line(1419,466,1419,327,C.blueDark,2.4);
line(1419,327,1430,327,C.blueDark,2.4,true);
line(1652, 472, 1652, 491, C.goldDark, 2.2, true);
line(1652, 691, 1652, 711, C.goldDark, 2.2, true);

// Frozen protocol lane and contribution boundary.
rect(292, 866, 1094, 172, C.blueFill, C.blueDark, 11, 1.5, "8 6");
text("FROZEN ANALYSIS / EVIDENCE BOUNDARY", 315, 882, 590, 23, { size: 16, bold: true, color: C.blueDark });
text("Outcome-blind; adverse results retained", 1052, 884, 310, 20, { size: 12.5, color: C.muted, align: "right" });
const lockItems = [
  [315, 927, 173, "VascX weights"],
  [503, 927, 202, "UDCT + FIVES weights"],
  [720, 927, 153, "data splits"],
  [888, 927, 215, "readout definitions"],
  [1118, 927, 244, "all 12 exploratory endpoints"],
];
for (const [x, y, w, value] of lockItems) labelCard(value, x, y, w, 49, C.goldDark, { size: 13, bold: true, fill: C.white, width: 1.2 });
for (let i = 0; i < lockItems.length - 1; i += 1) line(lockItems[i][0] + lockItems[i][2], 952, lockItems[i + 1][0], 952, C.blueDark, 1.8, true);
text("Current: exploratory ICC + MAD + SEM + LoA audit. Future: untouched-cohort dual gate.\nMass-preserving names the readout; its residual update does not enforce exact mass conservation.", 315, 986, 1047, 38, { size: 12.5, color: C.muted, align: "center" });

rect(1408, 866, 488, 172, C.goldFill, C.goldDark, 11, 1.6);
text("RMDB INCREMENT OVER VASCX", 1431, 881, 442, 23, { size: 16, bold: true, color: C.goldDark });
text("Which frozen probability readout\nproduces a usable measurement?", 1431, 909, 442, 38, { size: 14.5, bold: true, color: C.ink, align: "center", leading: 1.05 });
labelCard("SEPARATES / EXPOSES\nstructural fidelity ↔ repeatability trade-offs", 1431, 950, 442, 53, C.blueDark, { size: 14.5, bold: true, fill: C.white, width: 1.3 });
text("No replacement, universal gain or cross-camera clinical claim", 1431, 1011, 442, 16, { size: 11.5, color: C.muted, align: "center" });

text("B/A/V/X = background / artery / vein / crossing   ·   solid = inference/measurement flow   ·   dashed = skip links or fixed analysis boundary", 24, 1050, 1872, 19, { size: 11.5, color: C.muted, align: "right" });

slide.speakerNotes.textFrame.setText("Scientific source of truth: final_measurement_controls.py, frozen gate protocol, VascX TorchScript configuration, train_udct_av_pilot.py, and the FIVES U-Net/ResNet34 checkpoint. Stage 1 embeds project-original images from Fundus-AVSeg, FIVES, mBRSET, and HRF-IQA. Stage 4 embeds an original/manual Fundus-AVSeg pair and a held-out same-eye mBRSET test pair. Derived retinal maps illustrate computation only and are not new results. The preceding complete figure is used only as a structure/content reference and is not embedded. The palette adapts the 2019 fifth-series RMB 10 colour hierarchy for scientific use; it does not reproduce banknote artwork or claim official colour specifications.");

svg.push("</svg>");
await fs.writeFile(FINAL_SVG, svg.join(""), "utf8");
await write("07_svg_export/svg_render_notes.md", "Native SVG mirror with editable text, banknote-inspired blue/gold framework rules, model glyphs, formulas, arrows, database originals, and data-derived retinal-map examples. The preceding figure reference is not embedded.\n");

const candidate = path.join(BUILD, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const finalization = await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: ROOT,
  candidatePath: candidate,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "18288000,10287000", "--validate-bullet-geometry", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, "finalization_v6.validation.json"),
});
await json("06_editable_pptx/pptx_validation.json", finalization);
const quickPreview = await presentation.export({ slide, format: "png", scale: 1.25 });
await fs.writeFile(path.join(PACKAGE, "08_previews/preview_pptx_fallback.png"), new Uint8Array(await quickPreview.arrayBuffer()));
console.log(JSON.stringify({ package: PACKAGE, pptx: FINAL_PPTX, svg: FINAL_SVG, font, finalization }, null, 2));
