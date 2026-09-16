import fs from 'node:fs/promises';
import sharp from '/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/dist/index.mjs';
const p='/Users/tongyue/Documents/ppf/papers/RMDB_资料汇总_20260909/实验项目/artifacts/scientific_figures/rmdb_birap_architecture_final_20260914_v5';
const a=await sharp(p+'/01_inputs/source_images/previous_complete_architecture.png').resize(1600,900,{fit:'contain',background:'#FAF8F0'}).png().toBuffer();
const b=await sharp(p+'/08_previews/preview_final.png').resize(1600,900,{fit:'contain',background:'#FAF8F0'}).png().toBuffer();
await sharp({create:{width:3200,height:900,channels:3,background:'#FAF8F0'}}).composite([{input:a,left:0,top:0},{input:b,left:1600,top:0}]).png().toFile(p+'/08_previews/comparison_master_vs_editable.png');
await sharp(p+'/07_svg_export/RMDB_Architecture_Final.svg').png().toFile(p+'/08_previews/preview_svg.png');
await fs.copyFile(p+'/08_previews/preview_final.png',p+'/08_previews/contact_sheet.png');
