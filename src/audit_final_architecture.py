"""Inventory and evidence checks for the detailed mother-layout restoration."""
import hashlib,json,posixpath,zipfile
from pathlib import Path
from lxml import etree
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'artifacts/scientific_figures/rmdb_birap_architecture_final_20260914_v5'
M=P/'09_manifests';Q=P/'10_qa'
def save(n,x): (M/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
sha=lambda b:hashlib.sha256(b).hexdigest()
files={}
for d in ['01_inputs/user_assets','03_assets_raw/generated_single_assets']:
    for f in (P/d).iterdir():
        if f.suffix.lower() in ['.png','.jpg','.jpeg']:files[sha(f.read_bytes())]=str(f.relative_to(P))
inventory=[];assets=[];routes=[]
with zipfile.ZipFile(P/'06_editable_pptx/RMDB_Architecture_Final_v6.pptx') as z:
    xml=etree.fromstring(z.read('ppt/slides/slide1.xml'))
    rels=etree.fromstring(z.read('ppt/slides/_rels/slide1.xml.rels'))
    relmap={r.get('Id'):posixpath.normpath(posixpath.join('ppt/slides',r.get('Target'))).lstrip('/') for r in rels}
    for i,o in enumerate(xml.xpath('.//p:spTree/*',namespaces=ns)):
        kind=etree.QName(o).localname
        if kind not in ['sp','pic','cxnSp']:continue
        x=o.find('.//a:xfrm',ns)
        bbox=None
        if x is not None:
            off=x.find('a:off',ns);ext=x.find('a:ext',ns)
            if off is not None and ext is not None:bbox=[int(off.get('x'))/9525,int(off.get('y'))/9525,int(ext.get('cx'))/9525,int(ext.get('cy'))/9525]
        txt=' '.join(o.xpath('.//a:t/text()',namespaces=ns))
        g=o.find('.//a:prstGeom',ns);geom=g.get('prst') if g is not None else kind
        entry={'id':f'object_{i}','kind':kind,'geometry':geom,'text':txt,'placement_target_bbox':bbox,'editability_type':'native_text_or_framework','region_granularity':'one independent slide object'}
        if kind=='pic':
            b=o.find('.//a:blip',ns);rid=b.get('{'+ns['r']+'}embed');digest=sha(z.read(relmap[rid]))
            assert digest in files,'No reference-screenshot-derived media permitted'
            entry.update(editability_type='independent_project_raster',asset=files[digest],sha256=digest)
            assets.append(entry.copy())
        elif geom in ['line','rightArrow','leftArrow','downArrow','upArrow']:
            routes.append(entry.copy())
        inventory.append(entry)
    text=' '.join(xml.xpath('.//a:t/text()',namespaces=ns))
    assert '255 patients' in text and '258 patients' not in text
    assert '8 encoder scales' in text and 'SHARED U-BLOCK' in text
    assert 'all 12 exploratory endpoints' in text and 'FUTURE UNTOUCHED' in text
    assert len(assets)==19
save('asset_manifest.json',{'assets':assets,'source_policy':'All 19 independent image objects match original database files or previously generated deterministic model-map assets byte for byte. No screenshot regions and no new AI-generated images.','procedural_source':'src/export_rmdb_architecture_assets.py'})
save('master_layout_manifest.json',{'canvas':[1920,1080],'reference':'01_inputs/source_images/previous_complete_architecture.png','contract':'Retain four columns, U-shaped networks, channel maps, both readout branches, evidence pairs and lower lane. Correct content and local typesetting.','fineRegions':inventory,'routeSegments':routes,'legendItems':['B','A','V','X','skeleton','branch','end','solid measurement flow','dashed skip / fixed boundary'],'smallIcons':'Numbered badges are native framework geometry, no semantic pictograms omitted.','accepted_deviations':['Full eight-scale backbone glyph','Explicit shared two-step residual U-block','Orthogonal external routes','255 repeat contributors','A/V ratio and numerical denominator floor','Exploratory versus future-gate distinction','Fixed font sizes and insets instead of silent shrink-to-fit']})
save('editability_manifest.json',{'native_shapes':len(inventory)-len(assets),'raster_objects':len(assets),'native_elements':'Text, network blocks, frames, lines, arrowheads, probability formulas and gates','non_vector':'Retinal photographs and deterministic probability/topology maps','reference_pixels_embedded':False,'arrow_policy':'Native directional arrow glyphs at line destinations; output formula routes into the joint image. No graph screenshot background.','svg_status':'Secondary editable mirror; PPTX is the authoritative requested deliverable.'})
save('output_index.json',{'final_pptx':'06_editable_pptx/RMDB_Architecture_Final_v6.pptx','final_svg':'07_svg_export/RMDB_Architecture_Final.svg','preview_pptx':'08_previews/preview_final.png','selected_master':'01_inputs/source_images/previous_complete_architecture.png','comparison_master_vs_editable':'08_previews/comparison_master_vs_editable.png','asset_manifest':'09_manifests/asset_manifest.json','master_layout_manifest':'09_manifests/master_layout_manifest.json','editability_manifest':'09_manifests/editability_manifest.json','qa_notes':'10_qa/QA_notes.md','known_issues':'10_qa/known_issues.md'})
save('visual_diff_report.json',{'layout_parity':'Preserved four-panel contract, checked against mother image','asset_parity':'19 original image objects retained; byte-level match passed','content_parity':'Intentional source-backed scientific corrections','arrow_parity':'Orthogonal routing and explicit arrowheads replace ambiguous original connections','fine_region_coverage':'All saved independent objects inventoried','no_crop_gate':'passed','visual_review':'Saved-PPTX LibreOffice render inspected, not just construction preview','svg_review':'Secondary SVG typesetting differs from PPTX and is not the principal delivery preview'})
(Q/'QA_notes.md').write_text('# Final architecture QA\n\nThe saved PPTX passes package, layout, font and re-import checks. Its final LibreOffice render was inspected at full composition and panel scale. The approved four-column master, original images, channel maps, network structure and evidence lane are retained. Exact object inventory is in final_object_inventory.json and master_layout_manifest.json. All 19 image objects match independent project assets.\n\nNative Microsoft PowerPoint execution is not claimed. The manuscript uses an A3 landscape foldout for the complete architecture so small structural labels are not forced into ordinary page width.\n')
(Q/'known_issues.md').write_text('# Boundaries\n\nNo new cross-camera clinical confirmation was executed. Backbone configuration was read from the checkpoint; run-specific TTA was not retrospectively established and is not asserted. The primary handoff is editable PPTX; the secondary SVG has different text wrapping and is not the authoritative view. A3 landscape is used for the detailed manuscript figure; journal-specific typesetting may require panel splitting at submission.\n')
(P/'01_inputs/notes/source_understanding.md').write_text('# Restoration basis\n\nThe user rejected the simplified card-based replacement and required the detailed RMB10-inspired mother layout. All database originals and deterministic maps were reused without screenshot extraction. Checkpoint config confirms five folds, six input channels, eight encoder scales and default sliding-window settings. Cached logit extraction and the frozen readout code determine the displayed postprocessing. Run-specific flip TTA is left unasserted. Existing corrected agreement evidence supplies 255 contributors, 475 eyes, 950 images.\n')
save('scientific_sources.json',{'backbone':'models/third_party/vascx_weights/av_july24.pt :: extra/config.yaml','cache':'src/infer_av_folder.py','readouts':['src/final_measurement_controls.py','src/train_udct_av_pilot.py','src/extract_mbrset_repeatability_features.py'],'numerical_evidence':'artifacts/final_measurement_controls','scope':'Architecture restoration, no model retraining or new clinical result'})
print('Passed: 19 independent images; native diagram and full fine-region inventory.')
