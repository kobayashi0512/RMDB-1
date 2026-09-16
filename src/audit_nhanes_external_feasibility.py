#!/usr/bin/env python3
"""Outcome-blind audit of the public NHANES retinal-image/clinical-data path.

This script deliberately does not fit a model or use labels for model selection.  It
only answers whether public retinal pixels can be joined to the public clinical
files.  It also guards against a dangerous false match: the NHANES III public FTP
``XRays/Nhanes3`` directory contains TIFF radiographs whose names share SEQN-like
identifiers, but those files are not retinal photographs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "data" / "external" / "NHANES"
OUT = PROJECT / "artifacts" / "nhanes_external_feasibility"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def xpt_summary(name: str, columns: list[str]) -> dict:
    path = DATA / name
    df = pd.read_sas(path, format="xport", encoding="latin1")
    seqn = pd.to_numeric(df.get("SEQN"), errors="coerce").dropna()
    result = {
        "file": name,
        "sha256": sha256(path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_seqn": int(seqn.nunique()),
        "requested_columns_present": {c: c in df.columns for c in columns},
    }
    for c in columns:
        if c in df:
            counts = df[c].value_counts(dropna=False).head(12)
            result["top_values_" + c] = {str(k): int(v) for k, v in counts.items()}
    return result


def main() -> None:
    listing = Path("/tmp/nhanes3_listing.html")
    names = []
    if listing.exists():
        names = re.findall(r">(\d+[HK]\d+\.tiff)<", listing.read_text(errors="ignore"))
    image_seqn = sorted({int(re.match(r"(\d+)[HK]\d+\.tiff", n).group(1)) for n in names})

    retinal = [
        xpt_summary("OPXRET_D.XPT", ["SEQN", "OPASCST2", "OPDURET", "OPDUARMA", "OPDDRET", "OPDSRET"]),
        xpt_summary("OPXRET_E.XPT", ["SEQN", "OPASCST2", "OPDURET", "OPDUARMA", "OPDDRET", "OPDSRET"]),
    ]
    clinical = [
        xpt_summary("BPX_D.XPT", ["SEQN", "BPXSY1", "BPXDI1"]),
        xpt_summary("BPX_E.XPT", ["SEQN", "BPXSY1", "BPXDI1"]),
        xpt_summary("BPQ_D.XPT", ["SEQN"]),
        xpt_summary("BPQ_E.XPT", ["SEQN"]),
        xpt_summary("DEMO_D.XPT", ["SEQN", "RIDAGEYR", "RIAGENDR"]),
        xpt_summary("DEMO_E.XPT", ["SEQN", "RIDAGEYR", "RIAGENDR"]),
    ]

    radiograph_sample = None
    sample = Path("/tmp/nhanes_samples/10018H1_full.tiff")
    if sample.exists():
        with Image.open(sample) as im:
            radiograph_sample = {
                "file": sample.name,
                "format": im.format,
                "size": list(im.size),
                "mode": im.mode,
                "frames": int(getattr(im, "n_frames", 1)),
                "semantic_check": "radiograph (hand/wrist markers visible), not a retinal photograph",
            }

    # The public OPXRET documentation describes the captured and graded images,
    # but does not publish an image-file endpoint.  Keep this conclusion explicit
    # and separate from the availability of the grading table.
    result = {
        "audit_type": "outcome_blind_public_data_feasibility",
        "source_scope": {
            "retinal_documentation": [
                "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/OPXRET_D.htm",
                "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/OPXRET_E.htm",
            ],
            "nhanes3_image_listing_checked": "https://ftp.cdc.gov/pub/NHANES/XRays/Nhanes3/",
            "nhanes3_public_data_page": "https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx",
        },
        "public_retinal_grading_tables": retinal,
        "public_clinical_tables": clinical,
        "nhanes3_ftp_tiff_listing": {
            "listed_files": len(names),
            "unique_seqn_like_stems": len(image_seqn),
            "identifier_overlap_with_clinical_files": "all listed stems parse as NHANES III SEQN-like identifiers",
            "semantic_result": "not_retinal_images",
            "warning": "Do not use XRays/Nhanes3 TIFF files as fundus images; the tested file is a hand/wrist radiograph.",
            "sample": radiograph_sample,
        },
        "public_image_pixel_join": {
            "retinal_pixels_found_in_checked_public_endpoints": False,
            "retinal_grading_variables_found": True,
            "usable_for_frozen_retinal_model_inference": False,
            "reason": "OPXRET_D/E expose retinal grading/quality variables, while the checked public endpoint does not expose the original retinal pixels.",
        },
        "decision": {
            "independent_external_clinical_outcome_cohort_ready": False,
            "do_not_claim": "independent external clinical confirmation",
            "highest_feasibility_next_step": "Use a cohort with image pixels and outcomes under a permitted DUA or institution-secure compute; keep NHANES as a non-image reference only.",
        },
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({
        "retinal_xpt_rows": [x["rows"] for x in retinal],
        "clinical_files": len(clinical),
        "xray_tiffs": len(names),
        "retinal_pixels_publicly_joinable": result["public_image_pixel_join"]["retinal_pixels_found_in_checked_public_endpoints"],
        "decision": result["decision"]["independent_external_clinical_outcome_cohort_ready"],
        "output": str(OUT / "audit.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
