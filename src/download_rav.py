#!/usr/bin/env python3
"""Restartable parallel downloader for the public RAV artery/vein dataset."""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOI = "doi:10.34894/9OIMWY"
METADATA_URL = f"https://dataverse.nl/api/datasets/:persistentId/?persistentId={DOI}"
FILE_URL = "https://dataverse.nl/api/access/datafile/{file_id}"


def download(entry: dict, root: Path) -> tuple[str, str]:
    data = entry["dataFile"]
    directory = Path(entry.get("directoryLabel", "dataset"))
    target = root / directory / data["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = int(data.get("filesize", 0))
    if target.exists() and (expected == 0 or target.stat().st_size == expected):
        return str(target), "cached"
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        FILE_URL.format(file_id=data["id"]), headers={"User-Agent": "RAV-research-downloader/1.0"}
    )
    with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if expected and partial.stat().st_size != expected:
        raise IOError(f"{target}: expected {expected}, received {partial.stat().st_size}")
    partial.replace(target)
    return str(target), "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../data/external/RAV"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(METADATA_URL, timeout=60) as response:
        metadata = json.load(response)
    version = metadata["data"]["latestVersion"]
    entries = [
        entry for entry in version["files"]
        if entry.get("directoryLabel") in {"dataset/rgb", "dataset/av_masks"}
    ]
    failures = []
    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, entry, args.output): entry for entry in entries}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                path, status = future.result()
                counts[status] += 1
                if completed % 20 == 0 or completed == len(entries):
                    print(f"{completed}/{len(entries)} cached={counts['cached']} downloaded={counts['downloaded']}", flush=True)
            except Exception as error:  # continue so a later run can resume only failed files
                failures.append({"file": futures[future]["dataFile"]["filename"], "error": str(error)})
                print(f"FAILED {failures[-1]}", flush=True)

    rgb = {path.name for path in (args.output / "dataset/rgb").glob("*.png")}
    masks = {path.name for path in (args.output / "dataset/av_masks").glob("*.png")}
    summary = {
        "doi": DOI,
        "version": version["versionNumber"],
        "metadata_files": len(entries),
        "rgb": len(rgb),
        "masks": len(masks),
        "paired": len(rgb & masks),
        "unpaired": sorted(rgb ^ masks),
        "failures": failures,
        "complete": not failures and rgb == masks and len(rgb) == len(entries) // 2,
    }
    (args.output / "download_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
