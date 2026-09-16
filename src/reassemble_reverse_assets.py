#!/usr/bin/env python3
"""Verify, reassemble, and safely extract the frozen reverse-audit assets.

This helper intentionally excludes China Fundus CIMT and FIVES.  The reverse
audit restores only the open, outcome-free external cohorts and the published
frozen checkpoints.  Every byte is checked against the release manifests
before extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECT_ARCHIVES = (
    "AFIO_CC_BY_4.0.zip",
    "Fundus_AVSeg_CC_BY_4.0.zip",
    "RAV_CC_BY_NC_4.0.tar.gz",
)
MODEL_ARCHIVE = "RMDB_model_checkpoints_v1.0.0.tar.gz"
MODEL_PREFIX = MODEL_ARCHIVE + ".part"


def parse_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        hashes[name.strip()] = digest.lower()
    return hashes


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}: {observed} != {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def reassemble(parts: list[Path], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    temporary.replace(destination)


def safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"Unsafe archive member: {member_name}")
    return target


def extract_zip(archive: Path, destination: Path) -> int:
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            safe_target(destination, member.filename)
        handle.extractall(destination)
    return len(members)


def extract_tar(archive: Path, destination: Path) -> int:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            safe_target(destination, member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not accepted: {member.name}")
        handle.extractall(destination, filter="data")
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    assets = args.assets.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or output / "asset_restore_metadata.json"

    open_hashes = parse_hashes(ROOT / "data/OPEN_DATA_ASSETS_SHA256.txt")
    part_hashes = parse_hashes(ROOT / "LARGE_ASSET_PARTS_SHA256.txt")
    archive_hashes = parse_hashes(ROOT / "MODELS_AND_LARGE_ARCHIVES_SHA256.txt")
    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "reverse_audit_open_assets_only_v1",
        "cimt_restored": False,
        "fives_restored": False,
        "verified": [],
        "extracted": {},
    }

    for name in DIRECT_ARCHIVES:
        report["verified"].append(verify(assets / name, open_hashes[name]))

    model_parts = sorted(
        path for path in assets.glob(MODEL_PREFIX + "*")
        if path.name in part_hashes
    )
    expected_model_parts = sorted(
        name for name in part_hashes if name.startswith(MODEL_PREFIX)
    )
    if [path.name for path in model_parts] != expected_model_parts:
        missing = sorted(set(expected_model_parts) - {path.name for path in model_parts})
        raise FileNotFoundError(f"Missing checkpoint parts: {missing}")
    for part in model_parts:
        report["verified"].append(verify(part, part_hashes[part.name]))

    model_archive = assets / MODEL_ARCHIVE
    if not model_archive.is_file() or sha256(model_archive) != archive_hashes[MODEL_ARCHIVE]:
        reassemble(model_parts, model_archive)
    report["verified"].append(verify(model_archive, archive_hashes[MODEL_ARCHIVE]))

    extraction_plan = {
        "AFIO": assets / DIRECT_ARCHIVES[0],
        "Fundus_AVSeg": assets / DIRECT_ARCHIVES[1],
        "RAV": assets / DIRECT_ARCHIVES[2],
        "models": model_archive,
    }
    for label, archive in extraction_plan.items():
        destination = output / label
        marker = destination / ".verified_complete.json"
        if marker.is_file():
            report["extracted"][label] = json.loads(marker.read_text(encoding="utf-8"))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        count = extract_zip(archive, destination) if archive.suffix == ".zip" else extract_tar(archive, destination)
        entry = {
            "archive": archive.name,
            "archive_sha256": sha256(archive),
            "members": count,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        marker.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        report["extracted"][label] = entry

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
