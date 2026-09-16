#!/usr/bin/env python3
"""Validate an unseen, paired multi-camera cohort before any RMDB inference.

The audit reads metadata and optional native-image checksums only.  It deliberately
does not open segmentation outputs, phenotype values, annotations, or clinical data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED = (
    "image_id", "patient_id", "eye", "session_id", "acquisition_id", "camera_id",
    "original_path", "native_width", "native_height", "capture_time", "field_center",
    "dilation_state", "prior_project_use", "sha256", "quality_status",
)
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["patient_id"], row["eye"].lower(), row["session_id"]


def camera_config(row: dict[str, str]) -> str:
    return f"{row['camera_id']}__{row['native_width']}x{row['native_height']}"


def add_issue(issues: list[dict[str, Any]], code: str, message: str, image_id: str | None = None) -> None:
    issue: dict[str, Any] = {"code": code, "message": message}
    if image_id is not None:
        issue["image_id"] = image_id
    issues.append(issue)


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [field for field in REQUIRED if field not in fields]
        if missing:
            add_issue(issues, "missing_columns", f"Missing required columns: {', '.join(missing)}")
            return [], issues
        rows = [{field: (value or "").strip() for field, value in row.items()} for row in reader]
    if not rows:
        add_issue(issues, "empty_manifest", "Manifest contains no rows.")
    return rows, issues


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    rows, issues = load_rows(manifest)
    if not rows:
        return {"manifest": str(manifest), "overall_status": "ineligible", "issues": issues}

    seen_images: set[str] = set()
    seen_acquisitions: set[str] = set()
    accepted: list[dict[str, str]] = []
    source_root = Path(args.project_root).resolve() if args.project_root else manifest.parent

    for row in rows:
        image_id = row["image_id"]
        for field in REQUIRED:
            if not row[field]:
                add_issue(issues, "missing_value", f"{field} is empty", image_id)
        if image_id in seen_images:
            add_issue(issues, "duplicate_image_id", "image_id must be unique", image_id)
        seen_images.add(image_id)
        if row["acquisition_id"] in seen_acquisitions:
            add_issue(issues, "duplicate_acquisition_id", "acquisition_id must be globally unique", image_id)
        seen_acquisitions.add(row["acquisition_id"])
        if row["eye"].lower() not in {"left", "right"}:
            add_issue(issues, "invalid_eye", "eye must be left or right", image_id)
        if row["prior_project_use"].lower() not in FALSE_VALUES:
            add_issue(issues, "prior_project_use", "Every confirmation image must declare prior_project_use=false", image_id)
        try:
            width, height = int(row["native_width"]), int(row["native_height"])
            if width <= 0 or height <= 0:
                raise ValueError
            if max(width, height) < args.minimum_long_side:
                add_issue(issues, "not_high_resolution", f"Native long side is below {args.minimum_long_side}px", image_id)
        except ValueError:
            add_issue(issues, "invalid_dimensions", "native_width/native_height must be positive integers", image_id)
        if parse_time(row["capture_time"]) is None:
            add_issue(issues, "invalid_capture_time", "capture_time must be ISO-8601", image_id)
        if len(row["sha256"]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in row["sha256"]):
            add_issue(issues, "invalid_sha256", "sha256 must be a 64-character hexadecimal digest", image_id)
        if args.verify_files:
            image_path = Path(row["original_path"])
            if not image_path.is_absolute():
                image_path = source_root / image_path
            if not image_path.is_file():
                add_issue(issues, "missing_original_file", f"Original image not found: {image_path}", image_id)
            elif args.verify_sha256 and sha256_file(image_path).lower() != row["sha256"].lower():
                add_issue(issues, "sha256_mismatch", "Original image checksum does not match manifest", image_id)
        if row["quality_status"].lower() == "accepted":
            accepted.append(row)

    sessions: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in accepted:
        sessions[key(row)].append(row)

    fully_qualified: list[tuple[str, str, str]] = []
    pairing_summary: list[dict[str, Any]] = []
    for session_key, group in sorted(sessions.items()):
        configs: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            configs[camera_config(row)].append(row)
        valid_repeats = {name: items for name, items in configs.items() if len(items) >= 2}
        times = [parse_time(row["capture_time"]) for row in group]
        time_ok = all(times) and (max(times) - min(times)).total_seconds() <= args.maximum_minutes * 60
        field_ok = len({row["field_center"] for row in group}) == 1
        dilation_ok = len({row["dilation_state"] for row in group}) == 1
        is_qualified = len(configs) >= 2 and len(valid_repeats) == len(configs) and time_ok and field_ok and dilation_ok
        pairing_summary.append({
            "patient_id": session_key[0], "eye": session_key[1], "session_id": session_key[2],
            "camera_configurations": {name: len(items) for name, items in sorted(configs.items())},
            "session_window_minutes": None if not all(times) else round((max(times) - min(times)).total_seconds() / 60, 3),
            "field_center_consistent": field_ok, "dilation_state_consistent": dilation_ok,
            "fully_qualified": is_qualified,
        })
        if is_qualified:
            fully_qualified.append(session_key)

    qualified_patients = {item[0] for item in fully_qualified}
    if len(fully_qualified) < args.minimum_eyes:
        add_issue(issues, "insufficient_qualified_eyes", f"Need >= {args.minimum_eyes} fully qualified eye-sessions; found {len(fully_qualified)}")
    if len(qualified_patients) < args.minimum_patients:
        add_issue(issues, "insufficient_participants", f"Need >= {args.minimum_patients} participants; found {len(qualified_patients)}")
    if len(accepted) != len(rows):
        add_issue(issues, "nonaccepted_rows", "All rows in a confirmation manifest must be prospectively accepted; exclusions belong in a separate screening log.")

    status = "eligible" if not issues else "ineligible"
    return {
        "audit_id": "RMDB-confirmation-cohort-eligibility-v1",
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "overall_status": status,
        "requirements": {
            "minimum_native_long_side_px": args.minimum_long_side,
            "minimum_fully_qualified_eyes": args.minimum_eyes,
            "minimum_participants": args.minimum_patients,
            "maximum_session_window_minutes": args.maximum_minutes,
        },
        "counts": {
            "rows": len(rows), "accepted_rows": len(accepted), "participants": len({row['patient_id'] for row in accepted}),
            "eye_sessions": len(sessions), "fully_qualified_eye_sessions": len(fully_qualified),
            "participants_with_fully_qualified_eye_sessions": len(qualified_patients),
            "camera_models": dict(sorted(Counter(row["camera_id"] for row in accepted).items())),
        },
        "fully_qualified_keys": ["|".join(item) for item in fully_qualified],
        "pairing_summary": pairing_summary,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Candidate UTF-8 CSV manifest")
    parser.add_argument("--out", required=True, help="Eligibility report JSON")
    parser.add_argument("--project-root", help="Base directory for relative original_path values")
    parser.add_argument("--minimum-long-side", type=int, default=3000)
    parser.add_argument("--minimum-eyes", type=int, default=120)
    parser.add_argument("--minimum-patients", type=int, default=60)
    parser.add_argument("--maximum-minutes", type=float, default=30.0)
    parser.add_argument("--verify-files", action="store_true", help="Check each original_path exists")
    parser.add_argument("--verify-sha256", action="store_true", help="Hash each original image; implies --verify-files")
    args = parser.parse_args()
    if args.verify_sha256:
        args.verify_files = True
    report = audit(args)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['overall_status']}: {destination}")
    return 0 if report["overall_status"] == "eligible" else 2


if __name__ == "__main__":
    sys.exit(main())
