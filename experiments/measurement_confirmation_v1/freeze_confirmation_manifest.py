#!/usr/bin/env python3
"""Create the immutable confirmation manifest after metadata-only eligibility passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


FALSE_VALUES = {"0", "false", "f", "no", "n"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(row: dict[str, str]) -> str:
    return f"{row['camera_id']}__{row['native_width']}x{row['native_height']}"


def eye_session(row: dict[str, str]) -> str:
    return "|".join((row["patient_id"], row["eye"].lower(), row["session_id"]))


def require_provenance(value: Any, path: str = "model_provenance") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require_provenance(child, f"{path}.{key}")
    elif isinstance(value, str) and (value == "REQUIRED" or "ABSOLUTE_OR_PROJECT_RELATIVE_PATH" in value):
        raise ValueError(f"Unresolved provenance placeholder at {path}")


def choose_annotation_eyes(keys: list[str], per_camera: dict[str, set[str]], n: int) -> set[str]:
    """Select eye-sessions deterministically after multi-camera eligibility is verified."""
    if n > len(keys):
        raise ValueError(f"Annotation subset asks for {n} eyes but only {len(keys)} are eligible")
    # Each qualified eye-session contains every required camera configuration, so selection
    # by eye-session automatically retains the multi-camera strata without duplicate quotas.
    del per_camera
    ranked = sorted(keys, key=lambda item: hashlib.sha256(item.encode()).hexdigest())
    return set(ranked[:n])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eligibility-report", required=True)
    parser.add_argument("--model-provenance", required=True)
    parser.add_argument("--protocol", default=Path(__file__).with_name("protocol_v1.json"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotation-eyes", type=int, default=80)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    audit_path = Path(args.eligibility_report).resolve()
    provenance_path = Path(args.model_provenance).resolve()
    protocol_path = Path(args.protocol).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if audit.get("overall_status") != "eligible":
        raise ValueError("Eligibility audit is not eligible; refusing to freeze a confirmation manifest")
    if audit.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("Candidate manifest changed after eligibility audit")
    if protocol.get("status") != "pre-exposure":
        raise ValueError("Expected a pre-exposure protocol")
    require_provenance(provenance)

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    qualified = set(audit.get("fully_qualified_keys", []))
    locked = [row for row in rows if eye_session(row) in qualified]
    if not locked:
        raise ValueError("No qualified rows found in candidate manifest")
    if any(row["prior_project_use"].strip().lower() not in FALSE_VALUES for row in locked):
        raise ValueError("All locked rows must have prior_project_use=false")
    locked.sort(key=lambda row: (row["patient_id"], row["eye"].lower(), row["session_id"], row["camera_id"], row["capture_time"], row["acquisition_id"], row["image_id"]))
    for row in locked:
        row["camera_configuration"] = configuration(row)
        row["eye_session_key"] = eye_session(row)

    eye_keys = sorted({row["eye_session_key"] for row in locked})
    camera_eyes: dict[str, set[str]] = defaultdict(set)
    for row in locked:
        camera_eyes[row["camera_id"]].add(row["eye_session_key"])
    annotation_keys = choose_annotation_eyes(eye_keys, camera_eyes, args.annotation_eyes)
    for row in locked:
        row["annotation_selected"] = "true" if row["eye_session_key"] in annotation_keys else "false"
        row["annotation_status"] = "pending" if row["eye_session_key"] in annotation_keys else "not_selected"

    destination = Path(args.out_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing lock directory: {destination}")
    destination.mkdir(parents=True)
    out_manifest = destination / "confirmation_manifest.csv"
    fields = list(locked[0])
    with out_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(locked)
    out_provenance = destination / "model_provenance.json"
    shutil.copy2(provenance_path, out_provenance)
    lock = {
        "lock_id": "RMDB-confirmation-manifest-v1",
        "protocol_id": protocol["protocol_id"],
        "candidate_manifest_sha256": digest(manifest_path),
        "eligibility_report_sha256": digest(audit_path),
        "protocol_sha256": digest(protocol_path),
        "model_provenance_sha256": digest(out_provenance),
        "confirmation_manifest_sha256": digest(out_manifest),
        "locked_images": len(locked),
        "locked_eye_sessions": len(eye_keys),
        "locked_participants": len({row["patient_id"] for row in locked}),
        "annotation_eye_sessions": len(annotation_keys),
        "no_tuning_after_lock": protocol["no_tuning"],
    }
    (destination / "lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"locked {lock['locked_images']} images / {lock['locked_eye_sessions']} eye-sessions: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # CLI should leave no ambiguous successful lock.
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
