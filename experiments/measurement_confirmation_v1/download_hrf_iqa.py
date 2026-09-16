#!/usr/bin/env python3
"""Resumeless, checksum-recorded downloader for the public HRF IQA pair archive.

The official server supports HTTP byte ranges but is slow per connection.  The
script requests non-overlapping ranges in parallel, validates every range size,
then atomically joins them.  It refuses to overwrite a completed archive.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/allQuality.zip"


def request(url: str, headers: dict[str, str] | None = None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120)


def fetch_part(url: str, start: int, end: int, path: Path) -> tuple[int, int]:
    if path.exists() and path.stat().st_size == end - start + 1:
        return start, end
    temporary = path.with_suffix(path.suffix + ".partial")
    existing = temporary.stat().st_size if temporary.exists() else 0
    expected = end - start + 1
    if existing > expected:
        raise RuntimeError(f"Partial range {start}-{end} has too many bytes: {existing}")
    range_start = start + existing
    with request(url, {"Range": f"bytes={range_start}-{end}"}) as response, temporary.open("ab") as handle:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise RuntimeError(f"Server did not honour Range {range_start}-{end}; status={status}")
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if temporary.stat().st_size != expected:
        raise RuntimeError(f"Range {start}-{end} has {temporary.stat().st_size} bytes, expected {expected}")
    temporary.replace(path)
    return start, end


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError(f"Archive already exists: {destination}")
    with request(args.url) as response:
        length = int(response.headers["Content-Length"])
        accept_ranges = response.headers.get("Accept-Ranges", "").lower()
    if "bytes" not in accept_ranges:
        raise RuntimeError("Official server did not advertise HTTP byte-range support")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = (length + args.workers - 1) // args.workers
    jobs = [(index * chunk, min(length - 1, (index + 1) * chunk - 1), parts_dir / f"part_{index:03d}") for index in range(args.workers) if index * chunk < length]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(fetch_part, args.url, start, end, path) for start, end, path in jobs]
            for future in concurrent.futures.as_completed(futures):
                start, end = future.result()
                print(f"downloaded bytes {start}-{end}", flush=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("xb") as output:
            for _, _, part in jobs:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        if temporary.stat().st_size != length:
            raise RuntimeError(f"Joined archive has {temporary.stat().st_size} bytes, expected {length}")
        temporary.replace(destination)
        record = {"url": args.url, "bytes": length, "sha256": digest(destination), "workers": args.workers}
        destination.with_suffix(destination.suffix + ".source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
        return 0
    except Exception:
        # Keep parts for forensic inspection; never assemble or present a partial archive as data.
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
