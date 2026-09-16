#!/usr/bin/env python3
"""Resume selected private GitHub release assets without exposing credentials."""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_PATTERNS = (
    "AFIO_CC_BY_4.0.zip",
    "Fundus_AVSeg_CC_BY_4.0.zip",
    "RAV_CC_BY_NC_4.0.tar.gz",
    "RMDB_model_checkpoints_v1.0.0.tar.gz.part*",
)


def token() -> str:
    value = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    if not value:
        raise RuntimeError("GitHub CLI returned an empty token")
    return value


def request_json(url: str, credential: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {credential}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RMDB-reproducibility-downloader/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(asset: dict, destination: Path, credential: str, attempts: int) -> dict:
    target = destination / asset["name"]
    expected = int(asset["size"])
    for attempt in range(1, attempts + 1):
        existing = target.stat().st_size if target.exists() else 0
        if existing == expected:
            return {"name": target.name, "bytes": existing, "status": "cached"}
        if existing > expected:
            raise ValueError(f"{target.name}: local file exceeds release size")
        headers = {
            "Authorization": f"Bearer {credential}",
            "Accept": "application/octet-stream",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RMDB-reproducibility-downloader/1.0",
        }
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(asset["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                append = existing > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with target.open(mode) as handle:
                    while block := response.read(4 * 1024 * 1024):
                        handle.write(block)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** attempt, 30))
            continue
        observed = target.stat().st_size
        if observed == expected:
            return {"name": target.name, "bytes": observed, "status": "downloaded"}
        if attempt == attempts:
            raise IOError(f"{target.name}: expected {expected}, received {observed}")
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="kobayashi0512/RMDB")
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", action="append", dest="patterns")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    credential = token()
    release = request_json(
        f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}",
        credential,
    )
    patterns = tuple(args.patterns or DEFAULT_PATTERNS)
    assets = [
        asset for asset in release["assets"]
        if any(fnmatch.fnmatch(asset["name"], pattern) for pattern in patterns)
    ]
    if not assets:
        raise FileNotFoundError(f"No release assets matched {patterns}")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, asset, args.output, credential, args.attempts): asset["name"]
            for asset in assets
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    print(json.dumps({"assets": len(results), "bytes": sum(item["bytes"] for item in results)}, indent=2))


if __name__ == "__main__":
    main()
