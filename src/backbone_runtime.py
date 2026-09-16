"""Atomic checkpointing and process checks for local training jobs."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(cpu_copy(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def process_matches(pid, token: str) -> bool:
    if not pid:
        return False
    result = subprocess.run(["ps", "-p", str(int(pid)), "-o", "stat=,command="],
                            capture_output=True, text=True)
    status = result.stdout.strip()
    return bool(status and not status.startswith("Z") and token in status)


def progress(out: Path, **fields) -> None:
    atomic_json(out / "progress.json", {"updated_at_unix": time.time(),
                                        "pid": os.getpid(), **fields})
