"""Utility helpers for deterministic numerics and serialization.

Doctrine reference is centralized in constants.DOCTRINE.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def runtime_meta() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_config": np.__config__.show.__module__,
        "pid": os.getpid(),
    }
