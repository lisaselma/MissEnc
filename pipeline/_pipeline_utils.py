"""Shared pipeline utilities.

Tiny helpers that were copy-pasted across `main.py`, `00a_random_sampler.py`,
`02_tapas_input_prep.py`, and others. Centralised here so the pipeline files
only contain task-specific logic.

Nothing in this module touches network/disk beyond the obvious helpers
(`sha256`, `atomic_write_json`, `load_sibling_module`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PIPELINE_DIR.parent


def load_sibling_module(filename: str, alias: str | None = None) -> Any:
    """Import a sibling pipeline module by filename.

    Step files like `01_table_cleaning.py` start with a digit and can't
    be imported with a normal `import` statement, hence the importlib
    dance. The optional ``alias`` lets the caller pick a stable module
    name (defaults to the filename without `.py`).
    """
    path = PIPELINE_DIR / filename
    mod_name = alias or filename.rsplit(".", 1)[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load sibling module: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so tools that look the module up by name (e.g.
    # dataclasses resolving annotations via sys.modules[cls.__module__])
    # find it. Required for modules that define @dataclass classes.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    """Stream-hash a file with SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    """ISO-8601 timestamp in UTC. Used for run metadata files."""
    return datetime.now(timezone.utc).isoformat()


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    """Short git HEAD sha for traceability. Returns 'unknown' on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write `payload` to `path` via `<path>.tmp` rename so partial writes
    are never visible to consumers."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
