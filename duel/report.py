"""One result envelope for every run, so a file carries what it took to
make it: the resolved parameters (hashed), the code revision, and a
timestamp.  Every writer in the package wraps its payload with
`envelope` and commits it through `write_once`, which refuses to
overwrite a result that already exists.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def jsonable(x: object) -> object:
    """Coerce numpy scalars, arrays, and nested containers to plain JSON
    types so a payload serialises and hashes deterministically."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    return x


def params_hash(resolved: dict) -> str:
    """sha256 of the resolved parameter dict, sorted so key order does not
    change the digest."""
    blob = json.dumps(jsonable(resolved), sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def git_rev() -> str | None:
    """Current commit hash, or None outside a repository."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip()
        return rev or None
    except Exception:  # noqa: BLE001  (metadata, never fatal)
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def envelope(kind: str, cell: str, seed: int, n_eval: int, n_tune: int,
             resolved: dict, payload: object) -> dict:
    """Wrap a payload in the common result fields (spec 1.4)."""
    return dict(
        kind=kind, cell=cell, seed=seed, n_eval=n_eval, n_tune=n_tune,
        params_hash=params_hash(resolved), code=git_rev(),
        created_utc=_utc_now(), payload=jsonable(payload),
    )


def write_once(path: str, obj: object) -> None:
    """Write one JSON file, refusing to overwrite an existing result."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"result already exists: {path}")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(jsonable(obj), fh, indent=1)
