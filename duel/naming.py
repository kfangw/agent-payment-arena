"""Policy family names.

The manuscript writes A, A\\V, A\\W.  The code and result files use
A_full/A/A_noV/A_noW; A_full (the true-arrival oracle, formerly A1) stays in
results but is dropped from the manuscript tables.  Result files written
before the rename carry the old keys A1/A2/A2v/A2w; canon() maps them
forward so old and new files aggregate together without rewriting them.
"""

from __future__ import annotations

NEW = {"A1": "A_full", "A2": "A", "A2v": "A_noV", "A2w": "A_noW"}
OLD = {v: k for k, v in NEW.items()}
TABLE_EXCLUDE = {"A_full"}  # A1: kept in results, excluded from tables


def canon(name: str) -> str:
    """Old family key -> canonical name; anything else unchanged."""
    return NEW.get(name, name)


def canon_keys(d: dict) -> dict:
    """Canonicalize the family keys of a policies/means dict (read compat)."""
    return {canon(k): v for k, v in d.items()}
