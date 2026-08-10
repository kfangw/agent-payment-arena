#!/usr/bin/env python3
"""Fail if any tracked text file contains Hangul.

The repository is English only: code, comments, docs, commit messages, and
program output. This hook keeps that from eroding one convenient note at a
time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Codepoint ranges rather than a character class, so this file does not fail
# its own check.
HANGUL_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
)

BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".woff", ".woff2", ".zip"}
)


def _has_hangul(line: str) -> bool:
    return any(low <= ord(character) <= high for character in line for low, high in HANGUL_RANGES)


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def _offending_lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if _has_hangul(line)
    ]


def main(argv: list[str]) -> int:
    """Check the given paths, or every tracked file when none are given.

    Args:
        argv: Paths to check, excluding the program name.

    Returns:
        A process exit status: 0 when clean, 1 when Hangul was found.
    """
    paths = [Path(arg) for arg in argv] if argv else _tracked_files()

    failed = False
    for path in paths:
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        for number, line in _offending_lines(path):
            print(f"{path}:{number}: non-English text: {line.strip()}", file=sys.stderr)
            failed = True

    if failed:
        print("\nThis repository is English only.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
