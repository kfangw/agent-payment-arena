"""Runner convention: a result may only be produced from committed code.

Every producing runner calls require_clean_tree() at its entry.  If the
working tree carries any tracked change the runner exits non-zero rather
than write a result, so the `code` field the envelope records
(git rev-parse HEAD) is exactly the commit that made the file.  There is
no bypass flag: the only way past the check is to commit.
"""
from __future__ import annotations

import subprocess
import sys


def head() -> str:
    """Current commit hash."""
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()


def _porcelain() -> str:
    return subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                          text=True, check=False).stdout.strip()


def require_clean_tree() -> str:
    """Refuse to run on a dirty working tree; otherwise return HEAD.

    Result files are ignored, so a run does not dirty the tree; only an
    uncommitted code change does, and that is exactly what must not produce
    a result whose recorded hash would not rebuild it."""
    dirty = _porcelain()
    if dirty:
        print("refusing to run: working tree is dirty; commit the code first "
              "so the result records the commit that made it (T4).\n" + dirty,
              file=sys.stderr)
        raise SystemExit(1)
    return head()
