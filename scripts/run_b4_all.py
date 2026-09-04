"""Run B4 across the nine confirmatory cells, skipping any already done.

Cell-to-seed follows the main run's assignment so B4 sits on the same
draws as its base cell.  Each cell writes results/b4_{tag}.json; a cell
whose file already exists is skipped, so the driver is resumable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from duel.b4 import main as b4_main

CELL_SEED = [
    ("E-fast", "F1", 1),
    ("E-fast", "F2", 2),
    ("E-fast", "F3", 3),
    ("E-slow", "F1", 4),
    ("E-slow", "F2", 5),
    ("E-slow", "F3", 6),
    ("E-outage", "F1", 7),
    ("E-outage", "F2", 8),
    ("E-outage", "F3", 9),
]


def run(out="results"):
    for env, flow, seed in CELL_SEED:
        tag = f"{env}_{flow}_mid_s{seed}"
        path = Path(out) / f"b4_{tag}.json"
        if path.exists():
            print(f"skip {tag} (exists)", flush=True)
            continue
        print(f"=== {tag} ===", flush=True)
        b4_main(["--env", env, "--flow", flow, "--seed", str(seed), "--out", out])


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "results")
