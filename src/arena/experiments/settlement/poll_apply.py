"""Apply a poll summary to the channel constants.

The measured reorg bounds and regime rates do not overwrite the
environment automatically; this script rewrites the assignments in
gate.py so the change lands as a reviewable commit diff.  The E-fast
per-depth hazards take the rule-of-three upper bound (a conservative
choice makes the advantage claim a lower bound), and the E-outage
transition rates take the observed regime estimates.

Run:  python -m arena.experiments.settlement.poll_apply --summary poll/poll_summary_123.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

N_FAST = 40  # E-fast hazard array length [f_0..f_39]


def fast_hazards(summary: dict, submission_f0: float = 0.005) -> list[float]:
    """Length-N_FAST hazard array: f_0 is the submission-point failure,
    f_i (i>=1) is the depth-i reorg upper bound where observed, else the
    floor already in the file."""
    upper = {row["depth"]: row["upper95"] for row in summary.get("reorgs", [])}
    return [submission_f0] + [float(upper.get(i, 1e-6)) for i in range(1, N_FAST)]


def _fmt_array(values: list[float]) -> str:
    return "np.array([" + ", ".join(f"{v:.6g}" for v in values) + "])"


def rewrite(text: str, summary: dict) -> str:
    """Return gate.py source with the E-fast array and the E-outage
    transition rates replaced from the summary."""
    f_fast = fast_hazards(summary)
    new_fast = f"f_fast = {_fmt_array(f_fast)}"
    text, n1 = re.subn(r"f_fast = np\.array\([^\n]*\)", new_fast, text)
    if n1 != 1:
        raise ValueError(f"expected one f_fast assignment, found {n1}")

    reg = summary.get("regime", {})
    p01, p10 = float(reg["p01"]), float(reg["p10"])
    new_reg = f"p01={p01:.6g}, p10={p10:.6g}"
    text, n2 = re.subn(r"p01=[^,]+,\s*p10=[^)]+", new_reg, text)
    if n2 != 1:
        raise ValueError(f"expected one p01/p10 pair, found {n2}")
    return text


def apply_summary(
    summary_path: str,
    gate_path: str = "src/arena/experiments/settlement/gate.py",
    dry_run: bool = False,
) -> str:
    """Read a summary, rewrite gate.py (unless dry_run), return new text."""
    summary = json.loads(Path(summary_path).read_text())
    src = Path(gate_path).read_text()
    out = rewrite(src, summary)
    if not dry_run:
        Path(gate_path).write_text(out)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry: apply a poll summary to gate.py."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--gate", default="src/arena/experiments/settlement/gate.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    apply_summary(args.summary, args.gate, dry_run=args.dry_run)
    where = "(dry run, not written)" if args.dry_run else args.gate
    print(f"applied {args.summary} -> {where}")


if __name__ == "__main__":
    main()
