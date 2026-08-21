"""Aggregate the nine confirmatory cells into the summary and the tables.

Reads the duel envelopes in a results directory (never a pilot/ subtree),
forms every paired comparison by subtracting episode block sums, applies
the bootstrap interval and the sign-flip permutation p, corrects the nine
A2 - B1 comparisons with Holm, and renders three-tier tables under the
denominator rule.  It refuses to run on fewer than the nine cells.

Run:  python -m duel.aggregate --results results --out tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .design import EPS as DEFAULT_EPS, NINE_CELLS
from .report import write_once
from .stats import boot_ci, holm, perm_p, ratio_mean, units, verdict

# Comparison layout (spec 3.7): each pair is (minuend, subtrahend).
CONFIRMATORY = ("A2", "B1")
MAIN_ROWS = ("A2", "B1", "A1")
SUB1 = [("A2", "B2"), ("A2", "B3"), ("A2", "A2v"), ("A2", "A2w")]
SUB2 = [("A1", "A2")]
APPENDIX = [("A1", "C1"), ("A1", "C2"), ("A1", "C3"), ("A1", "C4")]


def load_cells(results_dir: str) -> dict:
    """Merge the envelopes per cell.  Multiple seeds for one cell pool by
    concatenating episode blocks; policies stay aligned because every
    policy in a file shares that file's episode partition."""
    cells: dict = {}
    for path in sorted(Path(results_dir).glob("duel_*.json")):
        env = json.loads(path.read_text())
        if env.get("kind") != "duel":
            continue
        cell = env["cell"]
        pay = env["payload"]
        rec = cells.setdefault(cell, dict(sums={}, counts=[], exp_num=0.0,
                                          exp_den=0.0, files=0))
        counts = np.asarray(pay["block_counts"], dtype=float)
        rec["counts"].append(counts)
        for name, obj in pay["policies"].items():
            rec["sums"].setdefault(name, []).append(
                np.asarray(obj["block_sums"], dtype=float))
        tot = float(counts.sum())
        rec["exp_num"] += pay["mean_exposure"] * tot
        rec["exp_den"] += tot
        rec["files"] += 1
    # concatenate the per-file blocks into one series per cell
    for rec in cells.values():
        rec["counts"] = np.concatenate(rec["counts"])
        rec["sums"] = {k: np.concatenate(v) for k, v in rec["sums"].items()}
        rec["mean_exposure"] = rec["exp_num"] / rec["exp_den"]
        rec["n_episodes"] = int(len(rec["counts"]))
    return cells


def _compare(rec: dict, a: str, b: str, seed: int) -> dict:
    """One paired comparison a - b on a cell's blocks."""
    diff = rec["sums"][a] - rec["sums"][b]
    counts = rec["counts"]
    mean = ratio_mean(diff, counts)
    lo, hi = boot_ci(diff, counts, seed=seed)
    p = perm_p(diff, counts, seed=seed + 1)
    u = units(mean, rec["mean_exposure"])
    return dict(minuend=a, subtrahend=b, mean=mean, ci95=[lo, hi],
                p_raw=p, per_1000=u["per_1000"], bp=u["bp"])


def analyze(cells: dict, eps: float) -> dict:
    """Build the full summary: per-cell means, every comparison, and the
    Holm-corrected confirmatory verdicts."""
    missing = [c for c in NINE_CELLS if c not in cells]
    if missing:
        raise ValueError(f"missing cells, refusing partial aggregate: {missing}")

    per_cell: dict = {}
    conf_p: list[float] = []
    for i, cell in enumerate(NINE_CELLS):
        rec = cells[cell]
        seed = 9000 + 10 * i
        a, b = CONFIRMATORY
        conf = _compare(rec, a, b, seed)
        conf_p.append(conf["p_raw"])
        means = {k: ratio_mean(v, rec["counts"]) for k, v in rec["sums"].items()}
        per_cell[cell] = dict(
            means=means, mean_exposure=rec["mean_exposure"],
            n_episodes=rec["n_episodes"], confirmatory=conf,
            sub1=[_compare(rec, x, y, seed + 2 + 2 * j)
                  for j, (x, y) in enumerate(SUB1)],
            sub2=[_compare(rec, x, y, seed + 20 + 2 * j)
                  for j, (x, y) in enumerate(SUB2)],
            appendix=[_compare(rec, x, y, seed + 30 + 2 * j)
                      for j, (x, y) in enumerate(APPENDIX)],
        )
    p_holm = holm(conf_p)
    for cell, ph in zip(NINE_CELLS, p_holm):
        c = per_cell[cell]["confirmatory"]
        c["p_holm"] = ph
        c["verdict"] = verdict(c["ci95"][0], c["ci95"][1], ph, eps)
    return dict(eps=eps, cells=per_cell, n_cells=len(NINE_CELLS))


# ------------------------------------------------------------ rendering
def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_md(summary: dict) -> str:
    """Three-tier markdown tables under the denominator rule (spec 3.7)."""
    eps = summary["eps"]
    lines = [f"# Confirmatory comparison (eps = {eps} $/payment)", ""]
    lines += ["## Main table (advantage = A2 - B1, dollars per 1000)", "",
              "| cell | A2 | B1 | A1 | adv/1000 | ci95/1000 | p_holm | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for cell in NINE_CELLS:
        c = summary["cells"][cell]
        m = c["means"]
        conf = c["confirmatory"]
        ci = [conf["ci95"][0] * 1000, conf["ci95"][1] * 1000]
        lines.append(
            f"| {cell} | {_fmt(m['A2'])} | {_fmt(m['B1'])} | {_fmt(m['A1'])} "
            f"| {_fmt(conf['per_1000'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] "
            f"| {_fmt(conf['p_holm'], 4)} | {conf['verdict']} |")

    lines += ["", "## Subtable 1 (tuned B family and eliminated A family)",
              "", "| cell | A2-B2/1000 | A2-B3/1000 | A2-A2v/1000 | A2-A2w/1000 |",
              "|---|---|---|---|---|"]
    for cell in NINE_CELLS:
        cols = [f"{r['per_1000']:.3f}" for r in summary["cells"][cell]["sub1"]]
        lines.append(f"| {cell} | " + " | ".join(cols) + " |")

    lines += ["", "## Subtable 2 (identification loss, A1 - A2, per 1000)", "",
              "| cell | A1-A2/1000 | ci95/1000 |", "|---|---|---|"]
    for cell in NINE_CELLS:
        r = summary["cells"][cell]["sub2"][0]
        ci = [r["ci95"][0] * 1000, r["ci95"][1] * 1000]
        lines.append(f"| {cell} | {_fmt(r['per_1000'])} "
                     f"| [{_fmt(ci[0])}, {_fmt(ci[1])}] |")

    lines += ["", "## Appendix (C family, denominator A1, per 1000)", "",
              "| cell | A1-C1 | A1-C2 | A1-C3 | A1-C4 |", "|---|---|---|---|---|"]
    for cell in NINE_CELLS:
        cols = [f"{r['per_1000']:.3f}" for r in summary["cells"][cell]["appendix"]]
        lines.append(f"| {cell} | " + " | ".join(cols) + " |")
    return "\n".join(lines) + "\n"


def render_tex(summary: dict) -> str:
    """The main table as a tex tabular; the same numbers as the markdown."""
    lines = [r"\begin{tabular}{lrrrrl}", r"\hline",
             r"cell & A2 & B1 & adv/1000 & $p_{\mathrm{holm}}$ & verdict \\",
             r"\hline"]
    for cell in NINE_CELLS:
        c = summary["cells"][cell]
        m, conf = c["means"], c["confirmatory"]
        lines.append(
            f"{cell} & {m['A2']:.3f} & {m['B1']:.3f} & {conf['per_1000']:.3f} "
            f"& {conf['p_holm']:.4f} & {conf['verdict']} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """CLI entry: read results, write summary.json and the tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--out", default="tables")
    args = ap.parse_args(argv)
    cells = load_cells(args.results)
    summary = analyze(cells, args.eps)
    out = Path(args.out)
    write_once(str(out / "summary.json"), summary)   # refuses to overwrite
    (out / "tables.md").write_text(render_md(summary))
    (out / "tables.tex").write_text(render_tex(summary))
    print(f"aggregated {summary['n_cells']} cells -> {out}/summary.json, "
          f"tables.md, tables.tex")


if __name__ == "__main__":
    main()
