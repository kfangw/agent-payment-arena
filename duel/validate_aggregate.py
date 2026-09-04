"""Acceptance checks for the aggregator (spec 3.9, and A7-2 exclusion).

Run:  python -m duel.validate_aggregate
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .aggregate import analyze, load_cells, main, render_md, render_tex
from .design import NINE_CELLS
from .report import envelope, write_once

POLICIES = ("A1", "A2", "A2v", "A2w", "B1", "B2", "B3", "C1", "C2", "C3", "C4")


def _write_cell(results: Path, cell: str, seed: int, advantage: float) -> None:
    """Fabricate one duel envelope with block sums that put A2 above B1 by
    `advantage` dollars per payment on average."""
    rng = np.random.default_rng(seed)
    n_ep, per = 60, 1000
    counts = np.full(n_ep, per)
    base = {p: rng.standard_normal(n_ep) * per * 0.01 for p in POLICIES}
    base["A2"] = base["B1"] + advantage * per + rng.standard_normal(n_ep) * per * 0.002
    payload = dict(
        cw="mid",
        block_counts=[int(c) for c in counts],
        policies={p: dict(block_sums=[float(x) for x in base[p]]) for p in POLICIES},
        n_episodes=n_ep,
        mean_exposure=50.0,
        means={p: float(base[p].sum() / counts.sum()) for p in POLICIES},
        calib={},
    )
    env, flow = cell.split(" x ")
    obj = envelope("duel", cell, seed, n_ep * per, 50_000, dict(cell=cell), payload)
    write_once(str(results / f"duel_{env}_{flow}_mid_s{seed}.json"), obj)


def full_nine_render() -> None:
    """A full nine-cell directory produces a summary and non-empty tables."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results"
        for i, cell in enumerate(NINE_CELLS):
            _write_cell(results, cell, 2000 + i, advantage=0.004)
        cells = load_cells(str(results))
        assert set(cells) == set(NINE_CELLS)
        summary = analyze(cells, eps=0.005)
        assert summary["n_cells"] == 9
        for cell in NINE_CELLS:
            c = summary["cells"][cell]["confirmatory"]
            assert "p_holm" in c and "verdict" in c
            assert c["p_holm"] >= c["p_raw"] - 1e-12  # holm never shrinks p
        md, tex = render_md(summary), render_tex(summary)
        assert "Main table" in md and "Subtable 1" in md and "Appendix" in md
        assert r"\begin{tabular}" in tex
        # advantage 0.004 $/payment = 4 $/1000; verdict should be superior
        v = [summary["cells"][c]["confirmatory"]["verdict"] for c in NINE_CELLS]
        assert v.count("superior") >= 7, v
        print(f"aggregate ok: 9 cells, verdicts {sorted(set(v))}")


def refuses_partial() -> None:
    """Fewer than nine cells refuses with the missing cells named."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results"
        for i, cell in enumerate(NINE_CELLS[:-1]):  # drop one
            _write_cell(results, cell, 2000 + i, advantage=0.004)
        cells = load_cells(str(results))
        try:
            analyze(cells, eps=0.005)
        except ValueError as exc:
            assert NINE_CELLS[-1] in str(exc), exc
            print(f"aggregate ok: refuses partial ({NINE_CELLS[-1]} named)")
            return
        raise AssertionError("partial aggregate was not refused")


def excludes_pilot_dir() -> None:
    """A7-2: files under a pilot/ subdirectory are not read as input."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results"
        for i, cell in enumerate(NINE_CELLS):
            _write_cell(results, cell, 2000 + i, advantage=0.004)
        pilot = results / "pilot"
        _write_cell(pilot, NINE_CELLS[0], 1000, advantage=99.0)  # pilot band
        cells = load_cells(str(results))
        # the pilot file (huge advantage) must not perturb the cell
        adv = summary_adv(cells, NINE_CELLS[0])
        assert adv < 1.0, f"pilot file leaked into input: adv={adv}"
        print("aggregate ok: pilot/ excluded from input")


def summary_adv(cells: dict, cell: str) -> float:
    from .stats import ratio_mean

    rec = cells[cell]
    return ratio_mean(rec["sums"]["A2"] - rec["sums"]["B1"], rec["counts"])


def cli_writes_files() -> None:
    """The CLI writes summary.json, tables.md, tables.tex."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results"
        for i, cell in enumerate(NINE_CELLS):
            _write_cell(results, cell, 2000 + i, advantage=0.004)
        out = Path(d) / "tables"
        main(["--results", str(results), "--eps", "0.005", "--out", str(out)])
        summ = json.loads((out / "summary.json").read_text())
        assert summ["n_cells"] == 9
        assert (out / "tables.md").exists() and (out / "tables.tex").exists()
        print("aggregate ok: CLI wrote summary.json, tables.md, tables.tex")


def main_checks() -> None:
    """Run every acceptance check; raise on the first failure."""
    full_nine_render()
    refuses_partial()
    excludes_pilot_dir()
    cli_writes_files()
    print("\nS5 aggregator: all checks passed")


if __name__ == "__main__":
    main_checks()
