"""Acceptance checks for the pilot power analysis (spec 5.8).

Run:  python -m arena.experiments.settlement.validate_pilot
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .aggregate import load_cells
from .design import band_of, in_band
from .pilot import episode_sd, pick_K, run_pilot, sample_size
from .stats import boot_ci


def a7_1_sizing_hits_target() -> None:
    """A7-1: a run at the sized N has a CI half-width at or below eps/2."""
    rng = np.random.default_rng(1)
    s_true, eps, ep_size = 0.5, 0.05, 50
    sd = episode_sd(s_true + 0.0 * rng.standard_normal(200))  # placeholder
    # size from a known SD upper bound
    size = sample_size(s_true, eps, ep_size)
    k = size["n_episodes"]
    # generate k episodes of per-episode mean ~ N(0, s_true^2)
    ep_mean = s_true * rng.standard_normal(k)
    counts = np.full(k, ep_size, dtype=float)
    sums = ep_mean * ep_size
    lo, hi = boot_ci(sums, counts, seed=2)
    half = (hi - lo) / 2.0
    assert half <= eps / 2.0 + 1e-9, f"half-width {half} exceeds {eps / 2}"
    assert sd["point"] >= 0.0
    print(f"A7-1 ok: k={k} gives half-width {half:.4f} <= {eps / 2:.4f}")


def pick_K_bends() -> None:
    """pick_K is 6 for a straight curve, 9 for a sharp bend."""
    straight = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    bent = [1.0, 1.0, 1.0, 0.2, 0.2, 0.2]
    assert pick_K(straight) == 6, pick_K(straight)
    assert pick_K(bent) == 9, pick_K(bent)
    print("pick_K ok: 6 for straight, 9 for a bend")


def a7_2_aggregator_skips_pilot() -> None:
    """A7-2: run_pilot writes under pilot/, which the aggregator ignores."""
    with tempfile.TemporaryDirectory() as d:
        summary = run_pilot("E-outage x F1", [1000, 1001], 3_000, 2_000, d, k_sweep=False)
        pilot_dir = Path(d) / "pilot"
        assert (pilot_dir / "pilot_summary.json").exists()
        assert list(pilot_dir.glob("pilot_*_s1000.json"))
        # a results dir that only holds the pilot subtree yields no cells
        cells = load_cells(d)
        assert cells == {}, f"pilot files leaked into aggregator input: {cells}"
        dec = summary["decisions"]
        assert dec["n_eval"] > 0 and dec["block_len"] >= 1
        print(
            f"A7-2 ok: pilot under pilot/, aggregator sees 0 cells; "
            f"block_len={dec['block_len']} n_eval={dec['n_eval']}"
        )


def a7_3_seed_bands_disjoint() -> None:
    """A7-3: pilot seeds are guarded, and the bands do not overlap."""
    # a main-band seed is refused by the pilot
    with tempfile.TemporaryDirectory() as d:
        try:
            run_pilot("E-outage x F1", [1], 2_000, 1_000, d, k_sweep=False)
        except ValueError:
            pass
        else:
            raise AssertionError("pilot accepted a non-pilot seed")
    assert in_band(1000, "pilot") and not in_band(1000, "exp1")
    assert band_of(1) == "exp1" and band_of(2001) == "exp2"
    assert band_of(1500) == "pilot" and band_of(9001) == "verify"
    # pilot band shares no seed with exp1 or exp2
    from .design import SEED_BANDS

    p = set(range(*(SEED_BANDS["pilot"][0], SEED_BANDS["pilot"][1] + 1)))
    e1 = set(range(*(SEED_BANDS["exp1"][0], SEED_BANDS["exp1"][1] + 1)))
    e2 = set(range(*(SEED_BANDS["exp2"][0], SEED_BANDS["exp2"][1] + 1)))
    assert not (p & e1) and not (p & e2)
    print("A7-3 ok: pilot guard rejects main seeds; bands disjoint")


def main() -> None:
    """Run every acceptance check; raise on the first failure."""
    a7_1_sizing_hits_target()
    pick_K_bends()
    a7_2_aggregator_skips_pilot()
    a7_3_seed_bands_disjoint()
    print("\nS7 pilot: all checks passed")


if __name__ == "__main__":
    main()
