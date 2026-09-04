"""Pilot power analysis: fix N, the equivalence margin, the block length,
and the axis point counts before the main run, from a small run on a
main-narrative cell.  Pilot data is used only to decide; it is written
under pilot/ so the aggregator never reads it, and the main run starts
over on a disjoint seed band.

Run:  python -m duel.pilot --cell "E-outage x F1" --seeds 1000 1001 1002
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .design import eps_for, in_band
from .flows import make_flows
from .gate import envs_for
from .inject import parse_cell
from .report import envelope, write_once
from .run import block_stats, run_chain, run_outage

Z95 = 1.96
SAFETY = 1.5


def episode_sd(ep_means, n_boot: int = 2000, seed: int = 0) -> dict:
    """Standard deviation of the per-episode mean difference, with a
    bootstrap upper bound (sizing on the upper bound guards against a
    variance underestimate)."""
    x = np.asarray(ep_means, dtype=float)
    s = float(np.std(x, ddof=1))
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = [float(np.std(x[rng.integers(0, n, n)], ddof=1)) for _ in range(n_boot)]
    return dict(point=s, upper95=float(np.quantile(boots, 0.975)))


def autocorr(series, max_lag: int = 10) -> list[float]:
    """Autocorrelation of a series at lags 1..max_lag."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = []
    for lag in range(1, max_lag + 1):
        if denom == 0 or lag >= len(x):
            out.append(0.0)
        else:
            out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return out


def block_len_from_acf(acf, threshold: float = 0.05) -> int:
    """First lag whose autocorrelation falls below the threshold; the
    block length in episodes."""
    for lag, a in enumerate(acf, start=1):
        if abs(a) < threshold:
            return lag
    return len(acf)


def sample_size(
    s_hi: float, eps: float, ep_size: int, safety: float = SAFETY, z: float = Z95
) -> dict:
    """Episodes k and payments needed for a CI half-width at or below
    eps/2, then multiplied by the safety factor."""
    k = (2.0 * z * s_hi / eps) ** 2
    k = int(np.ceil(k * safety))
    return dict(n_episodes=k, n_payments=k * ep_size)


def pick_K(curve) -> int:
    """Point count for an axis: 6 by default, 9 where the curve bends
    sharply relative to its range."""
    c = np.asarray(curve, dtype=float)
    if len(c) < 3:
        return 6
    sec = np.abs(c[:-2] - 2 * c[1:-1] + c[2:])
    rng = float(c.max() - c.min()) or 1.0
    return 9 if float(sec.max()) / rng > 0.5 else 6


def _coarse_K(cell, seed, n_eval, n_tune, cw, axis="lambda") -> int:
    """A coarse three-point sweep to gauge the curve's bend for pick_K."""
    from .inject import _chain_ctx, _chain_diff, _outage_ctx, _outage_diff

    env_name, flow_name = parse_cell(cell)
    kind, env, rho = envs_for(cw)[env_name]
    flow = make_flows()[flow_name]
    if kind == "chain":
        ctx = _chain_ctx(env, rho, flow, seed, n_tune, n_eval)
        ctx["seed"] = seed
        diff = _chain_diff
    else:
        ctx = _outage_ctx(env, flow, seed, n_tune, n_eval)
        ctx["seed"] = seed
        diff = _outage_diff
    levels = [0.5, 1.0, 2.0]
    curve = [float(diff(ctx, axis, lv).mean()) for lv in levels]
    return pick_K(curve)


def run_pilot(
    cell: str, seeds, n_eval: int, n_tune: int, out_dir: str, cw: str = "mid", k_sweep: bool = True
) -> dict:
    """Run the pilot, decide the design parameters, and write everything
    under out_dir/pilot/."""
    for s in seeds:
        if not in_band(s, "pilot"):
            raise ValueError(f"seed {s} is not in the pilot band")
    env_name, flow_name = parse_cell(cell)
    kind, env, rho = envs_for(cw)[env_name]
    flow = make_flows()[flow_name]

    ep_means: list = []
    exposures: list = []
    ep_size = 0
    first_series = None
    for s in seeds:
        if kind == "chain":
            out, d, _, episodes = run_chain(env, rho, flow, n_tune, n_eval, s)
        else:
            out, d, _, episodes = run_outage(env, flow, n_tune, n_eval, s)
        sums, counts = block_stats(out["A"] - out["B1"], episodes)
        em = sums / counts
        ep_means.append(em)
        exposures.append(float(np.mean(d.v)))
        ep_size = int(np.median(counts))
        if first_series is None:
            first_series = em
    ep_all = np.concatenate(ep_means)
    sd = episode_sd(ep_all, seed=1000)
    mean_exposure = float(np.mean(exposures))
    eps = eps_for(mean_exposure)  # frozen anchor, not a pilot output
    size = sample_size(sd["upper95"], eps, ep_size)
    acf = autocorr(first_series)
    block_len = block_len_from_acf(acf)
    k_by_axis = {}
    if k_sweep:
        k_by_axis["lambda"] = _coarse_K(cell, seeds[0], n_eval, n_tune, cw)

    decisions = dict(
        eps_per_payment=eps,
        eps_per_1000=eps * 1000.0,
        n_eval=size["n_payments"],
        n_episodes=size["n_episodes"],
        block_len=block_len,
        ep_size=ep_size,
        K=k_by_axis,
    )
    summary = dict(
        pilot_cell=cell,
        episode_sd=sd,
        autocorr=acf,
        decisions=decisions,
        safety_factor=SAFETY,
        mean_exposure=mean_exposure,
    )
    pilot_dir = Path(out_dir) / "pilot"
    resolved = dict(cell=cell, cw=cw, seeds=list(seeds))
    for s in seeds:
        obj = envelope(
            "pilot",
            cell,
            s,
            n_eval,
            n_tune,
            resolved,
            dict(note="pilot draw, excluded from the main analysis"),
        )
        write_once(str(pilot_dir / f"pilot_{env_name}_{flow_name}_s{s}.json"), obj)
    write_once(str(pilot_dir / "pilot_summary.json"), summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry: run the pilot and print the decisions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="E-outage x F1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--cw", default="mid", choices=["high", "mid", "low"])
    ap.add_argument("--n-eval", type=int, default=20_000)
    ap.add_argument("--n-tune", type=int, default=10_000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    summary = run_pilot(args.cell, args.seeds, args.n_eval, args.n_tune, args.out, cw=args.cw)
    import json

    print(json.dumps(summary["decisions"], indent=1))


if __name__ == "__main__":
    main()
