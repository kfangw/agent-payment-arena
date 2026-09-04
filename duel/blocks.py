"""T2: outage-event block bootstrap for the E-outage margin.

The bootstrap resamples 50-payment episodes.  If a single outage lasted
longer than an episode its payments would spread across blocks and the
interval would come out too narrow.  This module first measures the outage
duration distribution (the number of outage events is the effective sample
for the margin), then re-aggregates the three E-outage A minus B4 intervals
under enlarged fixed blocks and under a moving block of several episodes,
keeping the nine-cell simultaneous level.  The point estimate is a
rearrangement of the same sums and does not move; only the interval can.

    python -m duel.blocks --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .gate import envs_for
from .grid_aggregate import N_BOOT, SIM_LEVEL, build as grid_build
from .design import eps_for
from .naming import canon_keys
from .stats import boot_ci, holm, perm_p, ratio_mean, units, verdict

OUTAGE = ["E-outage x F1", "E-outage x F2", "E-outage x F3"]
CELL_SEED = {"E-outage x F1": 7, "E-outage x F2": 8, "E-outage x F3": 9}
EPISODE_LEN, PPE = 1440, 50


def outage_runs(seed, n_ep, env, chunk=4000):
    """Maximal consecutive r=1 run lengths (ticks) over n_ep independent
    episode regime paths, generated as a vectorized two-state Markov chain
    at the cell's (p01, p10).  Regenerated from `seed`; the durations are a
    property of the regime process, not of a particular payment draw."""
    L = EPISODE_LEN + env.H + 1
    rng = np.random.default_rng(seed)
    runs = []
    n_out = 0
    done = 0
    while done < n_ep:
        m = min(chunk, n_ep - done)
        path = np.empty((m, L), dtype=np.int8)
        path[:, 0] = (rng.random(m) < env.stationary_outage).astype(np.int8)
        u = rng.random((m, L - 1))
        for t in range(1, L):
            prev = path[:, t - 1]
            p = np.where(prev == 0, env.p01, env.p10)
            path[:, t] = np.where(u[:, t - 1] < p, 1 - prev, prev)
        # maximal r=1 run lengths across the chunk
        for row in path:
            d = np.diff(np.concatenate(([0], row, [0])))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            if len(starts):
                runs.extend((ends - starts).tolist())
                n_out += len(starts)
        done += m
    runs = np.asarray(runs)
    return dict(
        n_events=int(n_out),
        n_episodes=int(n_ep),
        median=float(np.median(runs)) if len(runs) else 0.0,
        p90=float(np.quantile(runs, 0.90)) if len(runs) else 0.0,
        p99=float(np.quantile(runs, 0.99)) if len(runs) else 0.0,
        max=int(runs.max()) if len(runs) else 0,
        episode_ticks=EPISODE_LEN,
    )


def _group(sums, counts, g):
    """Sum consecutive blocks in groups of g (fixed enlarged block)."""
    n = len(sums)
    idx = np.arange(n) // g
    return np.bincount(idx, weights=sums), np.bincount(idx, weights=counts)


def _moving_ci(sums, counts, b, seed, level, n_boot):
    """Moving-block bootstrap CI of the paired mean: resample ceil(n/b)
    overlapping blocks of b consecutive episodes."""
    n = len(sums)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / b))
    starts_all = n - b + 1
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    means = np.empty(n_boot)
    # precompute block prefix sums for speed
    csum = np.concatenate(([0.0], np.cumsum(sums)))
    ccnt = np.concatenate(([0.0], np.cumsum(counts)))
    for i in range(n_boot):
        st = rng.integers(0, starts_all, size=nb)
        s = (csum[st + b] - csum[st]).sum()
        c = (ccnt[st + b] - ccnt[st]).sum()
        means[i] = s / c
    return float(np.quantile(means, lo_q)), float(np.quantile(means, hi_q))


def _diff_blocks(cell, results):
    seed = CELL_SEED[cell]
    e, f = (s.strip() for s in cell.split("x"))
    base = json.load(open(Path(results) / f"duel_{e}_{f}_mid_s{seed}.json"))
    grid = json.load(open(Path(results) / f"b4_gridN_{e}_{f}_mid_s{seed}.json"))
    a2 = np.asarray(canon_keys(base["payload"]["policies"])["A"]["block_sums"], dtype=float)
    b4 = np.asarray(grid["payload"]["b4_block_sums"], dtype=float)
    counts = np.asarray(grid["payload"]["block_counts"], dtype=float)
    mexp = grid["payload"]["mean_exposure"]
    return a2 - b4, counts, mexp, seed


def build(results="results", g_list=(4, 16), b_moving=8):
    # chain cells' A-B4 perm p stays at the episode block (they carry no
    # outage), taken from the S1 aggregation for the Holm family.
    base_rows = {r["cell"]: r for r in grid_build(results, results)}
    chain_p = {c: base_rows[c]["perm_p"] for c in base_rows if c not in OUTAGE}

    out = {}
    for cell in OUTAGE:
        diff, counts, mexp, seed = _diff_blocks(cell, results)
        mean = ratio_mean(diff, counts)
        eps = eps_for(mexp)
        schemes = {}
        # episode block (current)
        schemes["episode"] = dict(
            ci=list(boot_ci(diff, counts, n_boot=N_BOOT, seed=seed, level=SIM_LEVEL)),
            perm=perm_p(diff, counts, seed=seed + 1),
        )
        # enlarged fixed blocks
        for g in g_list:
            gs, gc = _group(diff, counts, g)
            schemes[f"fixed_x{g}"] = dict(
                ci=list(boot_ci(gs, gc, n_boot=N_BOOT, seed=seed, level=SIM_LEVEL)),
                perm=perm_p(gs, gc, seed=seed + 1),
            )
        # moving block of b episodes
        schemes[f"moving_{b_moving}"] = dict(
            ci=list(_moving_ci(diff, counts, b_moving, seed, SIM_LEVEL, N_BOOT)),
            perm=perm_p(*_group(diff, counts, b_moving), seed=seed + 1),
        )
        out[cell] = dict(
            cell=cell, mean=mean, bp=units(mean, mexp)["bp"], eps=eps, seed=seed, schemes=schemes
        )

    # verdicts: Holm each scheme's outage perm p with the chain family
    for scheme in next(iter(out.values()))["schemes"]:
        pvals = [chain_p[c] for c in chain_p] + [out[c]["schemes"][scheme]["perm"] for c in OUTAGE]
        adj = holm(pvals)
        adj_out = adj[len(chain_p) :]
        for c, ph in zip(OUTAGE, adj_out):
            s = out[c]["schemes"][scheme]
            s["holm_p"] = ph
            s["verdict"] = verdict(s["ci"][0], s["ci"][1], ph, out[c]["eps"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    print(f"== outage duration (ticks; episode spans {EPISODE_LEN} ticks = {PPE} payments) ==")
    for cell in OUTAGE:
        seed = CELL_SEED[cell]
        _, env, _ = envs_for("mid")["E-outage"]
        # n_ep matches the confirmatory eval (5,663,400 / 50)
        st = outage_runs(seed, 5_663_400 // PPE, env)
        print(
            f"{cell}: events={st['n_events']} over {st['n_episodes']} episodes "
            f"| median={st['median']:.0f} p90={st['p90']:.0f} p99={st['p99']:.0f} "
            f"max={st['max']} ticks  (p99 = {st['p99'] / EPISODE_LEN:.3f} episode)"
        )

    out = build(args.results)
    print(f"\n== A-B4 under re-blocking (nine-cell simultaneous {SIM_LEVEL:.4f}) ==")
    schemes = list(next(iter(out.values()))["schemes"])
    for cell in OUTAGE:
        r = out[cell]
        print(f"\n{cell}: mean={r['mean']:.5f} ({r['bp']:.1f} bp), eps={r['eps']:.6f}")
        for s in schemes:
            sc = r["schemes"][s]
            print(
                f"  {s:11} ci=[{sc['ci'][0]:.6f}, {sc['ci'][1]:.6f}] "
                f"holm_p={sc['holm_p']:.4f}  {sc['verdict']}"
            )


if __name__ == "__main__":
    main()
