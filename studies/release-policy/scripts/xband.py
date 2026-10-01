"""B4-band: B4 with (k, a, b) tuned separately per exposure band.

Draws are generated exactly as in b4.py / run.py (same seed; tuning batch
first, then evaluation batch, from one default_rng(seed)), so evaluation
payments and block sums pair with the E-fitted main run (b4ref files,
refined 161-point grid, k in {0,1,2,4,8,16,30}) and with the compiled
policy's saved block sums (wrong-constants run, key "true").

Tuning: for every k and each terminal action (grant, reject, verify) the
forced watch policy is replayed once on the tuning draws (as tune_b4 does).
For a band, the payoff of candidate (k, a, b) is the sum over its payments of
grant if pi0 < a, verify if a <= pi0 <= b, reject if pi0 > b, computed with
cumulative sums over payments sorted by pi0.  argmax over the flattened
(k ascending, grid order) array returns the first maximum, which is B4's
tie rule (strict > in the same iteration order).

Evaluation: for every k chosen by any band (any band count) and every
action, the forced policy is replayed on the evaluation draws with a copy of
replay_outage that also records whether the resource was released (grant,
or verify answered by an honest payer with a positive grant leg; same
definition as security_replay.replay_outage_fields).  Each band policy's
per-payment payoff is then a selection among these arrays, which equals
replaying the band policy itself (the same argument as b4.py's tuning).
B4 (the b4ref parameters) is formed the same way and must reproduce the
saved b4ref block sums; this is the reproduction check.

Run with Python 3.11:
  PYTHONPATH=src python3 scripts/xband.py --flow F2 --seed 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from paths import EXPOSURE_BAND, REFERENCE_B4, WRONG_CONSTANTS
from arena.experiments.settlement.core import GRANT, REJECT, VERIFY
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (
    draw_outage_batch,
    replay_outage,
    survival,
    window_AD,
    _settle_from,
)
from arena.experiments.settlement.policies import suspicion_grid
from arena.experiments.settlement.watch import FixedActionOutageWatchPolicy as Force
from arena.experiments.settlement.stats import boot_ci, perm_p, ratio_mean

PPE = 50
ACTS = (("G", GRANT), ("R", REJECT), ("V", VERIFY))
MAIN = REFERENCE_B4          # b4ref_E-fitted_F*_mid_s*.json (shipped)
WRONG = WRONG_CONSTANTS      # wrong_E-fitted_F*_s*.json, key "true" (written by scripts/wrong_constants.py)
V_DECL = 49.50


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def replay_release(env, d, policy, ex_pos):
    """replay_outage plus a released flag (copy of the repo loop)."""
    from arena.experiments.settlement.core import WAIT

    n = len(d)
    N, H, FIN = env.N, env.H, env.N + 1
    out = np.zeros(n)
    rel = np.zeros(n, dtype=bool)
    for k in range(n):
        v, pi = float(d.v[k]), float(d.pi0[k])
        payoff = 0.0
        released = False
        i, t = 0, 0
        while True:
            if i > N:
                i = FIN
            l = H - t
            r = int(d.paths[k, t])
            if l <= 0 and i < FIN:
                break
            a = policy(i, max(l, 0), r, v, pi)
            if a == GRANT:
                won = 1 if i == FIN else _settle_from(env, d, k, i, t)
                payoff += v * ((env.m if d.theta[k] == 0 else -env.h) if won else -1.0)
                released = True
                break
            if a == REJECT:
                break
            if a == VERIFY:
                payoff -= env.C
                w = min(env.tau, max(l, 0))
                s = 0
                while s < w:
                    payoff -= env.cw * v
                    s += 1
                    if d.t_ans[k] == s:
                        li, lr = H - t, int(d.paths[k, t])
                        if d.theta[k] == 0 and ex_pos[i, li, lr] > 0:
                            won = 1 if i == FIN else _settle_from(env, d, k, i, t)
                            payoff += v * (env.m if won else -1.0)
                            released = True
                        break
                    if int(d.paths[k, t]) == 0 and i <= N:
                        if d.u_stage[k, i] < env.f[i]:
                            break
                        i += 1
                        if i > N:
                            i = FIN
                    t += 1
                    if H - t <= 0:
                        break
                break
            if a == WAIT:
                payoff -= env.cw * v
                if r == 0 and i <= N:
                    if d.u_stage[k, i] < env.f[i]:
                        break
                    i += 1
                t += 1
                continue
            break
        out[k] = payoff
        rel[k] = released
    return out, rel


def band_of(v, inner_edges):
    """Band j holds e_{j-1} < v <= e_j (inner edges from the tuning sample)."""
    return np.searchsorted(inner_edges, v, side="left")


def tune_band(T, pi, grid_a, grid_b, ks):
    """T[k_idx][act] payoff arrays restricted to one band. Returns
    (k, a, b), best total, per-k best rows. First maximum in (k, grid) order."""
    order = np.argsort(pi, kind="stable")
    s = pi[order]
    n = len(s)
    nA = np.searchsorted(s, grid_a, side="left")  # count pi < a
    nB = np.searchsorted(s, grid_b, side="right")  # count pi <= b
    vals = np.empty((len(ks), len(grid_a)))
    for ki in range(len(ks)):
        cg = np.concatenate([[0.0], np.cumsum(T[ki]["G"][order])])
        cv = np.concatenate([[0.0], np.cumsum(T[ki]["V"][order])])
        cr = np.concatenate([[0.0], np.cumsum(T[ki]["R"][order])])
        vals[ki] = cg[nA] + (cv[nB] - cv[nA]) + (cr[n] - cr[nB])
    flat = int(np.argmax(vals))
    ki, gi = divmod(flat, len(grid_a))
    rows = [dict(k=int(ks[j]), a=float(grid_a[int(np.argmax(vals[j]))]),
                 b=float(grid_b[int(np.argmax(vals[j]))]), total=float(vals[j].max()))
            for j in range(len(ks))]
    return (int(ks[ki]), float(grid_a[gi]), float(grid_b[gi])), float(vals[ki, gi]), rows


def select(F, band, params, pi):
    """Per-payment payoff and release under band-specific (k, a, b)."""
    n = len(pi)
    pay = np.zeros(n)
    rel = np.zeros(n, dtype=bool)
    for j, (k, a, b) in enumerate(params):
        m = band == j
        g = pi < a
        r = pi > b
        for nm, sel in (("G", g), ("R", r), ("V", ~g & ~r)):
            idx = m & sel
            pay[idx] = F[k][nm][0][idx]
            rel[idx] = F[k][nm][1][idx]
    return pay, rel


def blocks(x):
    n = len(x) // PPE * PPE
    assert n == len(x)
    return x.reshape(-1, PPE).sum(axis=1)


def diff_stats(a_sums, b_sums, counts):
    d = a_sums - b_sums
    m = ratio_mean(d, counts)
    lo, hi = boot_ci(d, counts)
    return dict(mean=m, ci95=[lo, hi], bp=m / V_DECL * 1e4,
                bp_ci95=[lo / V_DECL * 1e4, hi / V_DECL * 1e4], perm_p=perm_p(d, counts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    ap.add_argument("--b3-n", type=int, default=161)
    ap.add_argument("--bands", default="1,3,4,5")
    ap.add_argument("--cache", default=str(EXPOSURE_BAND / "cache"))
    ap.add_argument("--out", default=str(EXPOSURE_BAND / "runs"))
    ap.add_argument("--smoke", action="store_true", help="slice run: compare 1 band with the repo tuner")
    a = ap.parse_args(argv)

    _, env, _ = envs_for("mid")["E-fitted"]
    flow = make_flows()[a.flow]
    grid = suspicion_grid(a.b3_n)
    grid_a = np.array([g[0] for g in grid])
    grid_b = np.array([g[1] for g in grid])
    ks = sorted({0, 1, 2, 4, 8, 16, env.H})
    assert ks == [0, 1, 2, 4, 8, 16, 30]
    nbs = [int(x) for x in a.bands.split(",")]
    cache = Path(a.cache) / f"{a.flow}_s{a.seed}_{a.n_tune}_{a.n_eval}"
    cache.mkdir(parents=True, exist_ok=True)
    timing = {}

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tune_d = draw_outage_batch(env, flow, a.n_tune, rng, payments_per_episode=PPE)
    log(f"{a.flow} tune draws done")
    eval_d = draw_outage_batch(env, flow, a.n_eval, rng, payments_per_episode=PPE)
    _, _, ex = window_AD(env, survival(env))
    timing["draws_s"] = time.time() - t0
    log(f"{a.flow} eval draws done {timing['draws_s']:.0f}s")

    # ---- tuning pass: forced replays per (k, action) on the tuning draws
    t1 = time.time()
    T = []
    for k in ks:
        row = {}
        for nm, act in ACTS:
            f = cache / f"tune_k{k}_{nm}.npy"
            if f.exists():
                row[nm] = np.load(f)
            else:
                row[nm] = replay_outage(env, tune_d, Force(k, act, env.H, env.N), ex)
                np.save(f, row[nm])
        T.append(row)
        log(f"{a.flow} tune k={k} done")
    timing["tune_replay_s"] = time.time() - t1

    vt, pt = tune_d.v, tune_d.pi0
    configs = {}
    for nb in nbs:
        inner = np.quantile(vt, [j / nb for j in range(1, nb)]) if nb > 1 else np.array([])
        bt = band_of(vt, inner)
        params, rows_all, totals = [], [], []
        for j in range(nb):
            m = bt == j
            Tj = [{nm: T[ki][nm][m] for nm, _ in ACTS} for ki in range(len(ks))]
            p, tot, rows = tune_band(Tj, pt[m], grid_a, grid_b, ks)
            params.append(p)
            rows_all.append(rows)
            totals.append(tot)
        configs[nb] = dict(inner_edges=inner.tolist(), params=params, tune_totals=totals,
                           tune_counts=[int((bt == j).sum()) for j in range(nb)],
                           tune_mean=float(sum(totals) / len(vt)), k_rows=rows_all)
        log(f"{a.flow} bands={nb} params={params}")

    # B4 itself on this tuning sample (1 band) and the saved refined B4
    b4_tune = configs[1]["params"][0] if 1 in configs else None
    ref = None
    ref_path = Path(MAIN) / f"b4ref_E-fitted_{a.flow}_mid_s{a.seed}.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))["payload"]
    check = {}
    if a.smoke:
        from arena.experiments.settlement.watch import tune_watch_policy
        best, best_val, _ = tune_watch_policy(
            lambda k, act: T[ks.index(k)][{GRANT: "G", REJECT: "R", VERIFY: "V"}[act]],
            grid, ks, pt)
        check["repo_tuner"] = dict(k=best[0], a=best[1], b=best[2], tune_mean=best_val)
        check["band1"] = dict(params=b4_tune, tune_mean=configs[1]["tune_mean"])
        check["band1_equals_repo"] = (tuple(best) == tuple(b4_tune)
                                      and abs(best_val - configs[1]["tune_mean"]) < 1e-9 * max(1, abs(best_val)))
        log(f"smoke check {check}")
    elif ref is not None and b4_tune is not None:
        rb = ref["b4"]
        check["band1_equals_b4ref"] = (b4_tune == (rb["k"], rb["a"], rb["b"])
                                       and abs(configs[1]["tune_mean"] - rb["tune_mean"]) < 1e-9 * abs(rb["tune_mean"]))
        check["b4ref"] = rb
        log(f"{a.flow} band1 vs b4ref {check['band1_equals_b4ref']}")
    for nb in nbs:
        configs[nb]["tune_gain_over_b4"] = configs[nb]["tune_mean"] - configs[1]["tune_mean"]
        assert configs[nb]["tune_gain_over_b4"] >= -1e-9, "band tuning below B4 on tuning sample"

    # ---- evaluation: forced replays for every k any config chose
    b4_params = [(ref["b4"]["k"], ref["b4"]["a"], ref["b4"]["b"])] if (ref and not a.smoke) else [b4_tune]
    need = sorted({p[0] for c in configs.values() for p in c["params"]} | {b4_params[0][0]})
    t2 = time.time()
    F = {}
    for k in need:
        F[k] = {}
        for nm, act in ACTS:
            fp, fr = cache / f"eval_k{k}_{nm}_pay.npy", cache / f"eval_k{k}_{nm}_rel.npy"
            if fp.exists() and fr.exists():
                F[k][nm] = (np.load(fp), np.load(fr))
            else:
                F[k][nm] = replay_release(env, eval_d, Force(k, act, env.H, env.N), ex)
                np.save(fp, F[k][nm][0])
                np.save(fr, F[k][nm][1])
            log(f"{a.flow} eval k={k} {nm} done")
    timing["eval_replay_s"] = time.time() - t2
    if a.smoke:
        gaps = {}
        for k in need:
            for nm, act in ACTS:
                x = replay_outage(env, eval_d, Force(k, act, env.H, env.N), ex)
                gaps[f"k{k}{nm}"] = float(np.abs(x - F[k][nm][0]).max())
        check["replay_release_vs_repo_maxgap"] = gaps
        from arena.experiments.settlement.watch import OutageWatchBandPolicy as OB4
        k0, a0, b0 = b4_tune
        x = replay_outage(env, eval_d, OB4(k0, a0, b0, env.H, env.N), ex)
        y, _ = select(F, np.zeros(len(x), dtype=int), [b4_tune], eval_d.pi0)
        check["select_vs_repo_B4_maxgap"] = float(np.abs(x - y).max())
        log(f"smoke replay checks {gaps} select gap {check['select_vs_repo_B4_maxgap']}")

    ve, pe, th = eval_d.v, eval_d.pi0, eval_d.theta
    mis = th == 1
    n_eval = len(ve)

    def sec(pay, rel):
        return dict(mean=float(pay.mean()),
                    misuse_grant_rate=float(rel[mis].sum() / mis.sum()),
                    misuse_exposure=float((ve * (rel & mis)).mean()),
                    release_rate=float(rel.mean()),
                    legit_refusal_rate=float((~rel[~mis]).mean()))

    b4_pay, b4_rel = select(F, np.zeros(n_eval, dtype=int), b4_params, pe)
    b4_sums = blocks(b4_pay)
    counts = np.full(len(b4_sums), float(PPE))
    res = dict(flow=a.flow, seed=a.seed, n_tune=a.n_tune, n_eval=n_eval, b3_n=a.b3_n, k_grid=ks,
               b4_params=b4_params[0], b4=sec(b4_pay, b4_rel), check=check, timing=timing)
    comp_sums = None
    if not a.smoke:
        rs = np.asarray(ref["b4_block_sums"])[: len(b4_sums)]
        res["b4_reproduces_b4ref_maxgap"] = float(np.abs(rs - b4_sums).max())
        w = json.load(open(Path(WRONG) / f"wrong_E-fitted_{a.flow}_s{a.seed}.json"))
        comp_sums = np.asarray(w["true"])[: len(b4_sums)]
        res["compiled_mean"] = float(comp_sums.sum() / counts.sum())
        res["compiled_minus_b4"] = diff_stats(comp_sums, b4_sums, counts)
    for nb in nbs:
        c = configs[nb]
        be = band_of(ve, np.asarray(c["inner_edges"]))
        pay, rel = select(F, be, c["params"], pe)
        sums = blocks(pay)
        c["eval_counts"] = [int((be == j).sum()) for j in range(nb)]
        c["eval"] = sec(pay, rel)
        c["eval_per_band_mean"] = [float(pay[be == j].mean()) for j in range(nb)]
        c["band_minus_b4"] = diff_stats(sums, b4_sums, counts)
        if comp_sums is not None:
            c["compiled_minus_band"] = diff_stats(comp_sums, sums, counts)
        c["block_sums"] = sums.tolist()
        log(f"{a.flow} bands={nb} eval mean {c['eval']['mean']:.6f} "
            f"band-B4 {c['band_minus_b4']['bp']:.2f} bp"
            + (f" comp-band {c['compiled_minus_band']['bp']:.2f} bp" if comp_sums is not None else ""))
    res["configs"] = {str(k): v for k, v in configs.items()}
    res["timing"]["total_s"] = time.time() - t0
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = f"xband_{'smoke_' if a.smoke else ''}E-fitted_{a.flow}_s{a.seed}.json"
    Path(a.out, name).write_text(json.dumps(res, indent=1, default=float))
    log(f"{a.flow} wrote {name} total {res['timing']['total_s']:.0f}s")


if __name__ == "__main__":
    main()
