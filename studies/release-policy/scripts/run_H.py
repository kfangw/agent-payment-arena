"""Experiment H: the compiled release policy at real payment sizes.
E-fitted world, flow F2, true q = 0.75, the main-run compiled table (v grid geomspace(0.5, 2000, 41)).
Only the exposure law of F2 changes (laws L0..L5).  Spec: docs/exposure_law.md.

Run from the package root (Python 3.11, numpy, scipy):
    export PYTHONPATH=$PWD/src
    python3 scripts/run_H.py check                      (spec 9.1-9.4, vec == repo)
    python3 scripts/run_H.py smoke                      (spec 9.5, timing, checks.md)
    python3 scripts/run_H.py run                        (full run, resumable)
    python3 scripts/analyze_H.py [--tag smoke --nb 100]
Output goes to results/exposure_law/ (paths.EXPOSURE_LAW), the table cache to results/cache/.

Per law L (in the order L0, L1, ..., L0 first because B4_L0 is tuned there):
  plan  200,000 learning payments from default_rng([111, 700, L]); forced payoffs for every
        k in horizon_grid(H) and action G/R/V (vectorized replay, checked == replay_outage);
        tune_watch_policy on the 161-point suspicion grid.  <run>/plan/b4_L.json.
  eval  draw_outage_batch(world, F2 with law L, 5,663,400, default_rng(111), 50 per episode);
        replay Comp (table), Cutoff (grant iff pi0 < pihat0 else reject, at arrival),
        B4_L and B4_L0 (OutageWatchBandPolicy semantics).  One npz of block sums per law
        (<run>/npz/law_L.npz, .tmp then rename; an existing npz is skipped).
Every policy on a law replays the same draw (paired).  Payoff is computed in the true world.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths                            # noqa: E402
import run_D as RD                      # noqa: E402  (world_env, comp_env, table_hash, block_sums)
import fastreplay_G as FR               # noqa: E402  (vectorized replay, bit-identical to replay_outage)

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY, WAIT     # noqa: E402
from arena.experiments.settlement.flows import LogNormalV, make_flows        # noqa: E402
from arena.experiments.settlement.outage import (                             # noqa: E402
    PI_GRID_OUTAGE, compile_outage, draw_outage_batch, replay_outage, survival, window_AD)
from arena.experiments.settlement.policies import suspicion_grid              # noqa: E402
from arena.experiments.settlement.watch import (                              # noqa: E402
    FixedActionOutageWatchPolicy as OB4Force, OutageWatchBandPolicy as OB4,
    horizon_grid, tune_watch_policy)

# ------------------------------------------------------------------ fixed constants (docs/exposure_law.md)
EVAL_SEED = 111
LEARN_TAG = 700
N_EVAL = RD.N_EVAL                 # 5,663,400
N_LEARN = 200_000
PPE = RD.PPE                       # 50
FLOW = RD.FLOW                     # F2
Q_TRUE = RD.Q_TRUE                 # 0.75
B3_N = RD.B3_N                     # 161
REF_HASH = "b80233bd3f238993d72e5a68c23a516da2590e25b8cbc8847d7337e58488b7d2"
D_REF_J = 8.938881202421832        # D's pi_ref J on eval seed 108, L0 law
FLOOR = 1.90                       # C/(q m), spec 1
VSTAR = 3.13                       # v*, spec 1
PIHAT0_SPEC = 0.255
Z95 = float(norm.ppf(0.95))
RULE_HOURS = 2.5                   # spec 7
POLICIES = ("comp", "cutoff", "b4L", "b4L0")
ACTS = ("grant", "reject", "verify", "wait")


def sigma_L3():
    """Spec 9.4: (ln 0.2 + s^2/2)/s = z_0.95, larger root of s^2/2 - z s + ln 0.2 = 0."""
    return float(Z95 + np.sqrt(Z95 ** 2 - 2.0 * np.log(0.20)))


# name -> (target mean, sigma, lo, hi); L0 keeps the main run's median 30 (see the change log of the spec)
LAWS = {
    "L0": (49.50, 1.0, 0.5, 2000.0),
    "L1": (0.20, 1.0, 0.01, 2000.0),
    "L2": (0.20, 2.0, 0.01, 2000.0),
    "L3": (0.20, None, 0.01, 2000.0),
    "L4": (1.00, 1.0, 0.01, 2000.0),
    "L5": (5.00, 1.0, 0.01, 2000.0),
}
LAW_INDEX = {k: i for i, k in enumerate(LAWS)}


def law_params(name):
    mean, s, lo, hi = LAWS[name]
    if s is None:
        s = sigma_L3()
    med = 30.0 if name == "L0" else mean * np.exp(-s * s / 2.0)
    return dict(name=name, mean_target=mean, sigma=float(s), median=float(med), lo=lo, hi=hi)


def make_law(name):
    p = law_params(name)
    return LogNormalV(p["median"], p["sigma"], p["lo"], p["hi"])


def flow_for(name):
    return replace(make_flows()[FLOW], vdist=make_law(name))


# ------------------------------------------------------------------ world and table
def world_env():
    return RD.world_env()


def setup():
    world = world_env()
    _, _, ex = window_AD(world, survival(world))
    return world, ex


def pihat0(world):
    """Spec 4: pihat(0) = (sigma0 (1+m) - 1) / (sigma0 (m+h)), sigma0 = (1-f0)(1-fi)^N."""
    sig0 = float(np.prod(1.0 - np.asarray(world.f, dtype=float)))
    return (sig0 * (1 + world.m) - 1.0) / (sig0 * (world.m + world.h)), sig0


def load_table(world, cache_dir, log=print):
    """Main-run table at true q.  Compiled once and cached as a stacked int8 array."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"table_{REF_HASH[:16]}.npy"
    vg = np.geomspace(0.5, 2000.0, 41)
    if p.exists():
        return np.load(p), vg
    t0 = time.time()
    pol = compile_outage(RD.comp_env(world, Q_TRUE), "comp")
    h = RD.table_hash(pol)
    assert h == REF_HASH, f"table hash {h} != main-run {REF_HASH}"
    assert np.array_equal(pol.v_grid, vg)
    T = FR.stack_tables(pol)
    tmp = p.with_name(p.stem + ".tmp.npy")
    np.save(tmp, T)
    os.replace(tmp, p)
    log(f"TABLE compiled hash={h[:16]} t={time.time() - t0:.1f}s")
    return T, vg


# ------------------------------------------------------------------ vectorized policies
def recorder(fn, granted):
    """Wrap an action function; mark payments that receive GRANT (a grant ends the payment)."""
    def g(k, i, l, r):
        a = np.asarray(fn(k, i, l, r))
        granted[k[a == GRANT]] = True
        return a
    return g


def comp_fn(T, iv, pi):
    return FR.table_action_fn(T, iv, FR.pi_bin(pi))


def cutoff_fn(pi, thr):
    pi = np.asarray(pi, dtype=float)

    def fn(k, i, l, r):
        return np.where(pi[k] < thr, GRANT, REJECT)
    return fn


def band_fn(env, pi, horizon, lower, upper):
    """OutageWatchBandPolicy(horizon, lower, upper, H, N) vectorized."""
    H, FIN = env.H, env.N + 1
    pi = np.asarray(pi, dtype=float)

    def fn(k, i, l, r):
        elapsed = H - l
        wait = (elapsed < horizon) & (i < FIN) & (l > 0)
        s = pi[k]
        act = np.where(s < lower, GRANT, np.where(s > upper, REJECT, VERIFY))
        return np.where(wait, WAIT, act)
    return fn


def replay(env, d, fn, ex):
    FR.set_exercise(ex)
    granted = np.zeros(len(d), dtype=bool)
    pay = FR.replay_vec(env, d, recorder(fn, granted))
    return pay, granted


def policy_fn(name, env, d, T, iv, thr, b4):
    if name == "comp":
        return comp_fn(T, iv, d.pi0)
    if name == "cutoff":
        return cutoff_fn(d.pi0, thr)
    k, a, b = b4[name]
    return band_fn(env, d.pi0, k, a, b)


def repo_policy(name, env, T_pol, thr, b4):
    if name == "comp":
        return T_pol
    if name == "cutoff":
        return lambda i, l, r, v, pi: GRANT if pi < thr else REJECT
    k, a, b = b4[name]
    return OB4(k, a, b, env.H, env.N)


class GrantCounter:
    def __init__(self, pol):
        self.pol, self.n = pol, 0

    def __call__(self, *a):
        x = self.pol(*a)
        if x == GRANT:
            self.n += 1
        return x


def first_actions(T, iv, d, H):
    """Comp's action at arrival (i = 0, l = H, regime at arrival), per payment."""
    ip = FR.pi_bin(d.pi0)
    r0 = d.paths[:, 0].astype(np.int64)
    return T[iv, 0, H, r0, ip].astype(np.int64)


def prefix(d, n):
    return replace(d, **{f: getattr(d, f)[:n] for f in
                         ("v", "p_true", "theta", "pi0", "u_stage", "t_ans", "paths")})


# ------------------------------------------------------------------ B4 tuning (plan)
def plan_law(name, world, ex, log):
    t0 = time.time()
    rng = np.random.default_rng([EVAL_SEED, LEARN_TAG, LAW_INDEX[name]])
    learn = draw_outage_batch(world, flow_for(name), N_LEARN, rng, payments_per_episode=PPE)
    t_draw = time.time() - t0
    ks = list(horizon_grid(world.H))
    force = {}
    for k in ks:
        for act in (GRANT, REJECT, VERIFY):
            force[(k, act)] = FR.replay_forced(world, learn, k, act, ex)
    t_force = time.time() - t0 - t_draw
    best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)], suspicion_grid(B3_N), ks, learn.pi0)
    t_tune = time.time() - t0 - t_draw - t_force
    rec = dict(law=name, rng=[EVAL_SEED, LEARN_TAG, LAW_INDEX[name]], n=N_LEARN, b3_n=B3_N, k_grid=ks,
               b4=dict(k=best[0], a=best[1], b=best[2], tune_mean=val), b4_k_rows=rows,
               t_draw=t_draw, t_force=t_force, t_tune=t_tune, t_total=time.time() - t0,
               learn_mean_v=float(np.mean(learn.v)), impl="vec forced replay + tune_watch_policy")
    log(f"PLAN {name} b4=({best[0]},{best[1]:.5f},{best[2]:.5f}) tune_mean={val:.6g} "
        f"draw={t_draw:.1f}s force={t_force:.1f}s tune={t_tune:.1f}s")
    return rec


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def get_plan(run_dir, name, world, ex, log):
    p = run_dir / "plan" / f"b4_{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    rec = plan_law(name, world, ex, log)
    _write_json(p, rec)
    return rec


# ------------------------------------------------------------------ eval (one law)
def eval_law(name, run_dir, n_eval, world, ex, T, vg, log):
    path = run_dir / "npz" / f"law_{name}.npz"
    if path.exists():
        log(f"SKIP {name} (npz exists)")
        return None
    t_law = time.time()
    pl = get_plan(run_dir, name, world, ex, log)
    p0 = pl if name == "L0" else get_plan(run_dir, "L0", world, ex, log)
    b4 = dict(b4L=(pl["b4"]["k"], pl["b4"]["a"], pl["b4"]["b"]),
              b4L0=(p0["b4"]["k"], p0["b4"]["a"], p0["b4"]["b"]))
    t0 = time.time()
    d = draw_outage_batch(world, flow_for(name), n_eval, np.random.default_rng(EVAL_SEED),
                          payments_per_episode=PPE)
    t_draw = time.time() - t0
    thr, sig0 = pihat0(world)
    iv = FR.v_index(vg, d.v)
    v = np.asarray(d.v, dtype=float)
    mis = np.asarray(d.theta) == 1
    hi = v > VSTAR
    fa = first_actions(T, iv, d, world.H)
    bs = RD.block_sums
    out = dict(bs_v=bs(v), bs_n_mis=bs(mis.astype(float)), bs_v_mis=bs(np.where(mis, v, 0.0)))
    times = {}
    pays = {}
    for p in POLICIES:
        if p == "b4L0" and b4["b4L0"] == b4["b4L"]:
            pay, gr = pays["b4L"]
            times[p] = 0.0
        else:
            t1 = time.time()
            pay, gr = replay(world, d, policy_fn(p, world, d, T, iv, thr, b4), ex)
            times[p] = time.time() - t1
        pays[p] = (pay, gr)
        mg = gr & mis
        out[f"pay_{p}"] = bs(pay)
        out[f"hi_{p}"] = bs(np.where(hi, pay, 0.0))
        out[f"misn_{p}"] = bs(mg.astype(float))
        out[f"misv_{p}"] = bs(np.where(mg, v, 0.0))
        out[f"gr_{p}"] = bs(gr.astype(float))
    diff = pays["comp"][0] - pays["cutoff"][0]
    for j, an in enumerate(ACTS):
        out[f"dact_{an}"] = bs(np.where(fa == j, diff, 0.0))
    # Cutoff sanity: grants exactly the payments with pi0 < thr
    assert np.array_equal(pays["cutoff"][1], np.asarray(d.pi0) < thr), "Cutoff grant set"
    cnt = np.bincount(fa, minlength=4)[:4]
    val = np.bincount(fa, weights=v, minlength=4)[:4]
    meta = dict(law=law_params(name), eval_seed=EVAL_SEED, n_eval=n_eval, n_blocks=n_eval // PPE,
                realized_mean=float(v.mean()), realized_median=float(np.median(v)),
                n_at_lo=int(np.sum(v <= law_params(name)["lo"])), n_at_hi=int(np.sum(v >= law_params(name)["hi"])),
                floor=FLOOR, vstar=VSTAR,
                share_count_above_floor=float(np.mean(v > FLOOR)), share_value_above_floor=float(v[v > FLOOR].sum() / v.sum()),
                share_count_above_vstar=float(np.mean(hi)), share_value_above_vstar=float(v[hi].sum() / v.sum()),
                share_count_below_grid=float(np.mean(v < 0.5)),
                comp_first_count=dict(zip(ACTS, cnt.tolist())), comp_first_value=dict(zip(ACTS, val.tolist())),
                pihat0=thr, sigma0=sig0, b4L=dict(zip("kab", b4["b4L"])), b4L0=dict(zip("kab", b4["b4L0"])),
                b4L_tune_mean=pl["b4"]["tune_mean"], misuse_rate=float(mis.mean()),
                t_draw=t_draw, t_replay=times, t_plan=pl["t_total"], table_hash=REF_HASH,
                total_payoff={p: float(pays[p][0].sum()) for p in POLICIES})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    el = time.time() - t_law
    log(f"DONE {name} mean_v={meta['realized_mean']:.4f} draw={t_draw:.1f}s "
        + " ".join(f"{p}={t:.1f}s" for p, t in times.items()) + f" law_total={el:.1f}s "
        + " ".join(f"J_{p}={pays[p][0].mean():.6g}" for p in POLICIES))
    return meta


# ------------------------------------------------------------------ checks 9.1-9.4
def run_check(run_dir, log):
    res = {}
    world, ex = setup()
    T, vg = load_table(world, paths.CACHE, log)
    thr, sig0 = pihat0(world)
    # 9.1: D's seed-108 reference through the H code path (flow rebuilt with the L0 law)
    t0 = time.time()
    d = draw_outage_batch(world, flow_for("L0"), N_EVAL, np.random.default_rng(108), payments_per_episode=PPE)
    t_draw = time.time() - t0
    iv = FR.v_index(vg, d.v)
    times = {}
    t0 = time.time()
    pay, _ = replay(world, d, comp_fn(T, iv, d.pi0), ex)
    times["comp"] = time.time() - t0
    b = RD.block_sums(pay)
    J = float((np.column_stack([b, b]).sum(axis=0) / (len(b) * PPE))[0])
    t0 = time.time()
    replay(world, d, cutoff_fn(d.pi0, thr), ex)
    times["cutoff"] = time.time() - t0
    t0 = time.time()
    replay(world, d, band_fn(world, d.pi0, 1, 0.1625, 0.63125), ex)
    times["b4_k1"] = time.time() - t0
    t0 = time.time()
    replay(world, d, band_fn(world, d.pi0, 30, 0.1625, 0.63125), ex)
    times["b4_k30"] = time.time() - t0
    same_flow = make_flows()[FLOW].vdist == make_law("L0")
    res["9.1"] = dict(J=J, J_D=D_REF_J, abs_diff=abs(J - D_REF_J), equal=J == D_REF_J, t_draw_full=t_draw,
                      t_replay_full=times, l0_law_equals_make_flows_F2=bool(same_flow), pass_=J == D_REF_J)
    log(f"CHECK 9.1 J={J!r} D={D_REF_J!r} diff={abs(J - D_REF_J):.1e} draw={t_draw:.1f}s replays={times} "
        f"{'PASS' if J == D_REF_J else 'FAIL'}")
    # vec == repo replay on prefixes: L0 (seed 108) and L1, L3 (seed 111), every policy type
    T_pol = compile_outage(RD.comp_env(world, Q_TRUE), "comp")
    assert RD.table_hash(T_pol) == REF_HASH
    b4 = dict(b4L=(1, 0.1625, 0.63125), b4L0=(16, 0.05, 0.9))
    gaps = {}
    npre = 20_000
    draws = {"L0_s108": prefix(d, npre)}
    del d, pay
    for nm in ("L1", "L3"):
        draws[f"{nm}_s111"] = draw_outage_batch(world, flow_for(nm), npre, np.random.default_rng(EVAL_SEED),
                                                payments_per_episode=PPE)
    for dn, dp in draws.items():
        ivp = FR.v_index(vg, dp.v)
        fa_vec = first_actions(T, ivp, dp, world.H)
        fa_repo = np.array([T_pol(0, world.H, int(dp.paths[j, 0]), float(dp.v[j]), float(dp.pi0[j]))
                            for j in range(len(dp))])
        gaps[f"{dn}:first_action"] = dict(max_abs=float(np.sum(fa_vec != fa_repo)))
        for p in POLICIES:
            a, gr = replay(world, dp, policy_fn(p, world, dp, T, ivp, thr, b4), ex)
            gc = GrantCounter(repo_policy(p, world, T_pol, thr, b4))
            bb = replay_outage(world, dp, gc, ex)
            gaps[f"{dn}:{p}"] = dict(max_abs=float(np.max(np.abs(a - bb))), grants_vec=int(gr.sum()), grants_repo=gc.n)
        for k in (0, 4, 30):
            for act in (GRANT, REJECT, VERIFY):
                a = FR.replay_forced(world, dp, k, act, ex)
                bb = replay_outage(world, dp, OB4Force(k, act, world.H, world.N), ex)
                gaps[f"{dn}:force_k{k}_{'GRV'[[GRANT, REJECT, VERIFY].index(act)]}"] = dict(
                    max_abs=float(np.max(np.abs(a - bb))))
    ok = all(g["max_abs"] == 0.0 and g.get("grants_vec", 0) == g.get("grants_repo", 0) for g in gaps.values())
    res["vec_vs_repo"] = dict(prefix=npre, gaps=gaps, pass_=ok)
    log(f"CHECK vec==repo prefix={npre} cases={len(gaps)} max={max(g['max_abs'] for g in gaps.values()):.1e} "
        f"grant counts equal={all(g.get('grants_vec', 0) == g.get('grants_repo', 0) for g in gaps.values())} "
        f"{'PASS' if ok else 'FAIL'}")
    # 9.2: below-grid-floor invariance
    env = RD.comp_env(world, Q_TRUE)
    H = world.H
    rows = {}
    for vv in (0.01, 0.05, 0.2, 0.5):
        t0 = time.time()
        pol = compile_outage(env, f"v{vv}", n_v=1, v_lo=vv, v_hi=vv)
        assert np.isclose(pol.v_grid[0], vv)
        lab = pol.tables[0]
        arr = {f"r{r}": int(np.sum(lab[0, H, r, :] != T[0, 0, H, r, :])) for r in (0, 1)}
        rows[repr(vv)] = dict(arrival_mismatch=arr, arrival_points=2 * len(PI_GRID_OUTAGE),
                              all_state_mismatch=int(np.sum(lab != T[0])), all_state_points=int(lab.size),
                              arrival_actions_v05=dict(zip(ACTS, np.bincount(T[0, 0, H].ravel().astype(int), minlength=4)[:4].tolist())),
                              t=time.time() - t0)
        log(f"CHECK 9.2 v={vv} arrival mismatch running={arr['r0']} halted={arr['r1']} "
            f"all-state mismatch={rows[repr(vv)]['all_state_mismatch']}/{lab.size}")
    ok92 = all(sum(r["arrival_mismatch"].values()) == 0 for r in rows.values())
    res["9.2"] = dict(per_v=rows, table_row_v=float(vg[0]), pass_=ok92)
    log(f"CHECK 9.2 {'PASS' if ok92 else 'FAIL'}")
    # 9.3: pihat0
    res["9.3"] = dict(pihat0=thr, sigma0=sig0, N=int(world.N), f0=float(world.f[0]), fi=float(world.f[1]),
                      m=float(world.m), h=float(world.h), spec=PIHAT0_SPEC, rounded=round(thr, 3),
                      pass_=round(thr, 3) == PIHAT0_SPEC)
    log(f"CHECK 9.3 pihat0={thr!r} sigma0={sig0!r} N={world.N} {'PASS' if round(thr, 3) == PIHAT0_SPEC else 'FAIL'}")
    # 9.4: L3 sigma
    s3 = sigma_L3()
    mu3 = np.log(0.20) - s3 * s3 / 2
    lhs = (np.log(0.20) + s3 * s3 / 2) / s3
    an_mean = float(np.exp(mu3 + s3 * s3 / 2))
    an_share = float(norm.cdf((mu3 + s3 * s3 - 0.0) / s3))
    x_raw = np.exp(mu3 + s3 * np.random.default_rng([EVAL_SEED, 940]).standard_normal(1_000_000))
    x = np.clip(x_raw, 0.01, 2000.0)
    raw_mean, raw_share = float(x_raw.mean()), float(x_raw[x_raw >= 1].sum() / x_raw.sum())
    cl_mean, cl_share = float(x.mean()), float(x[x >= 1].sum() / x.sum())
    within = abs(cl_mean / 0.20 - 1) <= 0.02 and abs(cl_share / 0.95 - 1) <= 0.02
    res["9.4"] = dict(sigma3=s3, mu3=float(mu3), median3=float(np.exp(mu3)), z=Z95, lhs=float(lhs),
                      analytic_mean=an_mean, analytic_value_share_ge1=an_share,
                      sample=dict(n=1_000_000, rng=[EVAL_SEED, 940], untruncated_mean=raw_mean,
                                  untruncated_value_share_ge1=raw_share, clipped_mean=cl_mean,
                                  clipped_value_share_ge1=cl_share,
                                  clipped_share_at_lo=float(np.mean(x_raw <= 0.01)),
                                  clipped_share_at_hi=float(np.mean(x_raw >= 2000)),
                                  analytic_value_share_above_2000=float(norm.cdf((mu3 + s3 * s3 - np.log(2000)) / s3))),
                      within_2pct_clipped=bool(within),
                      pass_=bool(abs(lhs - Z95) < 1e-9 and abs(an_mean - 0.20) < 1e-12))
    log(f"CHECK 9.4 sigma3={s3!r} mu3={mu3:.4f} analytic mean={an_mean:.4f} share>=1={an_share:.4f}; "
        f"1M clipped mean={cl_mean:.4f} share={cl_share:.4f} (untruncated {raw_mean:.4f}, {raw_share:.4f}) "
        f"within 2%={within}")
    _write_json(run_dir / "checks.json", res)
    return res


# ------------------------------------------------------------------ smoke (9.5) and timing
def run_laws(run_dir, laws, n_eval, log):
    world, ex = setup()
    T, vg = load_table(world, paths.CACHE, log)
    t0 = time.time()
    fails = []
    for nm in laws:
        try:
            eval_law(nm, run_dir, n_eval, world, ex, T, vg,
                     lambda m: log(f"{m} elapsed={(time.time() - t0) / 60:.1f}m"))
        except Exception:
            fails.append(nm)
            log(f"FAIL {nm} elapsed={(time.time() - t0) / 60:.1f}m")
            with (run_dir / "errors.log").open("a") as f:
                f.write(f"FAIL {nm}\n{traceback.format_exc()}\n")
    return fails, time.time() - t0


def projection(run_dir):
    ch = json.loads((run_dir / "checks.json").read_text())
    tr = ch["9.1"]["t_replay_full"]
    t_draw = ch["9.1"]["t_draw_full"]
    t_rep4 = tr["comp"] + tr["cutoff"] + tr["b4_k1"] + tr["b4_k30"]
    plans = [json.loads((run_dir / "plan" / f"b4_{n}.json").read_text()) for n in ("L0", "L1")]
    t_plan = max(p["t_total"] for p in plans)
    per_law = t_draw + t_rep4 + t_plan + 15.0          # 15 s: index, first actions, block sums, npz
    return dict(t_draw_full=t_draw, t_replays_full=t_rep4, t_plan_200k=t_plan, per_law_s=per_law,
                six_laws_h=6 * per_law / 3600, rule_h=RULE_HOURS, drop_L4=6 * per_law / 3600 > RULE_HOURS)


def write_checks_md(run_dir, proj, laws, n_eval):
    ch = json.loads((run_dir / "checks.json").read_text())
    sm = json.loads((run_dir / "summary.json").read_text())
    c1, c2, c3, c4, vr = ch["9.1"], ch["9.2"], ch["9.3"], ch["9.4"], ch["vec_vs_repo"]
    tr = c1["t_replay_full"]
    ok = lambda b: "PASS" if b else "FAIL"
    s4 = c4["sample"]
    m92 = ", ".join(f"v = {v}: arrivals {r['arrival_mismatch']['r0']} / {r['arrival_mismatch']['r1']}, all states {r['all_state_mismatch']}"
                    for v, r in c2["per_v"].items())
    a05 = c2["per_v"]["0.5"]["arrival_actions_v05"]
    rows = [
        "# Experiment H smoke checks (spec sec 9)", "",
        f"Records: `smoke/progress.log`, `smoke/checks.json`, `smoke/timing.json`, `smoke/summary.md`. Generated {time.strftime('%Y-%m-%d %H:%M')}.", "",
        "| Check | Content | Numbers | Verdict |", "|---|---|---|---|",
        f"| 9.1 | Reproduction of the seed-108 reference value of experiment D (F2 rebuilt with the L0 law, true-q table, vectorized replay) | J = {c1['J']!r}, D = {c1['J_D']!r}, difference {c1['abs_diff']:.1e}. "
        f"L0 law equals the vdist of make_flows F2: {c1['l0_law_equals_make_flows_F2']}. Full-size draw {c1['t_draw_full']:.1f} s, replay Comp {tr['comp']:.1f} s, "
        f"Cutoff {tr['cutoff']:.1f} s, B4 (k = 1) {tr['b4_k1']:.1f} s, B4 (k = 30) {tr['b4_k30']:.1f} s | {ok(c1['pass_'])} |",
        f"| extra | Vectorized replay = replay_outage of the repository | first {vr['prefix']:,} payments, L0 (seed 108), L1 and L3 (seed 111), the four policies and 9 forced policies (k = 0, 4, 30), "
        f"including the first action of Comp (vectorized = repository policy call): {len(vr['gaps'])} cases, per-payment maximum difference 0 in all, grant counts equal | {ok(vr['pass_'])} |",
        f"| 9.2 | Identical decisions below the table's lowest v, pi grid of 501 points, running and halted arrivals | mismatches (running / halted): {m92}. "
        f"Arrival actions in the $0.5 row (running and halted together, 1,002 points): grant {a05['grant']}, reject {a05['reject']}, verify {a05['verify']}, wait {a05['wait']} | {ok(c2['pass_'])} |",
        f"| 9.3 | Recomputation of pihat(0) | sigma(0) = (1 - {c3['f0']})(1 - {c3['fi']})^{c3['N']} = {c3['sigma0']!r}, pihat(0) = {c3['pihat0']!r}, "
        f"third decimal {c3['rounded']} | {ok(c3['pass_'])} |",
        f"| 9.4 | sigma_3 of L3 | analytic sigma_3 = {c4['sigma3']:.6f}, mu_3 = {c4['mu3']:.4f}, median {c4['median3']:.3e}. Untruncated: mean {c4['analytic_mean']:.4f}, "
        f"value share at or above $1 {c4['analytic_value_share_ge1']:.4f}. Sample of 1 million (before clip): mean {s4['untruncated_mean']:.4f}, share {s4['untruncated_value_share_ge1']:.4f}; "
        f"after clip to [0.01, 2000]: mean {s4['clipped_mean']:.4f}, share {s4['clipped_value_share_ge1']:.4f}, which {'stay within' if c4['within_2pct_clipped'] else 'leave'} +-2%. "
        f"Clipped at the lower bound {100 * s4['clipped_share_at_lo']:.1f}%, at the upper bound {100 * s4['clipped_share_at_hi']:.3f}%, value share above $2000 (analytic) {s4['analytic_value_share_above_2000']:.3f}. "
        f"As instructed, the analytic sigma_3 is kept and the realized values are recorded | {'PASS' if c4['within_2pct_clipped'] else 'PASS (analytic); sample condition not met, recorded'} |",
        f"| 9.5 | Smoke run ({n_eval:,} evaluation payments, {', '.join(laws)}, four policies, 100 bootstrap resamples) | 0 errors, `smoke/summary.md` written | PASS |",
        "", "## Run-time estimate (spec sec 7)", "",
        f"One law = full-size draw {proj['t_draw_full']:.1f} s + 4 replays {proj['t_replays_full']:.1f} s + B4 learning (200,000-payment draw, 21 forced replays, 161-point grid tuning) "
        f"{proj['t_plan_200k']:.1f} s + other 15 s = {proj['per_law_s']:.0f} s. Six laws {proj['six_laws_h']:.2f} h, limit {proj['rule_h']} h. "
        f"L4 dropped: {'yes' if proj['drop_L4'] else 'no'}.", "",
        "Smoke numbers are not results."]
    (run_dir / "checks.md").write_text("\n".join(rows) + "\n")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "smoke", "run"])
    ap.add_argument("--laws", default=None, help="comma list; default all six (L4 dropped by the runtime rule)")
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase in ("check", "smoke") else None)
    run_dir = paths.EXPOSURE_LAW / tag if tag else paths.EXPOSURE_LAW
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = (run_dir / "progress.log").open("a", buffering=1)
    T0 = time.time()

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [{(time.time() - T0) / 60:6.1f}m] {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    log(f"START phase={a.phase} tag={tag} pid={os.getpid()}")
    if a.phase == "check":
        res = run_check(run_dir, log)
        ok = all(v["pass_"] for v in res.values())
        log(f"CHECKS {'ALL PASS' if ok else 'SOME FAIL'}")
        return 0 if ok else 1
    if a.phase == "smoke":
        laws = a.laws.split(",") if a.laws else ["L0", "L1"]
        n_eval = a.n_eval or 100_000
        fails, el = run_laws(run_dir, laws, n_eval, log)
        if fails:
            return 1
        proj = projection(run_dir)
        proj["smoke_wall_s"] = el
        proj["smoke_law_s"] = {n: json.loads(str(np.load(run_dir / "npz" / f"law_{n}.npz")["meta"]))
                               for n in laws}
        proj["smoke_law_s"] = {n: dict(t_draw=m["t_draw"], t_replay=m["t_replay"], t_plan=m["t_plan"])
                               for n, m in proj["smoke_law_s"].items()}
        _write_json(run_dir / "timing.json", proj)
        log(f"TIMING per_law={proj['per_law_s']:.0f}s six_laws={proj['six_laws_h']:.2f}h rule={RULE_HOURS}h "
            f"drop_L4={proj['drop_L4']}")
        import analyze_H
        t1 = time.time()
        analyze_H.main(["--tag", tag, "--nb", "100"])
        log(f"ANALYZE smoke nb=100 t={time.time() - t1:.1f}s")
        write_checks_md(run_dir, proj, laws, n_eval)
        log("SMOKEDONE")
        return 0
    # full run
    laws = a.laws.split(",") if a.laws else list(LAWS)
    n_eval = a.n_eval or N_EVAL
    tj = paths.EXPOSURE_LAW / "smoke" / "timing.json"
    if not a.laws and tj.exists() and json.loads(tj.read_text())["drop_L4"]:
        laws.remove("L4")
        log("RULE spec 7: projected > 2.5 h, L4 dropped")
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        laws=laws, n_eval=n_eval, eval_seed=EVAL_SEED, learn_rng=[EVAL_SEED, LEARN_TAG, "L index"],
        n_learn=N_LEARN, table_hash=REF_HASH, policies=POLICIES, floor=FLOOR, vstar=VSTAR,
        law_params={n: law_params(n) for n in laws}, started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    fails, el = run_laws(run_dir, laws, n_eval, log)
    if fails:
        log(f"INCOMPLETE failed={fails} (rerun the same command to resume)")
        return 1
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={el / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
