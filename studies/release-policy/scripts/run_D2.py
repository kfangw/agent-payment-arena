"""Experiment D2: the compiled release policy without the verify action (Comp_nv).
E-fitted world, flows F1, F2, F3, true constants.  Spec: docs/spec-D2.md.

Run from the package root:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_D2.py smoke      (100,000 evaluation payments, F2, checks)
    python3 scripts/run_D2.py run        (full, resumable)
    python3 scripts/run_D2.py analyze [--tag smoke]

Output goes to results/no_channel/ (paths.NO_CHANNEL); the compiled-table caches go to results/cache/.

Evaluation set per flow (main-run design, see the change log of the spec): rng = default_rng(seed_f), seed_f = 7, 8, 9
for F1, F2, F3; the first draw is the 200,000-payment tuning split, the second the 5,663,400-payment
evaluation split, both with 50 payments per episode.  B4_main is retuned on that tuning split
(161-point grid, k in horizon_grid) to reproduce the main run's B4.
Policies replayed on the same draw (paired): Comp (main-run table), Comp_nv (same compiler,
drop=(VERIFY,)), Cutoff (grant iff pi0 < pihat0 at arrival, else reject), B4_main.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths                            # noqa: E402
import run_D as RD                      # noqa: E402
import fastreplay_G as FR               # noqa: E402
import run_H as HH                      # noqa: E402

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY                 # noqa: E402
from arena.experiments.settlement.flows import make_flows                            # noqa: E402
from arena.experiments.settlement.outage import (                                    # noqa: E402
    compile_outage, draw_outage_batch, replay_outage)
from arena.experiments.settlement.policies import suspicion_grid                     # noqa: E402
from arena.experiments.settlement.watch import (                                     # noqa: E402
    FixedActionOutageWatchPolicy as OB4Force, OutageWatchBandPolicy as OB4, horizon_grid,
    tune_watch_policy)

FLOWS = ("F1", "F2", "F3")
SEEDS = {"F1": 7, "F2": 8, "F3": 9}
N_TUNE = 200_000
N_EVAL = RD.N_EVAL
PPE = RD.PPE
Q_TRUE = RD.Q_TRUE
REF_HASH = HH.REF_HASH
DEN = 49.5
NB, BSEED = 1_000, 20260930
POLICIES = ("comp", "comp_nv", "cutoff", "b4main")
ACTS = ("grant", "reject", "verify", "wait")
# main run, results/generated/lead_cells.json (Comp - B4, bp, 95% interval)
MAIN_LEAD = {"F1": (3.8, 3.5, 4.1), "F2": (4.3, 3.7, 4.8), "F3": (1.4, 1.0, 1.8)}
MAIN_B4_F2 = dict(k=0, a=0.1125, b=0.8, tune_mean=8.872650462538848)   # repro record of run_D.py


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ------------------------------------------------------------------ tables
def tables(world, cache, log):
    cache.mkdir(parents=True, exist_ok=True)
    T, vg = HH.load_table(world, cache, log)
    p = cache / "table_nv.npy"
    hp = cache / "table_nv.json"
    if p.exists():
        return T, np.load(p), vg, json.loads(hp.read_text())
    t0 = time.time()
    pol = compile_outage(RD.comp_env(world, Q_TRUE), "comp_nv", drop=(VERIFY,))
    info = dict(hash=RD.table_hash(pol), compile_s=time.time() - t0)
    assert np.array_equal(pol.v_grid, vg)
    Tn = FR.stack_tables(pol)
    info["n_verify_labels"] = int(np.sum(Tn == VERIFY))
    info["label_counts"] = dict(zip(ACTS, np.bincount(Tn.ravel().astype(int), minlength=4)[:4].tolist()))
    assert info["n_verify_labels"] == 0
    np.save(p, Tn)
    write_json(hp, info)
    log(f"TABLE_NV hash={info['hash'][:16]} t={info['compile_s']:.1f}s verify_labels=0")
    return T, Tn, vg, info


# ------------------------------------------------------------------ draws (main-run design)
def draw_main(world, flow, n_eval, n_tune=N_TUNE):
    """Main-run draws: one rng per flow, tuning split first, evaluation split second."""
    rng = np.random.default_rng(SEEDS[flow])
    fl = make_flows()[flow]
    tune = draw_outage_batch(world, fl, N_TUNE, rng, payments_per_episode=PPE)
    if n_tune < N_TUNE:
        tune = HH.prefix(tune, n_tune)
    ev = draw_outage_batch(world, fl, n_eval, rng, payments_per_episode=PPE)
    return tune, ev


def tune_b4(world, ex, learn):
    """Same tuner as run_H.plan_law: vectorized forced replays, 161-point grid, horizon_grid(H)."""
    t0 = time.time()
    ks = list(horizon_grid(world.H))
    force = {}
    for k in ks:
        for act in (GRANT, REJECT, VERIFY):
            force[(k, act)] = FR.replay_forced(world, learn, k, act, ex)
    t_force = time.time() - t0
    best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)], suspicion_grid(HH.B3_N), ks, learn.pi0)
    return dict(k=int(best[0]), a=float(best[1]), b=float(best[2]), tune_mean=float(val), k_rows=rows,
                t_force=t_force, t_tune=time.time() - t0 - t_force, n=len(learn))


def arrival_actions(T, iv, d, H):
    return HH.first_actions(T, iv, d, H)


# ------------------------------------------------------------------ one flow
def run_flow(flow, run_dir, n_eval, n_tune, world, ex, T, Tn, vg, log):
    path = run_dir / "npz" / f"flow_{flow}.npz"
    if path.exists():
        log(f"SKIP {flow}")
        return
    t0 = time.time()
    tune, d = draw_main(world, flow, n_eval, n_tune)
    t_draw = time.time() - t0
    pj = run_dir / "plan" / f"b4main_{flow}.json"
    if pj.exists():
        b4 = json.loads(pj.read_text())
    else:
        b4 = tune_b4(world, ex, tune)
        b4.update(flow=flow, rng=SEEDS[flow], split="first draw of the main-run rng")
        write_json(pj, b4)
    log(f"B4MAIN {flow} ({b4['k']},{b4['a']!r},{b4['b']!r}) tune_mean={b4['tune_mean']!r} "
        f"force={b4['t_force']:.1f}s tune={b4['t_tune']:.1f}s")
    del tune
    thr, sig0 = HH.pihat0(world)
    iv = FR.v_index(vg, d.v)
    v = np.asarray(d.v, dtype=float)
    mis = np.asarray(d.theta) == 1
    r0 = d.paths[:, 0].astype(np.int64)
    bs = RD.block_sums
    out = dict(bs_v=bs(v), bs_n_mis=bs(mis.astype(float)), bs_v_mis=bs(np.where(mis, v, 0.0)))
    fns = dict(comp=HH.comp_fn(T, iv, d.pi0), comp_nv=HH.comp_fn(Tn, iv, d.pi0),
               cutoff=HH.cutoff_fn(d.pi0, thr), b4main=HH.band_fn(world, d.pi0, b4["k"], b4["a"], b4["b"]))
    times, J, grants = {}, {}, {}
    for p in POLICIES:
        t1 = time.time()
        pay, gr = HH.replay(world, d, fns[p], ex)
        times[p] = time.time() - t1
        mg = gr & mis
        out[f"pay_{p}"] = bs(pay)
        out[f"misn_{p}"] = bs(mg.astype(float))
        out[f"misv_{p}"] = bs(np.where(mg, v, 0.0))
        out[f"gr_{p}"] = bs(gr.astype(float))
        J[p] = float(pay.mean())
        grants[p] = gr
        del pay
    assert np.array_equal(grants["cutoff"], np.asarray(d.pi0) < thr), "Cutoff grant set"
    # first actions at arrival
    fa_c = arrival_actions(T, iv, d, world.H)
    fa_n = arrival_actions(Tn, iv, d, world.H)
    fa_cut = np.where(np.asarray(d.pi0) < thr, GRANT, REJECT)
    first = {}
    for nm, fa in (("comp", fa_c), ("comp_nv", fa_n)):
        first[nm] = {reg: dict(zip(ACTS, np.bincount(fa[r0 == ri], minlength=4)[:4].tolist()))
                     for reg, ri in (("running", 0), ("halted", 1))}
    mism = {}
    pi0 = np.asarray(d.pi0, dtype=float)
    ip = FR.pi_bin(pi0)
    for reg, ri in (("running", 0), ("halted", 1)):
        sel = r0 == ri
        diff = sel & (fa_n != fa_cut)
        rec = dict(n_arrivals=int(sel.sum()), n_mismatch=int(diff.sum()))
        if diff.any():
            pairs = {}
            for a_n, a_c in zip(fa_n[diff], fa_cut[diff]):
                key = f"nv_{ACTS[a_n]}_cut_{ACTS[a_c]}"
                pairs[key] = pairs.get(key, 0) + 1
            rec["pairs"] = pairs
            rec["pi0_range"] = [float(pi0[diff].min()), float(pi0[diff].max())]
            rec["pi_bin_values"] = sorted({float(x) for x in (ip[diff] / 500.0)})
            rec["v_range"] = [float(v[diff].min()), float(v[diff].max())]
        mism[reg] = rec
    # Comp_nv decision boundary at running arrivals, per v grid row: smallest pi bin that is not GRANT
    H = world.H
    bnd = []
    for j in range(len(vg)):
        row = Tn[j, 0, H, 0, :]
        ng = np.nonzero(row != GRANT)[0]
        bnd.append(dict(v=float(vg[j]), first_non_grant_bin=int(ng[0]) if len(ng) else None,
                        pi=float(ng[0] / 500.0) if len(ng) else None,
                        actions_above=sorted({ACTS[int(x)] for x in row[ng]}) if len(ng) else []))
    meta = dict(flow=flow, eval_rng=SEEDS[flow], draw="default_rng(seed): tuning 200,000 then evaluation",
                n_eval=n_eval, n_tune=n_tune, n_blocks=n_eval // PPE, pihat0=thr, sigma0=sig0,
                b4main=dict(k=b4["k"], a=b4["a"], b=b4["b"], tune_mean=b4["tune_mean"]),
                J=J, t_draw=t_draw, t_replay=times, first_actions=first, mismatch_nv_vs_cutoff=mism,
                nv_running_boundary=bnd, misuse_rate=float(mis.mean()), realized_mean=float(v.mean()),
                table_hash=REF_HASH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    log(f"DONE {flow} draw={t_draw:.1f}s " + " ".join(f"{p}={t:.1f}s" for p, t in times.items())
        + " " + " ".join(f"J_{p}={J[p]:.8g}" for p in POLICIES)
        + f" mismatch_running={mism['running']['n_mismatch']} halted={mism['halted']['n_mismatch']}"
        + f" flow_total={time.time() - t0:.1f}s")


# ------------------------------------------------------------------ smoke checks
def smoke_checks(world, ex, T, Tn, vg, log, npre=20_000):
    """Vectorized replay == repo replay_outage for Comp_nv and Cutoff on a prefix."""
    _, d = draw_main(world, "F2", npre, n_tune=1000)
    thr, _ = HH.pihat0(world)
    pol_nv = compile_outage(RD.comp_env(world, Q_TRUE), "comp_nv", drop=(VERIFY,))
    iv = FR.v_index(vg, d.v)
    res = {}
    for nm, fn, rp in (("comp_nv", HH.comp_fn(Tn, iv, d.pi0), pol_nv),
                       ("cutoff", HH.cutoff_fn(d.pi0, thr), lambda i, l, r, v, pi: GRANT if pi < thr else REJECT)):
        a, gr = HH.replay(world, d, fn, ex)
        gc = HH.GrantCounter(rp)
        b = replay_outage(world, d, gc, ex)
        res[nm] = dict(max_abs=float(np.max(np.abs(a - b))), grants_vec=int(gr.sum()), grants_repo=gc.n)
    fa_vec = HH.first_actions(Tn, iv, d, world.H)
    fa_repo = np.array([pol_nv(0, world.H, int(d.paths[j, 0]), float(d.v[j]), float(d.pi0[j])) for j in range(len(d))])
    res["first_action_nv"] = dict(mismatch=int(np.sum(fa_vec != fa_repo)))
    ok = all(r.get("max_abs", 0) == 0 and r.get("grants_vec", 0) == r.get("grants_repo", 0)
             for r in res.values()) and res["first_action_nv"]["mismatch"] == 0
    log(f"CHECK vec==repo prefix={npre} {res} {'PASS' if ok else 'FAIL'}")
    return dict(prefix=npre, cases=res, pass_=ok)


# ------------------------------------------------------------------ analysis
def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def boot_sums(M, nb, seed):
    nblk = M.shape[0]
    rng = np.random.default_rng(seed)
    S = np.empty((nb, M.shape[1]))
    for j0 in range(0, nb, 20):
        m = min(20, nb - j0)
        idx = rng.integers(0, nblk, (m, nblk))
        W = np.bincount((idx + (np.arange(m) * nblk)[:, None]).ravel(),
                        minlength=m * nblk).reshape(m, nblk).astype(float)
        S[j0:j0 + m] = W @ M
    return S


def analyze(run_dir, nb=NB):
    res = dict(bootstrap=dict(resamples=nb, seed=BSEED, note="paired block bootstrap, same resamples for every "
                              "policy of a flow, default_rng(seed) restarted per flow, 95% percentile"),
               unit="bp of $49.50", flows={})
    for flow in FLOWS:
        p = run_dir / "npz" / f"flow_{flow}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        cols = [k for k in z.files if k != "meta"]
        ix = {k: i for i, k in enumerate(cols)}
        M = np.column_stack([z[k].astype(float) for k in cols])
        n = M.shape[0] * PPE
        S = boot_sums(M, nb, BSEED)
        P = M.sum(axis=0)
        c = lambda s, k: s[..., ix[k]]

        def stat(f):
            return dict(point=float(f(P)), ci=ci(f(S)))
        bp = lambda x: x / DEN * 1e4
        pairs = {"comp-comp_nv": ("comp", "comp_nv"), "comp_nv-cutoff": ("comp_nv", "cutoff"),
                 "comp_nv-b4main": ("comp_nv", "b4main"), "comp-b4main": ("comp", "b4main"),
                 "comp-cutoff": ("comp", "cutoff")}
        r = dict(meta=meta, J={q: stat(lambda s, q=q: c(s, f"pay_{q}") / n) for q in POLICIES},
                 diff_bp={k: stat(lambda s, a=a, b=b: bp((c(s, f"pay_{a}") - c(s, f"pay_{b}")) / n))
                          for k, (a, b) in pairs.items()},
                 misuse_granted_pct={q: stat(lambda s, q=q: 100.0 * c(s, f"misn_{q}") / c(s, "bs_n_mis"))
                                     for q in POLICIES},
                 misuse_exposure_usd_per_payment={q: stat(lambda s, q=q: c(s, f"misv_{q}") / n) for q in POLICIES},
                 misuse_exposure_usd_total={q: float(P[ix[f"misv_{q}"]]) for q in POLICIES},
                 misuse_total=dict(n=float(P[ix["bs_n_mis"]]), usd=float(P[ix["bs_v_mis"]])),
                 grant_pct={q: float(100.0 * P[ix[f"gr_{q}"]] / n) for q in POLICIES})
        fa = meta["first_actions"]["comp_nv"]
        tot = sum(sum(x.values()) for x in fa.values())
        r["comp_nv_first_share_pct"] = {reg: {a: 100.0 * fa[reg][a] / tot for a in ACTS} for reg in fa}
        r["comp_nv_first_share_pct"]["all"] = {a: 100.0 * sum(fa[reg][a] for reg in fa) / tot for a in ACTS}
        ml = MAIN_LEAD[flow]
        got = r["diff_bp"]["comp-b4main"]
        r["main_check"] = dict(main_comp_minus_b4=list(ml), got=[round(got["point"], 1)] + [round(x, 1) for x in got["ci"]],
                               point_match_0p1=round(got["point"], 1) == ml[0])
        if flow == "F2":
            b = meta["b4main"]
            r["main_check"]["b4_params_match_D_repro"] = (b["k"], b["a"], b["b"], b["tune_mean"]) == (
                MAIN_B4_F2["k"], MAIN_B4_F2["a"], MAIN_B4_F2["b"], MAIN_B4_F2["tune_mean"])
        res["flows"][flow] = r
    write_json(run_dir / "summary.json", res)
    return res


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "run", "analyze"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--flows", default=None)
    ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase == "smoke" else None)
    run_dir = paths.NO_CHANNEL / tag if tag else paths.NO_CHANNEL
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.phase == "analyze":
        analyze(run_dir, a.nb)
        return 0
    prog = (run_dir / "progress.log").open("a", buffering=1)
    T0 = time.time()

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [{(time.time() - T0) / 60:6.1f}m] {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    log(f"START phase={a.phase} tag={tag} pid={os.getpid()}")
    world, ex = HH.setup()
    T, Tn, vg, info = tables(world, paths.CACHE, log)
    flows = a.flows.split(",") if a.flows else (["F2"] if a.phase == "smoke" else list(FLOWS))
    if a.phase == "smoke":
        chk = smoke_checks(world, ex, T, Tn, vg, log)
        write_json(run_dir / "checks.json", dict(vec_vs_repo=chk, table_nv=info))
        n_eval, n_tune = 100_000, 20_000
    else:
        n_eval, n_tune = N_EVAL, N_TUNE
    fails = []
    for fl in flows:
        try:
            run_flow(fl, run_dir, n_eval, n_tune, world, ex, T, Tn, vg, log)
        except Exception:
            fails.append(fl)
            log(f"FAIL {fl}")
            with (run_dir / "errors.log").open("a") as f:
                f.write(f"FAIL {fl}\n{traceback.format_exc()}\n")
    if fails:
        log(f"INCOMPLETE {fails}")
        return 1
    t1 = time.time()
    analyze(run_dir, 100 if a.phase == "smoke" else a.nb)
    log(f"ANALYZE t={time.time() - t1:.1f}s")
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={(time.time() - T0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
