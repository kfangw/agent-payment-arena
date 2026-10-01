"""Per-draw input-error experiments J, C, K in the E-fitted world.

Run from the package root:
    PYTHONPATH=src python3 scripts/run_queue.py [--workers 2]        
    PYTHONPATH=src python3 scripts/run_queue.py --only F2:nominal      (single item)

Every item compiles one policy and replays it on the full evaluation draw
(5,663,400 payments, 113,268 blocks of 50) of its flow, then writes one npz with
the per-block payoff sums (float64) and metadata.  Items whose npz exists are
skipped, so the queue is resumable.  Draw generation, tuning split, q_hat and the
replicate RNG follow wrong_constants.py exactly (seed, order, clipping).

Payoff is always computed from the true world env: replay_outage(world, ...)
reads m, h, C, cw, f, tau, H from the env passed to it; the compiled policy is a
label table only.  Cost errors (K) therefore change only the compiler input.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from arena.experiments.settlement.core import rho_hat_from_q
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (
    PI_GRID_OUTAGE, compile_outage, draw_outage_batch, replay_outage, survival, window_AD)
from paths import DRAW_INJECTION, WRONG_CONSTANTS

OUT = DRAW_INJECTION
NPZ = OUT / "npz"
REF = WRONG_CONSTANTS
PPE = 50
N_EVAL = 5_663_400
N_TUNE = 200_000
SEEDS = {"F1": 7, "F2": 8, "F3": 9}
ETAS = [0.05, 0.1, 0.15, 0.2]
R = 20
K_CONSTS = ["C", "m", "h", "cw"]          # OutageEnv field names, draw order C, m, h, c_w
K_FACTORS = [0.5, 0.8, 1.25, 2.0]
K_ETAS = [0.1, 0.2]
K_R = 10


# ------------------------------------------------------------------ items
def item_name(it):
    flow, exp, mode, eta, r = it
    if exp == "nominal":
        return f"{flow}/nominal.npz"
    if exp == "K" and mode.startswith("oat"):
        return f"{flow}/K_{mode}.npz"                      # e.g. K_oat-C-x0.5
    return f"{flow}/{exp}_{mode}_eta{eta:.2f}_r{r:02d}.npz"


def build_queue():
    """Fixed order (decided before any result)."""
    q = []
    J = lambda f, e: [(f, "J", "joint", e, r) for r in range(R)]
    Cq = lambda e: [("F2", "C", "q", e, r) for r in range(R)]
    Ch = lambda e: [("F2", "C", "haz", e, r) for r in range(R)]
    # 1) F2 nominal + joint 0.1 (repro check) + joint 0.2
    q += [("F2", "nominal", "-", 0.0, 0)] + J("F2", 0.1) + J("F2", 0.2)
    n_check = 1 + R
    # 2) F2 C q-only and hazard-only at 0.1, 0.2
    for e in (0.1, 0.2):
        q += Cq(e) + Ch(e)
    # 3) F2 joint and C at 0.05, 0.15
    for e in (0.05, 0.15):
        q += J("F2", e) + Cq(e) + Ch(e)
    # 4) K on F2 (order change only, no item added or removed): one-at-a-time then joint lognormal
    for c in K_CONSTS:
        for a in K_FACTORS:
            q += [("F2", "K", f"oat-{c}-x{a:g}", 0.0, 0)]
    for e in K_ETAS:
        q += [("F2", "K", "joint", e, r) for r in range(K_R)]
    # 5) F1 and F3: nominal + joint at all four levels
    for f in ("F1", "F3"):
        q += [(f, "nominal", "-", 0.0, 0)]
        for e in ETAS:
            q += J(f, e)
    return q, n_check


# ------------------------------------------------------------------ worker
_CACHE = {}


def load_flow(flow):
    if flow in _CACHE:
        return _CACHE[flow]
    _CACHE.clear()
    gc.collect()
    t0 = time.time()
    envs = envs_for("mid")
    _, world, _ = envs["E-fitted"]
    fl = make_flows()[flow]
    seed = SEEDS[flow]
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(world, fl, N_TUNE, rng, payments_per_episode=PPE)
    eval_d = draw_outage_batch(world, fl, N_EVAL, rng, payments_per_episode=PPE)
    q_hat = float((tune_d.t_ans <= world.tau).mean())
    del tune_d
    _, _, ex = window_AD(world, survival(world))
    env_hat = replace(world, rho=rho_hat_from_q(q_hat, world.tau))
    _CACHE[flow] = dict(world=world, eval_d=eval_d, ex=ex, q_hat=q_hat, env_hat=env_hat,
                        seed=seed, load_s=time.time() - t0)
    return _CACHE[flow]


def block_sums(pay):
    n = len(pay) // PPE * PPE
    return pay[:n].reshape(-1, PPE).sum(axis=1)


def first_action_counts(pol, d, H):
    """Counts of GRANT/REJECT/VERIFY/WAIT at arrival state (i=0, l=H, r=path[0]),
    same lookup rule as CompiledOutagePolicy.__call__ (vectorised)."""
    lv = np.log(pol.v_grid)
    ip = np.round(d.pi0 * (len(PI_GRID_OUTAGE) - 1)).astype(np.int64)
    r0 = d.paths[:, 0].astype(np.int64)
    tabs = np.stack([t[0, H] for t in pol.tables])        # (n_v, 2, n_pi)
    cnt = np.zeros(4, dtype=np.int64)
    for s in range(0, len(d), 500_000):
        sl = slice(s, s + 500_000)
        iv = np.argmin(np.abs(lv[None, :] - np.log(np.maximum(d.v[sl], 1e-12))[:, None]), axis=1)
        a = tabs[iv, r0[sl], ip[sl]]
        cnt += np.bincount(a, minlength=4)[:4]
    return cnt


def make_env(it, c):
    """Compiler input env for an item plus metadata."""
    flow, exp, mode, eta, r = it
    world, q_hat, seed = c["world"], c["q_hat"], c["seed"]
    meta = dict(flow=flow, exp=exp, mode=mode, eta=eta, r=r, seed=seed, q_hat=q_hat)
    f_hat, qh = world.f.copy(), q_hat
    cost = dict(C=world.C, m=world.m, h=world.h, cw=world.cw)
    fac = dict(C=1.0, m=1.0, h=1.0, cw=1.0)
    if exp in ("J", "C"):
        g = np.random.default_rng([seed, 700, r, int(eta * 1000)])
        z = g.standard_normal(len(world.f))                # hazards first
        zq = g.standard_normal()                           # then q
        raw_f = world.f * np.exp(eta * z)
        raw_q = q_hat * np.exp(eta * zq)
        use_f = exp == "J" or mode == "haz"
        use_q = exp == "J" or mode == "q"
        if use_f:
            f_hat = np.clip(raw_f, 0.0, 1.0)
        if use_q:
            qh = float(np.clip(raw_q, 0.0, 1.0))
        meta.update(z=z.tolist(), z_q=float(zq), perturb_f=use_f, perturb_q=use_q,
                    clip_f=int(((raw_f > 1) | (raw_f < 0)).sum()) if use_f else 0,
                    clip_q=int(raw_q > 1 or raw_q < 0) if use_q else 0)
    elif exp == "K":
        if mode.startswith("oat"):
            _, name, x = mode.split("-")
            fac[name] = float(x[1:])
        else:
            g = np.random.default_rng([seed, 900, r, int(eta * 1000)])
            zk = g.standard_normal(4)                      # order C, m, h, cw
            for k, zz in zip(K_CONSTS, zk):
                fac[k] = float(np.exp(eta * zz))
            meta.update(z_cost=zk.tolist())
    env = replace(world, f=f_hat, rho=rho_hat_from_q(qh, world.tau),
                  **{k: cost[k] * fac[k] for k in K_CONSTS})
    meta.update(f_hat=np.asarray(f_hat).tolist(), q_hat_prime=qh, rho_hat=env.rho,
                cost_factors=fac, cost_values={k: getattr(env, k) for k in K_CONSTS},
                f_hat_min=float(np.min(f_hat)), f_hat_max=float(np.max(f_hat)),
                f_hat_logratio_mean=float(np.mean(np.log(np.asarray(f_hat) / world.f))))
    return env, meta


def run_item(it):
    try:
        path = NPZ / item_name(it)
        if path.exists():
            return it, "SKIP", 0.0, None
        t_all = time.time()
        c = load_flow(it[0])
        env, meta = make_env(it, c)
        t0 = time.time()
        pol = compile_outage(env, f"{it[1]}-{it[2]}")
        t1 = time.time()
        pay = replay_outage(c["world"], c["eval_d"], pol, c["ex"])   # payoff from the TRUE world
        t2 = time.time()
        bs = block_sums(pay)
        acts = first_action_counts(pol, c["eval_d"], c["world"].H)
        meta.update(compile_s=t1 - t0, replay_s=t2 - t1, load_s=c["load_s"],
                    n_payments=len(pay), total_payoff=float(pay.sum()),
                    first_action_counts=dict(zip(["grant", "reject", "verify", "wait"],
                                                 acts.tolist())))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, block_sums=bs, first_action_counts=acts,
                            meta=np.array(json.dumps(meta)))
        os.replace(tmp, path)
        return it, "DONE", time.time() - t_all, None
    except Exception:
        return it, "FAIL", 0.0, traceback.format_exc()


# ------------------------------------------------------------------ checks
def load_bs(it):
    return np.load(NPZ / item_name(it))["block_sums"]


def check_nominal(flow):
    ref = np.asarray(json.loads((REF / f"wrong_E-fitted_{flow}_s{SEEDS[flow]}.json").read_text())["true"])
    got = load_bs((flow, "nominal", "-", 0.0, 0))
    d = float(np.max(np.abs(got - ref)))
    ok = len(got) == len(ref) and np.allclose(got, ref, rtol=1e-12, atol=1e-9)
    return ok, f"nominal {flow} vs JSON true: n={len(got)} max|diff|={d:.3e}"


def check_eta01():
    ref = np.asarray(json.loads((REF / "wrong_E-fitted_F2_s8.json").read_text())["eta0.1"])
    got = np.mean([load_bs(("F2", "J", "joint", 0.1, r)) for r in range(R)], axis=0)
    d = float(np.max(np.abs(got - ref)))
    ok = len(got) == len(ref) and np.allclose(got, ref, rtol=1e-9, atol=1e-8)
    return ok, f"mean of 20 F2 joint eta=0.1 draws vs JSON eta0.1: max|diff|={d:.3e}"


# ------------------------------------------------------------------ main
def fmt_it(it):
    flow, exp, mode, eta, r = it
    e = "-" if exp == "nominal" or mode.startswith("oat") else f"{eta:g}"
    return f"{flow} {exp} {mode} eta={e} r={r}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default=None, help="run one item: F2:nominal")
    a = ap.parse_args(argv)
    NPZ.mkdir(parents=True, exist_ok=True)
    if a.only:
        flow, what = a.only.split(":")
        it, st, t, tb = run_item((flow, "nominal", "-", 0.0, 0) if what == "nominal" else eval(what))
        print(st, fmt_it(it), f"t={t:.0f}", tb or "")
        if what == "nominal":
            print(check_nominal(flow))
        return
    queue, n_check = build_queue()
    prog = open(OUT / "progress.log", "a", buffering=1)
    errs = open(OUT / "errors.log", "a", buffering=1)
    t_start = time.time()
    prog.write(f"START {time.strftime('%Y-%m-%d %H:%M:%S')} items={len(queue)} workers={a.workers}\n")
    check_set = set(range(n_check))
    done_idx, failed = set(), False
    nominal_idx = {i: it[0] for i, it in enumerate(queue) if it[1] == "nominal"}
    checked = False
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, maxtasksperchild=None) as pool:
        results = pool.imap_unordered(_run_idx, list(enumerate(queue)), chunksize=1)
        for i, (it, st, t, tb) in results:
            el = time.time() - t_start
            hhmm = f"{int(el // 3600):02d}:{int(el % 3600 // 60):02d}"
            if st == "FAIL":
                prog.write(f"FAIL {fmt_it(it)} elapsed={hhmm}\n")
                errs.write(f"FAIL {fmt_it(it)}\n{tb}\n")
                if i in check_set:
                    failed = True
            else:
                prog.write(f"{st} {fmt_it(it)} t={t:.0f} elapsed={hhmm}\n")
                done_idx.add(i)
                if i in nominal_idx:
                    ok, msg = check_nominal(nominal_idx[i])
                    prog.write(f"CHECK1 {'PASS' if ok else 'FAIL'} {msg}\n")
                    if not ok and nominal_idx[i] == "F2":
                        failed = True
            if not checked and check_set <= done_idx:
                checked = True
                ok, msg = check_eta01()
                prog.write(f"CHECK2 {'PASS' if ok else 'FAIL'} {msg}\n")
                (OUT / "check2.txt").write_text(f"{'PASS' if ok else 'FAIL'} {msg}\n")
                if not ok:
                    failed = True
            if failed:
                prog.write("STOPPED reproduction check failed or check item failed\n")
                pool.terminate()
                return 1
    prog.write(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={(time.time() - t_start) / 3600:.2f}h\n")
    return 0


def _run_idx(pair):
    i, it = pair
    return i, run_item(it)


if __name__ == "__main__":
    sys.exit(main())
