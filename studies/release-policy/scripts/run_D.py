"""Experiment D: policy loss of the compiled release policy against the number n
of observed answers used to estimate the answer rate q, and the B4 rule tuned on
the same n payments.  E-fitted world, flow F2, true q = 0.75.

Run (Python 3.11):
    PYTHONPATH=src python3 scripts/run_D.py plan        
    PYTHONPATH=src python3 scripts/run_D.py eval --workers 2
    PYTHONPATH=src python3 scripts/run_D.py all --workers 2   (plan, then eval)
    PYTHONPATH=src python3 scripts/run_D.py repro             (reproduction checks)
    PYTHONPATH=src python3 scripts/run_D.py all --workers 1 --seeds 1,2 \
        --n-eval 100000 --tag smoke                                                (smoke test)

Phases
  plan  per learning seed s: draw 5,000 learning payments from default_rng([108, 500, s]),
        read the pilot answer indicators (only field: t_ans <= tau) for Comp, and tune B4
        on the n-prefix with the existing tuner.  One json per seed in <run>/plan/.
  eval  draw the evaluation set (default_rng(108), 5,663,400 payments, 50 per episode),
        compile every distinct q (plan q-hats, fixed 0.60/0.75/0.90, and K/130 for every K
        with Binomial(130, 0.75) pmf > 1e-6), group q values whose compiled label tables are
        identical (sha256), replay one policy per distinct table and one per distinct B4
        (k, a, b).  One npz per evaluated policy (block sums of 50), written via .tmp rename.
        Existing npz are skipped (resumable).  progress.log gets DONE / FAIL / ALLDONE lines.

Payoff is always computed in the true world env; the compiled policy is a label table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import binom

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY, rho_hat_from_q
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (
    PI_GRID_OUTAGE, compile_outage, draw_outage_batch, replay_outage, survival, window_AD)
from arena.experiments.settlement.policies import suspicion_grid
from arena.experiments.settlement.watch import (
    FixedActionOutageWatchPolicy as OB4Force, OutageWatchBandPolicy as OB4,
    horizon_grid, tune_watch_policy)
from paths import D_ANSWERS, REFERENCE_B4, WRONG_CONSTANTS

OUT = D_ANSWERS
PPE = 50
N_EVAL = 5_663_400
EVAL_SEED = 108
LEARN_TAG = 500
N_LEARN = 5_000
NS = (50, 130, 500, 5000)
SEEDS = tuple(range(1, 31))
Q_TRUE = 0.75
Q_FIXED = (0.60, 0.75, 0.90)
B3_N = 161
BINOM_N, BINOM_EPS = 130, 1e-6
FLOW = "F2"

# old (development) evaluation set for the reproduction check
REF_WRONG = WRONG_CONSTANTS / "wrong_E-fitted_F2_s8.json"
REF_B4 = REFERENCE_B4 / "b4ref_E-fitted_F2_mid_s8.json"
OLD_SEED, OLD_TUNE = 8, 200_000


# ------------------------------------------------------------------ world
def world_env():
    _, world, _ = envs_for("mid")["E-fitted"]
    return world


def block_sums(pay):
    n = len(pay) // PPE * PPE
    return pay[:n].reshape(-1, PPE).sum(axis=1)


# ------------------------------------------------------------------ info contract
class FieldGuard:
    """Read-only view of a draw batch that records which fields are read."""

    def __init__(self, d):
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "read", set())

    def __getattr__(self, name):
        self.read.add(name)
        return getattr(self._d, name)

    def __setattr__(self, name, value):
        raise AttributeError("read-only")


def pilot_answers(draws, tau):
    """What the pilot records when it asks every payment: did the answer arrive
    within the deadline tau.  Reads t_ans only (no payoff, intent or value field)."""
    return np.asarray(draws.t_ans <= tau, dtype=bool)


def q_hat_from_answers(answers):
    """q-hat = answers within tau / n.  Timeouts stay in the denominator.
    q-hat = 0 and 1 are passed to the compiler unchanged (rho = 0 or 1 are valid
    inputs; checked in the smoke test)."""
    a = np.asarray(answers, dtype=bool)
    return int(a.sum()) / len(a)


def comp_env(world, q):
    """Compiler input: the world constants with rho from the declared or estimated q."""
    return replace(world, rho=rho_hat_from_q(q, world.tau))


def table_hash(pol):
    h = hashlib.sha256()
    for t in pol.tables:
        h.update(np.ascontiguousarray(t).tobytes())
    h.update(np.asarray(pol.v_grid).tobytes())
    return h.hexdigest()


def qkey(q):
    return repr(float(q))


def binom_ks():
    ks = np.arange(BINOM_N + 1)
    pmf = binom.pmf(ks, BINOM_N, Q_TRUE)
    return [int(k) for k in ks[pmf > BINOM_EPS]]


# ------------------------------------------------------------------ plan phase
def plan_seed(s, world, ex, ab_grid, ks, n_list=NS):
    g = np.random.default_rng([EVAL_SEED, LEARN_TAG, s])
    learn = draw_outage_batch(world, make_flows()[FLOW], N_LEARN, g, payments_per_episode=PPE)
    # Comp path: only the pilot answer indicators
    guard = FieldGuard(learn)
    answers = pilot_answers(guard, world.tau)
    assert guard.read == {"t_ans"}, f"Comp path read fields {guard.read}"
    rec = dict(seed=s, rng=[EVAL_SEED, LEARN_TAG, s], n_learn=N_LEARN, comp_fields_read=sorted(guard.read),
               per_n={})
    t0 = time.time()
    # B4 path: forced payoffs on the whole learning set; the n-prefix of a replay equals the
    # replay of the n-prefix (payments are replayed independently).
    force = {}
    for k in ks:
        for act in (GRANT, REJECT, VERIFY):
            force[(k, act)] = replay_outage(world, learn, OB4Force(k, act, world.H, world.N), ex)
    t_force = time.time() - t0
    for n in n_list:
        K = int(answers[:n].sum())
        q = q_hat_from_answers(answers[:n])
        t1 = time.time()
        best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)][:n], ab_grid, ks,
                                            learn.pi0[:n])
        rec["per_n"][str(n)] = dict(n=n, K=K, q_hat=q, b4=dict(k=best[0], a=best[1], b=best[2],
                                    tune_mean=val), b4_k_rows=rows, t_tune=time.time() - t1)
    rec["t_force"] = t_force
    rec["t_total"] = time.time() - t0
    return rec


def run_plan(run_dir, seeds, log):
    pdir = run_dir / "plan"
    pdir.mkdir(parents=True, exist_ok=True)
    world = world_env()
    _, _, ex = window_AD(world, survival(world))
    ab_grid = suspicion_grid(B3_N)
    ks = horizon_grid(world.H)
    for s in seeds:
        path = pdir / f"seed_{s:02d}.json"
        if path.exists():
            continue
        t0 = time.time()
        rec = plan_seed(s, world, ex, ab_grid, ks)
        rec.update(b3_n=B3_N, k_grid=ks, n_grid=len(ab_grid))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, path)
        log(f"PLAN seed={s} t={time.time() - t0:.0f} " + " ".join(
            f"n{n}:q={v['q_hat']:.4f},b4=({v['b4']['k']},{v['b4']['a']:.4f},{v['b4']['b']:.4f})"
            for n, v in rec["per_n"].items()))


def load_plan(run_dir, seeds):
    return {s: json.loads((run_dir / "plan" / f"seed_{s:02d}.json").read_text()) for s in seeds}


def distinct_items(plan):
    qs = {qkey(q): float(q) for q in Q_FIXED}
    for k in binom_ks():
        qs.setdefault(qkey(k / BINOM_N), k / BINOM_N)
    b4 = {}
    for rec in plan.values():
        for v in rec["per_n"].values():
            qs.setdefault(qkey(v["q_hat"]), float(v["q_hat"]))
            p = v["b4"]
            b4[b4_name(p["k"], p["a"], p["b"])] = (int(p["k"]), float(p["a"]), float(p["b"]))
    return qs, b4


def b4_name(k, a, b):
    return f"b4_k{int(k)}_a{float(a)!r}_b{float(b)!r}"


# ------------------------------------------------------------------ eval phase (workers)
_G = {}


def _init_eval(n_eval):
    world = world_env()
    t0 = time.time()
    d = draw_outage_batch(world, make_flows()[FLOW], n_eval, np.random.default_rng(EVAL_SEED),
                          payments_per_episode=PPE)
    _, _, ex = window_AD(world, survival(world))
    _G.update(world=world, eval_d=d, ex=ex, draw_s=time.time() - t0, n_eval=n_eval)


def first_action_counts(pol, d, H):
    lv = np.log(pol.v_grid)
    ip = np.round(d.pi0 * (len(PI_GRID_OUTAGE) - 1)).astype(np.int64)
    r0 = d.paths[:, 0].astype(np.int64)
    tabs = np.stack([t[0, H] for t in pol.tables])
    cnt = np.zeros(4, dtype=np.int64)
    for s in range(0, len(d), 500_000):
        sl = slice(s, s + 500_000)
        iv = np.argmin(np.abs(lv[None, :] - np.log(np.maximum(d.v[sl], 1e-12))[:, None]), axis=1)
        cnt += np.bincount(tabs[iv, r0[sl], ip[sl]], minlength=4)[:4]
    return cnt


def job_hash(q):
    try:
        t0 = time.time()
        pol = compile_outage(comp_env(_G["world"], q), "comp")
        return q, table_hash(pol), time.time() - t0, None
    except Exception:
        return q, None, 0.0, traceback.format_exc()


def _save(path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def job_eval(item):
    kind, name, spec = item
    try:
        path = _G["npz"] / f"{name}.npz"
        if path.exists():
            return item, "SKIP", {}, None
        world, d, ex = _G["world"], _G["eval_d"], _G["ex"]
        t0 = time.time()
        if kind == "comp":
            q = spec["q"]
            pol = compile_outage(comp_env(world, q), "comp")
            assert table_hash(pol) == spec["hash"], "table hash changed between phases"
            t1 = time.time()
            pay = replay_outage(world, d, pol, ex)
            t2 = time.time()
            acts = first_action_counts(pol, d, world.H)
            meta = dict(kind=kind, q_compiled=q, q_all=spec["q_all"], hash=spec["hash"],
                        first_action_counts=dict(zip(["grant", "reject", "verify", "wait"],
                                                     acts.tolist())))
        else:
            k, a, b = spec["k"], spec["a"], spec["b"]
            pol = OB4(k, a, b, world.H, world.N)
            t1 = time.time()
            pay = replay_outage(world, d, pol, ex)
            t2 = time.time()
            meta = dict(kind=kind, k=k, a=a, b=b)
        meta.update(compile_s=t1 - t0, replay_s=t2 - t1, n_payments=len(pay),
                    total_payoff=float(pay.sum()), eval_seed=EVAL_SEED, n_eval=_G["n_eval"])
        _save(path, block_sums=block_sums(pay), meta=np.array(json.dumps(meta)))
        return item, "DONE", dict(compile_s=t1 - t0, replay_s=t2 - t1), None
    except Exception:
        return item, "FAIL", {}, traceback.format_exc()


def run_eval(run_dir, seeds, n_eval, workers, log):
    plan = load_plan(run_dir, seeds)
    qs, b4 = distinct_items(plan)
    npz = run_dir / "npz"
    npz.mkdir(parents=True, exist_ok=True)
    _init_eval(n_eval)
    _G["npz"] = npz
    log(f"EVALSET seed={EVAL_SEED} n={n_eval} draw_s={_G['draw_s']:.0f}")
    qmap_path = run_dir / "qmap.json"
    qmap = json.loads(qmap_path.read_text()) if qmap_path.exists() else {}
    todo = [q for key, q in sorted(qs.items(), key=lambda kv: kv[1]) if key not in qmap]
    ctx = mp.get_context("fork")
    t_start = time.time()
    with ctx.Pool(workers) as pool:
        for q, h, t, tb in pool.imap_unordered(job_hash, todo, chunksize=1):
            if h is None:
                log(f"FAIL hash q={q!r}")
                (run_dir / "errors.log").open("a").write(f"FAIL hash q={q!r}\n{tb}\n")
                return 1
            qmap[qkey(q)] = h
            tmp = qmap_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(qmap, indent=0))
            os.replace(tmp, qmap_path)
        # group q by table
        groups = {}
        for key, q in qs.items():
            groups.setdefault(qmap[key], []).append(q)
        ref_hash = qmap[qkey(Q_TRUE)]
        items = []
        for h, ql in sorted(groups.items(), key=lambda kv: (kv[0] != ref_hash, min(kv[1]))):
            ql = sorted(ql)
            items.append(("comp", f"comp_{h[:16]}", dict(q=ql[0], q_all=ql, hash=h)))
        for name, (k, a, bb) in sorted(b4.items()):
            items.append(("b4", name, dict(k=k, a=a, b=bb)))
        manifest = dict(eval_seed=EVAL_SEED, n_eval=n_eval, seeds=list(seeds), ns=list(NS),
                        n_q_distinct=len(qs), n_tables_distinct=len(groups), n_b4_distinct=len(b4),
                        ref_hash=ref_hash, binom_ks=binom_ks(),
                        items=[dict(kind=k, name=n, **{kk: vv for kk, vv in sp.items()})
                               for k, n, sp in items])
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        log(f"ITEMS q_distinct={len(qs)} tables={len(groups)} b4={len(b4)} total={len(items)} "
            f"hash_phase_s={time.time() - t_start:.0f}")
        for it, st, tm, tb in pool.imap_unordered(job_eval, items, chunksize=1):
            el = time.time() - t_start
            if st == "FAIL":
                log(f"FAIL {it[1]} elapsed={el / 60:.1f}m")
                (run_dir / "errors.log").open("a").write(f"FAIL {it[1]}\n{tb}\n")
            else:
                extra = " ".join(f"{k}={v:.1f}" for k, v in tm.items())
                log(f"{st} {it[1]} {extra} elapsed={el / 60:.1f}m")
    missing = [n for _, n, _ in items if not (npz / f"{n}.npz").exists()]
    if missing:
        log(f"INCOMPLETE missing={len(missing)}")
        return 1
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={(time.time() - t_start) / 60:.1f}m")
    return 0


# ------------------------------------------------------------------ reproduction checks
def run_repro(run_dir, log, b4_subset=None):
    world = world_env()
    fl = make_flows()[FLOW]
    rng = np.random.default_rng(OLD_SEED)
    t0 = time.time()
    tune_d = draw_outage_batch(world, fl, OLD_TUNE, rng, payments_per_episode=PPE)
    eval_d = draw_outage_batch(world, fl, N_EVAL, rng, payments_per_episode=PPE)
    _, _, ex = window_AD(world, survival(world))
    log(f"REPRO draws s={time.time() - t0:.0f}")
    res = {}
    # 1a) Comp at q-hat estimated from the old tuning split, through the D code path
    guard = FieldGuard(tune_d)
    q = q_hat_from_answers(pilot_answers(guard, world.tau))
    assert guard.read == {"t_ans"}
    q_old = float((tune_d.t_ans <= world.tau).mean())
    t0 = time.time()
    pol = compile_outage(comp_env(world, q), "true")
    pay = replay_outage(world, eval_d, pol, ex)
    got = block_sums(pay)
    ref = np.asarray(json.loads(REF_WRONG.read_text())["true"])
    diff = float(np.max(np.abs(got - ref)))
    res["comp"] = dict(q_hat=q, q_hat_wrong_constants=q_old, same_q=q == q_old, n_blocks=len(got),
                       n_blocks_ref=len(ref), max_abs_diff=diff, t=time.time() - t0,
                       pass_=bool(len(got) == len(ref) and diff <= 1e-9))
    log(f"REPRO comp q_hat={q!r} (wrong_constants {q_old!r}) blocks={len(got)} max|diff|={diff:.3e} "
        f"t={time.time() - t0:.0f}s {'PASS' if res['comp']['pass_'] else 'FAIL'}")
    del eval_d, pay
    # 1b) B4 tuning on the standard tuning split (or a documented prefix)
    src = tune_d
    n_used = OLD_TUNE if b4_subset is None else b4_subset
    ks = horizon_grid(world.H)
    ab = suspicion_grid(B3_N)
    t0 = time.time()
    force = {}
    for k in ks:
        for act in (GRANT, REJECT, VERIFY):
            force[(k, act)] = replay_outage(world, src, OB4Force(k, act, world.H, world.N), ex)[:n_used]
    t_force = time.time() - t0
    best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)], ab, ks, src.pi0[:n_used])
    t_tune = time.time() - t0 - t_force
    rp = json.loads(REF_B4.read_text())["payload"]
    want = rp["b4"]
    ok = (best[0] == want["k"] and best[1] == want["a"] and best[2] == want["b"])
    rows_ok = all(r["k"] == w["k"] and r["a"] == w["a"] and r["b"] == w["b"]
                  and abs(r["tune_mean"] - w["tune_mean"]) <= 1e-12 for r, w in zip(rows, rp["b4_k_rows"]))
    res["b4"] = dict(n_tune_used=n_used, got=dict(k=best[0], a=best[1], b=best[2], tune_mean=val),
                     want=want, params_equal=ok, tune_mean_diff=abs(val - want["tune_mean"]),
                     k_rows_equal=rows_ok, b3_n=rp["b3_n"], t_force=t_force, t_grid=t_tune,
                     pass_=bool(ok))
    log(f"REPRO b4 n_tune={n_used} got=({best[0]},{best[1]!r},{best[2]!r}) tune_mean={val!r} "
        f"want=({want['k']},{want['a']!r},{want['b']!r}) tune_mean={want['tune_mean']!r} "
        f"rows_equal={rows_ok} t_force={t_force:.0f}s t_grid={t_tune:.0f}s {'PASS' if ok else 'FAIL'}")
    (run_dir / "repro.json").write_text(json.dumps(res, indent=1))
    return 0 if res["comp"]["pass_"] and res["b4"]["pass_"] else 1


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["plan", "eval", "all", "repro"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default=None, help="comma list; default 1..30")
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--tag", default=None, help="write under <OUT>/<tag>/ instead of <OUT>/")
    ap.add_argument("--b4-subset", type=int, default=None, help="repro: tune B4 on a prefix")
    a = ap.parse_args(argv)
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
    run_dir = OUT / a.tag if a.tag else OUT
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = (run_dir / "progress.log").open("a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    log(f"START phase={a.phase} seeds={seeds[0]}..{seeds[-1]}({len(seeds)}) n_eval={a.n_eval} "
        f"workers={a.workers} pid={os.getpid()}")
    if a.phase == "repro":
        return run_repro(run_dir, log, a.b4_subset)
    if a.phase in ("plan", "all"):
        run_plan(run_dir, seeds, log)
        log("PLANDONE")
    if a.phase in ("eval", "all"):
        return run_eval(run_dir, seeds, a.n_eval, a.workers, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
