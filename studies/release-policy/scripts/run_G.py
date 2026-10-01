"""Experiment G: re-estimating the score calibration from the pilot answers.
E-fitted world, flow F2, true q = 0.75.  The policy reads s = clip(zeta * pi, 0, 1); the
simulator keeps the true pi and intent (experiment 2's prior-miscalibration axis:
replace(draws, pi0=clip(level * pi0, 0, 1)), inject.py / inject_b4.py).  Spec: docs/calibration_pilot.md.

Run from the package root (Python 3.11, numpy, scipy):
    export PYTHONPATH=$PWD/src
    python3 scripts/run_G.py check                       (spec 9.1-9.4)
    python3 scripts/run_G.py all --tag smoke --n-eval 100000 --seeds 1,2,3 \
        --ns 130,2000 --zetas 1,2 --workers 2                        (spec 9.5)
    python3 scripts/run_G.py all --workers 2              (full run)
    python3 scripts/analyze_G.py [--tag smoke]
Output goes to results/calibration_pilot/ (paths.CALIBRATION_PILOT).

Phases
  plan   per learning seed s (1..30): 5,000 learning payments from default_rng([110, 500, s]).
         Comp_naive: pilot answers only (FieldGuard: {t_ans}).  Comp_cal: CalGuard, reads
         exactly {t_ans, theta (answered payments only), pi0 (= s)}; zeta-hat by the spec-5 MLE.
         B4_n: forced payoffs (repo replay_outage) and tune_watch_policy on s, 161-point grid.
         One json per seed in <run>/plan/.  B4_200k(zeta): 200,000 payments from
         default_rng([110, 700]) (<run>/plan/b4_200k.json).
  eval   evaluation set default_rng(110), 5,663,400 payments (50 per episode).  Compile every
         distinct q-hat once (sha256 of the tables), then replay one policy per distinct
         (table, zeta, zeta-hat); B4 policies are per-payment selections among the forced
         arrays of their k (forced policies ignore the score), checked against a direct
         replay.  Runtime rule of spec 6 decides the evaluation size before any replay.
         One npz of block sums per policy (.tmp then rename; existing npz are skipped).
Replay: fastreplay_G.replay_vec, bit-identical to replay_outage (checked in `check` and at
the start of every eval run; --impl repo uses replay_outage itself).
"""
from __future__ import annotations

import argparse
import datetime as dt
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths                            # noqa: E402
OUT = paths.CALIBRATION_PILOT
D_DIR = paths.D_ANSWERS                 # results of run_D.py (qmap.json), used only for a cross-check

import run_D as RD                      # noqa: E402  (world, FieldGuard, q-hat, hashes, block sums)
import run_E1 as RE1                    # noqa: E402  (sel_pay, _save_npy)
import fastreplay_G as FR               # noqa: E402

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY          # noqa: E402
from arena.experiments.settlement.flows import make_flows                    # noqa: E402
from arena.experiments.settlement.outage import (                            # noqa: E402
    compile_outage, draw_outage_batch, replay_outage, survival, window_AD)
from arena.experiments.settlement.policies import suspicion_grid             # noqa: E402
from arena.experiments.settlement.watch import (                             # noqa: E402
    FixedActionOutageWatchPolicy as OB4Force, OutageWatchBandPolicy as OB4,
    horizon_grid, tune_watch_policy)

# ------------------------------------------------------------------ fixed constants (docs/calibration_pilot.md)
EVAL_SEED = 110
LEARN_ROOT = 110
LEARN_TAG = RD.LEARN_TAG           # 500
N_LEARN = RD.N_LEARN               # 5,000
N_EVAL = RD.N_EVAL                 # 5,663,400
N_EVAL_SMALL = 2_000_000           # runtime rule fallback (40,000 blocks)
RULE_HOURS = 5.0
NS = (130, 500, 2000, 5000)
SEEDS = tuple(range(1, 31))
ZETAS = (0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
Q_TRUE = RD.Q_TRUE                 # 0.75
B3_N = RD.B3_N                     # 161
PPE = RD.PPE                       # 50
FLOW = RD.FLOW                     # F2
N_200K, TAG_200K = 200_000, 700    # B4_200k: default_rng([110, 700])
SYN_N, SYN_TAG = 100_000, 900      # check 9.3: default_rng([110, 900])
KGRID = np.arange(-200, 201)
ZGRID = np.exp(KGRID * 0.01)       # zeta-hat grid exp(0.01 k), 0.135 .. 7.39
EPS_LOG = 1e-12
D_REF_J = 8.938881202421832        # D's pi_ref J on eval seed 108
N_CHECK_PREFIX = 500_000           # direct repo-replay checks run on this evaluation prefix

FIELDS_NAIVE = {"t_ans"}
FIELDS_CAL = {"t_ans", "theta", "pi0"}


def zkey(z):
    return repr(float(z))


# ------------------------------------------------------------------ world
def world_env():
    return RD.world_env()


def setup():
    world = world_env()
    _, _, ex = window_AD(world, survival(world))
    return world, ex, list(horizon_grid(world.H)), suspicion_grid(B3_N)


def distort(d, z):
    """The score the policy reads: s = clip(zeta * pi, 0, 1), as experiment 2's kappa axis.
    theta, t_ans, v and the paths stay the true world."""
    return replace(d, pi0=np.clip(z * d.pi0, 0.0, 1.0))


def prefix(d, n):
    return replace(d, **{f: getattr(d, f)[:n] for f in
                         ("v", "p_true", "theta", "pi0", "u_stage", "t_ans", "paths")})


# ------------------------------------------------------------------ information contract
class LabelView:
    """Intent labels behind a gate: only payments whose answer arrived within tau may be read.
    Any other access (unanswered index, whole-array conversion, iteration) raises."""

    def __init__(self, arr, allowed):
        self._a, self._ok = arr, allowed
        self.n_read = 0

    def __len__(self):
        return len(self._a)

    def __getitem__(self, idx):
        pos = np.arange(len(self._a))[idx]
        bad = ~self._ok[pos]
        if np.any(bad):
            raise AssertionError(f"information contract: intent of {int(np.sum(bad))} unanswered "
                                 f"payment(s) read")
        self.n_read += int(np.size(pos))
        return self._a[idx]

    def __array__(self, *a, **k):
        raise AssertionError("information contract: whole intent array read")

    def __iter__(self):
        raise AssertionError("information contract: intent array iterated")


class CalGuard(RD.FieldGuard):
    """FieldGuard for Comp_cal: records fields read; theta is a LabelView gated on t_ans <= tau.
    The gate is computed by the harness from the true draw, not through the guard."""

    def __init__(self, d, tau):
        super().__init__(d)
        object.__setattr__(self, "_theta", LabelView(d.theta, np.asarray(d.t_ans <= tau)))

    def __getattr__(self, name):
        self.read.add(name)
        if name == "theta":
            return self._theta
        return getattr(self._d, name)


def zeta_mle(s, y):
    """Spec 5: MLE of zeta in P(y=1|s) = min(s/zeta, 1) on the grid exp(0.01k), k=-200..200.
    log 0 guarded with 1e-12.  No misuse -> top of the grid; all misuse -> bottom.
    Ties: first maximum (smallest zeta).  Returns (k, flag)."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=bool)
    if len(y) == 0 or not y.any():
        return int(KGRID[-1]), "no_misuse"
    if y.all():
        return int(KGRID[0]), "all_misuse"
    ll = np.zeros(len(ZGRID))
    for c in range(0, len(s), 20_000):
        sc, yc = s[c:c + 20_000], y[c:c + 20_000]
        p = np.minimum(sc[None, :] / ZGRID[:, None], 1.0)
        lp = np.log(np.maximum(p, EPS_LOG))
        l1p = np.log(np.maximum(1.0 - p, EPS_LOG))
        ll += np.where(yc[None, :], lp, l1p).sum(axis=1)
    j = int(np.argmax(ll))
    flag = "grid_edge" if j in (0, len(ZGRID) - 1) else "ok"
    return int(KGRID[j]), flag


def cal_learn(g, n, tau):
    """Comp_cal learning on the first n payments of guarded draws g (distorted world).
    Reads t_ans, theta of answered payments, pi0 (= s).  Returns q-hat inputs and zeta-hat."""
    t = np.asarray(g.t_ans)[:n]
    ans = t <= tau
    K = int(ans.sum())
    idx = np.nonzero(ans)[0]
    y = np.asarray(g.theta[idx]) == 1          # 'misuse' answer
    s = np.asarray(g.pi0, dtype=float)[:n][idx]
    k, flag = zeta_mle(s, y)
    return dict(K=K, q_hat=K / n, zhat_k=k, zhat=float(ZGRID[k + 200]), n_ans=int(len(idx)),
                n_mis_ans=int(y.sum()), flag=flag)


# ------------------------------------------------------------------ plan phase
def plan_seed(s, ns, zetas):
    world, ex, ks, ab = setup()
    t0 = time.time()
    learn = draw_outage_batch(world, make_flows()[FLOW], N_LEARN,
                              np.random.default_rng([LEARN_ROOT, LEARN_TAG, s]), payments_per_episode=PPE)
    guard = RD.FieldGuard(learn)
    answers = RD.pilot_answers(guard, world.tau)
    assert guard.read == FIELDS_NAIVE, f"Comp_naive read {guard.read}"
    force = {}
    for k in ks:
        for act in (GRANT, REJECT, VERIFY):
            force[(k, act)] = replay_outage(world, learn, OB4Force(k, act, world.H, world.N), ex)
    t_force = time.time() - t0
    rec = dict(seed=s, rng=[LEARN_ROOT, LEARN_TAG, s], n_learn=N_LEARN, naive_fields=sorted(guard.read),
               per_n={}, t_force=t_force)
    for n in ns:
        K = int(answers[:n].sum())
        q = RD.q_hat_from_answers(answers[:n])
        pz = {}
        for z in zetas:
            dz = distort(learn, z)
            g = CalGuard(dz, world.tau)
            cal = cal_learn(g, n, world.tau)
            assert g.read == FIELDS_CAL, f"Comp_cal read {g.read}"
            assert cal["K"] == K and cal["q_hat"] == q, "Comp_cal and Comp_naive q-hat differ"
            t1 = time.time()
            s_n = np.asarray(dz.pi0[:n], dtype=float)
            best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)][:n], ab, ks, s_n)
            pz[zkey(z)] = dict(cal=dict(cal, fields=sorted(g.read), theta_reads=g._theta.n_read),
                               b4=dict(k=best[0], a=best[1], b=best[2], tune_mean=val),
                               b4_k_rows=rows, t_tune=time.time() - t1)
        rec["per_n"][str(n)] = dict(n=n, K=K, q_hat=q, per_zeta=pz)
    rec["t_total"] = time.time() - t0
    return rec


def plan_200k(zetas):
    world, ex, ks, ab = setup()
    t0 = time.time()
    d = draw_outage_batch(world, make_flows()[FLOW], N_200K,
                          np.random.default_rng([LEARN_ROOT, TAG_200K]), payments_per_episode=PPE)
    force = {}
    for k in ks:
        for act in (GRANT, REJECT, VERIFY):
            force[(k, act)] = replay_outage(world, d, OB4Force(k, act, world.H, world.N), ex)
    rec = dict(rng=[LEARN_ROOT, TAG_200K], n=N_200K, t_force=time.time() - t0, per_zeta={})
    for z in zetas:
        t1 = time.time()
        sz = np.clip(z * d.pi0, 0.0, 1.0)
        best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)], ab, ks, sz)
        rec["per_zeta"][zkey(z)] = dict(b4=dict(k=best[0], a=best[1], b=best[2], tune_mean=val),
                                        b4_k_rows=rows, t_tune=time.time() - t1)
    rec["t_total"] = time.time() - t0
    return rec


def _plan_job(item):
    kind, s, ns, zetas = item
    try:
        rec = plan_seed(s, ns, zetas) if kind == "seed" else plan_200k(zetas)
        return item, rec, None
    except Exception:
        return item, None, traceback.format_exc()


def _write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def plan_path(run_dir, kind, s):
    return run_dir / "plan" / (f"seed_{s:02d}.json" if kind == "seed" else "b4_200k.json")


def run_plan(run_dir, seeds, ns, zetas, workers, log):
    (run_dir / "plan").mkdir(parents=True, exist_ok=True)
    jobs = [("b200k", 0, ns, zetas)] + [("seed", s, ns, zetas) for s in seeds]
    jobs = [j for j in jobs if not plan_path(run_dir, j[0], j[1]).exists()]
    if not jobs:
        return 0
    t0 = time.time()
    fails = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for item, rec, tb in pool.imap_unordered(_plan_job, jobs, chunksize=1):
            el = (time.time() - t0) / 60
            if rec is None:
                fails += 1
                log(f"FAIL plan {item[0]} {item[1]} elapsed={el:.1f}m")
                with (run_dir / "errors.log").open("a") as f:
                    f.write(f"FAIL plan {item[:2]}\n{tb}\n")
                continue
            rec.update(ns=list(ns), zetas=list(zetas))
            _write_json(plan_path(run_dir, item[0], item[1]), rec)
            if item[0] == "seed":
                log(f"PLAN seed={item[1]} t={rec['t_total']:.0f}s elapsed={el:.1f}m " + " ".join(
                    f"n{n}:q={v['q_hat']:.4f},zh[" + ",".join(
                        f"{float(z):g}:{pz['cal']['zhat']:.3f}" for z, pz in v["per_zeta"].items()) + "]"
                    for n, v in rec["per_n"].items()))
            else:
                log(f"PLAN b4_200k t={rec['t_total']:.0f}s elapsed={el:.1f}m " + " ".join(
                    f"z{float(z):g}:({v['b4']['k']},{v['b4']['a']:.4f},{v['b4']['b']:.4f})"
                    for z, v in rec["per_zeta"].items()))
    return 1 if fails else 0


def load_plan(run_dir, seeds):
    plan = {s: json.loads(plan_path(run_dir, "seed", s).read_text()) for s in seeds}
    b200 = json.loads(plan_path(run_dir, "b200k", 0).read_text())
    return plan, b200


# ------------------------------------------------------------------ policy names
def comp_name(h, z, k):
    return f"c_{h[:16]}_z{float(z)!r}_k{int(k):+d}"


def b4_name(z, k, a, b):
    return f"b_z{float(z)!r}_k{int(k)}_a{float(a)!r}_b{float(b)!r}"


def force_name(k, act):
    return f"force_k{int(k)}_{'GRV'[[GRANT, REJECT, VERIFY].index(act)]}"


# ------------------------------------------------------------------ eval workers
_G = {}


def _init_eval(n_eval_draw, impl):
    world, ex, _, _ = setup()
    t0 = time.time()
    d = draw_outage_batch(world, make_flows()[FLOW], n_eval_draw, np.random.default_rng(EVAL_SEED),
                          payments_per_episode=PPE)
    _G.update(world=world, ex=ex, full=d, draw_s=time.time() - t0, impl=impl, tables={})


def _use_prefix(n_used):
    d = _G["full"] if n_used == len(_G["full"]) else prefix(_G["full"], n_used)
    iv = FR.v_index(np.geomspace(0.5, 2000.0, 41), d.v)
    _G.update(d=d, iv=iv, n_used=n_used)


def _table(h):
    T = _G["tables"].get(h)
    if T is None:
        T = np.load(_G["tdir"] / f"{h}.npy")
        if len(_G["tables"]) >= 4:
            _G["tables"].pop(next(iter(_G["tables"])))
        _G["tables"][h] = T
    return T


def job_hash(q):
    try:
        t0 = time.time()
        pol = compile_outage(RD.comp_env(_G["world"], q), "comp")
        h = RD.table_hash(pol)
        assert np.allclose(pol.v_grid, np.geomspace(0.5, 2000.0, 41))
        p = _G["tdir"] / f"{h}.npy"
        if not p.exists():
            RE1._save_npy(p, FR.stack_tables(pol))
        return q, h, time.time() - t0, None
    except Exception:
        return q, None, 0.0, traceback.format_exc()


def pi_seen(z, k):
    """Score the compiled policy reads: min(clip(zeta*pi, 0, 1) / zeta-hat, 1).  k = 0 -> /1.0."""
    s = np.clip(z * _G["d"].pi0, 0.0, 1.0)
    return np.minimum(s / ZGRID[k + 200], 1.0) if k != 0 else s


def replay_comp(h, z, k, d=None, iv=None):
    world, ex = _G["world"], _G["ex"]
    d = _G["d"] if d is None else d
    ps = pi_seen(z, k)[:len(d)]
    if _G["impl"] == "vec":
        iv = _G["iv"][:len(d)] if iv is None else iv
        return FR.replay_table(world, d, _table(h), iv, ps, ex)
    pol = compile_outage(RD.comp_env(world, _G["qmap_inv"][h]), "comp")
    return replay_outage(world, replace(d, pi0=ps), pol, ex)


def replay_force(k, act):
    world, ex, d = _G["world"], _G["ex"], _G["d"]
    if _G["impl"] == "vec":
        return FR.replay_forced(world, d, k, act, ex)
    return replay_outage(world, d, OB4Force(k, act, world.H, world.N), ex)


def select_b4(z, k, a, b):
    s = np.clip(z * _G["d"].pi0, 0.0, 1.0)
    F = [np.load(_G["cache"] / f"{force_name(k, act)}.npy", mmap_mode="r") for act in (GRANT, REJECT, VERIFY)]
    return RE1.sel_pay(F[0], F[2], F[1], s, a, b)     # grant if s < a, reject if s > b, else verify


def job_eval(item):
    kind, name, spec = item
    try:
        if kind == "force":
            p = _G["cache"] / f"{name}.npy"
            if p.exists():
                return item, "SKIP", {}, None
            t0 = time.time()
            pay = replay_force(spec["k"], spec["act"])
            RE1._save_npy(p, np.asarray(pay, dtype=float))
            return item, "DONE", dict(replay_s=time.time() - t0), None
        path = _G["npz"] / f"{name}.npz"
        if path.exists():
            return item, "SKIP", {}, None
        t0 = time.time()
        if kind == "comp":
            pay = replay_comp(spec["hash"], spec["zeta"], spec["k"])
        else:
            pay = select_b4(spec["zeta"], spec["k"], spec["a"], spec["b"])
        t1 = time.time()
        meta = dict(spec, kind=kind, name=name, replay_s=t1 - t0, n_payments=len(pay),
                    total_payoff=float(np.sum(pay)), eval_seed=EVAL_SEED, n_eval=_G["n_used"], impl=_G["impl"])
        RD._save(path, block_sums=RD.block_sums(np.asarray(pay)), meta=np.array(json.dumps(meta)))
        return item, "DONE", dict(replay_s=t1 - t0), None
    except Exception:
        return item, "FAIL", {}, traceback.format_exc()


# ------------------------------------------------------------------ eval planning
def build_items(plan, b200, ns, zetas, qmap):
    """Distinct evaluation items with their priority (zeta = 1 first, then by n)."""
    ref_h = qmap[RD.qkey(Q_TRUE)]
    comp, b4, used = {}, {}, {}
    zord = {zkey(z): i for i, z in enumerate(sorted(zetas, key=lambda z: (z != 1.0, z)))}

    def add(dct, name, spec, prio):
        if name not in dct or prio < dct[name][1]:
            dct[name] = (spec, prio)

    add(comp, comp_name(ref_h, 1.0, 0), dict(hash=ref_h, q=Q_TRUE, zeta=1.0, k=0), (0, 0, 0))
    for s, rec in plan.items():
        for n in ns:
            v = rec["per_n"][str(n)]
            h = qmap[RD.qkey(v["q_hat"])]
            for z in zetas:
                pz = v["per_zeta"][zkey(z)]
                pr = (0 if z == 1.0 else 1, ns.index(n), zord[zkey(z)])
                add(comp, comp_name(h, z, 0), dict(hash=h, q=v["q_hat"], zeta=z, k=0), pr)
                kk = pz["cal"]["zhat_k"]
                add(comp, comp_name(h, z, kk), dict(hash=h, q=v["q_hat"], zeta=z, k=kk), pr)
                p = pz["b4"]
                add(b4, b4_name(z, p["k"], p["a"], p["b"]),
                    dict(zeta=z, k=int(p["k"]), a=float(p["a"]), b=float(p["b"])), pr)
    for z in zetas:
        p = b200["per_zeta"][zkey(z)]["b4"]
        add(b4, b4_name(z, p["k"], p["a"], p["b"]), dict(zeta=z, k=int(p["k"]), a=float(p["a"]),
                                                          b=float(p["b"])), (0, -1, zord[zkey(z)]))
    ks = sorted({sp["k"] for sp, _ in b4.values()})
    return comp, b4, ks, ref_h


def filter_ns(plan, ns):
    return {s: dict(rec, per_n={k: v for k, v in rec["per_n"].items() if int(k) in ns})
            for s, rec in plan.items()}


# ------------------------------------------------------------------ eval phase
def run_eval(run_dir, seeds, ns, zetas, n_eval, workers, impl, deadline, log, allow_drop=True):
    plan, b200 = load_plan(run_dir, seeds)
    npz, cache = run_dir / "npz", run_dir / "cache"
    tdir = cache / "tables"
    for p in (npz, cache, tdir):
        p.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "manifest.json"
    man_old = json.loads(mpath.read_text()) if mpath.exists() else {}
    _init_eval(n_eval, impl)
    _G.update(npz=npz, cache=cache, tdir=tdir)
    log(f"EVALSET seed={EVAL_SEED} n={n_eval} draw_s={_G['draw_s']:.0f} impl={impl}")
    # ---- compile every distinct q-hat (plus the true q); hash the tables
    qs = {RD.qkey(Q_TRUE): Q_TRUE}
    for rec in plan.values():
        for n in ns:
            q = rec["per_n"][str(n)]["q_hat"]
            qs.setdefault(RD.qkey(q), float(q))
    qmap_path = run_dir / "qmap.json"
    qmap = json.loads(qmap_path.read_text()) if qmap_path.exists() else {}
    todo = [q for key, q in sorted(qs.items(), key=lambda kv: kv[1])
            if key not in qmap or not (tdir / f"{qmap[key]}.npy").exists()]
    t_start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for q, h, t, tb in pool.imap_unordered(job_hash, todo, chunksize=1):
            if h is None:
                log(f"FAIL hash q={q!r}")
                with (run_dir / "errors.log").open("a") as f:
                    f.write(f"FAIL hash q={q!r}\n{tb}\n")
                return 1
            qmap[RD.qkey(q)] = h
            _write_json(qmap_path, qmap)
    _G["qmap_inv"] = {h: float(k) for k, h in qmap.items()}
    log(f"HASHED q_distinct={len(qs)} tables={len(set(qmap[k] for k in qs))} t={time.time() - t_start:.0f}s")

    # ---- check 9.2: tables equal D's for the same q-hat
    dq = json.loads((D_DIR / "qmap.json").read_text()) if (D_DIR / "qmap.json").exists() else {}
    common = sorted(set(dq) & set(qs), key=float)
    hash_eq = {k: dq[k] == qmap[k] for k in common}
    log(f"CHECK 9.2 table hash vs D qmap: common q={len(common)} equal={sum(hash_eq.values())} "
        f"{('PASS' if common and all(hash_eq.values()) else 'FAIL') if dq else 'SKIP (no results/D_answers/qmap.json; run scripts/run_D.py first)'}")

    # ---- runtime rule (spec 6): measure one replay, count distinct replays, decide
    comp, b4, ks, ref_h = build_items(plan, b200, ns, zetas, qmap)
    _use_prefix(n_eval)
    replay_comp(ref_h, 1.0, 0, d=prefix(_G["d"], min(50_000, n_eval)))      # warm-up (page faults, caches)
    t0 = time.time()
    replay_comp(ref_h, 1.0, 0)
    t_rep_run = time.time() - t0
    # spec 6: the replay time measured at the smoke stage, one full-size (5,663,400) replay:
    # vec -> smoke/checks.json 9.1 (pi_ref on eval seed 108); repo -> repro/repro_D.json (G0)
    if impl == "vec":
        sm = OUT / "smoke" / "checks.json"
        t_rep_smoke = json.loads(sm.read_text())["9.1"]["t_replay_vec_full"] if sm.exists() else None
    else:
        sm = OUT / "repro" / "repro_D.json"
        t_rep_smoke = json.loads(sm.read_text())["replay_s"] if sm.exists() else None
    t_rep = t_rep_smoke if t_rep_smoke is not None else t_rep_run * N_EVAL / n_eval
    n_rep = len(comp) + 3 * len(ks)
    est_h = n_rep * t_rep / workers / 3600
    if "decision" in man_old and man_old["decision"].get("n_eval_used") and n_eval == man_old.get("n_eval_draw"):
        dec = man_old["decision"]
        log(f"RULE reuse earlier decision n_eval_used={dec['n_eval_used']} (resume)")
    else:
        n_used = N_EVAL if (n_eval < N_EVAL or est_h <= RULE_HOURS) else N_EVAL_SMALL
        if n_eval < N_EVAL:
            n_used = n_eval
        dec = dict(replays=n_rep, comp_replays=len(comp), forced_replays=3 * len(ks), b4_selections=len(b4),
                   t_replay_full_s=t_rep, t_replay_source=(str(sm) if t_rep_smoke is not None else "this run, scaled"),
                   t_replay_this_run_s=t_rep_run, n_eval_measured=n_eval, workers=workers,
                   est_hours_full=est_h, rule_hours=RULE_HOURS, n_eval_used=n_used, impl=impl,
                   decided_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        scale = n_used / N_EVAL
        dec["est_hours_used"] = est_h * scale
        end = dt.datetime.now() + dt.timedelta(hours=dec["est_hours_used"] + 0.1)
        dec["projected_end"] = end.strftime("%Y-%m-%d %H:%M")
        dec["dropped_n5000"] = False
        if deadline and allow_drop and 5000 in ns:
            hh, mm = map(int, deadline.split(":"))
            dl = end.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if end > dl:
                dec["dropped_n5000"] = True
        log(f"RULE replays={n_rep} (comp {len(comp)}, forced {3 * len(ks)}) t_replay={t_rep:.1f}s "
            f"[{dec['t_replay_source']}] workers={workers} est={est_h:.2f}h vs {RULE_HOURS}h -> "
            f"n_eval_used={n_used} ({'full' if n_used == N_EVAL else 'prefix'}), est_used={dec['est_hours_used']:.2f}h, "
            f"projected_end={dec['projected_end']} deadline={deadline} drop_n5000={dec['dropped_n5000']}")
    n_used = dec["n_eval_used"]
    ns_eval = tuple(n for n in ns if not (dec.get("dropped_n5000") and n == 5000))
    if ns_eval != tuple(ns):
        comp, b4, ks, ref_h = build_items(filter_ns(plan, ns_eval), b200, ns_eval, zetas, qmap)
    if n_used != len(_G["d"]):
        _use_prefix(n_used)

    # ---- item list in the fixed order (zeta = 1 first, then n = 130, 500, 2000, 5000)
    force_items = [("force", force_name(k, act), dict(k=k, act=act)) for k in sorted(ks, reverse=True)
                   for act in (GRANT, REJECT, VERIFY)]
    ordered = sorted([("comp", n_, sp, pr) for n_, (sp, pr) in comp.items()] +
                     [("b4", n_, sp, pr) for n_, (sp, pr) in b4.items()],
                     key=lambda x: (x[3], x[0] != "comp", x[1]))
    items = [(k_, n_, sp) for k_, n_, sp, _ in ordered]
    manifest = dict(eval_seed=EVAL_SEED, learn_root=LEARN_ROOT, n_eval_draw=n_eval, n_eval_used=n_used,
                    n_blocks=n_used // PPE, seeds=list(seeds), ns=list(ns), ns_evaluated=list(ns_eval),
                    zetas=list(zetas), decision=dec, ref=comp_name(ref_h, 1.0, 0), ref_hash=ref_h,
                    b4_200k=dict(rng=[LEARN_ROOT, TAG_200K], n=N_200K), force_k=ks,
                    check_9_2=dict(common_q=common, equal=hash_eq,
                                   pass_=bool(common) and all(hash_eq.values())),
                    items=[dict(kind=k_, name=n_, **sp) for k_, n_, sp in items],
                    forced=[n_ for _, n_, _ in force_items], zgrid="exp(0.01k), k=-200..200",
                    started=man_old.get("started", time.strftime("%Y-%m-%d %H:%M:%S")))
    _write_json(mpath, manifest)
    log(f"ITEMS comp={len(comp)} b4={len(b4)} forced={len(force_items)} n_used={n_used} ns={ns_eval}")

    fails = []
    t_start = time.time()
    with ctx.Pool(workers) as pool:
        for phase, lst in (("forced", force_items), ("policies", items)):
            for it, st, tm, tb in pool.imap(job_eval, lst, chunksize=1):
                el = time.time() - t_start
                if st == "FAIL":
                    fails.append(it[1])
                    log(f"FAIL {it[1]} elapsed={el / 60:.1f}m")
                    with (run_dir / "errors.log").open("a") as f:
                        f.write(f"FAIL {it[1]}\n{tb}\n")
                elif st == "DONE":
                    log(f"DONE {it[1]} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items())
                        + f" elapsed={el / 60:.1f}m")
            log(f"PHASE {phase} finished elapsed={(time.time() - t_start) / 60:.1f}m")
    if fails:
        log(f"INCOMPLETE failed={len(fails)} (rerun the same command to resume)")
        return 1
    checks = direct_checks(manifest, log)
    manifest["direct_checks"] = checks
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(mpath, manifest)
    if not all(c["pass_"] for c in checks):
        log("FAIL direct replay checks; results not valid")
        return 1
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={(time.time() - t_start) / 60:.1f}m")
    return 0


def direct_checks(manifest, log):
    """Repo replay_outage against the stored block sums on an evaluation prefix: pi_ref, one
    Comp_cal with zeta != 1, and B4 selections (one per zeta)."""
    world, ex = _G["world"], _G["ex"]
    npre = min(N_CHECK_PREFIX, _G["n_used"])
    dp = prefix(_G["d"], npre)
    nb = npre // PPE
    out = []

    def cmp(name, pay):
        y = np.load(_G["npz"] / f"{name}.npz")["block_sums"][:nb]
        gap = float(np.max(np.abs(RD.block_sums(pay) - y)))
        out.append(dict(name=name, prefix=npre, max_abs_block_gap=gap, pass_=gap == 0.0))
        log(f"CHECK direct repo replay {name} prefix={npre} gap={gap:.3e} {'PASS' if gap == 0.0 else 'FAIL'}")

    items = manifest["items"]
    comps = [it for it in items if it["kind"] == "comp"]
    pick = [comps[0]] + [it for it in comps if it["zeta"] != 1.0 and it["k"] != 0][:1]
    for it in pick:
        pol = compile_outage(RD.comp_env(world, it["q"]), "comp")
        assert RD.table_hash(pol) == it["hash"]
        s = np.clip(it["zeta"] * dp.pi0, 0.0, 1.0)
        ps = np.minimum(s / ZGRID[it["k"] + 200], 1.0) if it["k"] != 0 else s
        cmp(it["name"], replay_outage(world, replace(dp, pi0=ps), pol, ex))
    seen = set()
    for it in items:
        if it["kind"] == "b4" and it["zeta"] not in seen:
            seen.add(it["zeta"])
            s = np.clip(it["zeta"] * dp.pi0, 0.0, 1.0)
            cmp(it["name"], replay_outage(world, replace(dp, pi0=s),
                                          OB4(it["k"], it["a"], it["b"], world.H, world.N), ex))
    return out


# ------------------------------------------------------------------ checks 9.1-9.4
def run_check(run_dir, log):
    res = {}
    world, ex, _, _ = setup()
    # 9.1: D reference through the G code path (zeta = 1, zeta-hat = 1, true q), eval seed 108
    t0 = time.time()
    d = draw_outage_batch(world, make_flows()[FLOW], N_EVAL, np.random.default_rng(108), payments_per_episode=PPE)
    t_draw = time.time() - t0
    pol = compile_outage(RD.comp_env(world, Q_TRUE), "comp")
    _G.update(world=world, ex=ex, d=d, impl="vec", tables={"ref": FR.stack_tables(pol)},
              iv=FR.v_index(pol.v_grid, d.v), n_used=N_EVAL)
    t0 = time.time()
    pay = replay_comp("ref", 1.0, 0)
    t_vec = time.time() - t0
    bs = RD.block_sums(pay)
    J = float((np.column_stack([bs, bs]).sum(axis=0) / (len(bs) * PPE))[0])  # analyze_D's aggregation
    res["9.1"] = dict(J=J, J_D=D_REF_J, equal=J == D_REF_J, abs_diff=abs(J - D_REF_J), t_draw=t_draw,
                      t_replay_vec_full=t_vec, table_hash=RD.table_hash(pol),
                      pass_=J == D_REF_J)
    log(f"CHECK 9.1 J={J!r} D={D_REF_J!r} diff={abs(J - D_REF_J):.1e} t_replay_vec={t_vec:.1f}s "
        f"{'PASS' if J == D_REF_J else 'FAIL'}")
    # vec vs repo on a 300,000 prefix: pi_ref and a distorted Comp_cal (zeta 2, zeta-hat exp(0.5))
    dp = prefix(d, 300_000)
    gaps = {}
    for z, k in ((1.0, 0), (2.0, 50), (0.5, -80)):
        s = np.clip(z * dp.pi0, 0.0, 1.0)
        ps = np.minimum(s / ZGRID[k + 200], 1.0) if k != 0 else s
        a = replay_outage(world, replace(dp, pi0=ps), pol, ex)
        b = FR.replay_table(world, dp, _G["tables"]["ref"], _G["iv"][:300_000], ps, ex)
        gaps[f"z{z}_k{k}"] = float(np.max(np.abs(a - b)))
    res["vec_vs_repo_300k"] = dict(max_abs_diff=gaps, pass_=all(g == 0.0 for g in gaps.values()))
    log(f"CHECK vec==repo (300k prefix) {gaps} {'PASS' if res['vec_vs_repo_300k']['pass_'] else 'FAIL'}")
    del d, dp, pay
    _G.clear()
    # 9.3: estimator recovery on 100,000 synthetic payments, answered only
    syn = draw_outage_batch(world, make_flows()[FLOW], SYN_N, np.random.default_rng([LEARN_ROOT, SYN_TAG]),
                            payments_per_episode=PPE)
    rec = {}
    for z in (0.7, 1.4):
        g = CalGuard(distort(syn, z), world.tau)
        c = cal_learn(g, SYN_N, world.tau)
        dlog = abs(np.log(c["zhat"]) - np.log(z))
        rec[zkey(z)] = dict(c, k_true=float(100 * np.log(z)), abs_log_err=float(dlog), pass_=bool(dlog <= 0.02 + 1e-12))
        log(f"CHECK 9.3 zeta={z} zeta_hat={c['zhat']:.4f} (k={c['zhat_k']}, k_true={100 * np.log(z):.2f}) "
            f"|log err|={dlog:.4f} n_ans={c['n_ans']} n_mis={c['n_mis_ans']} {'PASS' if dlog <= 0.02 + 1e-12 else 'FAIL'}")
    res["9.3"] = dict(per_zeta=rec, rng=[LEARN_ROOT, SYN_TAG], n=SYN_N, pass_=all(v["pass_"] for v in rec.values()))
    # 9.4: information contract (the deliberate-violation test lives in tests_G.py)
    g = CalGuard(distort(syn, 2.0), world.tau)
    cal_learn(g, 5000, world.tau)
    ok = g.read == FIELDS_CAL
    res["9.4_contract"] = dict(fields=sorted(g.read), pass_=ok)
    log(f"CHECK 9.4 Comp_cal fields={sorted(g.read)} {'PASS' if ok else 'FAIL'} (violation test: tests_G.py)")
    (run_dir / "checks.json").write_text(json.dumps(res, indent=1))
    return 0 if all(v.get("pass_", True) for v in res.values()) else 1


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["plan", "eval", "all", "check"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--ns", default=None)
    ap.add_argument("--zetas", default=None)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--impl", choices=["vec", "repo"], default="vec")
    ap.add_argument("--deadline", default="21:00", help="spec 6: drop n = 5000 if projected end is later")
    ap.add_argument("--no-drop", action="store_true")
    a = ap.parse_args(argv)
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
    ns = tuple(int(x) for x in a.ns.split(",")) if a.ns else NS
    zetas = tuple(float(x) for x in a.zetas.split(",")) if a.zetas else ZETAS
    run_dir = OUT / a.tag if a.tag else OUT
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = (run_dir / "progress.log").open("a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    log(f"START phase={a.phase} seeds={seeds[0]}..{seeds[-1]}({len(seeds)}) ns={ns} zetas={zetas} "
        f"n_eval={a.n_eval} workers={a.workers} impl={a.impl} pid={os.getpid()}")
    if a.phase == "check":
        return run_check(run_dir, log)
    if a.phase in ("plan", "all"):
        if run_plan(run_dir, seeds, ns, zetas, a.workers, log):
            log("FAIL plan")
            return 1
        log("PLANDONE")
    if a.phase in ("eval", "all"):
        return run_eval(run_dir, seeds, ns, zetas, a.n_eval, a.workers, a.impl,
                        a.deadline, log, allow_drop=not a.no_drop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
