"""Experiment E1: regularized small-sample B4 against the compiled policy Comp_n.
E-fitted world, flow F2, true q = 0.75, n in {130, 500}, learning seeds 1..30 (as in run_D.py).
Design notes: docs/regularized_b4.md (fixed before any E1 run).

In comments, "D" means experiment D (scripts/run_D.py, evaluation seed 108).
Imports the functions of scripts/run_D.py and scripts/xband.py instead of copying them.
Output goes to results/regularized_b4/ (paths.REGULARIZED_B4; ARENA_RESULTS moves the tree).
The plan files of run_D.py are read for the reproduction check from results/D_answers/plan/
if present, else from the shipped copy results/response_records/plan/.

Commands (from the package root, Python 3.11 or later, numpy and scipy installed):

    export PYTHONPATH=$PWD/src
    python3 scripts/run_E1.py --dry-run
    python3 scripts/run_E1.py repro
    python3 scripts/run_E1.py all --tag smoke --n-eval 100000 --seeds 1,2 --workers 2
    python3 scripts/run_E1.py all --workers 2
    python3 scripts/run_E1.py all --confirm --workers 2    (optional; seed 109)

Phases
  plan     per learning seed s: draw D's 5,000 learning payments (default_rng([108, 500, s])),
           read the pilot answers for Comp (FieldGuard: only t_ans), replay the forced B4
           payoffs, and run the tuners R0 (D's B4, tune_watch_policy), R1 (coarse grid),
           R2 (shrinkage; c = 1 primary, 0.5 and 2 sensitivity) and R3 (bagged) on the
           n-prefix, n in {130, 500}.  One json per seed in <run>/plan/.  Checks R0, K and
           q-hat against D's plan jsons when they are available.
  eval     draw the evaluation set (default_rng(108), 5,663,400 payments, 50 per episode).
           Workers: compile and replay pi_ref (q = 0.75) and every distinct Comp q-hat with a
           released flag; replay the forced policies (k, grant/reject/verify) for every k any
           tuner chose.  Parent: every tuner policy (k, a, b) is a per-payment selection among
           the three forced arrays of its k (same argument as xband.py; checked against a
           direct OutageWatchBandPolicy replay).  Block sums of 50 per policy -> <run>/npz/.
           Resumable: existing npz / npy are skipped.
  analyze  analyze_E1.py: summary.json, summary.md.
  all      plan, eval, analyze.
  repro    plan for seed 1 only, compared with D's plan/seed_01.json (all four n), plus the
           vectorized scorer against tune_watch_policy.  No evaluation data are drawn.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from paths import D_ANSWERS, D_PLAN, D_PLAN_SHIPPED, REGULARIZED_B4, RESULTS

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

D_SUMMARY_SHIPPED = RESULTS / "response_records" / "summary.json"

# ------------------------------------------------------------------ fixed constants (docs/regularized_b4.md, section 4)
EVAL_SEED = 108            # D's evaluation seed (already used by D -> exploratory label)
CONFIRM_SEED = 109         # optional fresh evaluation + learning root (--confirm)
N_EVAL = 5_663_400
SEEDS = tuple(range(1, 31))
NS_E1 = (130, 500)
Q_TRUE = 0.75
K_R1 = (0, 1, 2, 4)        # R1 horizon subset
GRID_R1 = 21               # R1 threshold grid points (suspicion_grid(21), 210 pairs)
CENTRE = (1.0 / 3.0, 2.0 / 3.0)   # neutral centre: centroid of {0 <= a < b <= 1}
C_LAMBDA = 1.0             # R2 primary: max penalty = 1 standard error of the tuning mean
C_LAMBDA_SENS = (0.5, 2.0)  # R2 sensitivity rows (not used by the reading rule)
B_BAG = 20                 # R3 bootstrap resamples
BAG_TAG = 600              # R3 rng: default_rng([learn_root, 600, s, n])
TIE_TOL = 1e-9             # $/payment; candidates within this of the best are tied (R1, R2)
MARGIN_BP = 10.0           # reading rule
PRIMARY = ("R1_coarse", "R2_shrink", "R3_bag")
TUNERS = ("R0_b4",) + PRIMARY + ("R2_shrink_c0.5", "R2_shrink_c2", "Rc_centre")
LABEL_EXPLORATORY = ("exploratory: pre-registered follow-up on the evaluation sample "
                     "(seed 108) already used by experiment D")
LABEL_CONFIRM = "confirmatory: fresh evaluation seed 109 and learning root 109"

# rough per-item times on 2 cores (from the logs of run_D.py and xband.py)
T_COMP_S = 72.0            # compile ~3.5 s + replay ~67 s of 5,663,400 payments
T_FORCE_S = lambda k: 35.0 + 8.0 * k   # forced replay with released flag (rough for k > 1)
T_SELECT_S = 0.5           # one (k, a, b) selection + block sums in the parent


# ------------------------------------------------------------------ lazy imports
def d_plan_dir():
    """Plan files of run_D.py: the regenerated ones if present, else the shipped copy."""
    return D_PLAN if D_PLAN.exists() else D_PLAN_SHIPPED


def d_summary_path():
    p = D_ANSWERS / "summary.json"
    return p if p.exists() else D_SUMMARY_SHIPPED


RD = XB = None     # run_D, xband modules
A = {}             # arena names


def load_modules():
    """Import run_D and xband (both in scripts/) and the arena modules."""
    global RD, XB
    RD = importlib.import_module("run_D")
    XB = importlib.import_module("xband")
    from arena.experiments.settlement.core import GRANT, REJECT, VERIFY
    from arena.experiments.settlement.flows import make_flows
    from arena.experiments.settlement.outage import (
        compile_outage, draw_outage_batch, replay_outage, survival, window_AD)
    from arena.experiments.settlement.policies import suspicion_grid
    from arena.experiments.settlement.watch import (
        FixedActionOutageWatchPolicy, OutageWatchBandPolicy, horizon_grid, tune_watch_policy)
    A.update(GRANT=GRANT, REJECT=REJECT, VERIFY=VERIFY, make_flows=make_flows,
             compile_outage=compile_outage, draw_outage_batch=draw_outage_batch,
             replay_outage=replay_outage, survival=survival, window_AD=window_AD,
             suspicion_grid=suspicion_grid, Force=FixedActionOutageWatchPolicy,
             OB4=OutageWatchBandPolicy, horizon_grid=horizon_grid,
             tune_watch_policy=tune_watch_policy)


# ------------------------------------------------------------------ tuner machinery
def score_totals(G, V, R, pi, ga, gb, w=None):
    """Total tuning payoff of every candidate: rows = horizons (G, V, R are (nk, n) forced
    payoffs), columns = threshold pairs (ga, gb).  grant if pi < a, verify if a <= pi <= b,
    reject if pi > b (OutageWatchBandPolicy's bands; same cumulative-sum method as
    xband.tune_band).  w: optional bootstrap counts per payment."""
    order = np.argsort(pi, kind="stable")
    s = pi[order]
    nA = np.searchsorted(s, ga, side="left")    # count pi < a
    nB = np.searchsorted(s, gb, side="right")   # count pi <= b
    if w is not None:
        G, V, R = G * w, V * w, R * w
    z = np.zeros((G.shape[0], 1))
    cg = np.concatenate([z, np.cumsum(G[:, order], axis=1)], axis=1)
    cv = np.concatenate([z, np.cumsum(V[:, order], axis=1)], axis=1)
    cr = np.concatenate([z, np.cumsum(R[:, order], axis=1)], axis=1)
    return cg[:, nA] + (cv[:, nB] - cv[:, nA]) + (cr[:, -1:] - cr[:, nB])


def pick_first(vals):
    """B4's rule: first maximum in (k ascending, grid order)."""
    ki, gi = divmod(int(np.argmax(vals)), vals.shape[1])
    return ki, gi


def pick_tie(obj, kvals, dist, ga, gb, tol=TIE_TOL):
    """R1/R2 rule: candidates within tol of the best are tied; among them the smallest k,
    then the pair closest to CENTRE, then the smallest a, then the smallest b."""
    m = obj.max()
    ki, gi = np.nonzero(obj >= m - tol)
    o = np.lexsort((gb[gi], ga[gi], dist[gi], np.asarray(kvals)[ki]))[0]
    return int(ki[o]), int(gi[o]), int(len(ki))


def grid_arrays(grid):
    ga = np.array([g[0] for g in grid], dtype=float)
    gb = np.array([g[1] for g in grid], dtype=float)
    dist = np.hypot(ga - CENTRE[0], gb - CENTRE[1])
    return ga, gb, dist, grid


def sel_pay(G, V, R, pi, a, b):
    return np.where(pi < a, G, np.where(pi > b, R, V))


def round_to(pts, x):
    """Nearest grid point; exact ties go to the lower point."""
    i = int(np.searchsorted(pts, x))
    if i <= 0:
        return 0
    if i >= len(pts):
        return len(pts) - 1
    return i - 1 if (x - pts[i - 1]) <= (pts[i] - x) else i


def tuners_on_prefix(n, force, ks, pi_all, g161, g21, s, learn_root):
    """All tuners on the first n learning payments.  force[(k, act)] are forced payoffs on
    the 5,000 learning payments.  Returns a dict of tuner records (k, a, b, tune_mean ...)."""
    GR, RJ, VF = A["GRANT"], A["REJECT"], A["VERIFY"]
    pi = np.asarray(pi_all[:n], dtype=float)
    G = np.stack([force[(k, GR)][:n] for k in ks])
    V = np.stack([force[(k, VF)][:n] for k in ks])
    R = np.stack([force[(k, RJ)][:n] for k in ks])
    ga, gb, dist, grid161 = g161          # grid161: suspicion_grid(161) as D passes it
    a21, b21, d21, _ = g21
    means = score_totals(G, V, R, pi, ga, gb) / n
    out = {}

    # R0: D's B4 (existing tuner, unchanged) and the vectorized scorer as a cross-check
    t0 = time.time()
    best, val, rows = A["tune_watch_policy"](lambda k, act: force[(k, act)][:n], grid161, list(ks),
                                             pi_all[:n])
    ki, gi = pick_first(means)
    vec = (int(ks[ki]), float(ga[gi]), float(gb[gi]))
    out["R0_b4"] = dict(k=int(best[0]), a=float(best[1]), b=float(best[2]), tune_mean=float(val),
                        k_rows=rows, vec_equal=bool(vec == (int(best[0]), float(best[1]), float(best[2]))),
                        vec_pick=list(vec), vec_tune_mean=float(means[ki, gi]), t=time.time() - t0)

    # R1: coarse grid, k in K_R1, 21-point thresholds, tie -> smallest k, closest to CENTRE
    kidx = [i for i, k in enumerate(ks) if k in K_R1]
    kv = [ks[i] for i in kidx]
    m21 = score_totals(G[kidx], V[kidx], R[kidx], pi, a21, b21) / n
    ki, gi, nt = pick_tie(m21, kv, d21, a21, b21)
    out["R1_coarse"] = dict(k=int(kv[ki]), a=float(a21[gi]), b=float(b21[gi]),
                            tune_mean=float(m21[ki, gi]), n_tied=nt, grid_points=GRID_R1,
                            k_set=list(K_R1))

    # R2: shrinkage toward CENTRE on D's full grid.  lambda_n = c * sigma_n / sqrt(n) per
    # unit of normalized distance (distance / d_max), sigma_n = sd (ddof 1) over the n
    # payments of the payoff of the centre policy (k = 0, grid pair nearest CENTRE).
    k0 = list(ks).index(0)
    gc = int(np.argmin(dist))
    pay_c = sel_pay(G[k0], V[k0], R[k0], pi, ga[gc], gb[gc])
    sigma = float(np.std(pay_c, ddof=1))
    dmax = float(dist.max())
    # reference: the centre policy itself (the limit of R2 as c grows); not a tuner
    out["Rc_centre"] = dict(k=0, a=float(ga[gc]), b=float(gb[gc]), tune_mean=float(means[k0, gc]),
                            reference=True)
    for c, name in ((C_LAMBDA, "R2_shrink"), (C_LAMBDA_SENS[0], "R2_shrink_c0.5"),
                    (C_LAMBDA_SENS[1], "R2_shrink_c2")):
        lam = c * sigma / math.sqrt(n)
        obj = means - lam * (dist / dmax)[None, :]
        ki, gi, nt = pick_tie(obj, ks, dist, ga, gb)
        out[name] = dict(k=int(ks[ki]), a=float(ga[gi]), b=float(gb[gi]),
                         tune_mean=float(means[ki, gi]), penalized=float(obj[ki, gi]),
                         penalty=float(lam * dist[gi] / dmax), c=c, sigma_centre=sigma,
                         lambda_=lam, d_max=dmax, centre_pair=[float(ga[gc]), float(gb[gc])],
                         n_tied=nt, sensitivity=name != "R2_shrink")

    # R3: bagged B4.  B resamples of the n payments (with replacement); on each, B4's rule on
    # D's full grid; k = majority (ties -> smaller k); a, b = mean of the picks, rounded to
    # the 161-point values (ties -> lower); if a >= b after rounding, b moves one point up
    # (or a one point down when b is already the top point).
    rb = np.random.default_rng([learn_root, BAG_TAG, s, n])
    idx = rb.integers(0, n, (B_BAG, n))
    picks = []
    for j in range(B_BAG):
        w = np.bincount(idx[j], minlength=n).astype(float)
        vj = score_totals(G, V, R, pi, ga, gb, w=w)
        kj, gj = pick_first(vj)
        picks.append((int(ks[kj]), float(ga[gj]), float(gb[gj])))
    kc = np.bincount([list(ks).index(p[0]) for p in picks], minlength=len(ks))
    k_bag = int(ks[int(np.argmax(kc))])            # argmax -> first (smallest k) on ties
    pts = np.unique(np.concatenate([ga, gb]))
    ia = round_to(pts, float(np.mean([p[1] for p in picks])))
    ib = round_to(pts, float(np.mean([p[2] for p in picks])))
    if ia >= ib:
        if ia + 1 < len(pts):
            ib = ia + 1
        else:
            ia, ib = len(pts) - 2, len(pts) - 1
    a_bag, b_bag = float(pts[ia]), float(pts[ib])
    lookup = {(float(x), float(y)): i for i, (x, y) in enumerate(grid161)}
    gi = lookup.get((a_bag, b_bag))
    assert gi is not None, f"R3 pair {(a_bag, b_bag)} not on D's grid"
    out["R3_bag"] = dict(k=k_bag, a=float(ga[gi]), b=float(gb[gi]),
                         tune_mean=float(means[list(ks).index(k_bag), gi]), B=B_BAG,
                         picks=[list(p) for p in picks], k_counts=kc.tolist(),
                         a_mean=float(np.mean([p[1] for p in picks])),
                         b_mean=float(np.mean([p[2] for p in picks])))
    return out


TUNER_INPUTS = {
    "Comp_n": "pilot answer indicators t_ans <= tau on the n payments (FieldGuard: {t_ans}); "
              "the compiler also has the model and every constant except q",
    "R0_b4": "forced counterfactual payoffs of every (k, grant/reject/verify) on the n payments "
             "(uses intent label, settlement outcome and answer time through the oracle replay) "
             "and pi0; no model constants",
    "R1_coarse": "same as R0_b4 (subset of candidates)",
    "R2_shrink": "same as R0_b4 plus sigma_n computed from the same n payoffs",
    "R3_bag": "same as R0_b4 on bootstrap resamples of the n payments",
    "Rc_centre": "nothing (fixed policy k = 0, grid pair nearest CENTRE); reference row",
}


# ------------------------------------------------------------------ plan phase
def arena_setup():
    world = RD.world_env()
    _, _, ex = A["window_AD"](world, A["survival"](world))
    ks = list(A["horizon_grid"](world.H))
    g161 = grid_arrays(A["suspicion_grid"](RD.B3_N))
    g21 = grid_arrays(A["suspicion_grid"](GRID_R1))
    assert 0 in ks and all(k in ks for k in K_R1), ks
    return world, ex, ks, g161, g21


def plan_seed(s, cfg, world, ex, ks, g161, g21, n_list=NS_E1):
    g = np.random.default_rng([cfg["learn_root"], RD.LEARN_TAG, s])
    learn = A["draw_outage_batch"](world, A["make_flows"]()[RD.FLOW], RD.N_LEARN, g,
                                   payments_per_episode=RD.PPE)
    guard = RD.FieldGuard(learn)
    answers = RD.pilot_answers(guard, world.tau)
    assert guard.read == {"t_ans"}, f"Comp path read fields {guard.read}"
    rec = dict(seed=s, rng=[cfg["learn_root"], RD.LEARN_TAG, s], n_learn=RD.N_LEARN,
               comp_fields_read=sorted(guard.read), per_n={}, inputs=TUNER_INPUTS)
    t0 = time.time()
    force = {}
    for k in ks:
        for act in (A["GRANT"], A["REJECT"], A["VERIFY"]):
            force[(k, act)] = A["replay_outage"](world, learn, A["Force"](k, act, world.H, world.N), ex)
    rec["t_force"] = time.time() - t0
    for n in n_list:
        t1 = time.time()
        K = int(answers[:n].sum())
        q = RD.q_hat_from_answers(answers[:n])
        tun = tuners_on_prefix(n, force, ks, learn.pi0, g161, g21, s, cfg["learn_root"])
        rec["per_n"][str(n)] = dict(n=n, K=K, q_hat=q, tuners=tun, t=time.time() - t1)
    rec["t_total"] = time.time() - t0
    return rec


def d_check(rec, d_plan_dir):
    """Compare K, q-hat and R0 with D's plan json (only meaningful for learn_root 108)."""
    p = d_plan_dir / f"seed_{rec['seed']:02d}.json"
    if not p.exists():
        return None
    dp = json.loads(p.read_text())
    res = {}
    for n, v in rec["per_n"].items():
        if n not in dp["per_n"]:
            continue
        w = dp["per_n"][n]
        r0 = v["tuners"]["R0_b4"]
        res[n] = dict(K=v["K"] == w["K"], q_hat=v["q_hat"] == w["q_hat"],
                      b4=(r0["k"] == w["b4"]["k"] and r0["a"] == w["b4"]["a"] and r0["b"] == w["b4"]["b"]),
                      tune_mean_diff=abs(r0["tune_mean"] - w["b4"]["tune_mean"]),
                      vec_equal=r0["vec_equal"])
        res[n]["pass_"] = bool(res[n]["K"] and res[n]["q_hat"] and res[n]["b4"]
                               and res[n]["tune_mean_diff"] <= 1e-12)
    return res


def run_plan(run_dir, seeds, cfg, log):
    pdir = run_dir / "plan"
    pdir.mkdir(parents=True, exist_ok=True)
    world, ex, ks, g161, g21 = arena_setup()
    d_plan = cfg["d_plan"]
    bad = []
    for s in seeds:
        path = pdir / f"seed_{s:02d}.json"
        if path.exists():
            continue
        t0 = time.time()
        rec = plan_seed(s, cfg, world, ex, ks, g161, g21)
        rec.update(constants=constants(cfg), k_grid=ks)
        if cfg["learn_root"] == EVAL_SEED:
            rec["d_check"] = d_check(rec, d_plan)
            if rec["d_check"] is not None and not all(v["pass_"] for v in rec["d_check"].values()):
                bad.append(s)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, path)
        log(f"PLAN seed={s} t={time.time() - t0:.0f}s " + " | ".join(
            f"n{n}: q={v['q_hat']:.4f} " + " ".join(
                f"{t.split('_')[0]}=({v['tuners'][t]['k']},{v['tuners'][t]['a']:.4f},{v['tuners'][t]['b']:.4f})"
                for t in ("R0_b4",) + PRIMARY)
            for n, v in rec["per_n"].items())
            + (f" d_check={'PASS' if s not in bad else 'FAIL'}" if rec.get("d_check") else ""))
    if bad and not cfg["no_d_check"]:
        log(f"FAIL d_check seeds={bad}: R0/K/q-hat differ from D's plan (environment differs?)")
        return 1
    return 0


def constants(cfg):
    return dict(eval_seed=cfg["eval_seed"], learn_root=cfg["learn_root"], ns=list(NS_E1),
                q_true=Q_TRUE, k_r1=list(K_R1), grid_r1=GRID_R1, grid_b4=RD.B3_N if RD else 161,
                centre=list(CENTRE), c_lambda=C_LAMBDA, c_lambda_sens=list(C_LAMBDA_SENS),
                b_bag=B_BAG, bag_tag=BAG_TAG, tie_tol=TIE_TOL, margin_bp=MARGIN_BP,
                primary=list(PRIMARY), tuners=list(TUNERS), label=cfg["label"])


def load_plan(run_dir, seeds):
    return {s: json.loads((run_dir / "plan" / f"seed_{s:02d}.json").read_text()) for s in seeds}


def collect(plan):
    """Distinct compiled q values and distinct tuner triples (deduplicated)."""
    qs = {repr(float(Q_TRUE)): float(Q_TRUE)}
    tri = {}
    for rec in plan.values():
        for n in map(str, NS_E1):
            v = rec["per_n"][n]
            qs.setdefault(repr(float(v["q_hat"])), float(v["q_hat"]))
            for t in TUNERS:
                p = v["tuners"][t]
                tri.setdefault(b4n(p["k"], p["a"], p["b"]), (int(p["k"]), float(p["a"]), float(p["b"])))
    return qs, tri


def b4n(k, a, b):
    return f"b4_k{int(k)}_a{float(a)!r}_b{float(b)!r}"   # same naming as run_D.b4_name


def comp_name(q):
    return f"comp_q{float(q)!r}"


# ------------------------------------------------------------------ eval phase
_G = {}


def _init_eval(cfg):
    world = RD.world_env()
    t0 = time.time()
    d = A["draw_outage_batch"](world, A["make_flows"]()[RD.FLOW], cfg["n_eval"],
                               np.random.default_rng(cfg["eval_seed"]), payments_per_episode=RD.PPE)
    _, _, ex = A["window_AD"](world, A["survival"](world))
    _G.update(world=world, eval_d=d, ex=ex, draw_s=time.time() - t0)


def release_stats(pay, rel, d):
    """Same definitions as xband.py sec(): misuse = theta == 1."""
    mis = np.asarray(d.theta) == 1
    v = np.asarray(d.v, dtype=float)
    n = len(pay)
    out = dict(n=n, total_payoff=float(pay.sum()), mean=float(pay.mean()), n_misuse=int(mis.sum()))
    if rel is None:
        out.update(release=None)
        return out
    out.update(misuse_released=int((rel & mis).sum()),
               misuse_grant_rate=float((rel & mis).sum() / max(1, mis.sum())),
               misuse_exposure=float((v * (rel & mis)).sum() / n),
               release_rate=float(rel.mean()),
               legit_refusal_rate=float((~rel[~mis]).mean()))
    return out


def _save_npy(path, arr):
    tmp = path.with_name(path.stem + ".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def job(item):
    kind, name, spec = item
    try:
        world, d, ex = _G["world"], _G["eval_d"], _G["ex"]
        t0 = time.time()
        if kind == "force":
            k, nm = spec["k"], spec["act"]
            pp, pr = _G["cache"] / f"{name}_pay.npy", _G["cache"] / f"{name}_rel.npy"
            if pp.exists() and pr.exists():
                return item, "SKIP", {}, None
            act = {"G": A["GRANT"], "R": A["REJECT"], "V": A["VERIFY"]}[nm]
            pay, rel = XB.replay_release(world, d, A["Force"](k, act, world.H, world.N), ex)
            _save_npy(pr, np.asarray(rel, dtype=bool))
            _save_npy(pp, np.asarray(pay, dtype=float))
            return item, "DONE", dict(replay_s=time.time() - t0), None
        # compiled policy
        path = _G["npz"] / f"{name}.npz"
        if path.exists():
            return item, "SKIP", {}, None
        q = spec["q"]
        pol = A["compile_outage"](RD.comp_env(world, q), "comp")
        h = RD.table_hash(pol)
        t1 = time.time()
        note, gap = None, None
        try:
            pay, rel = XB.replay_release(world, d, pol, ex)
            if spec.get("check"):
                pay2 = A["replay_outage"](world, d, pol, ex)
                gap = float(np.max(np.abs(pay - pay2)))
        except Exception:
            note = "replay_release failed for the compiled policy; payoff from replay_outage, " \
                   "no released flag:\n" + traceback.format_exc()
            pay, rel = A["replay_outage"](world, d, pol, ex), None
        t2 = time.time()
        meta = dict(kind="comp", q=q, hash=h, d_hash=spec.get("d_hash"),
                    d_hash_equal=(None if spec.get("d_hash") is None else h == spec["d_hash"]),
                    release_vs_replay_gap=gap, note=note, compile_s=t1 - t0, replay_s=t2 - t1,
                    stats=release_stats(pay, rel, d), eval_seed=_G["eval_seed"], n_eval=len(pay))
        RD._save(path, block_sums=RD.block_sums(pay), meta=np.array(json.dumps(meta)))
        return item, "DONE", dict(compile_s=t1 - t0, replay_s=t2 - t1), None
    except Exception:
        return item, "FAIL", {}, traceback.format_exc()


def check_names(tri, n_check):
    """Policies checked by a direct replay: one per chosen k (smallest k first), cycling."""
    by_k = {}
    for name, (k, _, _) in sorted(tri.items()):
        by_k.setdefault(k, []).append(name)
    out, j = [], 0
    while len(out) < min(n_check, len(tri)):
        for k in sorted(by_k):
            if j < len(by_k[k]) and len(out) < n_check:
                out.append(by_k[k][j])
        j += 1
    return out


def select_triples(tri, log, n_check):
    """Tuner policies as per-payment selections among the forced arrays of their k, then a
    direct OutageWatchBandPolicy replay of n_check of them through the repo replay_outage."""
    d = _G["eval_d"]
    pi = np.asarray(d.pi0, dtype=float)
    by_k = {}
    for name, (k, a, b) in tri.items():
        by_k.setdefault(k, []).append((name, a, b))
    for k in sorted(by_k):
        todo = [x for x in by_k[k] if not (_G["npz"] / f"{x[0]}.npz").exists()]
        if not todo:
            continue
        F = {nm: (np.load(_G["cache"] / f"force_k{k}_{nm}_pay.npy"),
                  np.load(_G["cache"] / f"force_k{k}_{nm}_rel.npy")) for nm in "GRV"}
        t0 = time.time()
        for name, a, b in todo:
            g, r = pi < a, pi > b
            pay = np.where(g, F["G"][0], np.where(r, F["R"][0], F["V"][0]))
            rel = np.where(g, F["G"][1], np.where(r, F["R"][1], F["V"][1]))
            meta = dict(kind="b4", k=k, a=a, b=b, stats=release_stats(pay, rel, d),
                        eval_seed=_G["eval_seed"], n_eval=len(pay), method="select")
            RD._save(_G["npz"] / f"{name}.npz", block_sums=RD.block_sums(pay),
                     meta=np.array(json.dumps(meta)))
        log(f"SELECT k={k} policies={len(todo)} t={time.time() - t0:.0f}s")
        del F
    checks = []
    w = _G["world"]
    for name in check_names(tri, n_check):
        k, a, b = tri[name]
        t1 = time.time()
        x = A["replay_outage"](w, d, A["OB4"](k, a, b, w.H, w.N), _G["ex"])
        y = np.load(_G["npz"] / f"{name}.npz")["block_sums"]
        gap = float(np.max(np.abs(RD.block_sums(x) - y)))
        checks.append(dict(name=name, k=k, a=a, b=b, max_abs_block_gap=gap, t=time.time() - t1,
                           pass_=gap <= 1e-9))
        log(f"CHECK select vs direct replay {name} gap={gap:.3e} {'PASS' if gap <= 1e-9 else 'FAIL'}")
    return checks


def run_eval(run_dir, seeds, cfg, log):
    plan = load_plan(run_dir, seeds)
    qs, tri = collect(plan)
    ks_need = sorted({k for k, _, _ in tri.values()})
    npz = run_dir / "npz"
    cache = run_dir / "cache"
    npz.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    _init_eval(cfg)
    _G.update(npz=npz, cache=cache, eval_seed=cfg["eval_seed"])
    log(f"EVALSET seed={cfg['eval_seed']} n={cfg['n_eval']} draw_s={_G['draw_s']:.0f}")
    dq = D_ANSWERS / "qmap.json"      # written by run_D.py eval (not shipped)
    dqmap = json.loads(dq.read_text()) if (dq.exists() and cfg["eval_seed"] == EVAL_SEED) else {}
    items = []
    ref_key = repr(float(Q_TRUE))
    for key, q in sorted(qs.items(), key=lambda kv: (kv[0] != ref_key, kv[1])):
        items.append(("comp", comp_name(q), dict(q=q, d_hash=dqmap.get(key),
                                                 check=(key == ref_key or cfg["check_all_comp"]))))
    force = [("force", f"force_k{k}_{nm}", dict(k=k, act=nm)) for k in sorted(ks_need, reverse=True)
             for nm in "GRV"]
    # the longest forced replays first, then the compiled policies
    jobs = force + items
    est = (sum(T_FORCE_S(k) * 3 for k in ks_need) + T_COMP_S * len(items)) / max(1, cfg["workers"])
    log(f"ITEMS comp={len(items)} force_k={ks_need} triples={len(tri)} "
        f"est_wall={est / 60:.0f}m(+select {len(tri) * T_SELECT_S / 60:.1f}m)")
    t_start = time.time()
    fails = []
    ctx = mp.get_context("fork")
    with ctx.Pool(cfg["workers"]) as pool:
        for it, st, tm, tb in pool.imap_unordered(job, jobs, chunksize=1):
            el = time.time() - t_start
            if st == "FAIL":
                fails.append(it[1])
                log(f"FAIL {it[1]} elapsed={el / 60:.1f}m")
                with (run_dir / "errors.log").open("a") as f:
                    f.write(f"FAIL {it[1]}\n{tb}\n")
            else:
                log(f"{st} {it[1]} " + " ".join(f"{k}={v:.1f}" for k, v in tm.items())
                    + f" elapsed={el / 60:.1f}m")
    if fails:
        log(f"INCOMPLETE failed={len(fails)} (rerun the same command to resume)")
        return 1
    # reference check: released-flag replay equals the repo replay for pi_ref
    ref_meta = json.loads(str(np.load(npz / f"{comp_name(Q_TRUE)}.npz")["meta"]))
    if ref_meta["note"] is None and not (ref_meta["release_vs_replay_gap"] is not None
                                         and ref_meta["release_vs_replay_gap"] <= 1e-9):
        log(f"FAIL pi_ref released-flag replay differs from replay_outage: "
            f"gap={ref_meta['release_vs_replay_gap']}")
        return 1
    comp_meta = {n: json.loads(str(np.load(npz / f"{n}.npz")["meta"])) for _, n, _ in items}
    bad = {n: m["release_vs_replay_gap"] for n, m in comp_meta.items()
           if m["release_vs_replay_gap"] is not None and m["release_vs_replay_gap"] > 1e-9}
    if bad:
        log(f"FAIL released-flag replay differs from replay_outage for {bad}")
        return 1
    checks = select_triples(tri, log, cfg["n_check"])
    manifest = dict(
        eval_seed=cfg["eval_seed"], learn_root=cfg["learn_root"], n_eval=cfg["n_eval"],
        seeds=list(seeds), ns=list(NS_E1), label=cfg["label"], constants=constants(cfg),
        ref=comp_name(Q_TRUE), comp={k: comp_name(q) for k, q in qs.items()},
        triples={n: list(t) for n, t in tri.items()}, force_k=ks_need,
        select_checks=checks,
        d_hash_equal={n: m["d_hash_equal"] for n, m in comp_meta.items()},
        comp_notes={n: m["note"] for n, m in comp_meta.items() if m["note"]},
        ref_release_gap=ref_meta["release_vs_replay_gap"])
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if any(not c["pass_"] for c in checks):
        log("FAIL select-vs-direct check; results not valid")
        return 1
    if any(v is False for v in manifest["d_hash_equal"].values()):
        log("WARN some compiled tables differ from D's qmap.json hashes (see manifest)")
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={(time.time() - t_start) / 60:.1f}m")
    return 0


# ------------------------------------------------------------------ repro / dry-run
def run_repro(run_dir, cfg, log):
    world, ex, ks, g161, g21 = arena_setup()
    t0 = time.time()
    rec = plan_seed(1, dict(cfg, learn_root=EVAL_SEED), world, ex, ks, g161, g21,
                    n_list=(50, 130, 500, 5000))
    chk = d_check(rec, cfg["d_plan"])
    if chk is None:
        log("REPRO FAIL: D's plan/seed_01.json not found")
        return 1
    for n, c in chk.items():
        r0 = rec["per_n"][n]["tuners"]["R0_b4"]
        log(f"REPRO n={n} R0=({r0['k']},{r0['a']!r},{r0['b']!r}) tune_mean={r0['tune_mean']!r} "
            f"K/q_hat equal={c['K'] and c['q_hat']} b4 equal={c['b4']} "
            f"tune_mean_diff={c['tune_mean_diff']:.1e} vec_equal={c['vec_equal']} "
            f"{'PASS' if c['pass_'] else 'FAIL'}")
    for n in map(str, NS_E1):
        v = rec["per_n"][n]["tuners"]
        log(f"REPRO n={n} picks (learning data only): " + " ".join(
            f"{t}=({v[t]['k']},{v[t]['a']:.4f},{v[t]['b']:.4f};J_tune={v[t]['tune_mean']:.4f})"
            for t in TUNERS))
    ok = all(c["pass_"] and c["vec_equal"] for c in chk.values())
    (run_dir / "repro.json").write_text(json.dumps(dict(check=chk, seed1=rec, t=time.time() - t0,
                                                        pass_=ok), indent=1))
    log(f"REPRO {'PASS' if ok else 'FAIL'} t={time.time() - t0:.0f}s")
    return 0 if ok else 1


def dry_run(run_dir, seeds, cfg):
    print(f"run_dir      {run_dir}")
    print(f"label        {cfg['label']}")
    print(f"D plan files {cfg['d_plan']}")
    print(f"eval         default_rng({cfg['eval_seed']}), {cfg['n_eval']:,} payments, blocks of 50")
    print(f"learning     default_rng([{cfg['learn_root']}, 500, s]) x 5,000, s = {seeds[0]}..{seeds[-1]} "
          f"({len(seeds)}), prefixes n = {NS_E1}")
    print(f"tuners       R0_b4 (D, 7 k x 12,880 pairs, first max) | R1_coarse k in {K_R1}, "
          f"{GRID_R1}-point grid | R2_shrink c = {C_LAMBDA} (sens {C_LAMBDA_SENS}) | R3_bag B = {B_BAG}")
    print(f"centre       {CENTRE}, tie tol {TIE_TOL} $/payment, margin {MARGIN_BP} bp, primary {PRIMARY}")
    pdir = run_dir / "plan"
    have = [s for s in seeds if (pdir / f"seed_{s:02d}.json").exists()]
    print(f"plan jsons   {len(have)}/{len(seeds)} present in {pdir}")
    if len(have) == len(seeds):
        plan = load_plan(run_dir, seeds)
        qs, tri = collect(plan)
        ks = sorted({k for k, _, _ in tri.values()})
        per_t = {t: len({b4n(**{x: rec["per_n"][str(n)]["tuners"][t][x] for x in "kab"})
                         for rec in plan.values() for n in NS_E1}) for t in TUNERS}
        est = (sum(T_FORCE_S(k) * 3 for k in ks) + T_COMP_S * len(qs)) / max(1, cfg["workers"])
        print(f"distinct     compiled q = {len(qs)} (incl. 0.75), triples = {len(tri)} {per_t}, force k = {ks}")
        print(f"estimate     ~{est / 60:.0f} min eval wall on {cfg['workers']} workers "
              f"(+ ~1 min draw, ~{len(tri) * T_SELECT_S / 60:.0f} min select, ~1 min check, 2-3 min analyze)")
    else:
        est = (3 * (T_FORCE_S(0) + T_FORCE_S(1)) + T_COMP_S * 37) / max(1, cfg["workers"])
        print(f"estimate     (no full plan yet) ~{est / 60:.0f} min eval wall on {cfg['workers']} workers "
              f"if only k in {{0,1}} are chosen; each extra k adds ~3 x (35 + 8k) s / workers")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="all", choices=["plan", "eval", "analyze", "all", "repro"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default=None, help="comma list; default 1..30")
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--tag", default=None, help="write under results/regularized_b4/<tag>/ instead of results/regularized_b4/")
    ap.add_argument("--confirm", action="store_true",
                    help=f"fresh evaluation seed {CONFIRM_SEED} and learning root {CONFIRM_SEED} "
                         f"(default tag confirm{CONFIRM_SEED})")
    ap.add_argument("--n-check", type=int, default=None,
                    help="number of tuner policies checked by a direct replay (default 1; smoke 3)")
    ap.add_argument("--check-all-comp", action="store_true",
                    help="also check every compiled replay against replay_outage (doubles comp time)")
    ap.add_argument("--no-d-check", action="store_true", help="do not stop when R0 differs from D's plan")
    ap.add_argument("--dry-run", action="store_true", help="print the configuration and plan size only")
    a = ap.parse_args(argv)
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
    tag = a.tag or (f"confirm{CONFIRM_SEED}" if a.confirm else None)
    run_dir = REGULARIZED_B4 / tag if tag else REGULARIZED_B4
    es = CONFIRM_SEED if a.confirm else EVAL_SEED
    cfg = dict(eval_seed=es, learn_root=es, n_eval=a.n_eval, workers=a.workers,
               label=LABEL_CONFIRM if a.confirm else LABEL_EXPLORATORY,
               n_check=a.n_check if a.n_check is not None else (3 if a.n_eval < N_EVAL else 1),
               check_all_comp=a.check_all_comp or a.n_eval < N_EVAL, no_d_check=a.no_d_check,
               d_plan=d_plan_dir())
    if a.dry_run:
        dry_run(run_dir, seeds, cfg)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = (run_dir / "progress.log").open("a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    log(f"START phase={a.phase} seeds={seeds[0]}..{seeds[-1]}({len(seeds)}) n_eval={a.n_eval} "
        f"eval_seed={es} workers={a.workers} pid={os.getpid()} label={cfg['label']!r}")
    if a.phase != "analyze":
        load_modules()
    if a.phase == "repro":
        return run_repro(run_dir, cfg, log)
    if a.phase in ("plan", "all"):
        if run_plan(run_dir, seeds, cfg, log):
            return 1
        log("PLANDONE")
    if a.phase in ("eval", "all"):
        if run_eval(run_dir, seeds, cfg, log):
            return 1
    if a.phase in ("analyze", "all"):
        import analyze_E1
        return analyze_E1.main(["--run-dir", str(run_dir)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
