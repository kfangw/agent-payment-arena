"""F4: compiled release policy on a flow built from a public labelled fraud set.
Run from the package root, after scripts/prep_F4.py:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_F4.py smoke
    python3 scripts/run_F4.py run
    python3 scripts/run_F4.py rowboot [--reps N]
Output goes to results/real_label_flow/ (paths.REAL_LABEL_FLOW); the compiled-table caches go to results/cache/.
Run phase: learn draw -> B4 tuning (plan/b4_F4.json), one 500k eval draw, four policies paired.
Rowboot: resample the eval-split rows with replacement, rebuild the flow, redraw, replay (B4 fixed).
Resumable: existing outputs are skipped.
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

import paths                             # noqa: E402
import run_H as HH                       # noqa: E402
RD, FR = HH.RD, HH.FR

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY            # noqa: E402
from arena.experiments.settlement.flows import make_flows                      # noqa: E402
from arena.experiments.settlement.outage import (                              # noqa: E402
    compile_outage, draw_outage_batch, replay_outage)
from arena.experiments.settlement.policies import suspicion_grid               # noqa: E402
from arena.experiments.settlement.watch import horizon_grid, tune_watch_policy  # noqa: E402

EVAL_SEED = 111
LEARN_RNG = [111, 700, 40]
N_LEARN = 200_000
N_EVAL = 500_000
N_EVAL_SMOKE = 100_000
PPE = RD.PPE
Q_TRUE = RD.Q_TRUE
A_BAND, B_BAND = 0.1625, 0.63125
POLICIES = ("comp", "comp_nv", "cutoff", "b4F4")
ACTS = ("grant", "reject", "verify", "wait")
SPEC_FILE = paths.ROOT / "docs" / "spec-F4.md"
OUT = paths.REAL_LABEL_FLOW


# ------------------------------------------------------------------ data and flow
def load_scores():
    z = np.load(paths.CARD_FRAUD / "F4_scores.npz")
    return z["pi0"].astype(float), z["label"].astype(np.int8)


class EmpiricalFlow:
    """Flow.sample(n, rng) -> (v, p_true, theta, pi0).
    Rng order: vdist.sample(n) first, then rng.integers(0, len(rows), n).  theta is the row's label.
    `rows` optionally selects/reorders the stored rows (a bootstrap resample of row indices)."""
    name = "F4"

    def __init__(self, pi0, label, vdist=None, rows=None):
        rows = np.arange(len(pi0)) if rows is None else np.asarray(rows)
        self.pi0 = np.asarray(pi0, dtype=float)[rows]
        self.label = np.asarray(label)[rows]
        self.vdist = vdist if vdist is not None else make_flows()["F1"].vdist

    def sample(self, n, rng):
        v = self.vdist.sample(n, rng)
        idx = rng.integers(0, len(self.pi0), size=n)
        p = self.pi0[idx]
        return v, p, self.label[idx].astype(np.int8), p.copy()


# ------------------------------------------------------------------ tables
def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def tables(world, log):
    """Main table (cached by run_H) and the no-verify table (compiled as in run_D2, cached here)."""
    T, vg = HH.load_table(world, paths.CACHE, log)
    cache = paths.CACHE
    cache.mkdir(parents=True, exist_ok=True)
    p, hp = cache / "table_nv.npy", cache / "table_nv.json"
    if p.exists():
        return T, np.load(p), vg, json.loads(hp.read_text())
    t0 = time.time()
    pol = compile_outage(RD.comp_env(world, Q_TRUE), "comp_nv", drop=(VERIFY,))
    info = dict(hash=RD.table_hash(pol), compile_s=time.time() - t0)
    assert np.array_equal(pol.v_grid, vg)
    Tn = FR.stack_tables(pol)
    info["n_verify_labels"] = int(np.sum(Tn == VERIFY))
    assert info["n_verify_labels"] == 0
    np.save(p, Tn)
    write_json(hp, info)
    log(f"TABLE_NV hash={info['hash'][:16]} t={info['compile_s']:.1f}s")
    return T, Tn, vg, info


# ------------------------------------------------------------------ plan (B4 tuning)
def get_plan(world, ex, flow, log):
    p = OUT / "plan" / "b4_F4.json"
    if p.exists():
        return json.loads(p.read_text())
    t0 = time.time()
    rng = np.random.default_rng(LEARN_RNG)
    learn = draw_outage_batch(world, flow, N_LEARN, rng, payments_per_episode=PPE)
    t_draw = time.time() - t0
    ks = list(horizon_grid(world.H))
    force = {(k, act): FR.replay_forced(world, learn, k, act, ex) for k in ks for act in (GRANT, REJECT, VERIFY)}
    t_force = time.time() - t0 - t_draw
    best, val, rows = tune_watch_policy(lambda k, act: force[(k, act)], suspicion_grid(HH.B3_N), ks, learn.pi0)
    rec = dict(rng=LEARN_RNG, n=N_LEARN, b3_n=HH.B3_N, k_grid=ks,
               b4=dict(k=int(best[0]), a=float(best[1]), b=float(best[2]), tune_mean=float(val)), b4_k_rows=rows,
               learn_base_rate=float(np.mean(learn.theta)), t_draw=t_draw, t_force=t_force,
               t_tune=time.time() - t0 - t_draw - t_force, t_total=time.time() - t0,
               impl="vec forced replay + tune_watch_policy (as run_H.plan_law)")
    write_json(p, rec)
    log(f"PLAN b4=({best[0]},{best[1]:.5f},{best[2]:.5f}) tune_mean={val:.6g} t={rec['t_total']:.1f}s")
    return rec


def b4_tuple(plan):
    return plan["b4"]["k"], plan["b4"]["a"], plan["b4"]["b"]


# ------------------------------------------------------------------ replays
def fns_for(world, d, T, Tn, iv, thr, b4):
    k, a, b = b4
    return dict(comp=HH.comp_fn(T, iv, d.pi0), comp_nv=HH.comp_fn(Tn, iv, d.pi0),
                cutoff=HH.cutoff_fn(d.pi0, thr), b4F4=HH.band_fn(world, d.pi0, k, a, b))


def draw_eval(world, flow, n):
    return draw_outage_batch(world, flow, n, np.random.default_rng(EVAL_SEED), payments_per_episode=PPE)


# ------------------------------------------------------------------ eval
def run_eval(run_dir, n_eval, world, ex, T, Tn, vg, tinfo, plan, flow, log):
    path = run_dir / "npz" / "F4.npz"
    if path.exists():
        log("SKIP eval (npz exists)")
        return
    t0 = time.time()
    d = draw_eval(world, flow, n_eval)
    t_draw = time.time() - t0
    thr, sig0 = HH.pihat0(world)
    iv = FR.v_index(vg, d.v)
    v = np.asarray(d.v, dtype=float)
    mis = np.asarray(d.theta) == 1
    pi0 = np.asarray(d.pi0)
    inband = (pi0 >= A_BAND) & (pi0 < B_BAND)
    fa = HH.first_actions(T, iv, d, world.H)
    fa_nv = HH.first_actions(Tn, iv, d, world.H)
    bs = RD.block_sums
    out = dict(bs_v=bs(v), bs_n_mis=bs(mis.astype(float)), bs_v_mis=bs(np.where(mis, v, 0.0)),
               bs_band=bs(inband.astype(float)), fa_verify_comp=bs((fa == VERIFY).astype(float)))
    fns = fns_for(world, d, T, Tn, iv, thr, b4_tuple(plan))
    times, tot, mean_pay, mis_pass, mis_usd = {}, {}, {}, {}, {}
    grants = {}
    for p in POLICIES:
        t1 = time.time()
        pay, gr = HH.replay(world, d, fns[p], ex)
        times[p] = time.time() - t1
        mg = gr & mis
        out[f"pay_{p}"] = bs(pay)
        out[f"misn_{p}"] = bs(mg.astype(float))
        out[f"misv_{p}"] = bs(np.where(mg, v, 0.0))
        out[f"gr_{p}"] = bs(gr.astype(float))
        tot[p], mean_pay[p] = float(pay.sum()), float(pay.mean())
        mis_pass[p], mis_usd[p] = int(mg.sum()), float(v[mg].sum())
        grants[p] = gr
    assert np.array_equal(grants["cutoff"], pi0 < thr), "Cutoff grant set"
    cnt = np.bincount(fa, minlength=4)[:4]
    cnt_nv = np.bincount(fa_nv, minlength=4)[:4]
    meta = dict(eval_seed=EVAL_SEED, n_eval=n_eval, n_blocks=n_eval // PPE, realized_mean=float(v.mean()),
                realized_median=float(np.median(v)), n_misuse=int(mis.sum()), misuse_rate=float(mis.mean()),
                n_band=int(inband.sum()), band_share=float(inband.mean()),
                band_misuse_rate=float(mis[inband].mean()) if inband.any() else None,
                total_payoff=tot, mean_payoff=mean_pay, misuse_pass_count=mis_pass, misuse_exposure_usd_total=mis_usd,
                comp_first_count=dict(zip(ACTS, cnt.tolist())), comp_nv_first_count=dict(zip(ACTS, cnt_nv.tolist())),
                comp_verify_share=float(cnt[VERIFY] / n_eval), pihat0=thr, sigma0=sig0,
                b4F4=dict(zip("kab", b4_tuple(plan))), b4F4_tune_mean=plan["b4"]["tune_mean"],
                table_hash=HH.REF_HASH, table_nv_hash=tinfo["hash"], t_draw=t_draw, t_replay=times,
                rng_order="vdist.sample then rng.integers (EmpiricalFlow)")
    tmp = path.with_name(path.stem + ".tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    log(f"EVAL done n={n_eval} draw={t_draw:.1f}s " + " ".join(f"{p}={t:.1f}s" for p, t in times.items())
        + " " + " ".join(f"J_{p}={mean_pay[p]:.6g}" for p in POLICIES)
        + f" band_share={meta['band_share']:.5f} verify_share={meta['comp_verify_share']:.5f}")


# ------------------------------------------------------------------ row bootstrap
def run_rowboot(run_dir, n_eval, reps, reps_explicit, world, ex, T, Tn, vg, plan, pi0_rows, lab_rows, log):
    path = run_dir / "npz" / "rowboot.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["rep"]] = r
    thr, _ = HH.pihat0(world)
    b4 = b4_tuple(plan)
    nrows = len(pi0_rows)
    r = 0
    rule_checked = False
    while r < reps:
        r += 1
        if r in done:
            continue
        t0 = time.time()
        rng_r = np.random.default_rng([111, 900, r])
        rows = rng_r.integers(0, nrows, size=nrows)
        flow = EmpiricalFlow(pi0_rows, lab_rows, rows=rows)
        d = draw_eval(world, flow, n_eval)
        iv = FR.v_index(vg, d.v)
        v = np.asarray(d.v, dtype=float)
        mis = np.asarray(d.theta) == 1
        pi0 = np.asarray(d.pi0)
        fns = fns_for(world, d, T, Tn, iv, thr, b4)
        rec = dict(rep=r, rng=[111, 900, r], mean_v=float(v.mean()), n_misuse=int(mis.sum()),
                   band_share=float(np.mean((pi0 >= A_BAND) & (pi0 < B_BAND))),
                   row_base_rate=float(lab_rows[rows].mean()), row_n_fraud=int(lab_rows[rows].sum()),
                   mean_payoff={}, misuse_pass_rate={}, misuse_exposure_usd={})
        for p in POLICIES:
            pay, gr = HH.replay(world, d, fns[p], ex)
            mg = gr & mis
            rec["mean_payoff"][p] = float(pay.mean())
            rec["misuse_pass_rate"][p] = float(mg.sum() / max(mis.sum(), 1))
            rec["misuse_exposure_usd"][p] = float(v[mg].sum() / n_eval)
        rec["wall_s"] = time.time() - t0
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        done[r] = rec
        log(f"ROWBOOT rep {r}/{reps} wall={rec['wall_s']:.1f}s comp-b4F4={(rec['mean_payoff']['comp'] - rec['mean_payoff']['b4F4']) / rec['mean_v'] * 1e4:+.3f}bp "
            f"n_fraud={rec['row_n_fraud']}")
        if (not reps_explicit) and (not rule_checked) and len(done) >= 3:
            rule_checked = True
            first3 = [done[k]["wall_s"] for k in sorted(done)[:3]]
            if np.mean(first3) > 30.0 and reps > 100:
                reps = 100
                log("RULE spec-F4 section 5: >30s/rep, reps=100")
    log(f"ROWBOOT complete reps={reps} total_logged={len(done)}")


# ------------------------------------------------------------------ smoke sanity (vec vs repo)
def vec_vs_repo(world, ex, T, Tn, vg, tinfo, plan, flow, log, npre=20_000):
    d = HH.prefix(draw_eval(world, flow, npre), npre)
    thr, _ = HH.pihat0(world)
    iv = FR.v_index(vg, d.v)
    t0 = time.time()
    Tpol = compile_outage(RD.comp_env(world, Q_TRUE), "comp")
    assert RD.table_hash(Tpol) == HH.REF_HASH
    Tnpol = compile_outage(RD.comp_env(world, Q_TRUE), "comp_nv", drop=(VERIFY,))
    assert RD.table_hash(Tnpol) == tinfo["hash"]
    b4 = b4_tuple(plan)
    repo = dict(comp=Tpol, comp_nv=Tnpol,
                cutoff=lambda i, l, r, v, pi: GRANT if pi < thr else REJECT,
                b4F4=HH.OB4(b4[0], b4[1], b4[2], world.H, world.N))
    fns = fns_for(world, d, T, Tn, iv, thr, b4)
    res = {}
    for p in POLICIES:
        a, gr = HH.replay(world, d, fns[p], ex)
        bb = replay_outage(world, d, repo[p], ex)
        res[p] = float(np.max(np.abs(a - bb)))
    ok = all(x == 0.0 for x in res.values())
    log(f"CHECK vec==repo prefix={npre} max_abs={res} {'PASS' if ok else 'FAIL'} t={time.time() - t0:.1f}s")
    return ok


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "run", "rowboot"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    a = ap.parse_args(argv)
    smoke = a.phase == "smoke"
    run_dir = OUT / "smoke" if smoke else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    prog = (OUT / "progress.log").open("a", buffering=1)
    T0 = time.time()

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [{(time.time() - T0) / 60:6.1f}m] {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    log(f"START phase={a.phase} pid={os.getpid()}")
    world, ex = HH.setup()
    T, Tn, vg, tinfo = tables(world, log)
    pi0_rows, lab_rows = load_scores()
    flow = EmpiricalFlow(pi0_rows, lab_rows)
    plan = get_plan(world, ex, flow, log)
    n_eval = a.n_eval or (N_EVAL_SMOKE if smoke else N_EVAL)
    try:
        if smoke:
            ok = vec_vs_repo(world, ex, T, Tn, vg, tinfo, plan, flow, log)
            run_eval(run_dir, n_eval, world, ex, T, Tn, vg, tinfo, plan, flow, log)
            run_rowboot(run_dir, n_eval, a.reps or 3, True, world, ex, T, Tn, vg, plan, pi0_rows, lab_rows, log)
            import analyze_F4
            analyze_F4.main(["--tag", "smoke", "--nb", "100"])
            log("SMOKEDONE" + ("" if ok else " (vec==repo FAILED)"))
            return 0 if ok else 1
        if a.phase == "run":
            run_eval(run_dir, n_eval, world, ex, T, Tn, vg, tinfo, plan, flow, log)
            log("RUN done")
        else:
            run_rowboot(run_dir, n_eval, a.reps or 200, a.reps is not None, world, ex, T, Tn, vg, plan,
                        pi0_rows, lab_rows, log)
    except Exception:
        log("FAIL " + traceback.format_exc().splitlines()[-1])
        with (OUT / "errors.log").open("a") as f:
            f.write(traceback.format_exc() + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
