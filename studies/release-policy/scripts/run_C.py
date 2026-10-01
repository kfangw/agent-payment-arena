"""Experiment C: variation of Comp - B4 across B4 tuning samples.
E-fitted world, flows F1, F2, F3, true constants.  Spec: docs/spec-C.md.

Run from the package root:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_C.py smoke     (100,000 evaluation, 20,000 learning payments, F2)
    python3 scripts/run_C.py run       (full, resumable)
    python3 scripts/run_C.py analyze [--tag smoke]

Output goes to results/tuning_variation/ (paths.TUNING_VARIATION); the table cache is results/cache/.

plan  per (flow, s), s = 1..5: 200,000 learning payments from default_rng([112, 700, s]) with that
      flow; B4_s tuned with the tuner of run_H (vectorized forced replays, 161-point grid,
      horizon_grid(H)).  <run>/plan/b4_<flow>_s<s>.json.  Two worker processes.
eval  per flow: the main-run evaluation split (run_D2.draw_main: default_rng(seed_f), tuning
      200,000 then evaluation 5,663,400); Comp (main-run table) once, B4_1..B4_5 each.
      Comp's block sums are compared with experiment D2's npz of the same draw.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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
import run_D2 as D2                     # noqa: E402
import run_D as RD                      # noqa: E402
import fastreplay_G as FR               # noqa: E402
import run_H as HH                      # noqa: E402

from arena.experiments.settlement.flows import make_flows                            # noqa: E402
from arena.experiments.settlement.outage import draw_outage_batch                    # noqa: E402

FLOWS = D2.FLOWS
S_LIST = (1, 2, 3, 4, 5)
LEARN_ROOT, LEARN_TAG = 112, 700
N_LEARN = 200_000
PPE = RD.PPE
DEN = 49.5
NB, BSEED = 1_000, 20260930
TSEED = [20260930, 5]          # generator for the resampling of tuning samples
MARGIN = 10.0


def plan_path(run_dir, flow, s):
    return run_dir / "plan" / f"b4_{flow}_s{s}.json"


def job_plan(args):
    run_dir, flow, s, n_learn = args
    try:
        p = plan_path(run_dir, flow, s)
        if p.exists():
            return flow, s, json.loads(p.read_text()), None
        world, ex = HH.setup()
        t0 = time.time()
        learn = draw_outage_batch(world, make_flows()[flow], n_learn,
                                  np.random.default_rng([LEARN_ROOT, LEARN_TAG, s]), payments_per_episode=PPE)
        t_draw = time.time() - t0
        rec = D2.tune_b4(world, ex, learn)
        rec.update(flow=flow, s=s, rng=[LEARN_ROOT, LEARN_TAG, s], t_draw=t_draw, n_learn=n_learn)
        D2.write_json(p, rec)
        return flow, s, rec, None
    except Exception:
        return flow, s, None, traceback.format_exc()


def run_plan(run_dir, flows, n_learn, workers, log):
    jobs = [(run_dir, f, s, n_learn) for f in flows for s in S_LIST]
    with mp.get_context("fork").Pool(workers) as pool:
        for flow, s, rec, tb in pool.imap_unordered(job_plan, jobs, chunksize=1):
            if rec is None:
                log(f"FAIL plan {flow} s={s}")
                (run_dir / "errors.log").open("a").write(f"FAIL plan {flow} s={s}\n{tb}\n")
                return 1
            log(f"PLAN {flow} s={s} b4=({rec['k']},{rec['a']!r},{rec['b']!r}) tune_mean={rec['tune_mean']:.8g} "
                f"force={rec['t_force']:.1f}s tune={rec['t_tune']:.1f}s")
    return 0


def eval_flow(flow, run_dir, n_eval, world, ex, T, vg, log):
    path = run_dir / "npz" / f"flow_{flow}.npz"
    if path.exists():
        log(f"SKIP {flow}")
        return
    t0 = time.time()
    tune, d = D2.draw_main(world, flow, n_eval)
    del tune
    t_draw = time.time() - t0
    iv = FR.v_index(vg, d.v)
    bs = RD.block_sums
    out, times, J, params = {}, {}, {}, {}
    t1 = time.time()
    pay, _ = HH.replay(world, d, HH.comp_fn(T, iv, d.pi0), ex)
    times["comp"] = time.time() - t1
    out["pay_comp"] = bs(pay)
    J["comp"] = float(pay.mean())
    del pay
    for s in S_LIST:
        rec = json.loads(plan_path(run_dir, flow, s).read_text())
        params[s] = dict(k=rec["k"], a=rec["a"], b=rec["b"], tune_mean=rec["tune_mean"])
        t1 = time.time()
        pay, _ = HH.replay(world, d, HH.band_fn(world, d.pi0, rec["k"], rec["a"], rec["b"]), ex)
        times[f"b4_s{s}"] = time.time() - t1
        out[f"pay_b4_s{s}"] = bs(pay)
        J[f"b4_s{s}"] = float(pay.mean())
        del pay
    # cross-check with run_D2.py (same draw, same table), if its block sums exist
    chk = dict(available=False)
    d2 = paths.NO_CHANNEL / "npz" / f"flow_{flow}.npz"
    if d2.exists() and n_eval == D2.N_EVAL:
        z = np.load(d2)
        ref = z["pay_comp"]
        chk = dict(available=True, max_abs_block_diff=float(np.max(np.abs(ref - out["pay_comp"]))),
                   n_blocks=len(ref))
        out["pay_b4main"] = z["pay_b4main"]
        chk["b4main"] = json.loads(str(z["meta"]))["b4main"]
    meta = dict(flow=flow, eval_rng=D2.SEEDS[flow], n_eval=n_eval, n_blocks=n_eval // PPE, J=J, b4=params,
                t_draw=t_draw, t_replay=times, d2_check=chk, table_hash=HH.REF_HASH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    log(f"DONE {flow} draw={t_draw:.1f}s " + " ".join(f"{k}={v:.1f}s" for k, v in times.items())
        + " " + " ".join(f"J_{k}={v:.8g}" for k, v in J.items()) + f" d2_check={chk.get('max_abs_block_diff')}"
        + f" flow_total={time.time() - t0:.1f}s")


# ------------------------------------------------------------------ analysis
def analyze(run_dir, nb=NB):
    res = dict(unit="bp of $49.50", bootstrap=dict(
        resamples=nb, block_seed=BSEED, tuning_seed=TSEED,
        note="block resample default_rng(20260930) restarted per flow (paired over all policies); "
             "two-way: each resample also draws 5 of the 5 tuning samples with replacement "
             "(default_rng([20260930, 5]) restarted per flow) and averages D_s over them"),
        margin=MARGIN, flows={})
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
        S = D2.boot_sums(M, nb, BSEED)
        P = M.sum(axis=0)
        bp = lambda x: x / n / DEN * 1e4
        Dpt = np.array([bp(P[ix["pay_comp"]] - P[ix[f"pay_b4_s{s}"]]) for s in S_LIST])
        Dbs = np.column_stack([bp(S[:, ix["pay_comp"]] - S[:, ix[f"pay_b4_s{s}"]]) for s in S_LIST])  # (nb, 5)
        tr = np.random.default_rng(TSEED)
        pick = tr.integers(0, len(S_LIST), (nb, len(S_LIST)))
        two = Dbs[np.arange(nb)[:, None], pick].mean(axis=1)
        lo, hi = D2.ci(two)
        mean = float(Dpt.mean())
        per_s = {str(s): dict(D=float(Dpt[j]), ci_block=D2.ci(Dbs[:, j]), **meta["b4"][str(s)])
                 for j, s in enumerate(S_LIST)}
        r = dict(J_comp=meta["J"]["comp"], per_s=per_s, mean=mean, min=float(Dpt.min()), max=float(Dpt.max()),
                 sd=float(Dpt.std(ddof=1)), two_way_ci=[lo, hi], two_way_half_width=(hi - lo) / 2,
                 block_only_ci_of_mean=D2.ci(Dbs.mean(axis=1)), d2_check=meta["d2_check"],
                 n_blocks=meta["n_blocks"], eval_rng=meta["eval_rng"])
        if "pay_b4main" in ix:
            r["main_b4"] = dict(D=float(bp(P[ix["pay_comp"]] - P[ix["pay_b4main"]])),
                                ci_block=D2.ci(bp(S[:, ix["pay_comp"]] - S[:, ix["pay_b4main"]])),
                                main_run_reported=list(D2.MAIN_LEAD[flow]), **meta["d2_check"]["b4main"])
        within = lo >= -MARGIN and hi <= MARGIN
        r["verdict"] = dict(within_margin=bool(within), sign_split=bool((Dpt < 0).any() and (Dpt > 0).any()),
                            n_negative=int((Dpt < 0).sum()),
                            reach=float(max(abs(lo), abs(hi))))
        res["flows"][flow] = r
    fl = res["flows"]
    if fl:
        allin = all(r["verdict"]["within_margin"] for r in fl.values())
        res["verdict"] = dict(
            all_within=allin,
            rule=("keep 'equal within the margin'; caption: half-width with tuning variation <= x bp"
                  if allin else "'on average within the margin; the interval with tuning variation reaches ±y'; "
                  "abstract 'matches in payoff' gets 'on average'"),
            x_half_width_max=max(r["two_way_half_width"] for r in fl.values()),
            y_reach_max=max(r["verdict"]["reach"] for r in fl.values()),
            sign_split_flows=[f for f, r in fl.items() if r["verdict"]["sign_split"]])
    D2.write_json(run_dir / "summary.json", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "run", "analyze"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase == "smoke" else None)
    run_dir = paths.TUNING_VARIATION / tag if tag else paths.TUNING_VARIATION
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
    flows = ["F2"] if a.phase == "smoke" else list(FLOWS)
    n_eval, n_learn = (100_000, 20_000) if a.phase == "smoke" else (D2.N_EVAL, N_LEARN)
    if run_plan(run_dir, flows, n_learn, a.workers, log):
        return 1
    log("PLANDONE")
    world, ex = HH.setup()
    T, vg = HH.load_table(world, paths.CACHE, log)
    for fl in flows:
        try:
            eval_flow(fl, run_dir, n_eval, world, ex, T, vg, log)
        except Exception:
            log(f"FAIL {fl}")
            (run_dir / "errors.log").open("a").write(f"FAIL {fl}\n{traceback.format_exc()}\n")
            return 1
    t1 = time.time()
    analyze(run_dir, 100 if a.phase == "smoke" else a.nb)
    log(f"ANALYZE t={time.time() - t1:.1f}s")
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={(time.time() - T0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
