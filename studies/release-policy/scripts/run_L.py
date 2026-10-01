"""Experiment L: misuse intent answers at a lower hazard, rho_m = lam * rho.
Spec: docs/spec-L.md.  World E-fitted, flow F2, main-run constants, main-run rng (seed 8: tuning
200,000 first, evaluation 5,663,400 second), draws with draw_outage_batch(..., rho_m=lam*rho).
lam in {0, 0.25, 0.5, 1}; lam = 0 passes rho_m = 0.0 (the draw clips the hazard to 1e-9,
so every misuse answer falls far beyond the deadline; checked).

Run from the package root:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_L.py check      (vectorized == repo replay on a prefix; writes results/misuse_answer/smoke/)
    python3 scripts/run_L.py smoke      (100,000 evaluation payments)
    python3 scripts/run_L.py run        (full, resumable)
    python3 scripts/run_L.py analyze [--tag smoke]

Output goes to results/misuse_answer/ (paths.MISUSE_ANSWER); the compiled-table caches go to results/cache/.

Policies on each draw (paired): Comp (main-run table, not recompiled), B4_lam (tuned on that
draw's tuning split, 161-point grid), Cutoff (grant iff pi0 < pihat0 at arrival), and B4_main
(the lam = 1 tuning, for reference).  Comp_lam is not built: compile_outage takes one answer
hazard (env.rho) for both intents (see the change log of the spec).
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
import run_D2 as D2                     # noqa: E402
import run_D as RD                      # noqa: E402
import fastreplay_G as FR               # noqa: E402
import run_H as HH                      # noqa: E402

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY               # noqa: E402
from arena.experiments.settlement.flows import make_flows                          # noqa: E402
from arena.experiments.settlement.outage import (                                  # noqa: E402
    compile_outage, draw_outage_batch, replay_outage)
from arena.experiments.settlement.watch import (                                   # noqa: E402
    FixedActionOutageWatchPolicy as OB4Force, OutageWatchBandPolicy as OB4)

LAMS = (1.0, 0.5, 0.25, 0.0)
FLOW, SEED = "F2", 8
N_TUNE, N_EVAL, PPE, DEN = D2.N_TUNE, D2.N_EVAL, D2.PPE, D2.DEN
NB, BSEED = 1_000, 20261001
MARGIN_BP = 10.0
POLICIES = ("comp", "b4lam", "cutoff", "b4main")
write_json = D2.write_json
FIELDS = ("v", "p_true", "theta", "pi0", "u_stage", "t_ans", "paths")


def ltag(lam):
    return f"lam{lam:.2f}".replace(".", "p")


def rho_m_of(world, lam):
    return 0.0 if lam == 0 else lam * float(world.rho)


def draw_lam(world, lam, n_eval, n_tune=N_TUNE):
    rng = np.random.default_rng(SEED)
    fl = make_flows()[FLOW]
    rm = rho_m_of(world, lam)
    tune = draw_outage_batch(world, fl, N_TUNE, rng, payments_per_episode=PPE, rho_m=rm)
    if n_tune < N_TUNE:
        tune = HH.prefix(tune, n_tune)
    ev = draw_outage_batch(world, fl, n_eval, rng, payments_per_episode=PPE, rho_m=rm)
    return tune, ev


def replay_rec(env, d, fn, ex):
    """Vectorized replay; records grants, and for VERIFY the window w = min(tau, l)."""
    FR.set_exercise(ex)
    n = len(d)
    granted = np.zeros(n, dtype=bool)
    ver = np.zeros(n, dtype=bool)
    ver_w = np.zeros(n, dtype=np.int64)

    def g(k, i, l, r):
        a = np.asarray(fn(k, i, l, r))
        granted[k[a == GRANT]] = True
        vm = a == VERIFY
        if vm.any():
            ver[k[vm]] = True
            ver_w[k[vm]] = np.minimum(env.tau, l[vm])
        return a
    pay = FR.replay_vec(env, d, g)
    return pay, granted, ver, ver_w


def load_b4main():
    b = json.loads((paths.NO_CHANNEL / "plan" / f"b4main_{FLOW}.json").read_text())
    return dict(k=b["k"], a=b["a"], b=b["b"], tune_mean=b["tune_mean"])


def ref_d2(run_dir):
    sub = paths.NO_CHANNEL / run_dir.name if run_dir != paths.MISUSE_ANSWER else paths.NO_CHANNEL
    if not (sub / "summary.json").exists():       # no D2 run with the same tag: fall back to the full D2 run
        sub = paths.NO_CHANNEL
    s = json.loads((sub / "summary.json").read_text())["flows"][FLOW]
    return dict(J_comp=s["J"]["comp"]["point"], J_cutoff=s["J"]["cutoff"]["point"],
                J_b4main=s["J"]["b4main"]["point"], b4main=s["meta"]["b4main"])


# ------------------------------------------------------------------ one lambda
def run_lam(lam, run_dir, n_eval, n_tune, world, ex, T, vg, log):
    tg = ltag(lam)
    path = run_dir / "npz" / f"{tg}.npz"
    if path.exists():
        log(f"SKIP {tg}")
        return
    t0 = time.time()
    tune, d = draw_lam(world, lam, n_eval, n_tune)
    t_draw = time.time() - t0
    pj = run_dir / "plan" / f"b4_{tg}.json"
    if pj.exists():
        b4 = json.loads(pj.read_text())
    else:
        b4 = D2.tune_b4(world, ex, tune)
        b4.update(lam=lam, rho_m=rho_m_of(world, lam), rng=SEED)
        write_json(pj, b4)
    log(f"B4 {tg} ({b4['k']},{b4['a']!r},{b4['b']!r}) tune_mean={b4['tune_mean']!r} "
        f"force={b4['t_force']:.1f}s tune={b4['t_tune']:.1f}s")
    del tune
    b4m = load_b4main()
    thr, _ = HH.pihat0(world)
    iv = FR.v_index(vg, d.v)
    v = np.asarray(d.v, dtype=float)
    mis = np.asarray(d.theta) == 1
    t_ans = np.asarray(d.t_ans)
    bs = RD.block_sums
    out = dict(bs_v=bs(v), bs_n_mis=bs(mis.astype(float)), bs_v_mis=bs(np.where(mis, v, 0.0)))
    fns = dict(comp=HH.comp_fn(T, iv, d.pi0), cutoff=HH.cutoff_fn(d.pi0, thr),
               b4lam=HH.band_fn(world, d.pi0, b4["k"], b4["a"], b4["b"]),
               b4main=HH.band_fn(world, d.pi0, b4m["k"], b4m["a"], b4m["b"]))
    J, times, vstat = {}, {}, {}
    for p in POLICIES:
        t1 = time.time()
        pay, gr, ver, vw = replay_rec(world, d, fns[p], ex)
        times[p] = time.time() - t1
        mg = gr & mis
        out[f"pay_{p}"] = bs(pay)
        out[f"misn_{p}"] = bs(mg.astype(float))
        out[f"misv_{p}"] = bs(np.where(mg, v, 0.0))
        out[f"gr_{p}"] = bs(gr.astype(float))
        J[p] = float(pay.mean())
        noans = ver & (t_ans > vw)
        rec = dict(n_verify=int(ver.sum()), n_noans=int(noans.sum()))
        for nm, sel in (("honest", ~mis), ("misuse", mis)):
            rec[f"n_verify_{nm}"] = int((ver & sel).sum())
            rec[f"n_noans_{nm}"] = int((noans & sel).sum())
        rec["noans_share"] = rec["n_noans"] / max(1, rec["n_verify"])
        rec["noans_share_honest"] = rec["n_noans_honest"] / max(1, rec["n_verify_honest"])
        rec["noans_share_misuse"] = rec["n_noans_misuse"] / max(1, rec["n_verify_misuse"])
        rec["misuse_granted_after_verify"] = int((ver & mis & gr).sum())
        rec["verify_share_of_misuse"] = float((ver & mis).sum() / max(1, mis.sum()))
        vstat[p] = rec
        if p == "cutoff":
            assert np.array_equal(gr, np.asarray(d.pi0) < thr), "Cutoff grant set"
        del pay, gr, ver, vw
    ma = mis & (t_ans <= world.tau)
    meta = dict(lam=lam, rho=float(world.rho), rho_m=rho_m_of(world, lam), flow=FLOW, eval_rng=SEED,
                n_eval=n_eval, n_tune=n_tune, n_blocks=n_eval // PPE, pihat0=thr,
                b4lam=dict(k=b4["k"], a=b4["a"], b=b4["b"], tune_mean=b4["tune_mean"]), b4main=b4m,
                J=J, t_draw=t_draw, t_replay=times, verify=vstat, misuse_rate=float(mis.mean()),
                misuse_answer_within_tau_share=float(ma.sum() / max(1, mis.sum())),
                misuse_t_ans_min=int(t_ans[mis].min()) if mis.any() else None,
                realized_mean=float(v.mean()), ref_d2=ref_d2(run_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    log(f"DONE {tg} draw={t_draw:.1f}s " + " ".join(f"{p}={t:.1f}s" for p, t in times.items())
        + " " + " ".join(f"J_{p}={J[p]:.8g}" for p in POLICIES)
        + f" comp_noans={vstat['comp']['noans_share']:.4f} total={time.time() - t0:.1f}s")


# ------------------------------------------------------------------ checks
def run_check(world, ex, T, vg, log, npre=20_000):
    res = {}
    rng = np.random.default_rng(SEED)
    fl = make_flows()[FLOW]
    base = draw_outage_batch(world, fl, npre, np.random.default_rng(SEED), payments_per_episode=PPE)
    d1 = draw_outage_batch(world, fl, npre, np.random.default_rng(SEED), payments_per_episode=PPE,
                           rho_m=float(world.rho))
    res["lam1_equals_default_prefix"] = bool(all(np.array_equal(getattr(base, f), getattr(d1, f)) for f in FIELDS))
    d0 = draw_outage_batch(world, fl, npre, np.random.default_rng(SEED), payments_per_episode=PPE, rho_m=0.0)
    mis = np.asarray(d0.theta) == 1
    res["lam0_accepts_zero"] = True
    res["lam0_misuse_t_ans_min"] = int(d0.t_ans[mis].min())
    res["lam0_all_misuse_beyond_tau"] = bool(np.all(d0.t_ans[mis] > world.tau))
    res["lam0_honest_t_ans_same"] = bool(np.array_equal(d0.t_ans[~mis], base.t_ans[~mis]))
    res["lam0_other_fields_same"] = bool(all(np.array_equal(getattr(base, f), getattr(d0, f))
                                             for f in FIELDS if f != "t_ans"))
    dq = draw_outage_batch(world, fl, npre, np.random.default_rng(SEED), payments_per_episode=PPE,
                           rho_m=0.25 * float(world.rho))
    mq = np.asarray(dq.theta) == 1
    res["lam0p25_misuse_answer_within_tau"] = float(np.mean(dq.t_ans[mq] <= world.tau))
    res["lam1_misuse_answer_within_tau"] = float(np.mean(base.t_ans[mis] <= world.tau))
    del rng
    pol = compile_outage(RD.comp_env(world, D2.Q_TRUE), "comp")
    res["table_hash_main"] = RD.table_hash(pol) == HH.REF_HASH
    thr, _ = HH.pihat0(world)
    cases = {}
    for dn, dd in (("lam0", d0), ("lam0p25", dq)):
        iv = FR.v_index(vg, dd.v)
        for nm, fn, rp in (("comp", HH.comp_fn(T, iv, dd.pi0), pol),
                           ("cutoff", HH.cutoff_fn(dd.pi0, thr), lambda i, l, r, v, pi: GRANT if pi < thr else REJECT),
                           ("b4_k0", HH.band_fn(world, dd.pi0, 0, 0.1125, 0.8), OB4(0, 0.1125, 0.8, world.H, world.N)),
                           ("b4_k4", HH.band_fn(world, dd.pi0, 4, 0.1125, 0.8), OB4(4, 0.1125, 0.8, world.H, world.N))):
            a, gr, _, _ = replay_rec(world, dd, fn, ex)
            gc = HH.GrantCounter(rp)
            bb = replay_outage(world, dd, gc, ex)
            cases[f"{dn}:{nm}"] = dict(max_abs=float(np.max(np.abs(a - bb))), grants_vec=int(gr.sum()), grants_repo=gc.n)
        for k in (0, 4):
            for act in (GRANT, REJECT, VERIFY):
                a = FR.replay_forced(world, dd, k, act, ex)
                bb = replay_outage(world, dd, OB4Force(k, act, world.H, world.N), ex)
                cases[f"{dn}:force_k{k}_{'GRV'[[GRANT, REJECT, VERIFY].index(act)]}"] = dict(
                    max_abs=float(np.max(np.abs(a - bb))))
    ok = (res["lam1_equals_default_prefix"] and res["lam0_all_misuse_beyond_tau"] and res["lam0_honest_t_ans_same"]
          and res["lam0_other_fields_same"] and res["table_hash_main"]
          and all(c["max_abs"] == 0 and c.get("grants_vec", 0) == c.get("grants_repo", 0) for c in cases.values()))
    res.update(prefix=npre, cases=cases, pass_=bool(ok))
    log(f"CHECK {json.dumps({k: v for k, v in res.items() if k != 'cases'})} cases={len(cases)} "
        f"max={max(c['max_abs'] for c in cases.values()):.1e} {'PASS' if ok else 'FAIL'}")
    return res


# ------------------------------------------------------------------ analysis
def analyze(run_dir, nb=NB):
    res = dict(bootstrap=dict(resamples=nb, seed=BSEED, note="paired block bootstrap (50-payment episodes), "
                              "same resamples for every policy of a lambda, default_rng(seed) restarted per "
                              "lambda, 95% percentile"),
               unit="bp of $49.50", comp_lam="not built: compile_outage takes one answer hazard for both intents",
               lams={})
    for lam in LAMS:
        tg = ltag(lam)
        p = run_dir / "npz" / f"{tg}.npz"
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
        c = lambda s, k: s[..., ix[k]]
        bp = lambda x: x / DEN * 1e4

        def stat(f):
            return dict(point=float(f(P)), ci=D2.ci(f(S)))
        pairs = {"comp-b4lam": ("comp", "b4lam"), "comp-cutoff": ("comp", "cutoff"),
                 "comp-b4main": ("comp", "b4main"), "b4lam-b4main": ("b4lam", "b4main")}
        r = dict(meta=meta, J={q: stat(lambda s, q=q: c(s, f"pay_{q}") / n) for q in POLICIES},
                 diff_bp={k: stat(lambda s, a=a, b=b: bp((c(s, f"pay_{a}") - c(s, f"pay_{b}")) / n))
                          for k, (a, b) in pairs.items()},
                 misuse_granted_pct={q: stat(lambda s, q=q: 100.0 * c(s, f"misn_{q}") / c(s, "bs_n_mis"))
                                     for q in POLICIES},
                 misuse_exposure_usd_per_payment={q: float(P[ix[f"misv_{q}"]] / n) for q in POLICIES},
                 grant_pct={q: float(100.0 * P[ix[f"gr_{q}"]] / n) for q in POLICIES},
                 comp_verify_noans_share=meta["verify"]["comp"]["noans_share"],
                 comp_verify_noans_share_misuse=meta["verify"]["comp"]["noans_share_misuse"],
                 comp_verify_noans_share_honest=meta["verify"]["comp"]["noans_share_honest"])
        if lam == 1.0:
            ref = meta["ref_d2"]
            r["check_lam1_vs_D2"] = dict(
                J_comp=r["J"]["comp"]["point"], J_comp_D2=ref["J_comp"],
                comp_equal=r["J"]["comp"]["point"] == ref["J_comp"],
                cutoff_equal=r["J"]["cutoff"]["point"] == ref["J_cutoff"],
                b4main_equal=r["J"]["b4main"]["point"] == ref["J_b4main"],
                b4lam_params_equal_D2=[meta["b4lam"][k] for k in ("k", "a", "b", "tune_mean")]
                == [ref["b4main"][k] for k in ("k", "a", "b", "tune_mean")])
            d2p = paths.NO_CHANNEL / "npz" / f"flow_{FLOW}.npz"
            if d2p.exists() and run_dir == paths.MISUSE_ANSWER:
                z2 = np.load(d2p)
                shared = sorted(k for k in z.files if k != "meta" and k in z2.files)
                neq = [k for k in shared if not np.array_equal(z[k], z2[k])]
                r["check_lam1_vs_D2"].update(block_arrays_compared=len(shared), block_arrays_differ=neq,
                                             block_sums_equal=not neq)
        res["lams"][tg] = r
    write_json(run_dir / "summary.json", res)
    write_md(run_dir, res)
    return res


def fmt(s):
    return f"{s['point']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"


def write_md(run_dir, res):
    L = ["# Experiment L: automatic summary", "", "Generated from summary.json. E-fitted, F2, seed 8. Unit: bp of the mean "
         "exposure $49.50; intervals are paired block bootstrap 95%. Comp_lambda is not built (the compiler does not "
         "take an answer hazard per intent).", ""]
    L.append("| lambda | rho_m | Comp - B4_lambda | Comp - Cutoff | Comp - B4_main | misuse granted, Comp / B4_lambda / Cutoff (%) | "
             "share of Comp verify payments without an answer (all / misuse / honest) | B4_lambda (k, a, b) |")
    L.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for tg, r in res["lams"].items():
        m, dm = r["meta"], r["misuse_granted_pct"]
        b = m["b4lam"]
        L.append(f"| {m['lam']:g} | {m['rho_m']:.5f} | {fmt(r['diff_bp']['comp-b4lam'])} | {fmt(r['diff_bp']['comp-cutoff'])} | "
                 f"{fmt(r['diff_bp']['comp-b4main'])} | {dm['comp']['point']:.2f} / {dm['b4lam']['point']:.2f} / "
                 f"{dm['cutoff']['point']:.2f} | {r['comp_verify_noans_share']:.4f} / "
                 f"{r['comp_verify_noans_share_misuse']:.4f} / {r['comp_verify_noans_share_honest']:.4f} | "
                 f"({b['k']}, {b['a']:.5g}, {b['b']:.5g}) |")
        rows.append((m["lam"], r))
    L.append("")
    L.append("## Reading-rule sentences")
    below = [lam for lam, r in rows if r["diff_bp"]["comp-b4lam"]["point"] < -MARGIN_BP]
    if below:
        L.append(f"- Largest lambda at which Comp - B4_lambda falls below -10 bp: {max(below):g}.")
        ok = [lam for lam, r in rows if lam > max(below)]
        if ok:
            L.append(f"- \"Even if (H2) fails down to lambda = {min(ok):g}, the compiled policy stays inside the margin.\"")
    else:
        L.append("- Comp - B4_lambda does not fall below -10 bp at any lambda (lambda = 0 included).")
    r0 = dict(rows).get(0.0)
    if r0 is not None and r0["diff_bp"]["comp-b4lam"]["point"] >= -MARGIN_BP:
        v = r0["meta"]["verify"]["comp"]
        L.append(f"- Inside the margin at lambda = 0 too. Check: at lambda = 0, {100 * r0['meta']['misuse_answer_within_tau_share']:.3f}% of misuse payments "
                 f"answer within tau (the misuse answer hazard is clipped to 1e-9; minimum t_ans = {r0['meta']['misuse_t_ans_min']}). "
                 f"Of the {v['n_verify_misuse']} misuse payments that Comp verified, {v['n_noans_misuse']} got no answer and "
                 f"{v['misuse_granted_after_verify']} were granted after the verify. A query that times out ends without a release, so it filters "
                 "misuse. In the replay rule an answered misuse payment is not released either (a grant after an answer goes "
                 "to honest intents only), so lambda changes only the waiting cost of misuse verify payments (at most tau ticks).")
    for tg, r in res["lams"].items():
        v = r["meta"]["verify"]["comp"]
        L.append(f"  - lambda = {r['meta']['lam']:g}: of {v['n_verify_misuse']} misuse payments verified by Comp, "
                 f"{v['misuse_granted_after_verify']} were granted after the verify; misuse no-answer share {v['noans_share_misuse']:.4f}.")
    lams_sorted = sorted(rows, key=lambda x: -x[0])
    for q, nm in (("comp", "Comp"), ("b4lam", "B4_lambda"), ("cutoff", "Cutoff")):
        seq = [r["misuse_granted_pct"][q]["point"] for _, r in lams_sorted]
        dif = np.diff(seq)
        if np.all(dif == 0):
            dirn = "stays the same"
        elif np.all(dif <= 0):
            dirn = "falls"
        elif np.all(dif >= 0):
            dirn = "rises"
        else:
            dirn = "is not monotone"
        L.append(f"- Misuse granted ({nm}) as lambda goes from 1 to 0: " + " -> ".join(f"{x:.3f}%" for x in seq) + f"; it {dirn}.")
    L.append("")
    L.append("## Checks")
    for tg, r in res["lams"].items():
        if "check_lam1_vs_D2" in r:
            c = r["check_lam1_vs_D2"]
            L.append(f"- lambda = 1 draw against the D2 F2 draw: J(Comp) {'equal' if c['comp_equal'] else 'different'}, J(Cutoff) "
                     f"{'equal' if c['cutoff_equal'] else 'different'}, J(B4_main) {'equal' if c['b4main_equal'] else 'different'}, "
                     f"B4_1 tuning result equals the D2 B4: {'yes' if c['b4lam_params_equal_D2'] else 'no'}.")
            if "block_sums_equal" in c:
                L.append(f"- lambda = 1 block sums ({c['block_arrays_compared']} per-episode arrays compared element by element with the D2 npz): "
                         f"{'all equal' if c['block_sums_equal'] else 'different: ' + ', '.join(c['block_arrays_differ'])}.")
    (run_dir / "summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "smoke", "run", "analyze"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lams", default=None)
    ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase in ("smoke", "check") else None)
    run_dir = paths.MISUSE_ANSWER / tag if tag else paths.MISUSE_ANSWER
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
    T, vg = HH.load_table(world, paths.CACHE, log)
    if a.phase == "check":
        chk = run_check(world, ex, T, vg, log)
        write_json(run_dir / "checks.json", chk)
        return 0 if chk["pass_"] else 1
    lams = [float(x) for x in a.lams.split(",")] if a.lams else list(LAMS)
    n_eval, n_tune = (100_000, 20_000) if a.phase == "smoke" else (N_EVAL, N_TUNE)
    fails = []
    for lam in lams:
        try:
            run_lam(lam, run_dir, n_eval, n_tune, world, ex, T, vg, log)
        except Exception:
            fails.append(lam)
            log(f"FAIL {lam}")
            with (run_dir / "errors.log").open("a") as f:
                f.write(f"FAIL {lam}\n{traceback.format_exc()}\n")
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
