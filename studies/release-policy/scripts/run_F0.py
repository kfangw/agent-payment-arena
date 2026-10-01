"""Experiment F0: the stage-0 submission failure f0 at its measured value.
Spec: docs/spec-F0.md.  World W_m = E-fitted with f[0] (p_sub) = 0.01216147, every other constant
unchanged; world W_d = E-fitted (f0 = 0.005).  Flows F1, F2, F3 with the main-run rng
(seed 7, 8, 9: tuning 200,000 first, evaluation 5,663,400 second).

Run from the package root:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_F0.py check      (vectorized == repo replay on a prefix; writes results/submission_hazard/smoke/)
    python3 scripts/run_F0.py smoke      (100,000 evaluation payments, F2)
    python3 scripts/run_F0.py run        (full, resumable)
    python3 scripts/run_F0.py analyze [--tag smoke]

Output goes to results/submission_hazard/ (paths.SUBMISSION_HAZARD); the compiled-table caches go to results/cache/.
The revert share is read from data/x402_amounts/revert_share.json.

The outage draw does not read f (only its length), so the W_m draw of a flow is the same
array set as the W_d draw; the check phase confirms this on a prefix and the run confirms
it through J(Comp_d) on W_d == the reference value.  Each world is replayed with its own
exercise rule ex (window_AD of that world), as the replay semantics require.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths                            # noqa: E402
import run_D2 as D2                     # noqa: E402
import run_D as RD                      # noqa: E402
import fastreplay_G as FR               # noqa: E402
import run_H as HH                      # noqa: E402

from arena.experiments.settlement.core import GRANT, REJECT, VERIFY, WAIT         # noqa: E402
from arena.experiments.settlement.outage import (                                  # noqa: E402
    compile_outage, replay_outage, survival, window_AD)
from arena.experiments.settlement.watch import (                                   # noqa: E402
    FixedActionOutageWatchPolicy as OB4Force, OutageWatchBandPolicy as OB4)

F0_MEAS = 0.01216147            # window B revert share, rounded (data/x402_amounts/revert_share.json)
F0_DECL = 0.005
MARGIN_BP = 10.0
FLOWS = D2.FLOWS
SEEDS = D2.SEEDS
N_TUNE, N_EVAL, PPE, DEN = D2.N_TUNE, D2.N_EVAL, D2.PPE, D2.DEN
NB, BSEED = 1_000, 20261001
ACTS = D2.ACTS
# replays: (world, policy)
RUNS = (("m", "comp_d"), ("m", "comp_m"), ("m", "b4m"), ("m", "b4d"), ("m", "cutoff_m"),
        ("d", "comp_d"), ("d", "comp_m"), ("d", "b4d"))
write_json = D2.write_json


# ------------------------------------------------------------------ worlds and tables
def worlds():
    wd, exd = HH.setup()
    f = np.array(wd.f, dtype=float)
    f[0] = F0_MEAS
    wm = replace(wd, f=f)
    exm = window_AD(wm, survival(wm))[2]
    return dict(d=(wd, exd), m=(wm, exm))


def tables(W, log):
    wd, _ = W["d"]
    wm, _ = W["m"]
    Td, vg = HH.load_table(wd, paths.CACHE, log)
    cache = paths.CACHE
    cache.mkdir(parents=True, exist_ok=True)
    p, hp = cache / "table_m.npy", cache / "table_m.json"
    if p.exists():
        return Td, np.load(p), vg, json.loads(hp.read_text())
    t0 = time.time()
    pol = compile_outage(RD.comp_env(wm, D2.Q_TRUE), "comp_m")
    assert np.array_equal(pol.v_grid, vg)
    Tm = FR.stack_tables(pol)
    info = dict(hash=RD.table_hash(pol), compile_s=time.time() - t0, f0=F0_MEAS,
                label_counts=dict(zip(ACTS, np.bincount(Tm.ravel().astype(int), minlength=4)[:4].tolist())),
                n_cells_changed_vs_d=int(np.sum(Tm != Td)))
    H = wm.H
    reg = []
    for j in range(len(vg)):
        row = Tm[j, 0, H, 0, :]
        w = np.nonzero(row == WAIT)[0]
        reg.append(dict(v=float(vg[j]), wait_bins=len(w),
                        wait_pi=[float(w.min() / 500.0), float(w.max() / 500.0)] if len(w) else None,
                        labels=sorted({ACTS[int(x)] for x in row})))
    info["running_arrival_wait_region"] = reg
    np.save(p, Tm)
    write_json(hp, info)
    log(f"TABLE_M hash={info['hash'][:16]} t={info['compile_s']:.1f}s labels={info['label_counts']} "
        f"changed={info['n_cells_changed_vs_d']}")
    return Td, Tm, vg, info


def load_b4d(flow):
    b = json.loads((paths.NO_CHANNEL / "plan" / f"b4main_{flow}.json").read_text())
    return dict(k=b["k"], a=b["a"], b=b["b"], tune_mean=b["tune_mean"])


def ref_d2(flow, run_dir):
    sub = paths.NO_CHANNEL / run_dir.name if run_dir != paths.SUBMISSION_HAZARD else paths.NO_CHANNEL
    if not (sub / "summary.json").exists():       # no D2 run with the same tag: fall back to the full D2 run
        sub = paths.NO_CHANNEL
    s = json.loads((sub / "summary.json").read_text())["flows"][flow]
    return dict(J_comp=s["J"]["comp"]["point"], J_b4main=s["J"]["b4main"]["point"])


# ------------------------------------------------------------------ replay with records
def replay_rec(env, d, fn, ex):
    FR.set_exercise(ex)
    n = len(d)
    granted = np.zeros(n, dtype=bool)

    def g(k, i, l, r):
        a = np.asarray(fn(k, i, l, r))
        granted[k[a == GRANT]] = True
        return a
    pay = FR.replay_vec(env, d, g)
    return pay, granted


def pol_fn(name, env, d, iv, Td, Tm, b4m, b4d):
    if name == "comp_d":
        return HH.comp_fn(Td, iv, d.pi0)
    if name == "comp_m":
        return HH.comp_fn(Tm, iv, d.pi0)
    if name == "cutoff_m":
        return HH.cutoff_fn(d.pi0, HH.pihat0(env)[0])
    b = b4m if name == "b4m" else b4d
    return HH.band_fn(env, d.pi0, b["k"], b["a"], b["b"])


# ------------------------------------------------------------------ one flow
def run_flow(flow, run_dir, n_eval, n_tune, W, Td, Tm, vg, log):
    path = run_dir / "npz" / f"flow_{flow}.npz"
    if path.exists():
        log(f"SKIP {flow}")
        return
    wm, exm = W["m"]
    wd, exd = W["d"]
    t0 = time.time()
    tune, d = D2.draw_main(wm, flow, n_eval, n_tune)
    t_draw = time.time() - t0
    pj = run_dir / "plan" / f"b4m_{flow}.json"
    if pj.exists():
        b4m = json.loads(pj.read_text())
    else:
        b4m = D2.tune_b4(wm, exm, tune)
        b4m.update(flow=flow, rng=SEEDS[flow], world="W_m", f0=F0_MEAS)
        write_json(pj, b4m)
    log(f"B4M {flow} ({b4m['k']},{b4m['a']!r},{b4m['b']!r}) tune_mean={b4m['tune_mean']!r} "
        f"force={b4m['t_force']:.1f}s tune={b4m['t_tune']:.1f}s")
    del tune
    b4d = load_b4d(flow)
    iv = FR.v_index(vg, d.v)
    v = np.asarray(d.v, dtype=float)
    pi0 = np.asarray(d.pi0, dtype=float)
    mis = np.asarray(d.theta) == 1
    r0 = d.paths[:, 0].astype(np.int64)
    bs = RD.block_sums
    out = dict(bs_v=bs(v), bs_n_mis=bs(mis.astype(float)), bs_v_mis=bs(np.where(mis, v, 0.0)))
    J, times = {}, {}
    for wk, p in RUNS:
        env, ex = W[wk]
        key = f"{wk}_{p}"
        t1 = time.time()
        pay, gr = replay_rec(env, d, pol_fn(p, env, d, iv, Td, Tm, b4m, b4d), ex)
        times[key] = time.time() - t1
        mg = gr & mis
        out[f"pay_{key}"] = bs(pay)
        out[f"misn_{key}"] = bs(mg.astype(float))
        out[f"misv_{key}"] = bs(np.where(mg, v, 0.0))
        out[f"gr_{key}"] = bs(gr.astype(float))
        J[key] = float(pay.mean())
        if p == "cutoff_m":
            assert np.array_equal(gr, pi0 < HH.pihat0(env)[0]), "Cutoff grant set"
        del pay, gr
    # first actions at arrival (the table and arrival state only; same in both worlds)
    first, wait_rec = {}, {}
    for nm, T in (("comp_m", Tm), ("comp_d", Td)):
        fa = HH.first_actions(T, iv, d, wd.H)
        first[nm] = {reg: dict(zip(ACTS, np.bincount(fa[r0 == ri], minlength=4)[:4].tolist()))
                     for reg, ri in (("running", 0), ("halted", 1))}
        rec = {}
        for reg, ri in (("running", 0), ("halted", 1)):
            sel = (r0 == ri) & (fa == WAIT)
            rec[reg] = dict(n=int(sel.sum()))
            if sel.any():
                rec[reg].update(pi0_range=[float(pi0[sel].min()), float(pi0[sel].max())],
                                v_range=[float(v[sel].min()), float(v[sel].max())],
                                misuse_share=float(mis[sel].mean()), v_sum=float(v[sel].sum()))
        wait_rec[nm] = rec
        if nm == "comp_m":
            fa_m = fa
        else:
            fa_d = fa
    ch = fa_m != fa_d
    pairs = {}
    if ch.any():
        a_m, a_d = fa_m[ch], fa_d[ch]
        for x in range(4):
            for y in range(4):
                c = int(np.sum((a_m == x) & (a_d == y)))
                if c:
                    pairs[f"m_{ACTS[x]}_d_{ACTS[y]}"] = c
    first_change = dict(n=int(ch.sum()), pairs=pairs)
    ref = ref_d2(flow, run_dir)
    meta = dict(flow=flow, eval_rng=SEEDS[flow], n_eval=n_eval, n_tune=n_tune, n_blocks=n_eval // PPE,
                f0_m=F0_MEAS, f0_d=F0_DECL, pihat0_m=HH.pihat0(wm)[0], pihat0_d=HH.pihat0(wd)[0],
                b4m=dict(k=b4m["k"], a=b4m["a"], b=b4m["b"], tune_mean=b4m["tune_mean"]), b4d=b4d,
                J=J, t_draw=t_draw, t_replay=times, first_actions=first, wait_payments=wait_rec,
                first_action_change_m_vs_d=first_change, misuse_rate=float(mis.mean()),
                realized_mean=float(v.mean()), ref_d2=ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    log(f"DONE {flow} draw={t_draw:.1f}s " + " ".join(f"{k}={t:.1f}s" for k, t in times.items())
        + " " + " ".join(f"J_{k}={x:.8g}" for k, x in J.items())
        + f" wait_m_running={wait_rec['comp_m']['running']['n']} flow_total={time.time() - t0:.1f}s")


# ------------------------------------------------------------------ checks
def run_check(W, Td, Tm, vg, log, npre=20_000):
    wm, exm = W["m"]
    wd, exd = W["d"]
    res = {}
    _, dm = D2.draw_main(wm, "F2", npre, n_tune=1000)
    _, dd = D2.draw_main(wd, "F2", npre, n_tune=1000)
    same = all(np.array_equal(getattr(dm, f), getattr(dd, f))
               for f in ("v", "p_true", "theta", "pi0", "u_stage", "t_ans", "paths"))
    res["draw_Wm_equals_Wd_prefix"] = bool(same)
    pol_m = compile_outage(RD.comp_env(wm, D2.Q_TRUE), "comp_m")
    res["table_m_equals_stack"] = bool(np.array_equal(FR.stack_tables(pol_m), Tm))
    pol_d = compile_outage(RD.comp_env(wd, D2.Q_TRUE), "comp")
    res["table_d_hash_main"] = RD.table_hash(pol_d) == HH.REF_HASH
    iv = FR.v_index(vg, dm.v)
    thr_m = HH.pihat0(wm)[0]
    b4 = dict(k=4, a=0.1125, b=0.8)
    cases = {}
    for wk, env, ex in (("m", wm, exm), ("d", wd, exd)):
        for nm, fn, rp in (("comp_m", HH.comp_fn(Tm, iv, dm.pi0), pol_m),
                           ("comp_d", HH.comp_fn(Td, iv, dm.pi0), pol_d),
                           ("cutoff_m", HH.cutoff_fn(dm.pi0, thr_m),
                            lambda i, l, r, v, pi: GRANT if pi < thr_m else REJECT),
                           ("b4_k4", HH.band_fn(env, dm.pi0, b4["k"], b4["a"], b4["b"]),
                            OB4(b4["k"], b4["a"], b4["b"], env.H, env.N))):
            a, gr = replay_rec(env, dm, fn, ex)
            gc = HH.GrantCounter(rp)
            bb = replay_outage(env, dm, gc, ex)
            cases[f"{wk}:{nm}"] = dict(max_abs=float(np.max(np.abs(a - bb))), grants_vec=int(gr.sum()),
                                       grants_repo=gc.n)
        for k in (0, 4):
            for act in (GRANT, REJECT, VERIFY):
                a = FR.replay_forced(env, dm, k, act, ex)
                bb = replay_outage(env, dm, OB4Force(k, act, env.H, env.N), ex)
                cases[f"{wk}:force_k{k}_{'GRV'[[GRANT, REJECT, VERIFY].index(act)]}"] = dict(
                    max_abs=float(np.max(np.abs(a - bb))))
    fa_vec = HH.first_actions(Tm, iv, dm, wm.H)
    fa_repo = np.array([pol_m(0, wm.H, int(dm.paths[j, 0]), float(dm.v[j]), float(dm.pi0[j]))
                        for j in range(len(dm))])
    cases["first_action_m"] = dict(max_abs=float(np.sum(fa_vec != fa_repo)))
    ok = same and res["table_m_equals_stack"] and res["table_d_hash_main"] and all(
        c["max_abs"] == 0 and c.get("grants_vec", 0) == c.get("grants_repo", 0) for c in cases.values())
    res.update(prefix=npre, cases=cases, ex_m_differs=bool(not np.array_equal(exm, exd)), pass_=bool(ok))
    log(f"CHECK draw_same={same} table_m={res['table_m_equals_stack']} hash_d={res['table_d_hash_main']} "
        f"cases={len(cases)} max={max(c['max_abs'] for c in cases.values()):.1e} {'PASS' if ok else 'FAIL'}")
    return res


# ------------------------------------------------------------------ analysis
def analyze(run_dir, nb=NB):
    res = dict(bootstrap=dict(resamples=nb, seed=BSEED, note="paired block bootstrap (50-payment episodes), "
                              "same resamples for every policy and both worlds of a flow, default_rng(seed) "
                              "restarted per flow, 95% percentile"),
               unit="bp of $49.50", f0_measured=F0_MEAS, f0_declared=F0_DECL, flows={})
    tbl = paths.CACHE / "table_m.json"
    if tbl.exists():
        res["table_m"] = json.loads(tbl.read_text())
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
        c = lambda s, k: s[..., ix[k]]
        bp = lambda x: x / DEN * 1e4

        def stat(f):
            return dict(point=float(f(P)), ci=D2.ci(f(S)))
        keys = [f"{w}_{q}" for w, q in RUNS]
        pairs = {
            "(1) W_m: comp_m - comp_d": ("m_comp_m", "m_comp_d"),
            "(2) W_d: comp_d - comp_m": ("d_comp_d", "d_comp_m"),
            "(3) W_m: comp_m - b4m": ("m_comp_m", "m_b4m"),
            "W_m: comp_d - b4m": ("m_comp_d", "m_b4m"),
            "W_m: comp_m - cutoff_m": ("m_comp_m", "m_cutoff_m"),
            "W_m: b4m - b4d": ("m_b4m", "m_b4d"),
            "W_d: comp_d - b4d": ("d_comp_d", "d_b4d"),
        }
        r = dict(meta=meta, J={q: stat(lambda s, q=q: c(s, f"pay_{q}") / n) for q in keys},
                 diff_bp={k: stat(lambda s, a=a, b=b: bp((c(s, f"pay_{a}") - c(s, f"pay_{b}")) / n))
                          for k, (a, b) in pairs.items()},
                 misuse_granted_pct={q: stat(lambda s, q=q: 100.0 * c(s, f"misn_{q}") / c(s, "bs_n_mis"))
                                     for q in keys},
                 misuse_exposure_usd_per_payment={q: float(P[ix[f"misv_{q}"]] / n) for q in keys},
                 grant_pct={q: float(100.0 * P[ix[f"gr_{q}"]] / n) for q in keys})
        tot = {}
        for nm, fa in meta["first_actions"].items():
            t = sum(sum(x.values()) for x in fa.values())
            sh = {reg: {a: 100.0 * fa[reg][a] / t for a in ACTS} for reg in fa}
            sh["all"] = {a: 100.0 * sum(fa[reg][a] for reg in fa) / t for a in ACTS}
            sh["within_regime"] = {reg: {a: 100.0 * fa[reg][a] / max(1, sum(fa[reg].values())) for a in ACTS}
                                   for reg in fa}
            tot[nm] = sh
        r["first_share_pct"] = tot
        ref = meta["ref_d2"]
        r["checks"] = dict(
            J_comp_d_on_W_d=r["J"]["d_comp_d"]["point"], J_comp_D2=ref["J_comp"],
            comp_d_on_W_d_equals_D2=r["J"]["d_comp_d"]["point"] == ref["J_comp"],
            J_b4d_on_W_d=r["J"]["d_b4d"]["point"], J_b4main_D2=ref["J_b4main"],
            b4d_on_W_d_equals_D2=r["J"]["d_b4d"]["point"] == ref["J_b4main"],
            note="J points from block sums / n, the same computation as the reference summary",
            own_table_ahead_W_m=meta["J"]["m_comp_m"] >= meta["J"]["m_comp_d"],
            own_table_ahead_W_d=meta["J"]["d_comp_d"] >= meta["J"]["d_comp_m"])
        res["flows"][flow] = r
    write_json(run_dir / "summary.json", res)
    write_md(run_dir, res)
    return res


def fmt(s):
    return f"{s['point']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]"


def write_md(run_dir, res):
    dat = json.loads((paths.X402_AMOUNTS / "revert_share.json").read_text())
    L = ["# Experiment F0: automatic summary", "",
         "Generated from summary.json. Unit: bp of the mean exposure $49.50; intervals are paired block bootstrap 95%.", ""]
    L.append("## Reading rule applied")
    outside = []
    for wn, w in dat["windows"].items():
        x = w["revert_share"] / F0_DECL
        inside = 0.5 <= x <= 2.0
        L.append(f"- Window {wn}: revert share {100 * w['revert_share']:.3f}% = {x:.2f} times the declared value, "
                 f"{'inside' if inside else 'outside'} the range 1/2 to 2 times.")
        if not inside:
            outside.append(wn)
    L.append(f"- Rule: {'(b) applies (window outside the range: ' + ', '.join(outside) + ')' if outside else '(a) applies'}. "
             f"Recompiled f0 = {F0_MEAS}.")
    L.append("")
    L.append("## Metrics by flow")
    L.append("| flow | (1) W_m: Comp_m - Comp_d | (2) W_d: Comp_d - Comp_m | (3) W_m: Comp_m - B4_m | W_m: Comp_d - B4_m |")
    L.append("|---|---|---|---|---|")
    over = []
    for fl, r in res["flows"].items():
        d = r["diff_bp"]
        L.append(f"| {fl} | {fmt(d['(1) W_m: comp_m - comp_d'])} | {fmt(d['(2) W_d: comp_d - comp_m'])} | "
                 f"{fmt(d['(3) W_m: comp_m - b4m'])} | {fmt(d['W_m: comp_d - b4m'])} |")
        for k in ("(1) W_m: comp_m - comp_d", "(2) W_d: comp_d - comp_m"):
            if d[k]["point"] > MARGIN_BP:
                over.append((fl, k, d[k]["point"]))
    L.append("")
    L.append("Misuse granted (%):")
    L.append("| flow | W_m Comp_d | W_m Comp_m | W_m B4_m | W_m Cutoff_m | W_d Comp_d | W_d Comp_m |")
    L.append("|---|---|---|---|---|---|---|")
    for fl, r in res["flows"].items():
        m = r["misuse_granted_pct"]
        L.append(f"| {fl} | " + " | ".join(f"{m[k]['point']:.2f}" for k in
                 ("m_comp_d", "m_comp_m", "m_b4m", "m_cutoff_m", "d_comp_d", "d_comp_m")) + " |")
    L.append("")
    L.append("First-action shares of Comp_m (%; of all payments, by arrival regime running/halted):")
    for fl, r in res["flows"].items():
        s = r["first_share_pct"]["comp_m"]
        wr = r["meta"]["wait_payments"]["comp_m"]
        L.append(f"- {fl}: running " + ", ".join(f"{a} {s['running'][a]:.3f}" for a in ACTS)
                 + "; halted " + ", ".join(f"{a} {s['halted'][a]:.3f}" for a in ACTS)
                 + f". wait in total {s['all']['wait']:.3f}%.")
        for reg in ("running", "halted"):
            w = wr[reg]
            if w["n"]:
                L.append(f"  - wait ({reg}): {w['n']} payments, pi0 in [{w['pi0_range'][0]:.4f}, {w['pi0_range'][1]:.4f}], "
                         f"v in [{w['v_range'][0]:.3f}, {w['v_range'][1]:.3f}]")
            else:
                L.append(f"  - wait ({reg}): 0 payments")
    L.append("")
    L.append("## Checks")
    for fl, r in res["flows"].items():
        c = r["checks"]
        L.append(f"- {fl}: J(Comp_d) on W_d {'equals' if c['comp_d_on_W_d_equals_D2'] else 'differs from'} the D2 J_comp "
                 f"(difference {c['J_comp_d_on_W_d'] - c['J_comp_D2']:.3g}); each world's own table is ahead: "
                 f"W_m {'yes' if c['own_table_ahead_W_m'] else 'no'}, W_d {'yes' if c['own_table_ahead_W_d'] else 'no'}.")
    L.append("")
    L.append("## Rule (c) sentence")
    if over:
        L.append("- The loss exceeds the 10 bp margin: " + "; ".join(f"{fl} {k} = {x:.2f} bp" for fl, k, x in over)
                 + ". State this as it is and narrow the scope of \"rail constants may be rough\".")
    else:
        mx = max(max(r["diff_bp"]["(1) W_m: comp_m - comp_d"]["point"], r["diff_bp"]["(2) W_d: comp_d - comp_m"]["point"])
                 for r in res["flows"].values())
        L.append(f"- In no flow and in neither direction does the misspecification loss exceed the 10 bp margin (maximum {mx:.2f} bp). "
                 "Report it with one summary table.")
    (run_dir / "summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "smoke", "run", "analyze"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--flows", default=None)
    ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase in ("smoke", "check") else None)
    run_dir = paths.SUBMISSION_HAZARD / tag if tag else paths.SUBMISSION_HAZARD
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
    W = worlds()
    Td, Tm, vg, info = tables(W, log)
    if a.phase == "check":
        chk = run_check(W, Td, Tm, vg, log)
        write_json(run_dir / "checks.json", dict(checks=chk, table_m=info))
        return 0 if chk["pass_"] else 1
    flows = a.flows.split(",") if a.flows else (["F2"] if a.phase == "smoke" else list(FLOWS))
    n_eval, n_tune = (100_000, 20_000) if a.phase == "smoke" else (N_EVAL, N_TUNE)
    fails = []
    for fl in flows:
        try:
            run_flow(fl, run_dir, n_eval, n_tune, W, Td, Tm, vg, log)
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
