"""Experiment HF: the harm-weight curve (payoff against misuse released).
E-fitted world, flow F2, main-run constants and draws (D2's draw_main, rng seed 8).  Spec: docs/spec-HF.md.

Run from the package root:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_HF.py check      (vectorized == repo replay on a prefix; writes results/harm_frontier/smoke/)
    python3 scripts/run_HF.py smoke      (100,000 evaluation, 20,000 tuning payments)
    python3 scripts/run_HF.py run        (full, resumable per h)
    python3 scripts/run_HF.py analyze [--tag smoke]

Output goes to results/harm_frontier/ (paths.HARM_FRONTIER); the compiled-table caches go to results/cache/.

For each h in HS: Comp_h is compiled with env.h = h (main-run grid); B4_h is tuned on the tuning split
with the payoff at weight h (161-point grid, horizon_grid(H), the D2 tuner).  Both are replayed on the
same evaluation split and scored twice: at weight h and at weight 1.  The weight enters the payoff only
through a settled misuse grant (-v h); the draws, the exercise rule of the verify window and every other
term do not depend on it (checked: ex is identical across h, and pay_h - pay_1 is zero outside the
settled misuse grants).
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
import run_D2 as D2                     # noqa: E402  (also imports run_D, fastreplay_G, run_H)

RD, FR, HH = D2.RD, D2.FR, D2.HH

from arena.experiments.settlement.core import GRANT, VERIFY                           # noqa: E402
from arena.experiments.settlement.outage import (                                      # noqa: E402
    compile_outage, replay_outage, survival, window_AD)
from arena.experiments.settlement.watch import OutageWatchBandPolicy as OB4            # noqa: E402

HS = (0.25, 0.5, 1.0, 2.0, 4.0)
FLOW = "F2"
N_TUNE = D2.N_TUNE
N_EVAL = D2.N_EVAL
PPE = D2.PPE
DEN = D2.DEN
NB, BSEED = D2.NB, D2.BSEED
Q_TRUE = D2.Q_TRUE
REF_HASH = D2.REF_HASH
D2_F2 = dict(J_comp=8.933993969941707, J_b4=8.91293001429241, k=0, a=0.1125, b=0.8,
             tune_mean=8.872650462538848)       # results/no_channel/summary.json, plan/b4main_F2.json
POLS = ("comp", "b4")


def hkey(h):
    return f"h{h:g}"


def write_json(path, obj):
    D2.write_json(path, obj)


# ------------------------------------------------------------------ instrumented verify window
_REC = {}
_ORIG_VERIFY = FR._verify


def _verify_rec(env, d, out, kv, iv, tv, ex_pos, v_all, theta, f, m, cw, tau):
    """FR._verify with one addition: the indices executed after an answer are recorded.
    Same float operations in the same order (checked against the unpatched replay)."""
    N, H, FIN = env.N, env.H, env.N + 1
    i, t = iv.copy(), tv.copy()
    w = np.minimum(tau, np.maximum(H - t, 0))
    s = np.zeros(len(kv), dtype=np.int64)
    live = s < w
    execd = []
    while live.any():
        a = np.nonzero(live)[0]
        k = kv[a]
        out[k] -= cw * v_all[k]
        s[a] += 1
        ans = d.t_ans[k] == s[a]
        if ans.any():
            aa, ka = a[ans], k[ans]
            li = H - t[aa]
            lr = d.paths[ka, t[aa]].astype(np.int64)
            ok = (theta[ka] == 0) & (ex_pos[i[aa], li, lr] > 0)
            if ok.any():
                ko, ao = ka[ok], aa[ok]
                execd.append(ko)
                won = np.where(i[ao] == FIN, 1, FR._settle(env, d, ko, i[ao], t[ao]))
                out[ko] += v_all[ko] * np.where(won == 1, m, -1.0)
            live[aa] = False
        na = ~ans
        a, k = a[na], k[na]
        up = (d.paths[k, t[a]] == 0) & (i[a] <= N)
        fail = np.zeros(len(a), dtype=bool)
        fail[up] = d.u_stage[k[up], i[a][up]] < f[i[a][up]]
        live[a[fail]] = False
        adv = a[up & ~fail]
        i[adv] += 1
        i[adv[i[adv] > N]] = FIN
        a2 = a[~fail]
        t[a2] += 1
        exp = (H - t[a2]) <= 0
        live[a2[exp]] = False
        a3 = a2[~exp]
        live[a3[s[a3] >= w[a3]]] = False
    _REC["exec"] = np.concatenate(execd) if execd else np.zeros(0, dtype=np.int64)


def replay_instr(env, d, fn, ex):
    """Vectorized replay; returns payoffs, grant mask, verify mask, executed-after-answer mask."""
    n = len(d)
    granted = np.zeros(n, dtype=bool)
    asked = np.zeros(n, dtype=bool)

    def g(k, i, l, r):
        a = np.asarray(fn(k, i, l, r))
        granted[k[a == GRANT]] = True
        asked[k[a == VERIFY]] = True
        return a
    FR.set_exercise(ex)
    _REC["exec"] = np.zeros(0, dtype=np.int64)
    FR._verify = _verify_rec
    try:
        pay = FR.replay_vec(env, d, g)
    finally:
        FR._verify = _ORIG_VERIFY
    execd = np.zeros(n, dtype=bool)
    execd[_REC["exec"]] = True
    return pay, granted, asked, execd


# ------------------------------------------------------------------ tables and B4
def world_h(world, h):
    return replace(world, h=float(h))


def table_h(world, h, cache, log):
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"table_{hkey(h)}.npy"
    hp = cache / f"table_{hkey(h)}.json"
    vg = np.geomspace(0.5, 2000.0, 41)
    if p.exists() and hp.exists():
        return np.load(p), json.loads(hp.read_text())
    t0 = time.time()
    pol = compile_outage(RD.comp_env(world_h(world, h), Q_TRUE), f"comp_{hkey(h)}")
    info = dict(h=h, hash=RD.table_hash(pol), compile_s=time.time() - t0)
    assert np.array_equal(pol.v_grid, vg)
    if h == 1.0:
        assert info["hash"] == REF_HASH, f"h=1 table hash {info['hash']} != main-run {REF_HASH}"
    T = FR.stack_tables(pol)
    info["label_counts"] = dict(zip(D2.ACTS, np.bincount(T.ravel().astype(int), minlength=4)[:4].tolist()))
    np.save(p, T)
    write_json(hp, info)
    log(f"TABLE {hkey(h)} hash={info['hash'][:16]} t={info['compile_s']:.1f}s")
    return T, info


def b4_h(world, ex, tune, h, run_dir, log):
    pj = run_dir / "plan" / f"b4_{hkey(h)}.json"
    if pj.exists():
        b4 = json.loads(pj.read_text())
    else:
        b4 = D2.tune_b4(world_h(world, h), ex, tune)
        b4.update(h=h, flow=FLOW, rng=D2.SEEDS[FLOW], split="first draw of the main-run rng",
                  payoff=f"weight h={h:g} on settled misuse grants")
        write_json(pj, b4)
    log(f"B4 {hkey(h)} ({b4['k']},{b4['a']!r},{b4['b']!r}) tune_mean={b4['tune_mean']!r} "
        f"force={b4['t_force']:.1f}s tune={b4['t_tune']:.1f}s")
    return b4


# ------------------------------------------------------------------ one h
def run_h(h, run_dir, world, ex, tune, d, iv, log):
    path = run_dir / "npz" / f"{hkey(h)}.npz"
    if path.exists():
        log(f"SKIP {hkey(h)}")
        return
    t0 = time.time()
    T, tinfo = table_h(world, h, paths.CACHE, log)
    b4 = b4_h(world, ex, tune, h, run_dir, log)
    wh, w1 = world_h(world, h), world_h(world, 1.0)
    v = np.asarray(d.v, dtype=float)
    mis = np.asarray(d.theta) == 1
    bs = RD.block_sums
    fns = dict(comp=HH.comp_fn(T, iv, d.pi0), b4=HH.band_fn(wh, d.pi0, b4["k"], b4["a"], b4["b"]))
    out, J, times, dec = {}, {}, {}, {}
    for p in POLS:
        t1 = time.time()
        pay_h, gr, asked, execd = replay_instr(wh, d, fns[p], ex)
        if h == 1.0:
            pay_1 = pay_h
        else:
            pay_1, gr1, asked1, ex1 = replay_instr(w1, d, fns[p], ex)
            assert np.array_equal(gr, gr1) and np.array_equal(asked, asked1) and np.array_equal(execd, ex1)
        times[p] = time.time() - t1
        mg = gr & mis
        hon = ~mis
        # the weight enters only through settled misuse grants
        diff = pay_h - pay_1
        outside = np.abs(diff[~mg]).max() if (~mg).any() else 0.0
        smg = mg & (diff != 0.0)                     # settled misuse grants (a failed grant pays -v at any h)
        resid = np.abs(diff[smg] - (1.0 - h) * v[smg]).max() if smg.any() else 0.0
        dec[p] = dict(max_abs_diff_outside_misuse_grants=float(outside),
                      max_abs_resid_settled=float(resid), n_settled_misuse=int(smg.sum()),
                      n_misuse_granted=int(mg.sum()))
        assert outside == 0.0, f"{p} {hkey(h)}: payoff depends on h outside misuse grants"
        out[f"payh_{p}"] = bs(pay_h)
        out[f"pay1_{p}"] = bs(pay_1)
        out[f"misn_{p}"] = bs(mg.astype(float))
        out[f"misv_{p}"] = bs(np.where(mg, v, 0.0))
        out[f"gr_{p}"] = bs(gr.astype(float))
        out[f"hrel_{p}"] = bs((hon & (gr | execd)).astype(float))
        out[f"hgr_{p}"] = bs((hon & gr).astype(float))
        out[f"hex_{p}"] = bs((hon & execd).astype(float))
        out[f"ask_{p}"] = bs(asked.astype(float))
        J[p] = dict(h=float(pay_h.mean()), one=float(pay_1.mean()))
        del pay_h, pay_1
    out["bs_n_mis"] = bs(mis.astype(float))
    out["bs_n_hon"] = bs((~mis).astype(float))
    out["bs_v_mis"] = bs(np.where(mis, v, 0.0))
    meta = dict(h=h, flow=FLOW, eval_rng=D2.SEEDS[FLOW], n_eval=len(d), n_tune=len(tune), n_blocks=len(d) // PPE,
                table=tinfo, b4=dict(k=b4["k"], a=b4["a"], b=b4["b"], tune_mean=b4["tune_mean"]),
                J=J, t_replay=times, decomposition=dec)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **out)
    os.replace(tmp, path)
    log(f"DONE {hkey(h)} " + " ".join(f"{p}={t:.1f}s" for p, t in times.items())
        + " " + " ".join(f"J_{p}(h)={J[p]['h']:.8g} J_{p}(1)={J[p]['one']:.8g}" for p in POLS)
        + f" total={time.time() - t0:.1f}s")


# ------------------------------------------------------------------ checks
def run_check(world, ex, log, npre=20_000):
    """(1) ex does not depend on h; (2) instrumented vectorized replay == repo replay_outage for
    Comp_h and a B4 band at h = 0.25 and 4 on a prefix; (3) patched == unpatched replay."""
    res = {}
    for h in HS:
        wh = world_h(world, h)
        _, _, exh = window_AD(wh, survival(wh))
        res.setdefault("ex_equal", {})[hkey(h)] = bool(np.array_equal(exh, ex))
    _, d = D2.draw_main(world, FLOW, npre, n_tune=1000)
    vg = np.geomspace(0.5, 2000.0, 41)
    iv = FR.v_index(vg, d.v)
    for h in (0.25, 4.0):
        wh = world_h(world, h)
        pol = compile_outage(RD.comp_env(wh, Q_TRUE), f"comp_{hkey(h)}")
        T = FR.stack_tables(pol)
        for nm, fn, rp in (("comp", HH.comp_fn(T, iv, d.pi0), pol),
                           ("b4_0_0.1125_0.8", HH.band_fn(wh, d.pi0, 0, 0.1125, 0.8), OB4(0, 0.1125, 0.8, wh.H, wh.N)),
                           ("b4_4_0.2_0.8", HH.band_fn(wh, d.pi0, 4, 0.2, 0.8), OB4(4, 0.2, 0.8, wh.H, wh.N))):
            a, gr, asked, execd = replay_instr(wh, d, fn, ex)
            a0, gr0 = HH.replay(wh, d, fn, ex)
            gc = HH.GrantCounter(rp)
            b = replay_outage(wh, d, gc, ex)
            res[f"{nm}_{hkey(h)}"] = dict(max_abs_vs_repo=float(np.max(np.abs(a - b))),
                                          max_abs_vs_unpatched=float(np.max(np.abs(a - a0))),
                                          grants_vec=int(gr.sum()), grants_repo=gc.n,
                                          asked=int(asked.sum()), exec_after_answer=int(execd.sum()))
    ok = all(res["ex_equal"].values()) and all(
        r["max_abs_vs_repo"] == 0 and r["max_abs_vs_unpatched"] == 0 and r["grants_vec"] == r["grants_repo"]
        for k, r in res.items() if k != "ex_equal")
    log(f"CHECK prefix={npre} {'PASS' if ok else 'FAIL'} {json.dumps(res)}")
    return dict(prefix=npre, cases=res, pass_=ok)


# ------------------------------------------------------------------ analysis
def ci(x):
    return D2.ci(x)


def analyze(run_dir, nb=NB):
    metas, cols, mats = {}, [], []
    for h in HS:
        p = run_dir / "npz" / f"{hkey(h)}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        metas[h] = json.loads(str(z["meta"]))
        for k in z.files:
            if k == "meta":
                continue
            cols.append((h, k))
            mats.append(z[k].astype(float))
    if not metas:
        return None
    M = np.column_stack(mats)
    ix = {c: i for i, c in enumerate(cols)}
    nblk = M.shape[0]
    n = nblk * PPE
    S = D2.boot_sums(M, nb, BSEED)
    P = M.sum(axis=0)
    c = lambda s, h, k: s[..., ix[(h, k)]]
    bp = lambda x: x / DEN * 1e4

    def stat(f):
        return dict(point=float(f(P)), ci=ci(f(S)))

    hs = [h for h in HS if h in metas]
    res = dict(bootstrap=dict(resamples=nb, seed=BSEED, note="paired block bootstrap, one set of resamples "
                              "for every policy and weight (same draw), 95% percentile"),
               unit=dict(J="$ per payment", bp="bp of $49.50", exposure="$ per input payment",
                         rates="% of misuse payments (misuse grant), % of honest payments (honest released), "
                               "% of all payments (query)"),
               n_eval=n, hs=hs, by_h={})
    for h in hs:
        m = metas[h]
        r = dict(b4=m["b4"], table_hash=m["table"]["hash"], decomposition=m["decomposition"])
        r["a_diff_bp_h"] = stat(lambda s, h=h: bp((c(s, h, "payh_comp") - c(s, h, "payh_b4")) / n))
        r["J_h"] = {p: stat(lambda s, h=h, p=p: c(s, h, f"payh_{p}") / n) for p in POLS}
        r["b_J1"] = {p: stat(lambda s, h=h, p=p: c(s, h, f"pay1_{p}") / n) for p in POLS}
        if 1.0 in metas:
            r["b_J1_minus_Comp1_bp"] = {p: stat(lambda s, h=h, p=p: bp((c(s, h, f"pay1_{p}")
                                                                          - c(s, 1.0, "pay1_comp")) / n))
                                        for p in POLS}
        r["b_J1_comp_minus_b4_bp"] = stat(lambda s, h=h: bp((c(s, h, "pay1_comp") - c(s, h, "pay1_b4")) / n))
        r["c_misuse_grant_pct"] = {p: stat(lambda s, h=h, p=p: 100.0 * c(s, h, f"misn_{p}") / c(s, h, "bs_n_mis"))
                                   for p in POLS}
        r["c_misuse_exposure_usd"] = {p: stat(lambda s, h=h, p=p: c(s, h, f"misv_{p}") / n) for p in POLS}
        r["d_honest_released_pct"] = {p: stat(lambda s, h=h, p=p: 100.0 * c(s, h, f"hrel_{p}") / c(s, h, "bs_n_hon"))
                                      for p in POLS}
        r["d_honest_grant_pct"] = {p: float(100.0 * P[ix[(h, f"hgr_{p}")]] / P[ix[(h, "bs_n_hon")]]) for p in POLS}
        r["d_honest_exec_after_answer_pct"] = {p: float(100.0 * P[ix[(h, f"hex_{p}")]] / P[ix[(h, "bs_n_hon")]])
                                               for p in POLS}
        r["d_query_pct"] = {p: stat(lambda s, h=h, p=p: 100.0 * c(s, h, f"ask_{p}") / n) for p in POLS}
        r["grant_pct"] = {p: float(100.0 * P[ix[(h, f"gr_{p}")]] / n) for p in POLS}
        res["by_h"][hkey(h)] = r
    # curve table (point values)
    curve = []
    for h in hs:
        r = res["by_h"][hkey(h)]
        for p in POLS:
            curve.append(dict(policy=f"{'Comp' if p == 'comp' else 'B4'}_{h:g}", h=h,
                              exposure_usd=r["c_misuse_exposure_usd"][p]["point"], J1=r["b_J1"][p]["point"],
                              misuse_grant_pct=r["c_misuse_grant_pct"][p]["point"],
                              query_pct=r["d_query_pct"][p]["point"]))
    res["curve"] = curve
    # frontier comparison: Comp_h curve (piecewise linear in exposure) at each B4_h exposure
    xs = lambda s: np.stack([c(s, h, "misv_comp") / n for h in hs], axis=-1)
    ys = lambda s: np.stack([c(s, h, "pay1_comp") / n for h in hs], axis=-1)

    def interp_at(xc, yc, x0):
        o = np.argsort(xc)
        return np.interp(x0, xc[o], yc[o], left=np.nan, right=np.nan)

    front = {}
    for h in hs:
        xb = lambda s, h=h: c(s, h, "misv_b4") / n
        yb = lambda s, h=h: c(s, h, "pay1_b4") / n
        pt = bp(interp_at(xs(P), ys(P), xb(P)) - yb(P))
        XS, YS, XB, YB = xs(S), ys(S), xb(S), yb(S)
        bsd = np.array([bp(interp_at(XS[j], YS[j], XB[j]) - YB[j]) for j in range(S.shape[0])])
        ok = np.isfinite(bsd)
        front[hkey(h)] = dict(b4_exposure=float(xb(P)), comp_curve_J1_minus_b4_J1_bp=float(pt),
                              ci=ci(bsd[ok]) if ok.sum() > 0.9 * len(bsd) else None,
                              resamples_in_range=int(ok.sum()),
                              in_range=bool(np.isfinite(pt)))
    res["frontier_at_b4_exposure"] = front
    # monotonicity (point values)
    exp_c = [res["by_h"][hkey(h)]["c_misuse_exposure_usd"]["comp"]["point"] for h in hs]
    j1_c = [res["by_h"][hkey(h)]["b_J1"]["comp"]["point"] for h in hs]
    exp_b = [res["by_h"][hkey(h)]["c_misuse_exposure_usd"]["b4"]["point"] for h in hs]
    j1_b = [res["by_h"][hkey(h)]["b_J1"]["b4"]["point"] for h in hs]
    dec = lambda a: all(a[i + 1] < a[i] for i in range(len(a) - 1))
    i1 = hs.index(1.0) if 1.0 in hs else None
    away = (lambda a: i1 is not None and all(a[i] < a[i + 1] for i in range(i1)) and
            all(a[i + 1] < a[i] for i in range(i1, len(a) - 1)))
    res["monotone"] = dict(comp_exposure_decreasing=dec(exp_c), comp_J1_decreasing=dec(j1_c),
                           comp_J1_peak_at_1_and_falling_away=away(j1_c),
                           b4_exposure_decreasing=dec(exp_b), b4_J1_decreasing=dec(j1_b),
                           b4_J1_peak_at_1_and_falling_away=away(j1_b))
    # checks against D2
    if 1.0 in metas:
        r1 = res["by_h"]["h1"]
        b = r1["b4"]
        res["check_vs_D2"] = dict(
            J_comp_h1=r1["b_J1"]["comp"]["point"], J_comp_D2=D2_F2["J_comp"],
            J_comp_equal=r1["b_J1"]["comp"]["point"] == D2_F2["J_comp"],
            J_b4_h1=r1["b_J1"]["b4"]["point"], J_b4_D2=D2_F2["J_b4"],
            J_b4_equal=r1["b_J1"]["b4"]["point"] == D2_F2["J_b4"],
            b4_h1=[b["k"], b["a"], b["b"], b["tune_mean"]],
            b4_D2=[D2_F2["k"], D2_F2["a"], D2_F2["b"], D2_F2["tune_mean"]],
            b4_equal=(b["k"], b["a"], b["b"], b["tune_mean"]) == (D2_F2["k"], D2_F2["a"], D2_F2["b"], D2_F2["tune_mean"]),
            table_hash_equal=r1["table_hash"] == REF_HASH)
    write_json(run_dir / "summary.json", res)
    write_md(run_dir, res)
    return res


def f2(x):
    return f"{x:.2f}"


def write_md(run_dir, res):
    hs = res["hs"]
    L = ["# Experiment HF: automatic summary", "",
         f"{res['n_eval']:,} evaluation payments (E-fitted, F2, main-run evaluation draw), paired block bootstrap "
         f"with {res['bootstrap']['resamples']} resamples, 95% intervals. bp is of $49.50. Exposure is in dollars "
         "per input payment.", "",
         "## Metrics by h", "",
         "| h | B4_h (k, a, b) | (a) J_h(Comp_h) - J_h(B4_h), bp | (b) J_1(Comp_h) | J_1(Comp_h) - J_1(Comp_1), bp | J_1(B4_h) |",
         "|---|---|---|---|---|---|"]
    for h in hs:
        r = res["by_h"][hkey(h)]
        b = r["b4"]
        a = r["a_diff_bp_h"]
        dd = r.get("b_J1_minus_Comp1_bp", {}).get("comp")
        L.append(f"| {h:g} | ({b['k']}, {b['a']:.4g}, {b['b']:.4g}) | {a['point']:.2f} [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}] | "
                 f"{r['b_J1']['comp']['point']:.6f} | "
                 + (f"{dd['point']:.2f} [{dd['ci'][0]:.2f}, {dd['ci'][1]:.2f}]" if dd else "-")
                 + f" | {r['b_J1']['b4']['point']:.6f} |")
    L += ["", "| h | policy | (c) misuse granted, % | (c) misuse exposure, $ per payment | "
          "(d) honest released, % (grant + executed after an answer) | (d) queries, % |",
          "|---|---|---|---|---|---|"]
    for h in hs:
        r = res["by_h"][hkey(h)]
        for p in POLS:
            mg, me = r["c_misuse_grant_pct"][p], r["c_misuse_exposure_usd"][p]
            hr, q = r["d_honest_released_pct"][p], r["d_query_pct"][p]
            L.append(f"| {h:g} | {'Comp' if p == 'comp' else 'B4'}_{h:g} | {mg['point']:.2f} [{mg['ci'][0]:.2f}, {mg['ci'][1]:.2f}] | "
                     f"{me['point']:.4f} [{me['ci'][0]:.4f}, {me['ci'][1]:.4f}] | {hr['point']:.2f} "
                     f"(grant {r['d_honest_grant_pct'][p]:.2f}, after answer {r['d_honest_exec_after_answer_pct'][p]:.2f}) | "
                     f"{q['point']:.2f} |")
    L += ["", "## The ten curve points (misuse exposure in $ per payment, J_1)", "",
          "| point | misuse exposure, $ per payment | J_1 | misuse granted, % | queries, % |", "|---|---|---|---|---|"]
    for pt in res["curve"]:
        L.append(f"| {pt['policy']} | {pt['exposure_usd']:.4f} | {pt['J1']:.6f} | {pt['misuse_grant_pct']:.2f} | {pt['query_pct']:.2f} |")
    L += ["", "## J_1 gap between the Comp_h curve and each B4_h point (same misuse exposure, curve interpolated linearly)", "",
          "| h | B4_h exposure, $ per payment | Comp curve J_1 - B4_h J_1 (bp) |", "|---|---|---|"]
    for h in hs:
        f = res["frontier_at_b4_exposure"][hkey(h)]
        if f["in_range"]:
            s = f"{f['comp_curve_J1_minus_b4_J1_bp']:.2f}" + (f" [{f['ci'][0]:.2f}, {f['ci'][1]:.2f}]" if f["ci"] else "")
        else:
            s = "outside the curve range"
        L.append(f"| {h:g} | {f['b4_exposure']:.4f} | {s} |")
    mo = res["monotone"]
    L += ["", "## Reading rules (spec section 3)", ""]
    if mo["comp_exposure_decreasing"] and mo["comp_J1_decreasing"]:
        L.append("- Monotonicity: misuse exposure and J_1 both decrease with h: h alone sets the amount of misuse "
                 "released, and that amount is priced in a loss of J_1.")
    else:
        L.append(f"- Monotonicity: the misuse exposure of Comp_h {'decreases' if mo['comp_exposure_decreasing'] else 'does not decrease'} with h. "
                 f"J_1(Comp_h) {'decreases' if mo['comp_J1_decreasing'] else 'does not decrease'} with h"
                 + (" (it is largest at h = 1 and falls on both sides)." if mo["comp_J1_peak_at_1_and_falling_away"] else ".")
                 + " The two conditions of the rule do not hold together, so the rule sentence is not used and this fact is reported instead.")
    L.append(f"- B4_h: misuse exposure {'decreasing' if mo['b4_exposure_decreasing'] else 'not monotone'}, "
             f"J_1 {'decreasing' if mo['b4_J1_decreasing'] else ('largest at h = 1, falling on both sides' if mo['b4_J1_peak_at_1_and_falling_away'] else 'not monotone')}.")
    fr = [(h, res["frontier_at_b4_exposure"][hkey(h)]) for h in hs]
    ins = [(h, f) for h, f in fr if f["in_range"]]
    pos = [h for h, f in ins if f["comp_curve_J1_minus_b4_J1_bp"] > 0]
    neg = [h for h, f in ins if f["comp_curve_J1_minus_b4_J1_bp"] <= 0]
    outr = [h for h, f in fr if not f["in_range"]]
    if ins and not neg:
        L.append("- Position: at every B4_h point inside the curve range, the Comp_h curve has the higher J_1 at the same misuse exposure. The Comp_h curve lies inside (on the favorable side of) the B4_h points.")
    elif ins and not pos:
        L.append("- Position: at every B4_h point inside the curve range, the Comp_h curve has a lower or equal J_1 at the same misuse exposure. The Comp_h curve lies outside (on the unfavorable side of) the B4_h points.")
    else:
        L.append(f"- Position: the Comp_h curve has a higher J_1 than the B4_h point for h = {', '.join(f'{x:g}' for x in pos)} "
                 f"and a lower or equal J_1 for h = {', '.join(f'{x:g}' for x in neg)}. The two cross between these.")
    if outr:
        L.append(f"- B4_h points outside the curve range: h = {', '.join(f'{x:g}' for x in outr)} (no interpolation; compare by point values in the tables).")
    if "check_vs_D2" in res:
        ck = res["check_vs_D2"]
        L += ["", "## Checks (h = 1 against experiment D2)", "",
              f"- table hash equal: {ck['table_hash_equal']}",
              f"- J_1(Comp_1) = {ck['J_comp_h1']!r}, D2 {ck['J_comp_D2']!r}, equal: {ck['J_comp_equal']}",
              f"- J_1(B4_1) = {ck['J_b4_h1']!r}, D2 {ck['J_b4_D2']!r}, equal: {ck['J_b4_equal']}",
              f"- B4_1 (k, a, b, tuning mean) = {ck['b4_h1']}, D2 {ck['b4_D2']}, equal: {ck['b4_equal']}"]
    (run_dir / "summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "smoke", "run", "analyze"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase in ("smoke", "check") else None)
    run_dir = paths.HARM_FRONTIER / tag if tag else paths.HARM_FRONTIER
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
    assert world.h == 1.0
    if a.phase == "check":
        chk = run_check(world, ex, log)
        write_json(run_dir / "checks.json", chk)
        return 0 if chk["pass_"] else 1
    n_eval, n_tune = (100_000, 20_000) if a.phase == "smoke" else (N_EVAL, N_TUNE)
    t0 = time.time()
    tune, d = D2.draw_main(world, FLOW, n_eval, n_tune)
    iv = FR.v_index(np.geomspace(0.5, 2000.0, 41), d.v)
    log(f"DRAW n_eval={len(d)} n_tune={len(tune)} t={time.time() - t0:.1f}s")
    fails = []
    for h in HS:
        try:
            run_h(h, run_dir, world, ex, tune, d, iv, log)
        except Exception:
            fails.append(h)
            log(f"FAIL {hkey(h)}")
            with (run_dir / "errors.log").open("a") as f:
                f.write(f"FAIL {hkey(h)}\n{traceback.format_exc()}\n")
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
