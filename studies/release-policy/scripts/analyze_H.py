"""Summarise experiment H (spec: docs/exposure_law.md sec 5, 6).  Reads results/exposure_law/[<tag>/]npz/law_*.npz.

    export PYTHONPATH=$PWD/src && python3 scripts/analyze_H.py [--tag smoke] [--nb 1000]

Per law: J of Comp, Cutoff, B4_L, B4_L0 ($ per payment = column sum of block sums / payments);
differences Comp - X in $ per payment and in bp of that law's realized mean exposure (the point
value of the mean on the evaluation draw, used as a fixed unit); misuse pass rate and misuse
exposure; (Comp - Cutoff) split into payments with v > v* and v <= v*, and by Comp's first action.
Bootstrap: paired block bootstrap (the same block resample for every policy and every statistic of a
law), 1,000 resamples, default_rng(20260930) restarted per law, 95% percentile intervals.
Reading rules of spec 6 are applied to the point estimates; intervals are reported beside them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

OUT = paths.EXPOSURE_LAW
NB, BSEED = 1_000, 20260930
MARGIN = 10.0
PPE = 50
POL = ("comp", "cutoff", "b4L", "b4L0")
PNAME = dict(comp="Comp", cutoff="Cutoff", b4L="B4_L", b4L0="B4_L0")
ACTS = ("grant", "reject", "verify", "wait")
ANAME = dict(grant="grant", reject="reject", verify="verify", wait="wait")
ORDER = ("L0", "L1", "L2", "L3", "L4", "L5")
SWEEP = ("L1", "L4", "L5", "L0")          # target means 0.20, 1, 5, 49.50


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyse_law(z, nb):
    meta = json.loads(str(z["meta"]))
    cols = [k for k in z.files if k != "meta"]
    M = np.column_stack([z[k].astype(float) for k in cols])
    ix = {k: i for i, k in enumerate(cols)}
    nblk = M.shape[0]
    n = nblk * PPE
    mv = meta["realized_mean"]
    rng = np.random.default_rng(BSEED)
    S = np.empty((nb, M.shape[1]))
    for j0 in range(0, nb, 20):
        m = min(20, nb - j0)
        idx = rng.integers(0, nblk, (m, nblk))
        W = np.bincount((idx + (np.arange(m) * nblk)[:, None]).ravel(),
                        minlength=m * nblk).reshape(m, nblk).astype(float)
        S[j0:j0 + m] = W @ M
    P = M.sum(axis=0)

    def stat(f):
        """f maps a vector of column sums to a scalar; returns point and CI."""
        pt = float(f(P))
        bs = np.array([f(S[b]) for b in range(nb)])
        return dict(point=pt, ci=ci(bs))

    c = lambda s, k: s[ix[k]]
    bp = lambda x: x / mv * 1e4
    r = dict(law=meta["law"], realized_mean=mv, realized_median=meta["realized_median"], n_eval=meta["n_eval"],
             n_blocks=nblk, eval_seed=meta["eval_seed"], n_at_lo=meta["n_at_lo"], n_at_hi=meta["n_at_hi"],
             dist=dict(count_above_floor=meta["share_count_above_floor"], value_above_floor=meta["share_value_above_floor"],
                       count_above_vstar=meta["share_count_above_vstar"], value_above_vstar=meta["share_value_above_vstar"],
                       count_below_grid=meta["share_count_below_grid"]),
             b4L=meta["b4L"], b4L0=meta["b4L0"], pihat0=meta["pihat0"], misuse_rate=meta["misuse_rate"])
    fc, fv = meta["comp_first_count"], meta["comp_first_value"]
    tc, tv = sum(fc.values()), sum(fv.values())
    r["comp_first"] = dict(count={a: fc[a] / tc for a in ACTS}, value={a: fv[a] / tv for a in ACTS},
                           wait_ge1_count=fc["wait"] / tc, wait_ge1_value=fv["wait"] / tv)
    r["J"] = {p: stat(lambda s, p=p: c(s, f"pay_{p}") / n) for p in POL}
    r["diff_usd"] = {p: stat(lambda s, p=p: (c(s, "pay_comp") - c(s, f"pay_{p}")) / n) for p in POL[1:]}
    r["diff_bp"] = {p: stat(lambda s, p=p: bp((c(s, "pay_comp") - c(s, f"pay_{p}")) / n)) for p in POL[1:]}
    r["misuse_pass_pct"] = {p: stat(lambda s, p=p: 100.0 * c(s, f"misn_{p}") / c(s, "bs_n_mis")) for p in POL}
    r["misuse_exposure_usd"] = {p: stat(lambda s, p=p: c(s, f"misv_{p}") / n) for p in POL}
    r["misuse_exposure_bp"] = {p: stat(lambda s, p=p: bp(c(s, f"misv_{p}") / n)) for p in POL}
    r["grant_pct"] = {p: float(100.0 * P[ix[f"gr_{p}"]] / n) for p in POL}
    hi = lambda s: (c(s, "hi_comp") - c(s, "hi_cutoff")) / n
    lo = lambda s: (c(s, "pay_comp") - c(s, "pay_cutoff") - c(s, "hi_comp") + c(s, "hi_cutoff")) / n
    tot = (P[ix["pay_comp"]] - P[ix["pay_cutoff"]]) / n
    r["decomp"] = dict(above_vstar_usd=stat(hi), below_vstar_usd=stat(lo),
                       above_vstar_bp=stat(lambda s: bp(hi(s))), below_vstar_bp=stat(lambda s: bp(lo(s))),
                       above_share_of_total=(float(hi(P) / tot) if tot != 0 else None))
    r["by_comp_action_usd"] = {a: stat(lambda s, a=a: c(s, f"dact_{a}") / n) for a in ACTS}
    r["by_comp_action_bp"] = {a: float(bp(P[ix[f"dact_{a}"]] / n)) for a in ACTS}
    return r


def fu(x, d=6):
    return f"{x:+.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nb", type=int, default=NB)
    a = ap.parse_args(argv)
    run = OUT / a.tag if a.tag else OUT
    laws = [L for L in ORDER if (run / "npz" / f"law_{L}.npz").exists()]
    res = dict(unit="$ per payment; bp of the law's realized mean exposure", margin_bp=MARGIN,
               bootstrap=dict(resamples=a.nb, seed=BSEED, kind="paired block (50 payments per block)"), laws={})
    for L in laws:
        res["laws"][L] = analyse_law(np.load(run / "npz" / f"law_{L}.npz"), a.nb)
    R = res["laws"]

    # ---------------- reading rules (spec 6), point estimates
    rr = {}
    if "L1" in R:
        d = R["L1"]["diff_bp"]["cutoff"]
        inside = abs(d["point"]) <= MARGIN
        ci_inside = -MARGIN <= d["ci"][0] and d["ci"][1] <= MARGIN
        src = max(ACTS, key=lambda x: abs(R["L1"]["by_comp_action_bp"][x]))
        if inside:
            txt = ("At realistic mean payment sizes the compiled policy equals a single cutoff at arrival. "
                   f"(At L1, Comp - Cutoff = {d['point']:+.2f} bp, 95% interval [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}], margin 10 bp)")
        else:
            txt = (f"At L1, Comp - Cutoff = {d['point']:+.2f} bp ({R['L1']['diff_usd']['cutoff']['point']:+.3e} $ per payment), "
                   f"outside the 10 bp margin. The largest part of the difference comes from payments where Comp's first action is {src} "
                   f"({R['L1']['by_comp_action_bp'][src]:+.2f} bp).")
        rr["ga"] = dict(diff_bp=d, inside_margin=inside, ci_inside_margin=ci_inside, largest_source=src,
                        by_action_bp=R["L1"]["by_comp_action_bp"], sentence=txt)
    nb_ = {}
    for L in ("L2", "L3"):
        if L in R:
            nb_[L] = R[L]["dist"]["value_above_vstar"]
    if "L3" in R:
        ok = nb_["L3"] >= 0.5
        rr["na"] = dict(value_share_above_vstar=nb_, L3_at_least_half=ok,
                        sentence=("Rare by count, but more than half of the value lies where the policy acts." if ok else None))
    elif nb_:
        rr["na"] = dict(value_share_above_vstar=nb_, L3_at_least_half=None, sentence=None)
    sw = [L for L in SWEEP if L in R]
    first = None
    for L in sw:
        if abs(R[L]["diff_bp"]["cutoff"]["point"]) > MARGIN:
            first = L
            break
    rr["da"] = dict(sweep=[dict(law=L, mean_target=R[L]["law"]["mean_target"], realized_mean=R[L]["realized_mean"],
                                diff_bp=R[L]["diff_bp"]["cutoff"]) for L in sw],
                    smallest_mean_over_margin=(None if first is None else dict(law=first, mean_target=R[first]["law"]["mean_target"])),
                    complete=all(L in R for L in SWEEP))
    res["reading_rules"] = rr
    (run / "summary.json").write_text(json.dumps(res, indent=1))

    # ---------------- markdown
    md = [f"# Experiment H summary (generated by analyze_H.py{', ' + a.tag if a.tag else ''})", ""]
    if a.tag:
        md += [f"This is a {a.tag} run. The numbers are not results (spec sec 9.5).", ""]
    any_l = R[laws[0]] if laws else None
    if any_l:
        md += [f"Evaluation seed {any_l['eval_seed']}, {any_l['n_eval']:,} payments per law ({any_l['n_blocks']:,} blocks). "
               f"The four policies replay the same paths on each law. Amounts are dollars per payment; bp is relative to the law's realized mean exposure. "
               f"Intervals are 95% percentile intervals of a paired block bootstrap ({a.nb} resamples, seed {BSEED}). The margin is 10 bp. "
               f"Cutoff threshold pihat(0) = {any_l['pihat0']:.6f}, floor $1.90, v* $3.13.", ""]
    md += ["## 1. Distribution", "",
           "| Law | Target mean | sigma | Median | Realized mean | Count clipped at lower bound | Count clipped at upper bound | Count above floor | Value above floor | Count above v* | Value above v* |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for L in laws:
        x = R[L]
        md.append(f"| {L} | {x['law']['mean_target']:g} | {x['law']['sigma']:.4f} | {x['law']['median']:.4g} | {x['realized_mean']:.4f} | "
                  f"{x['n_at_lo']:,} | {x['n_at_hi']:,} | {100 * x['dist']['count_above_floor']:.3f}% | {100 * x['dist']['value_above_floor']:.2f}% | "
                  f"{100 * x['dist']['count_above_vstar']:.3f}% | {100 * x['dist']['value_above_vstar']:.2f}% |")
    md += ["", "## 2. First action of Comp", "",
           "Shares by count and by value. The wait share is the share of payments that wait one tick or more.", "",
           "| Law | Basis | grant | reject | verify | wait |", "|---|---|---|---|---|---|"]
    for L in laws:
        f = R[L]["comp_first"]
        for k, nm in (("count", "count"), ("value", "value")):
            md.append(f"| {L} | {nm} | " + " | ".join(f"{100 * f[k][a]:.3f}%" for a in ACTS) + " |")
    md += ["", "## 3. Payoff", "", "Dollars per payment.", "",
           "| Law | J(Comp) | J(Cutoff) | J(B4_L) | J(B4_L0) | B4_L (k, a, b) | B4_L0 (k, a, b) |",
           "|---|---|---|---|---|---|---|"]
    for L in laws:
        x = R[L]
        md.append(f"| {L} | " + " | ".join(f"{x['J'][p]['point']:.6g}" for p in POL)
                  + f" | ({x['b4L']['k']}, {x['b4L']['a']:.4f}, {x['b4L']['b']:.4f}) | ({x['b4L0']['k']}, {x['b4L0']['a']:.4f}, {x['b4L0']['b']:.4f}) |")
    md += ["", "| Law | Comparison | $ per payment | 95% interval ($) | bp | 95% interval (bp) |", "|---|---|---|---|---|---|"]
    for L in laws:
        x = R[L]
        for p in POL[1:]:
            du, db = x["diff_usd"][p], x["diff_bp"][p]
            md.append(f"| {L} | Comp - {PNAME[p]} | {du['point']:+.4e} | [{du['ci'][0]:+.3e}, {du['ci'][1]:+.3e}] | "
                      f"{db['point']:+.2f} | [{db['ci'][0]:+.2f}, {db['ci'][1]:+.2f}] |")
    md += ["", "## 4. Misuse release", "", "The pass rate is the share of misuse payments that are granted; the exposure is the sum of v over granted misuse payments divided by the number of payments.", "",
           "| Law | Policy | Pass rate | 95% interval | Exposure ($ per payment) | 95% interval | Exposure (bp) | Share of all payments granted |",
           "|---|---|---|---|---|---|---|---|"]
    for L in laws:
        x = R[L]
        for p in POL:
            m_, e_, b_ = x["misuse_pass_pct"][p], x["misuse_exposure_usd"][p], x["misuse_exposure_bp"][p]
            md.append(f"| {L} | {PNAME[p]} | {m_['point']:.3f}% | [{m_['ci'][0]:.3f}, {m_['ci'][1]:.3f}] | {e_['point']:.4e} | "
                      f"[{e_['ci'][0]:.3e}, {e_['ci'][1]:.3e}] | {b_['point']:.1f} | {x['grant_pct'][p]:.3f}% |")
    md += ["", "## 5. Decomposition of Comp - Cutoff", "", "Dollar decomposition into payments above v* (v > 3.13) and at or below it, and decomposition by the first action of Comp.", "",
           "| Law | Above v* ($) | 95% interval | Above v* (bp) | At or below v* ($) | 95% interval | At or below v* (bp) | Share above v* |",
           "|---|---|---|---|---|---|---|---|"]
    for L in laws:
        dc = R[L]["decomp"]
        sh = dc["above_share_of_total"]
        md.append(f"| {L} | {dc['above_vstar_usd']['point']:+.4e} | [{dc['above_vstar_usd']['ci'][0]:+.3e}, {dc['above_vstar_usd']['ci'][1]:+.3e}] | "
                  f"{dc['above_vstar_bp']['point']:+.2f} | {dc['below_vstar_usd']['point']:+.4e} | "
                  f"[{dc['below_vstar_usd']['ci'][0]:+.3e}, {dc['below_vstar_usd']['ci'][1]:+.3e}] | {dc['below_vstar_bp']['point']:+.2f} | "
                  f"{'' if sh is None else f'{100 * sh:.1f}%'} |")
    md += ["", "| Law | " + " | ".join(f"Comp {a} (bp)" for a in ACTS) + " |", "|---|---|---|---|---|"]
    for L in laws:
        md.append(f"| {L} | " + " | ".join(f"{R[L]['by_comp_action_bp'][a]:+.2f}" for a in ACTS) + " |")
    md += ["", "## 6. Reading (spec sec 6, applied to the point estimate)", ""]
    if "ga" in rr:
        g = rr["ga"]
        md += [f"(a) {g['sentence']} The 95% interval {'lies inside' if g['ci_inside_margin'] else 'does not lie inside'} the margin.", ""]
    if "na" in rr:
        n_ = rr["na"]
        parts = [f"at {L} the value share of payments above v* is {100 * v:.2f}%" for L, v in n_["value_share_above_vstar"].items()]
        tail = (f" At L3 it is at least 50%, so the following sentence is used: {n_['sentence']}" if n_.get("L3_at_least_half")
                else " At L3 it is below 50%, so that sentence is not used." if n_.get("L3_at_least_half") is False else "")
        md += ["(b) " + ", ".join(parts) + "." + tail + " L3 is a tail sensitivity and is not used as the main basis of a conclusion.", ""]
    da = rr["da"]
    s = ", ".join(f"mean ${x['mean_target']:g} ({x['law']}) {x['diff_bp']['point']:+.2f} bp [{x['diff_bp']['ci'][0]:+.2f}, {x['diff_bp']['ci'][1]:+.2f}]"
                  for x in da["sweep"])
    if da["smallest_mean_over_margin"]:
        w = da["smallest_mean_over_margin"]
        concl = f"The smallest mean at which Comp - Cutoff exceeds the margin is ${w['mean_target']:g} ({w['law']})."
    else:
        concl = "At none of the measured means does Comp - Cutoff exceed the margin."
    md += [f"(c) In the mean sweep, Comp - Cutoff is {s}. {concl}"
           + ("" if da["complete"] else " Some laws of the sweep are missing, so the reading is provisional."), ""]
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    main()
