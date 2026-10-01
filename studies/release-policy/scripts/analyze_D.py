"""Summarise experiment D .  Reads <run>/plan/, <run>/qmap.json, <run>/npz/.

    python3 scripts/analyze_D.py [--tag smoke]      (from the package root)

Per learning seed s and n (all in bp of the $49.50 mean exposure; per-payment mean over
the evaluation blocks, all blocks hold 50 payments):
    L    = J(pi_ref) - J(Comp_n)      pi_ref = compiled with the true q = 0.75
    D    = J(Comp_n) - J(B4_n)
    L_B4 = J(pi_ref) - J(B4_n)
Bootstrap (2,000 resamples, seed 20260929), same block resamples for every policy (paired):
    CI_block : blocks resampled, learning seeds fixed      (evaluation variation only)
    CI_2way  : blocks and learning seeds resampled jointly (learning + evaluation variation);
               the same seed resample is used for all four n (prefix dependence kept)
Primary test: H0 E[D_n] = 0 for n in {50, 130, 500, 5000}, two-sided 2-way bootstrap p,
Holm at 0.05.  n = 130 exact binomial: E[L] = sum_K P(K) L(K/130), P(L > 14.3) likewise,
over K with Binomial(130, 0.75) pmf > 1e-6.  Writes summary.json, summary.md, D_results.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from paths import D_ANSWERS

OUT = D_ANSWERS
DEN = 49.5
NB, BSEED = 2_000, 20260929
BUDGET, DTHR = 14.3, -10.0
NS = (50, 130, 500, 5000)
Q_TRUE, Q_FIXED = 0.75, (0.60, 0.75, 0.90)
BINOM_N = 130


def qkey(q):
    return repr(float(q))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def summ(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),
                max=float(v.max()), n=int(len(v)))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    run = OUT / a.tag if a.tag else OUT
    man = json.loads((run / "manifest.json").read_text())
    qmap = json.loads((run / "qmap.json").read_text())
    seeds = man["seeds"]
    plan = {s: json.loads((run / "plan" / f"seed_{s:02d}.json").read_text()) for s in seeds}

    # ---- load one column per evaluated policy
    names = [it["name"] for it in man["items"]]
    cols = {}
    for n in names:
        z = np.load(run / "npz" / f"{n}.npz")
        cols[n] = z["block_sums"].astype(float)
    nblk = len(next(iter(cols.values())))
    assert all(len(v) == nblk for v in cols.values())
    denom = nblk * 50.0
    comp_of = {k: f"comp_{h[:16]}" for k, h in qmap.items()}
    ref = comp_of[qkey(Q_TRUE)]
    col_names = list(cols)
    ci_idx = {n: i for i, n in enumerate(col_names)}
    M = np.column_stack([cols[n] for n in col_names])          # (nblk, P)
    point = M.sum(axis=0) / denom                               # per-payment $ mean

    def bp(dollars):
        return dollars / DEN * 1e4

    # ---- bootstrap: per-policy resampled means, (NB, P)
    rng = np.random.default_rng(BSEED)
    Pb = np.empty((NB, M.shape[1]))
    chunk = 20
    for j0 in range(0, NB, chunk):
        m = min(chunk, NB - j0)
        idx = rng.integers(0, nblk, (m, nblk))
        W = np.bincount((idx + (np.arange(m) * nblk)[:, None]).ravel(),
                        minlength=m * nblk).reshape(m, nblk).astype(float)
        Pb[j0:j0 + m] = W @ M / denom
    seed_rs = rng.integers(0, len(seeds), (NB, len(seeds)))    # 2-way: seed resamples

    # ---- per (s, n) index arrays
    S = len(seeds)
    comp_i = np.zeros((S, len(NS)), int)
    b4_i = np.zeros((S, len(NS)), int)
    qh = np.zeros((S, len(NS)))
    for a_, s in enumerate(seeds):
        for b_, n in enumerate(NS):
            v = plan[s]["per_n"][str(n)]
            qh[a_, b_] = v["q_hat"]
            comp_i[a_, b_] = ci_idx[comp_of[qkey(v["q_hat"])]]
            p = v["b4"]
            b4_i[a_, b_] = ci_idx[f"b4_k{int(p['k'])}_a{float(p['a'])!r}_b{float(p['b'])!r}"]
    r_i = ci_idx[ref]
    L = bp(point[r_i] - point[comp_i])                         # (S, 4)
    D = bp(point[comp_i] - point[b4_i])
    LB = bp(point[r_i] - point[b4_i])
    Lb = bp(Pb[:, [r_i]][:, :, None] - Pb[:, comp_i])          # (NB, S, 4)
    Db = bp(Pb[:, comp_i] - Pb[:, b4_i])
    LBb = bp(Pb[:, [r_i]][:, :, None] - Pb[:, b4_i])

    def two_way(Xb):
        # Xb (NB, S, 4) -> mean over resampled seeds, per b
        return np.take_along_axis(Xb, seed_rs[:, :, None].repeat(Xb.shape[2], 2), axis=1).mean(axis=1)

    res = dict(eval_seed=man["eval_seed"], n_eval=man["n_eval"], n_blocks=nblk, seeds=seeds,
               unit="bp of $49.50 (x 0.00495 = $ per payment)", bootstrap=dict(resamples=NB, seed=BSEED),
               budget_bp=BUDGET, d_threshold_bp=DTHR,
               ref=dict(name=ref, J=float(point[r_i])),
               fixed={}, per_n={}, per_seed=[])
    for q in Q_FIXED:
        i = ci_idx[comp_of[qkey(q)]]
        x = bp(Pb[:, r_i] - Pb[:, i])
        res["fixed"][f"{q:.2f}"] = dict(L=float(bp(point[r_i] - point[i])), L_ci_block=ci(x),
                                        J=float(point[i]), policy=comp_of[qkey(q)])
    pvals = []
    L2, D2, LB2 = two_way(Lb), two_way(Db), two_way(LBb)
    for b_, n in enumerate(NS):
        Dm_b = Db[:, :, b_].mean(axis=1)
        d2 = D2[:, b_]
        p2 = min(1.0, 2 * min((np.sum(d2 <= 0) + 1) / (NB + 1), (np.sum(d2 >= 0) + 1) / (NB + 1)))
        pvals.append(p2)
        res["per_n"][str(n)] = dict(
            n=n,
            q_hat=summ(qh[:, b_]),
            n_distinct_q=int(len(set(qh[:, b_]))), n_distinct_comp=int(len(set(comp_i[:, b_]))),
            n_distinct_b4=int(len(set(b4_i[:, b_]))),
            L=dict(summ(L[:, b_]), ci_block=ci(Lb[:, :, b_].mean(axis=1)), ci_2way=ci(L2[:, b_]),
                   n_above_budget=int(np.sum(L[:, b_] > BUDGET))),
            D=dict(summ(D[:, b_]), ci_block=ci(Dm_b), ci_2way=ci(d2),
                   n_below_thr=int(np.sum(D[:, b_] < DTHR)), p_2way=p2),
            L_B4=dict(summ(LB[:, b_]), ci_block=ci(LBb[:, :, b_].mean(axis=1)), ci_2way=ci(LB2[:, b_]),
                      n_above_budget=int(np.sum(LB[:, b_] > BUDGET))),
            b4_params=sorted({(plan[s]["per_n"][str(n)]["b4"]["k"], plan[s]["per_n"][str(n)]["b4"]["a"],
                               plan[s]["per_n"][str(n)]["b4"]["b"]) for s in seeds}))
    adj = holm(np.array(pvals))
    for b_, n in enumerate(NS):
        res["per_n"][str(n)]["D"]["p_holm"] = float(adj[b_])
    for a_, s in enumerate(seeds):
        for b_, n in enumerate(NS):
            v = plan[s]["per_n"][str(n)]
            res["per_seed"].append(dict(
                seed=s, n=n, K=v["K"], q_hat=v["q_hat"], b4=v["b4"],
                L=float(L[a_, b_]), L_ci_block=ci(Lb[:, a_, b_]),
                D=float(D[a_, b_]), D_ci_block=ci(Db[:, a_, b_]),
                L_B4=float(LB[a_, b_])))

    # ---- n = 130 exact binomial over K
    Ks = np.array(man["binom_ks"])
    pmf = binom.pmf(Ks, BINOM_N, Q_TRUE)
    kc = np.array([ci_idx[comp_of[qkey(k / BINOM_N)]] for k in Ks])
    LK = bp(point[r_i] - point[kc])
    LKb = bp(Pb[:, [r_i]] - Pb[:, kc])                         # (NB, |K|)
    EL_b = LKb @ pmf
    PE_b = (LKb > BUDGET).astype(float) @ pmf
    res["binom130"] = dict(
        K=Ks.tolist(), pmf=pmf.tolist(), L=LK.tolist(), mass_included=float(pmf.sum()),
        E_L=float(pmf @ LK), E_L_ci_block=ci(EL_b),
        P_L_gt_budget=float(pmf @ (LK > BUDGET)), P_L_gt_budget_ci_block=ci(PE_b),
        K_above_budget=Ks[LK > BUDGET].tolist(),
        seeds_empirical=dict(mean_L=float(L[:, NS.index(130)].mean()),
                             frac_above=float(np.mean(L[:, NS.index(130)] > BUDGET))))
    (run / "summary.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(run / "D_results.npz", L=L, D=D, L_B4=LB, q_hat=qh, seeds=np.array(seeds),
                        ns=np.array(NS), binom_K=Ks, binom_pmf=pmf, binom_L=LK)

    # ---- markdown
    f = lambda x: f"{x:+.2f}"
    md = [f"# Experiment D summary (generated by analyze_D.py{', ' + a.tag if a.tag else ''})", "",
          f"Evaluation seed {man['eval_seed']}, {man['n_eval']:,} payments ({nblk:,} blocks), {len(seeds)} learning seeds. "
          f"Unit: bp of the $49.50 mean exposure. Bootstrap {NB} resamples (seed {BSEED}). "
          f"π_ref J = ${point[r_i]:.6f}/payment.", "",
          "| n | L mean | L median | L min | L max | L CI_2way | #L>14.3 | D mean | D median | D min | D max | D CI_block | D CI_2way | p_holm | #D<-10 | L_B4 mean |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in NS:
        g = res["per_n"][str(n)]
        Lg, Dg = g["L"], g["D"]
        md.append("| " + " | ".join([
            str(n), f(Lg["mean"]), f(Lg["median"]), f(Lg["min"]), f(Lg["max"]),
            f"[{Lg['ci_2way'][0]:+.2f}, {Lg['ci_2way'][1]:+.2f}]", str(Lg["n_above_budget"]),
            f(Dg["mean"]), f(Dg["median"]), f(Dg["min"]), f(Dg["max"]),
            f"[{Dg['ci_block'][0]:+.2f}, {Dg['ci_block'][1]:+.2f}]",
            f"[{Dg['ci_2way'][0]:+.2f}, {Dg['ci_2way'][1]:+.2f}]", f"{Dg['p_holm']:.4f}",
            str(Dg["n_below_thr"]), f(g["L_B4"]["mean"])]) + " |")
    md += ["", "Fixed Comp (no learning): " + ", ".join(
        f"q={q}: L={v['L']:+.2f} [{v['L_ci_block'][0]:+.2f}, {v['L_ci_block'][1]:+.2f}]"
        for q, v in res["fixed"].items()), ""]
    bb = res["binom130"]
    md += [f"n=130 exact binomial (K={Ks.min()}..{Ks.max()}, included mass {bb['mass_included']:.7f}): "
           f"E[L]={bb['E_L']:+.2f} [{bb['E_L_ci_block'][0]:+.2f}, {bb['E_L_ci_block'][1]:+.2f}], "
           f"P(L>14.3)={bb['P_L_gt_budget']:.4f} [{bb['P_L_gt_budget_ci_block'][0]:.4f}, "
           f"{bb['P_L_gt_budget_ci_block'][1]:.4f}]; 30-seed empirical mean L={bb['seeds_empirical']['mean_L']:+.2f}, "
           f"share above budget {bb['seeds_empirical']['frac_above']:.3f}", "",
           "| K | q̂ | P(K) | L(K/130) |", "|---|---|---|---|"]
    md += [f"| {k} | {k / BINOM_N:.4f} | {p:.2e} | {l:+.2f} |" for k, p, l in zip(Ks, pmf, LK)]
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12 + len(NS)]))


if __name__ == "__main__":
    main()
