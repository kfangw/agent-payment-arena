"""Summarise experiment E1 (design notes: docs/regularized_b4.md).
Reads <run>/manifest.json, <run>/plan/, <run>/npz/ with <run> = results/regularized_b4[/<tag>].

    PYTHONPATH=src python3 scripts/analyze_E1.py [--tag smoke]
    (also run by `run_E1.py analyze` and at the end of `run_E1.py all`; needs numpy and scipy,
     the latter because scripts/analyze_D.py provides the shared bootstrap helpers)

Per learning seed s, n in {130, 500} and tuner T (bp of the $49.50 mean exposure):
    L_T     = J(pi_ref) - J(T_n)        L_Comp = J(pi_ref) - J(Comp_n)
    Delta_T = J(Comp_n) - J(T_n)        (positive: Comp better; D's sign convention)
    overfit = J_tune(T_n) - J(T_n)      in-sample tuning mean minus evaluation mean
Bootstrap exactly as analyze_D.py (2,000 resamples, seed 20260929, same block resample for
every policy, CI_block and CI_2way with the seed resample drawn after the block resamples),
so with 30 seeds and the same evaluation set the resamples coincide with D's.
Reading rule (fixed): the stated claim changes only if a PRIMARY tuner (R1, R2 c = 1,
R3) has mean Delta <= 10 bp at n = 130 (point estimate over the 30 seeds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import REGULARIZED_B4  # noqa: E402
import analyze_D as AD  # noqa: E402
from run_E1 import (MARGIN_BP, NS_E1, PRIMARY, TUNERS, d_summary_path,  # noqa: E402
                    b4n, comp_name, EVAL_SEED, N_EVAL)

LABELS = {"Comp_n": "Comp_n (q-hat = K/n)", "R0_b4": "R0 B4_n (D, unregularized)",
          "R1_coarse": "R1 coarse grid", "R2_shrink": "R2 shrinkage c=1",
          "R3_bag": "R3 bagging B=20", "R2_shrink_c0.5": "R2 shrinkage c=0.5 (sensitivity)",
          "R2_shrink_c2": "R2 shrinkage c=2 (sensitivity)", "Rc_centre": "centre policy (reference, no tuning)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run_dir) if a.run_dir else (REGULARIZED_B4 / a.tag if a.tag else REGULARIZED_B4)
    NB, BSEED, DEN, ci, summ = AD.NB, AD.BSEED, AD.DEN, AD.ci, AD.summ

    man = json.loads((run / "manifest.json").read_text())
    seeds = man["seeds"]
    plan = {s: json.loads((run / "plan" / f"seed_{s:02d}.json").read_text()) for s in seeds}
    names = [man["ref"]] + sorted(set(man["comp"].values()) - {man["ref"]}) + sorted(man["triples"])
    cols, meta = {}, {}
    for n in names:
        z = np.load(run / "npz" / f"{n}.npz")
        cols[n] = z["block_sums"].astype(float)
        meta[n] = json.loads(str(z["meta"]))
    nblk = len(cols[names[0]])
    assert all(len(v) == nblk for v in cols.values())
    denom = nblk * 50.0
    ci_idx = {n: i for i, n in enumerate(names)}
    M = np.column_stack([cols[n] for n in names])
    point = M.sum(axis=0) / denom

    def bp(x):
        return x / DEN * 1e4

    # ---- bootstrap, same generator order as analyze_D.py
    rng = np.random.default_rng(BSEED)
    Pb = np.empty((NB, M.shape[1]))
    chunk = 20
    for j0 in range(0, NB, chunk):
        m = min(chunk, NB - j0)
        idx = rng.integers(0, nblk, (m, nblk))
        W = np.bincount((idx + (np.arange(m) * nblk)[:, None]).ravel(),
                        minlength=m * nblk).reshape(m, nblk).astype(float)
        Pb[j0:j0 + m] = W @ M / denom
    seed_rs = rng.integers(0, len(seeds), (NB, len(seeds)))

    def two_way(Xb):             # (NB, S) -> (NB,)
        return np.take_along_axis(Xb, seed_rs, axis=1).mean(axis=1)

    r_i = ci_idx[man["ref"]]
    S = len(seeds)
    res = dict(label=man["label"], eval_seed=man["eval_seed"], learn_root=man["learn_root"],
               n_eval=man["n_eval"], n_blocks=nblk, seeds=seeds, constants=man["constants"],
               unit="bp of $49.50 (x 0.00495 = $ per payment)", bootstrap=dict(resamples=NB, seed=BSEED),
               ref=dict(name=man["ref"], J=float(point[r_i]), stats=meta[man["ref"]]["stats"]),
               per_n={}, per_seed=[], checks=dict(select=man["select_checks"],
                                                   ref_release_gap=man["ref_release_gap"],
                                                   d_hash_equal=man["d_hash_equal"],
                                                   comp_notes=man["comp_notes"]))
    arrays = {}
    for n in NS_E1:
        comp_i = np.array([ci_idx[comp_name(plan[s]["per_n"][str(n)]["q_hat"])] for s in seeds])
        Lc = bp(point[r_i] - point[comp_i])
        Lcb = bp(Pb[:, [r_i]] - Pb[:, comp_i])
        g = dict(n=n, Comp_n=dict(L=dict(summ(Lc), ci_block=ci(Lcb.mean(axis=1)), ci_2way=ci(two_way(Lcb))),
                                  misuse=_mis([meta[names[i]]["stats"] for i in comp_i])),
                 tuners={})
        arrays[f"L_comp_{n}"] = Lc
        for t in TUNERS:
            P = [plan[s]["per_n"][str(n)]["tuners"][t] for s in seeds]
            t_i = np.array([ci_idx[b4n(p["k"], p["a"], p["b"])] for p in P])
            L = bp(point[r_i] - point[t_i])
            Dl = bp(point[comp_i] - point[t_i])
            Lb = bp(Pb[:, [r_i]] - Pb[:, t_i])
            Db = bp(Pb[:, comp_i] - Pb[:, t_i])
            jt = np.array([p["tune_mean"] for p in P])
            over = bp(jt - point[t_i])
            ltune = bp(point[r_i] - jt)
            ks = [int(p["k"]) for p in P]
            g["tuners"][t] = dict(
                L=dict(summ(L), ci_block=ci(Lb.mean(axis=1)), ci_2way=ci(two_way(Lb))),
                Delta=dict(summ(Dl), ci_block=ci(Db.mean(axis=1)), ci_2way=ci(two_way(Db)),
                           n_within_margin=int(np.sum(Dl <= MARGIN_BP))),
                overfit=summ(over), L_tune=summ(ltune),
                misuse=_mis([meta[names[i]]["stats"] for i in t_i]),
                n_distinct_params=int(len(set(t_i))),
                k_counts={str(k): ks.count(k) for k in sorted(set(ks))},
                a=summ([p["a"] for p in P]), b=summ([p["b"] for p in P]),
                primary=t in PRIMARY)
            arrays[f"L_{t}_{n}"], arrays[f"Delta_{t}_{n}"], arrays[f"overfit_{t}_{n}"] = L, Dl, over
            for j, s in enumerate(seeds):
                res["per_seed"].append(dict(seed=s, n=n, tuner=t, k=P[j]["k"], a=P[j]["a"], b=P[j]["b"],
                                            L=float(L[j]), Delta=float(Dl[j]), L_comp=float(Lc[j]),
                                            Delta_ci_block=ci(Db[:, j]), J_tune=float(jt[j]),
                                            J_eval=float(point[t_i[j]]), overfit=float(over[j])))
        res["per_n"][str(n)] = g

    # ---- reading rule (n = 130, primary tuners, 30-seed mean of Delta)
    g130 = res["per_n"]["130"]["tuners"]
    hits = [t for t in PRIMARY if g130[t]["Delta"]["mean"] <= MARGIN_BP]
    res["reading_rule"] = dict(
        rule=f"claim changes iff some primary tuner has mean Delta <= {MARGIN_BP} bp at n = 130",
        mean_Delta_130={t: g130[t]["Delta"]["mean"] for t in PRIMARY},
        hits=hits, changes_claim=bool(hits),
        sensitivity_hits=[t for t in TUNERS if t not in PRIMARY and t != "R0_b4"
                          and g130[t]["Delta"]["mean"] <= MARGIN_BP])

    # ---- reproduction of D (only on D's evaluation set and learning root)
    dsum = d_summary_path()
    if man["eval_seed"] == EVAL_SEED and man["learn_root"] == EVAL_SEED and man["n_eval"] == N_EVAL \
            and dsum.exists() and len(seeds) == 30:
        ds = json.loads(dsum.read_text())
        dps = {(r["seed"], r["n"]): r for r in ds["per_seed"]}
        gl, gb4 = [], []
        for r in res["per_seed"]:
            w = dps.get((r["seed"], r["n"]))
            if w is None:
                continue
            gl.append(abs(r["L_comp"] - w["L"]))
            if r["tuner"] == "R0_b4":
                gb4.append(abs(r["L"] - w["L_B4"]))
        res["checks"]["vs_D"] = dict(ref_J_diff=abs(float(point[r_i]) - ds["ref"]["J"]),
                                     max_abs_L_comp_diff_bp=max(gl), max_abs_L_R0_vs_D_L_B4_bp=max(gb4),
                                     pass_=bool(abs(float(point[r_i]) - ds["ref"]["J"]) <= 1e-9
                                                and max(gl) <= 1e-6 and max(gb4) <= 1e-6))
    (run / "summary.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(run / "E1_results.npz", seeds=np.array(seeds), **arrays)
    md = markdown(res, man)
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def _mis(stats):
    """Seed-mean of the released-misuse statistics (None when the flag is missing)."""
    keys = ("misuse_grant_rate", "misuse_exposure", "release_rate", "legit_refusal_rate")
    if any(s.get("release", 1) is None for s in stats):
        return None
    return {k: float(np.mean([s[k] for s in stats])) for k in keys}


def markdown(res, man):
    f = lambda x: f"{x:+.2f}"
    iv = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    md = [f"# Experiment E1 summary (generated by analyze_E1.py)", "",
          f"Status: {res['label']}", "",
          f"Evaluation seed {res['eval_seed']}, {res['n_eval']:,} payments ({res['n_blocks']:,} blocks), "
          f"learning root {res['learn_root']}, {len(res['seeds'])} learning seeds. Unit: bp of the mean "
          f"exposure $49.50. Bootstrap: {res['bootstrap']['resamples']} resamples (seed {res['bootstrap']['seed']}). "
          f"pi_ref J = ${res['ref']['J']:.6f} per payment.", ""]
    rr = res["reading_rule"]
    md += [f"**Reading rule (fixed in advance)**: {rr['rule']}. Mean Delta at n=130: "
           + ", ".join(f"{LABELS[t]} {v:+.2f}" for t, v in rr["mean_Delta_130"].items())
           + f" -> {'claim changes (' + ', '.join(rr['hits']) + ')' if rr['changes_claim'] else 'claim stands; report as a robustness check'}"
           + (f". Sensitivity rows inside the margin: {', '.join(rr['sensitivity_hits'])} (not used for the verdict)" if rr["sensitivity_hits"] else ""),
           ""]
    for n in NS_E1:
        g = res["per_n"][str(n)]
        md += [f"## n = {n}", "",
               "| policy | L mean | L median | L min | L max | L CI_2way | Delta mean | Delta CI_block | Delta CI_2way | #Delta<=10 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        c = g["Comp_n"]["L"]
        md.append(f"| {LABELS['Comp_n']} | {f(c['mean'])} | {f(c['median'])} | {f(c['min'])} | {f(c['max'])} "
                  f"| {iv(c['ci_2way'])} | 0 | – | – | – |")
        for t in TUNERS:
            x = g["tuners"][t]
            L, D = x["L"], x["Delta"]
            md.append(f"| {LABELS[t]} | {f(L['mean'])} | {f(L['median'])} | {f(L['min'])} | {f(L['max'])} "
                      f"| {iv(L['ci_2way'])} | {f(D['mean'])} | {iv(D['ci_block'])} | {iv(D['ci_2way'])} "
                      f"| {D['n_within_margin']} |")
        md += ["", "| policy | tuning J - evaluation J (overfit, mean) | pi_ref - tuning J (mean) | misuse grant rate | misuse exposure $/payment | distinct (k,a,b) | k counts | a mean | b mean |",
               "|---|---|---|---|---|---|---|---|---|"]
        mc = g["Comp_n"]["misuse"]
        md.append(f"| {LABELS['Comp_n']} | – | – | {_pct(mc, 'misuse_grant_rate')} | {_usd(mc)} | – | – | – | – |")
        for t in TUNERS:
            x = g["tuners"][t]
            md.append(f"| {LABELS[t]} | {f(x['overfit']['mean'])} | {f(x['L_tune']['mean'])} "
                      f"| {_pct(x['misuse'], 'misuse_grant_rate')} | {_usd(x['misuse'])} | {x['n_distinct_params']} "
                      f"| {x['k_counts']} | {x['a']['mean']:.4f} | {x['b']['mean']:.4f} |")
        md.append("")
    rs = res["ref"]["stats"]
    if rs.get("release", 1) is not None and "misuse_grant_rate" in rs:
        md.append(f"pi_ref misuse grant rate {100 * rs['misuse_grant_rate']:.3f}%, misuse exposure ${rs['misuse_exposure']:.5f} per payment.")
    ch = res["checks"]
    md += ["", "## Checks", "",
           f"- selection vs direct replay: " + ", ".join(f"{c['name']} gap {c['max_abs_block_gap']:.1e} "
                                                   f"{'PASS' if c['pass_'] else 'FAIL'}" for c in ch["select"]),
           f"- pi_ref released-flag replay vs replay_outage: gap {ch['ref_release_gap']}",
           f"- compiled-table hash equals the one in the qmap.json of run_D.py: "
           f"{sum(v is True for v in ch['d_hash_equal'].values())}/{len(ch['d_hash_equal'])} "
           f"(not comparable: {sum(v is None for v in ch['d_hash_equal'].values())})"]
    if ch.get("comp_notes"):
        md.append(f"- note: {len(ch['comp_notes'])} compiled policies were replayed without the released flag (comp_notes in the manifest)")
    if "vs_D" in ch:
        v = ch["vs_D"]
        md.append(f"- reproduction of D: pi_ref J difference {v['ref_J_diff']:.1e}, largest per-seed L(Comp) difference "
                  f"{v['max_abs_L_comp_diff_bp']:.1e} bp, largest L(R0) vs D's L_B4 difference "
                  f"{v['max_abs_L_R0_vs_D_L_B4_bp']:.1e} bp -> {'PASS' if v['pass_'] else 'FAIL'}")
    return md


def _pct(m, k):
    return "–" if m is None else f"{100 * m[k]:.3f}%"


def _usd(m):
    return "–" if m is None else f"{m['misuse_exposure']:.5f}"


if __name__ == "__main__":
    sys.exit(main())
