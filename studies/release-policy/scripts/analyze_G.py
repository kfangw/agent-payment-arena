"""Summarise experiment G (spec: docs/calibration_pilot.md sec 7, 8).  Reads <run>/plan/, <run>/manifest.json,
<run>/npz/ (run = results/calibration_pilot/[<tag>/]).

    export PYTHONPATH=$PWD/src && python3 scripts/analyze_G.py [--tag smoke]

Per true zeta, learning seed s and n (bp of the $49.50 mean exposure; $ per payment = bp x 0.00495):
    L_cal   = J(pi_ref) - J(Comp_cal)     main metric
    L_naive = J(pi_ref) - J(Comp_naive)
    L_B4    = J(pi_ref) - J(B4_n)
    D_cal   = J(Comp_cal) - J(B4_n)       positive: Comp_cal better
J is aggregated as in analyze_D.py (column sums of the block-sum matrix / payments).
Bootstrap (1,000 resamples, seed 20260930): the same block resample for every policy (paired);
2-way = block resample x learning-seed resample, the seed resample shared by every n and zeta.
Reading rule (spec 8): n* = smallest n with mean L_cal <= 10 bp and #(L_cal > 10) <= 3 of 30
for every true zeta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

OUT = paths.CALIBRATION_PILOT
DEN = 49.5
NB, BSEED = 1_000, 20260930
MARGIN = 10.0
MAX_EXCEED_FRAC = 0.10


def zkey(z):
    return repr(float(z))


def comp_name(h, z, k):
    return f"c_{h[:16]}_z{float(z)!r}_k{int(k):+d}"


def b4_name(z, k, a, b):
    return f"b_z{float(z)!r}_k{int(k)}_a{float(a)!r}_b{float(b)!r}"


def summ(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),
                max=float(v.max()), n=int(len(v)))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def bp(x):
    return x / DEN * 1e4


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--drop-n5000", action="store_true",
                    help="spec 6: analyse without n = 5000 when it did not finish by 21:00")
    a = ap.parse_args(argv)
    run = OUT / a.tag if a.tag else OUT
    man = json.loads((run / "manifest.json").read_text())
    qmap = json.loads((run / "qmap.json").read_text())
    seeds, zetas = man["seeds"], man["zetas"]
    ns = [n for n in man["ns_evaluated"] if not (a.drop_n5000 and n == 5000)]
    man["ns_evaluated"] = ns
    if a.drop_n5000:
        man["decision"]["dropped_n5000"] = True
    plan = {s: json.loads((run / "plan" / f"seed_{s:02d}.json").read_text()) for s in seeds}
    b200 = json.loads((run / "plan" / "b4_200k.json").read_text())

    names = [it["name"] for it in man["items"] if (run / "npz" / f"{it['name']}.npz").exists()]
    cols = [np.load(run / "npz" / f"{n}.npz")["block_sums"].astype(float) for n in names]
    nblk = len(cols[0])
    assert all(len(c) == nblk for c in cols)
    M = np.column_stack(cols)
    del cols
    denom = nblk * 50.0
    point = M.sum(axis=0) / denom
    ix = {n: i for i, n in enumerate(names)}
    r_i = ix[man["ref"]]

    Z, S, NN = len(zetas), len(seeds), len(ns)
    ci_, cn_, cb_ = (np.zeros((Z, S, NN), int) for _ in range(3))
    zh = np.zeros((Z, S, NN))
    zk = np.zeros((Z, S, NN), int)
    flag = np.empty((Z, S, NN), object)
    nmis = np.zeros((Z, S, NN), int)
    qh = np.zeros((S, NN))
    for zi, z in enumerate(zetas):
        for si, s in enumerate(seeds):
            for ni, n in enumerate(ns):
                v = plan[s]["per_n"][str(n)]
                pz = v["per_zeta"][zkey(z)]
                h = qmap[repr(float(v["q_hat"]))]
                qh[si, ni] = v["q_hat"]
                c = pz["cal"]
                ci_[zi, si, ni] = ix[comp_name(h, z, c["zhat_k"])]
                cn_[zi, si, ni] = ix[comp_name(h, z, 0)]
                p = pz["b4"]
                cb_[zi, si, ni] = ix[b4_name(z, p["k"], p["a"], p["b"])]
                zh[zi, si, ni], zk[zi, si, ni], flag[zi, si, ni] = c["zhat"], c["zhat_k"], c["flag"]
                nmis[zi, si, ni] = c["n_mis_ans"]
    b2_i = np.array([ix[b4_name(z, **{x: b200["per_zeta"][zkey(z)]["b4"][x] for x in "kab"})] for z in zetas])

    Lcal = bp(point[r_i] - point[ci_])
    Lnai = bp(point[r_i] - point[cn_])
    Lb4 = bp(point[r_i] - point[cb_])
    Dcal = bp(point[ci_] - point[cb_])
    L200 = bp(point[r_i] - point[b2_i])

    # ---- bootstrap
    rng = np.random.default_rng(BSEED)
    Pb = np.empty((NB, M.shape[1]))
    chunk = 20
    for j0 in range(0, NB, chunk):
        m = min(chunk, NB - j0)
        idx = rng.integers(0, nblk, (m, nblk))
        W = np.bincount((idx + (np.arange(m) * nblk)[:, None]).ravel(),
                        minlength=m * nblk).reshape(m, nblk).astype(float)
        Pb[j0:j0 + m] = W @ M / denom
    seed_rs = rng.integers(0, S, (NB, S))
    del M

    def two_way(Xb):      # (NB, Z, S, NN) -> (NB, Z, NN): mean over the resampled seeds
        g = seed_rs[:, None, :, None].repeat(Z, 1).repeat(NN, 3)
        return np.take_along_axis(Xb, g, axis=2).mean(axis=2)

    Lcal_b = bp(Pb[:, [r_i]][:, :, None, None] - Pb[:, ci_])
    Lnai_b = bp(Pb[:, [r_i]][:, :, None, None] - Pb[:, cn_])
    Lb4_b = bp(Pb[:, [r_i]][:, :, None, None] - Pb[:, cb_])
    Dcal_b = bp(Pb[:, ci_] - Pb[:, cb_])
    CN_b = bp(Pb[:, ci_] - Pb[:, cn_])                       # J(cal) - J(naive)
    L2 = {k: two_way(v) for k, v in dict(L_cal=Lcal_b, L_naive=Lnai_b, L_B4=Lb4_b, D_cal=Dcal_b,
                                          cal_minus_naive=CN_b).items()}
    L200_b = bp(Pb[:, [r_i]] - Pb[:, b2_i])

    res = dict(eval_seed=man["eval_seed"], n_eval_used=man["n_eval_used"], n_blocks=nblk,
               eval_size=("full 5,663,400" if man["n_eval_used"] == 5_663_400 else
                          f"prefix {man['n_eval_used']:,} (runtime rule, spec 6)" if man["n_eval_draw"] == 5_663_400
                          else f"prefix {man['n_eval_used']:,} (--n-eval)"),
               decision=man["decision"], seeds=seeds, zetas=zetas, ns=ns, ns_planned=man["ns"],
               unit="bp of $49.50 (x 0.00495 = $ per payment)", bootstrap=dict(resamples=NB, seed=BSEED),
               ref=dict(name=man["ref"], J=float(point[r_i])), per_cell={}, b4_200k={}, per_seed=[])
    X = dict(L_cal=Lcal, L_naive=Lnai, L_B4=Lb4, D_cal=Dcal)
    XB = dict(L_cal=Lcal_b, L_naive=Lnai_b, L_B4=Lb4_b, D_cal=Dcal_b)
    for zi, z in enumerate(zetas):
        for ni, n in enumerate(ns):
            cell = dict(zeta=z, n=n)
            for k, arr in X.items():
                v = arr[zi, :, ni]
                cell[k] = dict(summ(v), ci_block=ci(XB[k][:, zi, :, ni].mean(axis=1)),
                               ci_2way=ci(L2[k][:, zi, ni]),
                               n_gt_margin=int(np.sum(v > MARGIN)) if k != "D_cal" else None,
                               n_lt_minus_margin=int(np.sum(v < -MARGIN)) if k == "D_cal" else None)
            cmn = Lnai[zi, :, ni] - Lcal[zi, :, ni]              # = J(cal) - J(naive)
            cell["cal_minus_naive"] = dict(summ(cmn), ci_2way=ci(L2["cal_minus_naive"][:, zi, ni]))
            zz = zh[zi, :, ni]
            fl = list(flag[zi, :, ni])
            cell["zeta_hat"] = dict(median=float(np.median(zz)), q05=float(np.quantile(zz, 0.05)),
                                    q95=float(np.quantile(zz, 0.95)), mean_log_err=float(np.mean(np.log(zz) - np.log(z))),
                                    n_no_misuse=fl.count("no_misuse"), n_all_misuse=fl.count("all_misuse"),
                                    n_grid_edge=int(np.sum(np.abs(zk[zi, :, ni]) == 200)),
                                    misuse_labels=summ(nmis[zi, :, ni]))
            res["per_cell"][f"{zkey(z)}|{n}"] = cell
        res["b4_200k"][zkey(z)] = dict(L=float(L200[zi]), L_ci_block=ci(L200_b[:, zi]),
                                       params=b200["per_zeta"][zkey(z)]["b4"])
    for zi, z in enumerate(zetas):
        for si, s in enumerate(seeds):
            for ni, n in enumerate(ns):
                res["per_seed"].append(dict(zeta=z, seed=s, n=n, q_hat=float(qh[si, ni]), zeta_hat=float(zh[zi, si, ni]),
                                            zeta_hat_flag=flag[zi, si, ni], L_cal=float(Lcal[zi, si, ni]),
                                            L_naive=float(Lnai[zi, si, ni]), L_B4=float(Lb4[zi, si, ni]),
                                            D_cal=float(Dcal[zi, si, ni])))

    # ---- reading rule (spec 8)
    max_exceed = int(np.floor(MAX_EXCEED_FRAC * S + 1e-9))
    ok = {}
    for ni, n in enumerate(ns):
        per_z = {}
        for zi, z in enumerate(zetas):
            m_ = float(Lcal[zi, :, ni].mean())
            c_ = int(np.sum(Lcal[zi, :, ni] > MARGIN))
            per_z[zkey(z)] = dict(mean=m_, n_gt=c_, pass_=bool(m_ <= MARGIN and c_ <= max_exceed))
        ok[n] = dict(per_zeta=per_z, pass_=all(v["pass_"] for v in per_z.values()))
    nstar = next((n for n in ns if ok[n]["pass_"]), None)
    full_grid = len(zetas) == 6 and S == 30 and all(n in ns for n in (130, 500, 2000))
    if nstar is None:
        verdict = "No n*: reported as a negative result. The proposition stands as it is."
    elif nstar <= 500:
        verdict = "n* <= 500: the answer rate and the score calibration are estimated from the same pilot of a few hundred payments. The table does not change (case a)."
    else:
        verdict = "500 < n* <= 5000: the calibration needs thousands of labels (case b)."
    failed = {n: [z for z, v in ok[n]["per_zeta"].items() if not v["pass_"]] for n in ns}
    z1 = zetas.index(1.0) if 1.0 in zetas else None
    z1note = {}
    if z1 is not None:
        for ni, n in enumerate(ns):
            worse = float((Lcal[z1, :, ni] - Lnai[z1, :, ni]).mean())
            z1note[n] = dict(mean_Lcal_minus_Lnaive=worse, over_margin=bool(worse > MARGIN))
    res["reading_rule"] = dict(margin_bp=MARGIN, max_exceed_seeds=max_exceed, per_n=ok, n_star=nstar,
                               verdict=verdict, failed_zetas=failed, zeta1_cal_vs_naive=z1note,
                               complete_design=full_grid)
    (run / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    np.savez_compressed(run / "G_results.npz", L_cal=Lcal, L_naive=Lnai, L_B4=Lb4, D_cal=Dcal,
                        zeta_hat=zh, q_hat=qh, zetas=np.array(zetas), seeds=np.array(seeds), ns=np.array(ns))

    # ---- markdown
    f = lambda x: f"{x:+.2f}"
    rr = res["reading_rule"]
    md = [f"# Experiment G summary (generated by analyze_G.py{', ' + a.tag if a.tag else ''})", ""]
    if a.tag:
        md += [f"**This is a {a.tag} run. The numbers are not results (spec sec 9.5).**", ""]
    md += [f"Evaluation seed {man['eval_seed']}, {res['eval_size']} ({nblk:,} blocks), {S} learning seeds, true zeta {zetas}, n {ns}. "
           f"Unit bp (of the mean exposure $49.50; $ per payment = bp x 0.00495). Bootstrap {NB} resamples (seed {BSEED}), "
           f"two-way (block x seed; the seed resample is shared by all n and zeta). pi_ref J = ${point[r_i]:.6f} per payment.", "",
           f"Run-time rule: {man['decision']['replays']} replays x {man['decision']['t_replay_full_s']:.1f} s / "
           f"{man['decision']['workers']} workers = {man['decision']['est_hours_full']:.2f} h -> evaluation {res['eval_size']}. "
           f"n = 5000 dropped: {'yes' if man['decision'].get('dropped_n5000') else 'no'}.", "",
           "## Reading rule (spec sec 8)", "",
           f"- n* = {nstar if nstar is not None else 'none'}. {verdict}" + ("" if full_grid else " (only part of the design was run; the reading is void)"),
           "- Pass per condition (mean L_cal <= 10 bp, and at most "
           f"{max_exceed} seeds with L_cal > 10 bp):", "",
           "| n | " + " | ".join(f"zeta={z:g}" for z in zetas) + " | all |",
           "|---|" + "---|" * (len(zetas) + 1)]
    for n in ns:
        md.append(f"| {n} | " + " | ".join(
            f"{v['mean']:+.2f} / {v['n_gt']} {'pass' if v['pass_'] else 'fail'}" for v in ok[n]["per_zeta"].values())
            + f" | {'pass' if ok[n]['pass_'] else 'fail'} |")
    if z1note:
        md += ["", "At zeta = 1, the size by which the calibration estimate spoils a correct calibration (mean L_cal - L_naive, bp): " + ", ".join(
            f"n={n}: {v['mean_Lcal_minus_Lnaive']:+.2f}{' (above 10 bp, stated in the text)' if v['over_margin'] else ''}"
            for n, v in z1note.items())]
    for key, title in (("L_cal", "L_cal = J(pi_ref) - J(Comp_cal) (main metric)"),
                       ("L_naive", "L_naive = J(pi_ref) - J(Comp_naive) (no calibration)"),
                       ("L_B4", "L_B4 = J(pi_ref) - J(B4_n)"),
                       ("D_cal", "D_cal = J(Comp_cal) - J(B4_n) (positive: Comp_cal is better; secondary result)")):
        md += ["", f"## {title}", "",
               "| zeta | n | mean | median | min | max | CI_2way | " + ("#<-10" if key == "D_cal" else "#>10") + " |",
               "|---|---|---|---|---|---|---|---|"]
        for z in zetas:
            for n in ns:
                c = res["per_cell"][f"{zkey(z)}|{n}"][key]
                cnt = c["n_lt_minus_margin"] if key == "D_cal" else c["n_gt_margin"]
                md.append(f"| {z:g} | {n} | {f(c['mean'])} | {f(c['median'])} | {f(c['min'])} | {f(c['max'])} | "
                          f"[{c['ci_2way'][0]:+.2f}, {c['ci_2way'][1]:+.2f}] | {cnt} |")
    md += ["", "## Distribution of zeta-hat", "", "| zeta | n | median | 5% | 95% | mean log error | grid edge | no misuse | all misuse | median misuse labels with an answer |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for z in zetas:
        for n in ns:
            c = res["per_cell"][f"{zkey(z)}|{n}"]["zeta_hat"]
            md.append(f"| {z:g} | {n} | {c['median']:.3f} | {c['q05']:.3f} | {c['q95']:.3f} | {c['mean_log_err']:+.3f} | "
                      f"{c['n_grid_edge']} | {c['n_no_misuse']} | {c['n_all_misuse']} | {c['misuse_labels']['median']:.0f} |")
    md += ["", "## B4_200k (reference line)", "", "| zeta | L | CI_block | (k, a, b) |", "|---|---|---|---|"]
    for z in zetas:
        v = res["b4_200k"][zkey(z)]
        md.append(f"| {z:g} | {f(v['L'])} | [{v['L_ci_block'][0]:+.2f}, {v['L_ci_block'][1]:+.2f}] | "
                  f"({v['params']['k']}, {v['params']['a']:.4f}, {v['params']['b']:.4f}) |")
    md += ["", "## Notes for interpretation", "",
           "- Comp_cal knows the rail model, the constants other than q, and the functional form of the distortion (multiplicative). B4_n does not know these. D_cal is a comparison under this information asymmetry.",
           "- For zeta > 1 the information in pi vanishes at s = 1 and the estimated model min(s/zeta, 1) cannot explain those points. This is a property of the world (spec sec 5).",
           "- For F1 (misuse rate 0.005) at n = 5000 the expected number of misuse labels with an answer is 5000 x 0.75 x 0.005 = about 19, too few for calibration (computed only, spec sec 10).",
           "- pi_ref is the grid implementation's reference, not a theoretical optimum."]
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:14 + len(ns)]))


if __name__ == "__main__":
    main()
