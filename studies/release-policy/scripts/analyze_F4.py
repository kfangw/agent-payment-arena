"""Summarise F4.  Reads <run>/npz/F4.npz, <run>/npz/rowboot.jsonl, data/card_fraud/F4_calibration.json,
where <run> is results/real_label_flow/ (or its smoke/ subfolder with --tag smoke).
Run from the package root:
    PYTHONPATH=src python3 scripts/analyze_F4.py [--tag smoke] [--nb 1000]
Writes summary.json and summary.md into <run>.
bp = (difference in $ per payment) / (realized mean exposure of the evaluation draw) * 1e4, as analyze_H.
Draw-level interval: paired block bootstrap, 50-payment blocks (one episode), analyze_H resampling code.
Row-level interval: 2.5/97.5 percentiles of the per-rep differences (same unit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths             # noqa: E402
import analyze_H as AH   # noqa: E402  (ci, NB, BSEED, MARGIN, PPE)

POL = ("comp", "comp_nv", "cutoff", "b4F4")
PN = dict(comp="Comp", comp_nv="Comp_nv", cutoff="Cutoff", b4F4="B4_F4")
CMP = ("b4F4", "cutoff", "comp_nv")
NV_TOL_BP = 2.0          # "about 1 bp" in docs/spec-F4.md section 6, read as |diff| <= 2 bp


def draw_boot(z, nb):
    cols = [k for k in z.files if k != "meta"]
    M = np.column_stack([z[k].astype(float) for k in cols])
    ix = {k: i for i, k in enumerate(cols)}
    nblk = M.shape[0]
    rng = np.random.default_rng(AH.BSEED)
    S = np.empty((nb, M.shape[1]))
    for j0 in range(0, nb, 20):
        m = min(20, nb - j0)
        idx = rng.integers(0, nblk, (m, nblk))
        W = np.bincount((idx + (np.arange(m) * nblk)[:, None]).ravel(), minlength=m * nblk).reshape(m, nblk).astype(float)
        S[j0:j0 + m] = W @ M
    return M.sum(axis=0), S, ix, nblk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--nb", type=int, default=AH.NB)
    a = ap.parse_args(argv)
    run = paths.REAL_LABEL_FLOW / a.tag if a.tag else paths.REAL_LABEL_FLOW
    cal = json.loads((paths.CARD_FRAUD / "F4_calibration.json").read_text())
    z = np.load(run / "npz" / "F4.npz")
    meta = json.loads(str(z["meta"]))
    P, S, ix, nblk = draw_boot(z, a.nb)
    n = nblk * AH.PPE
    mv = meta["realized_mean"]
    c = lambda s, k: s[ix[k]]
    bp = lambda x: x / mv * 1e4

    def stat(f):
        return dict(point=float(f(P)), ci=AH.ci(np.array([f(S[b]) for b in range(a.nb)])))

    res = dict(tag=a.tag, n_eval=meta["n_eval"], realized_mean_v=mv,
               bp_formula="bp = (J_comp - J_other) [$/payment] / realized mean exposure of the eval draw * 1e4",
               draw_bootstrap=dict(resamples=a.nb, seed=AH.BSEED, block=f"{AH.PPE} payments (one episode), as analyze_H"))
    res["J"] = {p: stat(lambda s, p=p: c(s, f"pay_{p}") / n) for p in POL}
    res["diff_bp_draw"] = {p: stat(lambda s, p=p: bp((c(s, "pay_comp") - c(s, f"pay_{p}")) / n)) for p in CMP}
    res["misuse_pass_pct"] = {p: stat(lambda s, p=p: 100.0 * c(s, f"misn_{p}") / c(s, "bs_n_mis")) for p in POL}
    res["misuse_exposure_usd_per_payment"] = {p: stat(lambda s, p=p: c(s, f"misv_{p}") / n) for p in POL}
    res["misuse_exposure_usd_total"] = meta["misuse_exposure_usd_total"]
    res["misuse_pass_count"] = meta["misuse_pass_count"]
    res["n_misuse"] = meta["n_misuse"]
    res["grant_pct"] = {p: float(100.0 * P[ix[f"gr_{p}"]] / n) for p in POL}
    res["verify_share_comp"] = stat(lambda s: c(s, "fa_verify_comp") / n)
    res["comp_first_count"] = meta["comp_first_count"]
    res["comp_nv_first_count"] = meta["comp_nv_first_count"]
    res["band_share_draw"] = stat(lambda s: c(s, "bs_band") / n)
    res["b4F4"] = meta["b4F4"]

    # row-level
    rp = run / "npz" / "rowboot.jsonl"
    reps = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()] if rp.exists() else []
    row = None
    if reps:
        dd = {p: np.array([(r["mean_payoff"]["comp"] - r["mean_payoff"][p]) / r["mean_v"] * 1e4 for r in reps]) for p in CMP}
        row = dict(n_reps=len(reps), mean_rep_s=float(np.mean([r["wall_s"] for r in reps])),
                   diff_bp={p: dict(point_mean=float(dd[p].mean()), median=float(np.median(dd[p])), ci=AH.ci(dd[p]),
                                    sd=float(dd[p].std(ddof=1)) if len(reps) > 1 else None) for p in CMP},
                   misuse_pass_pct={p: dict(mean=float(100 * np.mean([r["misuse_pass_rate"][p] for r in reps])),
                                            ci=AH.ci(100 * np.array([r["misuse_pass_rate"][p] for r in reps]))) for p in POL},
                   band_share=dict(mean=float(np.mean([r["band_share"] for r in reps])),
                                   ci=AH.ci(np.array([r["band_share"] for r in reps]))),
                   n_fraud_rows=dict(mean=float(np.mean([r["row_n_fraud"] for r in reps])),
                                     min=int(min(r["row_n_fraud"] for r in reps)), max=int(max(r["row_n_fraud"] for r in reps))))
    res["row_bootstrap"] = row

    # reading rules (docs/spec-F4.md section 6)
    rr = {}
    zl, zh = cal["zeta_ci"]
    rr["zeta"] = dict(zeta_hat=cal["zeta_hat"], ci=cal["zeta_ci"], ci_contains_1=bool(zl <= 1.0 <= zh),
                      fired="R-zeta-1: calibration error inside the first grid point of the distortion-sensitivity curve" if zl <= 1.0 <= zh
                      else "R-zeta-2: zeta CI excludes 1; read the distortion-sensitivity curve at zeta_hat")
    if row:
        lo, hi = row["diff_bp"]["b4F4"]["ci"]
        pt = res["diff_bp_draw"]["b4F4"]["point"]
        m = AH.MARGIN
        below, above = lo < -m, hi > m
        if not below and not above:
            rule, txt = "R-b4-1", "equal within the margin (row CI inside [-10, +10] bp)"
        elif below and above:
            rule, txt = "R-b4-4", "no margin verdict possible with this data (row CI exceeds +-10 bp on both sides)"
        elif below:
            rule, txt = "R-b4-2", f"Comp trails B4_F4 (row CI lower end {lo:+.2f} bp < -10); report as 'behind by x bp', split by zeta_hat curve"
        else:
            rule, txt = "R-b4-3", f"Comp equal or above B4_F4 (row CI upper end {hi:+.2f} bp > +10); do not write 'ahead'"
        rr["comp_vs_b4"] = dict(rule=rule, text=txt, row_ci=[lo, hi], draw_point_bp=pt, margin_bp=m)
        ptn = res["diff_bp_draw"]["comp_nv"]["point"]
        if abs(ptn) <= NV_TOL_BP:
            rr["channel"] = dict(rule="R-ch-1", text=f"channel hardly used in F4 (|Comp - Comp_nv| = {abs(ptn):.2f} bp <= {NV_TOL_BP:g})", point_bp=ptn)
        else:
            rr["channel"] = dict(rule="R-ch-2", text=f"channel value {ptn:+.2f} bp > {NV_TOL_BP:g} bp; check the fixed-bin table for a thick band", point_bp=ptn)
    res["reading_rules"] = rr
    res["calibration"] = cal
    (run / "summary.json").write_text(json.dumps(res, indent=1))

    # ---------------- markdown
    f2 = lambda d: f"{d['point']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    md = [f"# F4 summary (auto{', ' + a.tag if a.tag else ''})", ""]
    if a.tag == "smoke":
        md += ["Smoke run; numbers are not results.", ""]
    md += [f"Eval draw: seed {meta['eval_seed']}, n={meta['n_eval']:,} ({nblk:,} blocks), realized mean exposure {mv:.4f}. "
           f"bp = (J_comp - J_other) / mean exposure * 1e4. Draw CI: paired block bootstrap, {a.nb} resamples, seed {AH.BSEED}, "
           f"50-payment blocks. Row CI: percentiles of per-rep differences"
           + (f" ({row['n_reps']} reps, mean {row['mean_rep_s']:.1f} s/rep)." if row else " (no reps).")
           + f" B4_F4 = (k={meta['b4F4']['k']}, a={meta['b4F4']['a']:.5f}, b={meta['b4F4']['b']:.5f}).", ""]
    bd = cal["band"]
    md += ["## 1. Band and calibration (eval split)", "",
           "| quantity | value |", "|---|---|",
           f"| band [{bd['a']}, {bd['b']}) share of eval rows | {100 * bd['share']:.4f}% ({bd['n']} rows) |",
           f"| band fraud rate vs mean pi0 | {bd['fraud_rate']:.4f} vs {bd['mean_pi0']:.4f} ({bd['n_fraud']} frauds) |",
           f"| zeta_hat [95% row-bootstrap CI] | {cal['zeta_hat']:.4f} [{cal['zeta_ci'][0]:.4f}, {cal['zeta_ci'][1]:.4f}] ({cal['zeta_boot_n']} reps) |",
           f"| band share in eval draw (payments) | {100 * res['band_share_draw']['point']:.4f}%"
           + (f"; row-boot mean {100 * row['band_share']['mean']:.4f}% [{100 * row['band_share']['ci'][0]:.4f}, {100 * row['band_share']['ci'][1]:.4f}]" if row else "") + " |",
           "", "Fixed-edge bins:", "", "| bin | n | mean pi0 | fraud rate | frauds |", "|---|---|---|---|---|"]
    for r in cal["bins"]:
        md.append(f"| [{r['lo']:g}, {r['hi']:g}{']' if r['hi'] == 1.0 else ')'} | {r['n']} | "
                  f"{'-' if r['mean_pi0'] is None else format(r['mean_pi0'], '.5f')} | "
                  f"{'-' if r['fraud_rate'] is None else format(r['fraud_rate'], '.5f')} | {r['n_fraud']} |")
    md += ["", "## 2. Payoff differences, Comp minus X (bp)", "",
           "| comparison | point (draw) [draw CI] | row-boot median [row CI] |", "|---|---|---|"]
    for p in CMP:
        rw = "" if not row else f"{row['diff_bp'][p]['median']:+.2f} [{row['diff_bp'][p]['ci'][0]:+.2f}, {row['diff_bp'][p]['ci'][1]:+.2f}]"
        md.append(f"| Comp - {PN[p]} | {f2(res['diff_bp_draw'][p])} | {rw} |")
    md += ["", "J ($ per payment): " + ", ".join(f"{PN[p]} {res['J'][p]['point']:.6f}" for p in POL), "",
           "## 3. Misuse and verify", "",
           "| policy | misuse pass rate (%) [draw CI] | row-boot mean [row CI] | misuse exposure ($ total) | per payment ($) | grant % |",
           "|---|---|---|---|---|---|"]
    for p in POL:
        m_ = res["misuse_pass_pct"][p]
        rw = "" if not row else f"{row['misuse_pass_pct'][p]['mean']:.3f} [{row['misuse_pass_pct'][p]['ci'][0]:.3f}, {row['misuse_pass_pct'][p]['ci'][1]:.3f}]"
        md.append(f"| {PN[p]} | {m_['point']:.3f} [{m_['ci'][0]:.3f}, {m_['ci'][1]:.3f}] | {rw} | "
                  f"{res['misuse_exposure_usd_total'][p]:.2f} | {res['misuse_exposure_usd_per_payment'][p]['point']:.5f} | {res['grant_pct'][p]:.3f} |")
    vs = res["verify_share_comp"]
    md += ["", f"Misuse payments in the draw: {meta['n_misuse']:,} (rate {100 * meta['misuse_rate']:.4f}%). "
           f"Comp verify share at arrival: {100 * vs['point']:.4f}% [{100 * vs['ci'][0]:.4f}, {100 * vs['ci'][1]:.4f}]. "
           f"Comp first actions (counts): {meta['comp_first_count']}.", "",
           "## 4. Classifier and split", "",
           f"ECE (fixed bins) {cal['ece_fixed_bins']:.6f}; AUC {cal['auc']:.4f}, AP {cal['ap']:.4f} (calibrated; raw scores AUC {cal['auc_raw']:.4f}, AP {cal['ap_raw']:.4f}). "
           f"Base rate (eval) {cal['base_rate_eval']:.5f}.", "",
           "| split | rows | frauds | Time range |", "|---|---|---|---|"]
    for k, s in cal["splits"].items():
        md.append(f"| {k} | {s['n']:,} | {s['n_fraud']} | {s['time_lo']:.0f} - {s['time_hi']:.0f} |")
    md += ["", "## 5. Reading rules (docs/spec-F4.md, section 6)", ""]
    rz = rr["zeta"]
    md.append(f"- zeta: {rz['fired']} (zeta_hat {rz['zeta_hat']:.4f}, CI [{rz['ci'][0]:.4f}, {rz['ci'][1]:.4f}]).")
    if "comp_vs_b4" in rr:
        md.append(f"- Comp vs B4_F4: {rr['comp_vs_b4']['rule']}: {rr['comp_vs_b4']['text']}.")
        md.append(f"- Channel: {rr['channel']['rule']}: {rr['channel']['text']}.")
    else:
        md.append("- No row-bootstrap reps yet; Comp vs B4_F4 and channel rules not applied.")
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    main()
