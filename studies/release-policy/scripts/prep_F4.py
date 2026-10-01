"""F4 prep: time-ordered split, classifier, isotonic calibration, calibration report.
Run from the package root, with data/card_fraud/creditcard.csv in place (see data/card_fraud/README.md):
    PYTHONPATH=src python3 scripts/prep_F4.py
Writes data/card_fraud/F4_scores.npz and data/card_fraud/F4_calibration.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths                                                     # noqa: E402
from run_G import zeta_mle, ZGRID                                # noqa: E402

A, B = 0.1625, 0.63125
EDGES = [0.0, 0.05, A, 0.3, B, 0.9, 1.0]
FRAC = (0.5, 0.25, 0.25)
SEED_BOOT = [111, 900, 0]
NBOOT = 200


def zhat(s, y):
    k, flag = zeta_mle(s, y)
    return float(np.exp(0.01 * k)), int(k), flag


def table(pi, y):
    rows = []
    for j in range(len(EDGES) - 1):
        lo, hi = EDGES[j], EDGES[j + 1]
        m = (pi >= lo) & ((pi < hi) if j < len(EDGES) - 2 else (pi <= hi))
        n = int(m.sum())
        rows.append(dict(lo=lo, hi=hi, n=n, mean_pi0=float(pi[m].mean()) if n else None,
                         fraud_rate=float(y[m].mean()) if n else None, n_fraud=int(y[m].sum())))
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv(paths.CARD_FRAUD / "creditcard.csv")
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    n_tr, n_ca = int(n * FRAC[0]), int(n * FRAC[1])
    cuts = [0, n_tr, n_tr + n_ca, n]
    X = df[[f"V{i}" for i in range(1, 29)]].copy()
    X["logAmount"] = np.log1p(df["Amount"].values)
    X = X.values
    y = df["Class"].values.astype(np.int8)
    sl = {nm: slice(cuts[i], cuts[i + 1]) for i, nm in enumerate(("train", "calib", "eval"))}
    splits = {nm: dict(n=int(s.stop - s.start), n_fraud=int(y[s].sum()),
                       time_lo=float(df["Time"].values[s.start]), time_hi=float(df["Time"].values[s.stop - 1]))
              for nm, s in sl.items()}
    assert splits["eval"]["n_fraud"] >= 60 and splits["calib"]["n_fraud"] >= 100, splits
    clf = HistGradientBoostingClassifier(random_state=0).fit(X[sl["train"]], y[sl["train"]])
    s_ca = clf.predict_proba(X[sl["calib"]])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s_ca, y[sl["calib"]])
    s_ev = clf.predict_proba(X[sl["eval"]])[:, 1]
    pi0 = np.clip(iso.predict(s_ev), 1e-9, 1 - 1e-9)
    ye = y[sl["eval"]]
    amt = df["Amount"].values[sl["eval"]]
    ne = len(ye)

    tab = table(pi0, ye)
    ece = float(sum(r["n"] * abs(r["mean_pi0"] - r["fraud_rate"]) for r in tab if r["n"]) / ne)
    inb = (pi0 >= A) & (pi0 < B)
    band = dict(a=A, b=B, n=int(inb.sum()), share=float(inb.mean()),
                n_fraud=int(ye[inb].sum()),
                fraud_rate=float(ye[inb].mean()) if inb.any() else None,
                mean_pi0=float(pi0[inb].mean()) if inb.any() else None)
    auc, ap = float(roc_auc_score(ye, pi0)), float(average_precision_score(ye, pi0))
    auc_raw, ap_raw = float(roc_auc_score(ye, s_ev)), float(average_precision_score(ye, s_ev))
    z, zk, zflag = zhat(pi0, ye)
    rng = np.random.default_rng(SEED_BOOT)
    zb = []
    for _ in range(NBOOT):
        ix = rng.integers(0, ne, size=ne)
        zb.append(zhat(pi0[ix], ye[ix])[0])
    zb = np.array(zb)
    zci = [float(np.percentile(zb, 2.5)), float(np.percentile(zb, 97.5))]
    cal = dict(splits=splits, split_fracs=FRAC, n_total=int(n), n_eval=int(ne), base_rate_eval=float(ye.mean()),
               n_fraud_eval=int(ye.sum()), bins=tab, ece_fixed_bins=ece, band=band,
               auc=auc, ap=ap, auc_raw=auc_raw, ap_raw=ap_raw,
               zeta_hat=z, zeta_k=zk, zeta_flag=zflag, zeta_ci=zci, zeta_boot_n=NBOOT,
               zeta_boot_rng=SEED_BOOT, zeta_boot_edge_hits=int(np.sum((zb <= ZGRID[0]) | (zb >= ZGRID[-1]))),
               zeta_convention="zeta_mle fits P(y=1|s)=min(s/zeta,1) on s=pi0",
               pi0_quantiles={str(q): float(np.quantile(pi0, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9995)},
               pi0_max=float(pi0.max()), prep_seconds=time.time() - t0)
    np.savez_compressed(paths.CARD_FRAUD / "F4_scores.npz", pi0=pi0, label=ye, amount=amt,
                        split_bounds=np.array(cuts), time_bounds=np.array(
                            [splits[k][t] for k in ("train", "calib", "eval") for t in ("time_lo", "time_hi")]))
    (paths.CARD_FRAUD / "F4_calibration.json").write_text(json.dumps(cal, indent=1))

    print(f"splits: " + ", ".join(f"{k} n={v['n']} fraud={v['n_fraud']}" for k, v in splits.items()))
    print(f"eval base rate {ye.mean():.5f}  AUC {auc:.4f} AP {ap:.4f} (raw AUC {auc_raw:.4f} AP {ap_raw:.4f})")
    print("bin           n     mean_pi0  fraud_rate  n_fraud")
    for r in tab:
        mp_ = "-" if r["mean_pi0"] is None else f"{r['mean_pi0']:.5f}"
        fr = "-" if r["fraud_rate"] is None else f"{r['fraud_rate']:.5f}"
        print(f"[{r['lo']:.5g},{r['hi']:.5g}) {r['n']:7d}  {mp_:>9}  {fr:>9}  {r['n_fraud']}")
    print(f"ECE(fixed bins) {ece:.6f}")
    print(f"band [{A},{B}): n={band['n']} share={band['share']:.5f} fraud={band['n_fraud']} "
          f"rate={band['fraud_rate']} mean_pi0={band['mean_pi0']}")
    print(f"zeta_hat {z:.4f} (k={zk}, {zflag}) CI [{zci[0]:.4f}, {zci[1]:.4f}] boot edge hits {cal['zeta_boot_edge_hits']}")
    print(f"pi0 quantiles {cal['pi0_quantiles']} max {pi0.max():.4f}  t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
