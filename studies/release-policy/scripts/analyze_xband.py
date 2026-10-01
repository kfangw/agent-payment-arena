"""Collect xband results into summary.json and summary.md.

Run after scripts/xband.py has written results/exposure_band/runs/xband_E-fitted_F*_s*.json:
    python3 scripts/analyze_xband.py [runs_subdir]
"""
import json
import sys
from pathlib import Path

from paths import EXPOSURE_BAND

D = EXPOSURE_BAND
CELLS = [("F1", 7), ("F2", 8), ("F3", 9)]
MARGIN = 10.0
RES = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = "" if RES == "runs" else RES + "-"


def f1(x):
    return f"{x:+.1f}"


def ci(c):
    return f"[{c[0]:+.1f}, {c[1]:+.1f}]"


def main():
    D.mkdir(parents=True, exist_ok=True)
    summ = {}
    for f, s in CELLS:
        p = D / RES / f"xband_E-fitted_{f}_s{s}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        r = json.load(open(p))
        e = dict(b4_params=r["b4_params"], b4_reproduces_b4ref_maxgap=r["b4_reproduces_b4ref_maxgap"],
                 band1_equals_b4ref=r["check"].get("band1_equals_b4ref"),
                 compiled_minus_b4=r["compiled_minus_b4"], b4_security=r["b4"], bands={})
        for nb, c in r["configs"].items():
            if nb == "1":
                continue
            e["bands"][nb] = dict(inner_edges=c["inner_edges"], params=c["params"],
                                  tune_gain_over_b4=c["tune_gain_over_b4"],
                                  compiled_minus_band=c["compiled_minus_band"],
                                  band_minus_b4=c["band_minus_b4"], security=c["eval"])
        summ[f] = e
    (D / f"{OUT}summary.json").write_text(json.dumps(summ, indent=1))

    L = ["# Exposure-band B4 (E-fitted, 161-point grid)", "",
         "bp is relative to the mean exposure of $49.50; brackets are 95% paired block-bootstrap intervals (blocks of 50, 10,000 resamples).", "",
         "## Difference to the compiled policy", "",
         "| flow | compiled − B4 | compiled − B4-band(4) | compiled − B4-band(3) | compiled − B4-band(5) |",
         "|---|---|---|---|---|"]
    for f, _ in CELLS:
        e = summ[f]
        row = [f, f"{f1(e['compiled_minus_b4']['bp'])} {ci(e['compiled_minus_b4']['bp_ci95'])}"]
        for nb in ("4", "3", "5"):
            c = e["bands"][nb]["compiled_minus_band"]
            row.append(f"{f1(c['bp'])} {ci(c['bp_ci95'])}")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## B4-band − B4", "", "| flow | 3 bands | 4 bands | 5 bands |", "|---|---|---|---|"]
    for f, _ in CELLS:
        e = summ[f]
        L.append("| " + " | ".join([f] + [f"{f1(e['bands'][nb]['band_minus_b4']['bp'])} "
                                           f"{ci(e['bands'][nb]['band_minus_b4']['bp_ci95'])}"
                                           for nb in ("3", "4", "5")]) + " |")
    L += ["", "## Verdict (main specification: 4 bands, margin 10 bp)", ""]
    for f, _ in CELLS:
        c = summ[f]["bands"]["4"]["compiled_minus_band"]["bp_ci95"]
        ok = c[0] > -MARGIN and c[1] < MARGIN
        L.append(f"- {f}: interval {ci(c)}, {'inside the margin' if ok else 'outside the margin'}")
    L += ["", "## Per-band choice (k, a, b)", ""]
    for f, _ in CELLS:
        e = summ[f]
        L.append(f"- {f} B4: {tuple(e['b4_params'])}")
        for nb in ("3", "4", "5"):
            b = e["bands"][nb]
            edges = ", ".join(f"{x:.2f}" for x in b["inner_edges"])
            ps = "; ".join(f"({p[0]}, {p[1]:.4f}, {p[2]:.4f})" for p in b["params"])
            L.append(f"  - {nb} bands (inner edges {edges} USD): {ps}")
    L += ["", "## Security: misuse payments released", "",
          "| flow | policy | share of misuse payments released | released misuse exposure per payment (USD) |", "|---|---|---|---|"]
    for f, _ in CELLS:
        e = summ[f]
        rows = [("B4", e["b4_security"])] + [(f"B4-band({nb})", e["bands"][nb]["security"]) for nb in ("3", "4", "5")]
        for nm, sct in rows:
            L.append(f"| {f} | {nm} | {sct['misuse_grant_rate']*100:.3f}% | {sct['misuse_exposure']:.5f} |")
    L += ["", "## Reproduction check", ""]
    for f, _ in CELLS:
        e = summ[f]
        L.append(f"- {f}: one band reproduces the b4ref (k, a, b) and tuning mean: {e['band1_equals_b4ref']}; "
                 f"largest gap of the B4 evaluation block sums to b4ref: {e['b4_reproduces_b4ref_maxgap']:.2e}")
    (D / f"{OUT}summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
