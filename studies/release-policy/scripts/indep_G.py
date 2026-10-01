"""Independent re-derivation of G headline numbers from npz block sums + plan json."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

R = paths.CALIBRATION_PILOT      # needs the qmap.json and npz/ written by run_G.py
man = json.loads((R / "manifest.json").read_text())
qmap = json.loads((R / "qmap.json").read_text())
ZG = np.exp(np.arange(-200, 201) * 0.01)


def J(name):
    b = np.load(R / "npz" / f"{name}.npz")["block_sums"].astype(float)
    return b.sum() / (len(b) * 50.0)


cache = {}
def Jc(name):
    if name not in cache:
        cache[name] = J(name)
    return cache[name]


bp = lambda x: x / 49.5 * 1e4
Jref = Jc(man["ref"])
print("ref", man["ref"], Jref)
zetas = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
ns = [130, 500, 2000, 5000]
out = {}
for z in zetas:
    for n in ns:
        Lc, Ln, Lb, zh = [], [], [], []
        for s in range(1, 31):
            p = json.loads((R / "plan" / f"seed_{s:02d}.json").read_text())
            v = p["per_n"][str(n)]
            pz = v["per_zeta"][repr(float(z))]
            h = qmap[repr(float(v["q_hat"]))][:16]
            k = pz["cal"]["zhat_k"]
            Lc.append(bp(Jref - Jc(f"c_{h}_z{float(z)!r}_k{int(k):+d}")))
            Ln.append(bp(Jref - Jc(f"c_{h}_z{float(z)!r}_k+0")))
            b = pz["b4"]
            Lb.append(bp(Jref - Jc(f"b_z{float(z)!r}_k{int(b['k'])}_a{float(b['a'])!r}_b{float(b['b'])!r}")))
            zh.append(ZG[k + 200])
        Lc, Ln, Lb, zh = map(np.array, (Lc, Ln, Lb, zh))
        out[f"{z}|{n}"] = dict(Lcal=Lc.mean(), Lnaive=Ln.mean(), LB4=Lb.mean(), ngt=int((Lc > 10).sum()),
                               zmed=float(np.median(zh)), z05=float(np.quantile(zh, .05)), z95=float(np.quantile(zh, .95)),
                               zlogbias=float(np.mean(np.log(zh / z))), cal_worse_naive=float((Lc - Ln).mean()),
                               nB4better=int((Lb < Lc).sum()), Lcal_max=float(Lc.max()))
        o = out[f"{z}|{n}"]
        print(f"z={z:<4} n={n:<5} Lcal={o['Lcal']:+7.2f} Lnaive={o['Lnaive']:+7.2f} LB4={o['LB4']:+7.2f} "
              f"#>10={o['ngt']:2d} max={o['Lcal_max']:6.2f} zh med={o['zmed']:.3f} [{o['z05']:.3f},{o['z95']:.3f}] "
              f"logbias={o['zlogbias']:+.4f} cal-naive={o['cal_worse_naive']:+.2f} #B4<cal={o['nB4better']}")

# compare with summary.json
S = json.loads((R / "summary.json").read_text())
md = 0.0
for key, o in out.items():
    z, n = key.split("|")
    c = S["per_cell"][f"{float(z)!r}|{n}"]
    for a, b in (("Lcal", c["L_cal"]["mean"]), ("Lnaive", c["L_naive"]["mean"]), ("LB4", c["L_B4"]["mean"]),
                 ("zmed", c["zeta_hat"]["median"]), ("z05", c["zeta_hat"]["q05"]), ("z95", c["zeta_hat"]["q95"])):
        md = max(md, abs(o[a] - b))
    assert o["ngt"] == c["L_cal"]["n_gt_margin"], key
print("max abs diff vs summary.json:", md)
json.dump(out, open(R / "indep_G.json", "w"), indent=1)
