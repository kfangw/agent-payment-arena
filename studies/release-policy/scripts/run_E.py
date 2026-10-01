"""Experiment E: the compiled release policy on measured x402 payment sizes.
Same world, table, policies, evaluation seed and metrics as experiment H (scripts/run_H.py).
Only the exposure law changes: L6 and L7 are empirical laws built from an on-chain histogram
(data/x402_amounts/hist.json; amounts rounded to 3 significant figures), sampled by inverse transform
over the point masses and clipped to [0.01, 2000].  Spec: docs/spec-E.md.

Run from the package root:
    export PYTHONPATH=$PWD/src
    python3 scripts/run_E.py smoke          (100,000 evaluation payments, tag smoke)
    python3 scripts/run_E.py run            (5,663,400 payments, resumable)
    python3 scripts/analyze_E.py [--tag smoke --nb 100]

B4_L is tuned on 200,000 learning payments from default_rng([111, 700, L]) with L = 6 (L6) or 7 (L7).
B4_L0 is experiment H's L0 tuning (results/exposure_law/plan/b4_L0.json, copied, not re-tuned).
Output goes to results/measured_streams/ (paths.MEASURED_STREAMS); the table cache is results/cache/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths                                               # noqa: E402
import run_H as RH                                         # noqa: E402
from arena.experiments.settlement.flows import make_flows  # noqa: E402

DATA = paths.X402_AMOUNTS / "hist.json"
LO, HI = 0.01, 2000.0
WINDOWS = {"L6": "A_all90d", "L7": "B_cb90d"}
LAW_INDEX = {"L6": 6, "L7": 7}


@dataclass(frozen=True)
class EmpiricalV:
    """Exposure in dollars: point masses at the histogram amounts, weights n / N, clipped to [lo, hi]."""
    points: tuple
    weights: tuple
    lo: float = LO
    hi: float = HI

    def sample(self, n, rng):
        pts = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        v = rng.choice(pts, size=n, p=w / w.sum())
        return np.clip(v, self.lo, self.hi)


def load_hist(win):
    rows = [r for r in json.loads(DATA.read_text())["rows"] if r["win"] == win]
    rows.sort(key=lambda r: r["amt3"])
    a = np.array([r["amt3"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    return a, n


_LAW_CACHE = {}


def make_law(name):
    if name not in _LAW_CACHE:
        a, n = load_hist(WINDOWS[name])
        _LAW_CACHE[name] = EmpiricalV(tuple(a.tolist()), tuple(n.tolist()))
    return _LAW_CACHE[name]


def law_params(name):
    if name not in WINDOWS:
        return RH_law_params(name)
    a, n = load_hist(WINDOWS[name])
    p = n / n.sum()
    c = np.clip(a, LO, HI)
    cdf = np.cumsum(p)
    med = float(c[np.searchsorted(cdf, 0.5)])
    return dict(name=name, kind="empirical", window=WINDOWS[name], n_points=int(len(a)), n_payments=int(n.sum()),
                mean_target=float((c * p).sum()), mean_unclipped=float((a * p).sum()), sigma=float("nan"),
                median=med, lo=LO, hi=HI,
                share_count_below_lo=float(p[a < LO].sum()), share_count_above_hi=float(p[a > HI].sum()))


RH_law_params = RH.law_params


def flow_for(name):
    if name not in WINDOWS:
        return replace(make_flows()[RH.FLOW], vdist=RH.make_law(name))
    return replace(make_flows()[RH.FLOW], vdist=make_law(name))


# Point experiment H's evaluation code at the empirical laws.
RH.flow_for = flow_for
RH.law_params = law_params
RH.LAW_INDEX = {**RH.LAW_INDEX, **LAW_INDEX}


def prepare(run_dir, log):
    """B4_L0 plan of experiment H, copied and never re-tuned (the table cache is built by run_laws)."""
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    src_p = paths.EXPOSURE_LAW / "plan" / "b4_L0.json"
    dst_p = run_dir / "plan" / "b4_L0.json"
    if not dst_p.exists():
        if not src_p.exists():
            log("B4_L0 plan not found in results/exposure_law/plan; L0 will be re-tuned with H's seed")
            return None
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_p, dst_p)
        b = json.loads(dst_p.read_text())["b4"]
        log(f"B4_L0 from results/exposure_law/plan/b4_L0.json k={b['k']} a={b['a']:.5f} b={b['b']:.5f} tune_mean={b['tune_mean']:.6g}")
    return json.loads(dst_p.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "run"])
    ap.add_argument("--laws", default="L6,L7")
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    tag = a.tag if a.tag is not None else ("smoke" if a.phase == "smoke" else None)
    run_dir = paths.MEASURED_STREAMS / tag if tag else paths.MEASURED_STREAMS
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = (run_dir / "progress.log").open("a", buffering=1)
    T0 = time.time()

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [{(time.time() - T0) / 60:6.1f}m] {msg}"
        prog.write(line + "\n")
        print(line, flush=True)

    laws = a.laws.split(",")
    n_eval = a.n_eval or (100_000 if a.phase == "smoke" else RH.N_EVAL)
    log(f"START phase={a.phase} tag={tag} laws={laws} n_eval={n_eval} pid={os.getpid()}")
    for L in laws:
        p = law_params(L)
        log(f"LAW {L} window={p['window']} points={p['n_points']} payments={p['n_payments']} "
            f"mean_clipped(exact)={p['mean_target']:.6f} mean_unclipped={p['mean_unclipped']:.6f} median={p['median']:g} "
            f"below_lo={p['share_count_below_lo']:.4f} above_hi={p['share_count_above_hi']:.2e}")
    prepare(run_dir, log)
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        laws=laws, n_eval=n_eval, eval_seed=RH.EVAL_SEED, learn_rng=[RH.EVAL_SEED, RH.LEARN_TAG, LAW_INDEX],
        n_learn=RH.N_LEARN, table_hash=RH.REF_HASH, policies=RH.POLICIES, floor=RH.FLOOR, vstar=RH.VSTAR,
        b4L0_source="experiment H results/exposure_law/plan/b4_L0.json", data=str(DATA.name),
        data_source=json.loads(DATA.read_text())["source"],
        law_params={n: law_params(n) for n in laws}, started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    fails, el = RH.run_laws(run_dir, laws, n_eval, log)
    if fails:
        log(f"INCOMPLETE failed={fails} (see errors.log; rerun to resume)")
        return 1
    log(f"ALLDONE {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={el / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
