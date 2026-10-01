"""Experiment R: compile with the fitted halt law, replay in the declared (E-outage) world.

Mirror image of wrong_constants.py.
World = E-outage (declared halt law). Tuning then evaluation payments are drawn
with one rng = default_rng(seed) in the same order as run.py:run_outage, so the
per-episode block sums pair with the archived E-outage block sums
(results/archived_blocks/blocks_E-outage_F*_mid_s*.npz; per-policy block sums of the main run,
regenerated from the main-run result files, not shipped).

Policies, both replayed in the E-outage world with the world's own ex_pos:
  nominal  compile_outage(env_hat), env_hat = world with rho from q_hat
           (must reproduce the archived A2 block sums)
  wrong    compile_outage(replace(env_hat, p01=fitted.p01, p10=fitted.p10))

Run (from the package root):
       PYTHONPATH=src python3 scripts/r_cross.py run --flows F1,F2,F3
       PYTHONPATH=src python3 scripts/r_cross.py analyze
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from arena.experiments.settlement.core import rho_hat_from_q
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import (
    PI_GRID_OUTAGE,
    compile_outage,
    draw_outage_batch,
    replay_outage,
    survival,
)
from arena.experiments.settlement.outage import window_AD as outage_window_AD
from paths import ARCHIVED_BLOCKS, R_CROSS

OUT = R_CROSS
ARCH = ARCHIVED_BLOCKS
PPE = 50
SEEDS = {"F1": 7, "F2": 8, "F3": 9}
ACTS = {0: "grant", 1: "reject", 2: "verify", 3: "wait"}  # core.GRANT, REJECT, VERIFY, WAIT


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "run-log.txt", "a") as fh:
        fh.write(line + "\n")


def block_sums(pay, ppe=PPE):
    n = len(pay) // ppe * ppe
    return pay[:n].reshape(-1, ppe).sum(axis=1)


def arrival_actions(pol, d, H, chunk=500_000):
    """Action each policy takes at arrival (i=0, l=H, r=paths[:,0]).  Same lookup
    as CompiledOutagePolicy.__call__ (nearest log-v grid point, rounded pi bin)."""
    lg = np.log(pol.v_grid)
    T = np.stack([t[0, H] for t in pol.tables])  # (n_v, 2, n_pi)
    acts = np.empty(len(d), dtype=np.int8)
    for s in range(0, len(d), chunk):
        v = np.maximum(d.v[s:s + chunk].astype(float), 1e-12)
        iv = np.abs(np.log(v)[:, None] - lg[None, :]).argmin(axis=1)
        ip = np.round(d.pi0[s:s + chunk].astype(float) * (len(PI_GRID_OUTAGE) - 1)).astype(int)
        r = d.paths[s:s + chunk, 0].astype(int)
        acts[s:s + chunk] = T[iv, r, ip]
    return acts


def shares(acts, mask=None):
    a = acts if mask is None else acts[mask]
    return {name: float((a == code).mean()) if len(a) else float("nan") for code, name in ACTS.items()} | {"n": int(len(a))}


def run_flow(flow_name, n_tune, n_eval):
    seed = SEEDS[flow_name]
    envs = envs_for("mid")
    _, world, _ = envs["E-outage"]
    _, fitted, _ = envs["E-fitted"]
    flow = make_flows()[flow_name]
    rng = np.random.default_rng(seed)
    tune_d = draw_outage_batch(world, flow, n_tune, rng, payments_per_episode=PPE)
    eval_d = draw_outage_batch(world, flow, n_eval, rng, payments_per_episode=PPE)
    _, _, ex = outage_window_AD(world, survival(world))
    q_hat = float((tune_d.t_ans <= world.tau).mean())
    env_hat = replace(world, rho=rho_hat_from_q(q_hat, world.tau))
    env_wrong = replace(env_hat, p01=fitted.p01, p10=fitted.p10)
    del tune_d

    # only rho (and, for wrong, p01/p10) may differ from the world
    for f in dataclasses.fields(world):
        a, b, c = getattr(world, f.name), getattr(env_hat, f.name), getattr(env_wrong, f.name)
        if f.name != "rho":
            assert np.array_equal(a, b), f.name
        if f.name not in ("rho", "p01", "p10"):
            assert np.array_equal(a, c), f.name
    assert env_wrong.rho == env_hat.rho

    arch = np.load(ARCH / f"blocks_E-outage_{flow_name}_mid_s{seed}.npz", allow_pickle=True)
    a_meta = json.loads(str(arch["meta"]))
    counts = block_sums(np.ones(n_eval))
    assert np.array_equal(counts, arch["block_counts"]), "block_counts differ"

    t0 = time.time()
    pol_n = compile_outage(env_hat, "nominal")
    nominal = block_sums(replay_outage(world, eval_d, pol_n, ex))
    t_nom = time.time() - t0
    diff = np.abs(nominal - arch["A2"])
    rep = dict(max_abs_diff=float(diff.max()), exact=bool(np.array_equal(nominal, arch["A2"])),
               mean_nominal=float(nominal.sum() / counts.sum()),
               mean_A2_archived=float(arch["A2"].sum() / counts.sum()),
               archived_meanA2_meta=a_meta["means"]["A2"])
    log(f"{flow_name} nominal replay {t_nom:.0f}s  q_hat={q_hat:.6f}  repro max|diff|={rep['max_abs_diff']:.3e} exact={rep['exact']}")
    if not np.allclose(nominal, arch["A2"], rtol=0, atol=1e-8):
        log(f"{flow_name} REPRODUCTION FAILED; stopping")
        np.savez(OUT / f"r_E-outage_{flow_name}_FAILED.npz", nominal=nominal, block_counts=counts)
        sys.exit(2)

    pol_w = compile_outage(env_wrong, "wrong")
    wrong = block_sums(replay_outage(world, eval_d, pol_w, ex))
    t_all = time.time() - t0
    log(f"{flow_name} wrong replay done, total {t_all:.0f}s")

    halt = eval_d.paths[:, 0] == 1
    an, aw = arrival_actions(pol_n, eval_d, world.H), arrival_actions(pol_w, eval_d, world.H)
    act = {"nominal": {"all": shares(an), "halt": shares(an, halt), "normal": shares(an, ~halt)},
           "wrong": {"all": shares(aw), "halt": shares(aw, halt), "normal": shares(aw, ~halt)},
           "changed_share_all": float((an != aw).mean()),
           "changed_share_halt": float((an[halt] != aw[halt]).mean()) if halt.any() else float("nan"),
           "changed_share_normal": float((an[~halt] != aw[~halt]).mean())}
    # full table difference (all states, all v, all pi)
    tn, tw = np.stack(pol_n.tables), np.stack(pol_w.tables)
    act["table_cells_changed"] = float((tn != tw).mean())
    act["table_cells_changed_by_regime"] = [float((tn[..., r, :] != tw[..., r, :]).mean()) for r in (0, 1)]

    meta = dict(flow=flow_name, seed=seed, world="E-outage", compile_halt_law="E-fitted",
                n_tune=n_tune, n_eval=n_eval, ppe=PPE, q_hat=q_hat, rho_hat=env_hat.rho,
                world_p01=world.p01, world_p10=world.p10, fitted_p01=fitted.p01, fitted_p10=fitted.p10,
                repro=rep, actions=act, t_nominal_s=t_nom, t_total_s=t_all,
                archived_meta=a_meta)
    np.savez(OUT / f"r_E-outage_{flow_name}.npz", nominal=nominal, wrong=wrong, block_counts=counts,
             B4_refined=arch["B4_refined"], meta=json.dumps(meta))
    return meta


# ------------------------------------------------------------ analysis
# fixed before looking at results
NB, DEN = 10_000, 49.5
FITTED_WORLD_L = {"F1": 0.718, "F2": 0.175, "F3": 0.688}  # opposite direction, reported


def ci_all(ds, c, rng, chunk=50):
    """Paired: the same resamples for every row of one flow."""
    n = len(c)
    bs = np.empty((len(ds), NB))
    for j0 in range(0, NB, chunk):
        i = rng.integers(0, n, (chunk, n))
        den = c[i].sum(axis=1)
        for r, d in enumerate(ds):
            bs[r, j0:j0 + chunk] = d[i].sum(axis=1) / den
    out = []
    for r, d in enumerate(ds):
        lo, hi = np.percentile(bs[r], [2.5, 97.5])
        out.append([float(x) / DEN * 1e4 for x in (d.sum() / c.sum(), lo, hi)])
    return out


def analyze():
    rng = np.random.default_rng(20260929)
    summary = {"convention": "bp of $49.50 mean exposure; 95% paired block bootstrap, "
                             "ratio of sums, 10,000 resamples, percentile, seed 20260929; "
                             "blocks = 50-payment episodes; NI margin -10 bp on D",
               "flows": {}}
    for f in ("F1", "F2", "F3"):
        z = np.load(OUT / f"r_E-outage_{f}.npz", allow_pickle=True)
        c, n, w, b4 = (np.asarray(z[k], float) for k in ("block_counts", "nominal", "wrong", "B4_refined"))
        meta = json.loads(str(z["meta"]))
        L, D0, D = ci_all([n - w, n - b4, w - b4], c, rng)
        summary["flows"][f] = dict(L=L, D0=D0, D=D, NI_D=bool(D[1] > -10.0), NI_D0=bool(D0[1] > -10.0),
                                   q_hat=meta["q_hat"], repro=meta["repro"], actions=meta["actions"],
                                   t_total_s=meta["t_total_s"], fitted_world_L=FITTED_WORLD_L[f])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for f, v in summary["flows"].items():
        print(f, {k: [round(x, 3) for x in v[k]] for k in ("L", "D0", "D")}, "NI", v["NI_D"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--flows", default="F1,F2,F3")
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "run":
        for f in a.flows.split(","):
            log(f"{f} start")
            run_flow(f, a.n_tune, a.n_eval)
            log(f"{f} end")
    else:
        analyze()


if __name__ == "__main__":
    main()
