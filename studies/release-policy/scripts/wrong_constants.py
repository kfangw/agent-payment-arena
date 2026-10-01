"""Evaluate policies compiled from wrong constants in the E-fitted world.

The world is E-fitted (halt law fitted to Base batch-inbox gaps). The tuning and
evaluation draws are generated exactly as in run.py (same seed, same order), so
per-episode block sums pair with the E-fitted settlement and B4 result files.

Policies, all replayed in the E-fitted world:
  true      compiled from the world's own constants (q estimated on the tuning split)
  declared  compiled with E-outage's declared halt law in place of the fitted one
  eta<s>    compiled with each stage hazard and the answer rate multiplied by an
            independent lognormal factor exp(s z), averaged over REPLICATES draws;
            the replicate policies are replayed on the first n_sub evaluation
            payments (whole episodes) to bound run time.

Run (from the package root; one call per flow, seeds 7, 8, 9 for F1, F2, F3):
    PYTHONPATH=src python3 scripts/wrong_constants.py --flow F2 --seed 8
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from arena.experiments.settlement.core import rho_hat_from_q
from arena.experiments.settlement.flows import make_flows
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import compile_outage, draw_outage_batch, replay_outage, survival
from arena.experiments.settlement.outage import window_AD as outage_window_AD
from paths import WRONG_CONSTANTS

PPE = 50


def block_sums(pay, ppe=PPE):
    n = len(pay) // ppe * ppe
    return pay[:n].reshape(-1, ppe).sum(axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, choices=["F1", "F2", "F3"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-eval", type=int, default=5_663_400)
    ap.add_argument("--n-tune", type=int, default=200_000)
    ap.add_argument("--n-sub", type=int, default=1_000_000)
    ap.add_argument("--levels", default="0.1,0.2")
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--out", default=str(WRONG_CONSTANTS))
    a = ap.parse_args(argv)

    envs = envs_for("mid")
    _, world, _ = envs["E-fitted"]
    _, declared, _ = envs["E-outage"]
    flow = make_flows()[a.flow]
    rng = np.random.default_rng(a.seed)
    tune_d = draw_outage_batch(world, flow, a.n_tune, rng, payments_per_episode=PPE)
    eval_d = draw_outage_batch(world, flow, a.n_eval, rng, payments_per_episode=PPE)
    _, _, ex = outage_window_AD(world, survival(world))
    q_hat = float((tune_d.t_ans <= world.tau).mean())
    env_hat = replace(world, rho=rho_hat_from_q(q_hat, world.tau))
    log = {"q_hat": q_hat}
    out = {}
    t0 = time.time()

    pol = compile_outage(env_hat, "true")
    out["true"] = block_sums(replay_outage(world, eval_d, pol, ex))
    pol = compile_outage(replace(env_hat, p01=declared.p01, p10=declared.p10), "declared")
    out["declared"] = block_sums(replay_outage(world, eval_d, pol, ex))
    log["t_full"] = time.time() - t0
    print(f"full-size policies done in {log['t_full']:.0f}s", flush=True)

    n_sub = a.n_sub // PPE * PPE
    sub = type(eval_d)(**{k: (v[:n_sub] if getattr(v, "shape", None) and v.shape[0] == len(eval_d) else v)
                          for k, v in eval_d.__dict__.items()})
    out["true_sub"] = block_sums(replay_outage(world, sub, compile_outage(env_hat, "true"), ex))
    for s in [float(x) for x in a.levels.split(",")]:
        acc = np.zeros(n_sub)
        for r in range(a.replicates):
            g = np.random.default_rng([a.seed, 700, r, int(s * 1000)])
            z = g.standard_normal(len(world.f))
            f_hat = np.clip(world.f * np.exp(s * z), 0.0, 1.0)
            qh = float(np.clip(q_hat * np.exp(s * g.standard_normal()), 0.0, 1.0))
            p = compile_outage(replace(world, f=f_hat, rho=rho_hat_from_q(qh, world.tau)), f"eta{s}")
            acc += replay_outage(world, sub, p, ex)
        out[f"eta{s:g}"] = block_sums(acc / a.replicates)
        print(f"eta {s} done at {time.time() - t0:.0f}s", flush=True)
    log["t_total"] = time.time() - t0
    payload = {k: v.tolist() for k, v in out.items()}
    payload.update(log=log, n_eval=a.n_eval, n_sub=n_sub, seed=a.seed, flow=a.flow,
                   replicates=a.replicates, world="E-fitted")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"wrong_E-fitted_{a.flow}_s{a.seed}.json").write_text(json.dumps(payload))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
