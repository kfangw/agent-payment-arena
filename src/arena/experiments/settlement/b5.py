"""Reproduce the B5 regime-aware comparison from declared seeds.

B5 rejects payments that arrive while the public settlement regime is
halted. In the normal regime, it applies the refined B3 suspicion thresholds.
The runner draws the tuning and evaluation samples itself, so it does not
depend on an archived base-result file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .core import GRANT, REJECT, VERIFY, rho_hat_from_q
from .design import eps_for
from .flows import make_flows
from .gate import envs_for
from .grid import _best_ab, _select, _terminals_outage
from .outage import compile_outage, draw_outage_batch, replay_outage, survival, window_AD
from .policies import suspicion_grid
from .report import envelope, write_once
from .stats import boot_ci, perm_p, ratio_mean, units
from .watch import PAYMENTS_PER_EPISODE, block_sums

CELL_SEED = {"F1": 7, "F2": 8, "F3": 9}


@dataclass(frozen=True)
class StateMatchedOutagePolicy:
    """Reject in a halted regime and apply B3 thresholds otherwise."""

    lower: float
    upper: float

    def __call__(
        self, stage: int, remaining: int, regime: int, value: float, suspicion: float
    ) -> int:
        del stage, remaining, value
        if regime == 1:
            return REJECT
        if suspicion < self.lower:
            return GRANT
        if suspicion > self.upper:
            return REJECT
        return VERIFY


def run_cell(
    flow_name: str,
    *,
    seed: int,
    n_tune: int,
    n_eval: int,
    b3_n: int = 161,
    n_boot: int = 20_000,
    n_perm: int = 10_000,
    payments_per_episode: int = PAYMENTS_PER_EPISODE,
) -> tuple[dict[str, object], dict[str, object]]:
    """Draw, tune, replay, and summarize one E-outage flow."""
    kind, env, _ = envs_for("mid")["E-outage"]
    if kind != "outage":
        raise RuntimeError("E-outage did not resolve to an outage environment")
    flow = make_flows()[flow_name]
    rng = np.random.default_rng(seed)
    tune = draw_outage_batch(
        env, flow, n_tune, rng, payments_per_episode=payments_per_episode
    )
    evaluation = draw_outage_batch(
        env, flow, n_eval, rng, payments_per_episode=payments_per_episode
    )
    _, _, exercise = window_AD(env, survival(env))

    grid = suspicion_grid(b3_n)
    grant_t, reject_t, verify_t = _terminals_outage(env, tune, exercise)
    (lower, upper), tune_mean = _best_ab(
        grant_t, reject_t, verify_t, tune.pi0, grid
    )
    grant_e, reject_e, verify_e = _terminals_outage(env, evaluation, exercise)
    b3_payoff = _select(
        grant_e, reject_e, verify_e, evaluation.pi0, lower, upper
    )
    b5_payoff = replay_outage(
        env,
        evaluation,
        StateMatchedOutagePolicy(lower, upper),
        exercise,
    )

    q_hat = float((tune.t_ans <= env.tau).mean())
    rho_hat = rho_hat_from_q(q_hat, env.tau)
    a_policy = compile_outage(replace(env, rho=rho_hat), "A")
    a_payoff = replay_outage(env, evaluation, a_policy, exercise)

    episodes = np.arange(n_eval) // payments_per_episode
    counts = np.bincount(episodes).astype(float)
    a_sums = block_sums(a_payoff, episodes)
    b3_sums = block_sums(b3_payoff, episodes)
    b5_sums = block_sums(b5_payoff, episodes)
    diff = a_sums - b5_sums
    mean_exposure = float(evaluation.v.mean())
    mean = ratio_mean(diff, counts)
    payload: dict[str, object] = {
        "policy": {
            "name": "B5",
            "lower": lower,
            "upper": upper,
            "halted_action": "reject",
            "normal_action": "B3",
            "b3_grid_points": b3_n,
            "b3_grid_candidates": len(grid),
        },
        "calibration": {"q_hat": q_hat, "rho_hat": rho_hat, "b3_tune_mean": tune_mean},
        "mean_exposure": mean_exposure,
        "means": {
            "A": ratio_mean(a_sums, counts),
            "B3_refined": ratio_mean(b3_sums, counts),
            "B5": ratio_mean(b5_sums, counts),
        },
        "a_minus_b5": {
            "mean": mean,
            "per_cell_ci95": boot_ci(diff, counts, n_boot=n_boot, seed=seed, level=0.95),
            "simultaneous_ci95_three_cells": boot_ci(
                diff, counts, n_boot=n_boot, seed=seed, level=1 - 0.05 / 3
            ),
            "bp": units(mean, mean_exposure)["bp"],
            "permutation_p": perm_p(diff, counts, n_perm=n_perm, seed=seed + 1),
            "equivalence_margin": eps_for(mean_exposure),
        },
        "block_counts": counts,
        "a_block_sums": a_sums,
        "b3_block_sums": b3_sums,
        "b5_block_sums": b5_sums,
        "a_minus_b5_block_sums": diff,
    }
    parameters: dict[str, object] = {
        "environment": "E-outage",
        "flow": flow_name,
        "cw": "mid",
        "seed": seed,
        "n_tune": n_tune,
        "n_eval": n_eval,
        "payments_per_episode": payments_per_episode,
        "b3_grid_points": b3_n,
        "bootstrap_replicates": n_boot,
        "permutation_replicates": n_perm,
    }
    return payload, parameters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True, choices=tuple(CELL_SEED))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-eval", type=int, default=5_663_400)
    parser.add_argument("--n-tune", type=int, default=200_000)
    parser.add_argument("--b3-n", type=int, default=161)
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("results_b5"))
    args = parser.parse_args(argv)
    seed = CELL_SEED[args.flow] if args.seed is None else args.seed
    payload, parameters = run_cell(
        args.flow,
        seed=seed,
        n_tune=args.n_tune,
        n_eval=args.n_eval,
        b3_n=args.b3_n,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
    )
    obj = envelope(
        "b5",
        f"E-outage x {args.flow}",
        seed,
        args.n_eval,
        args.n_tune,
        parameters,
        payload,
    )
    path = args.out / f"b5_E-outage_{args.flow}_mid_s{seed}.json"
    write_once(path, obj)
    print(json.dumps({"cell": obj["cell"], **payload["means"], "a_minus_b5": payload["a_minus_b5"]}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
