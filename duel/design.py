"""Frozen design constants shared across the experiment: seed bands, the
nine confirmatory cells, the equivalence margin, and the main-narrative
cells.  These are set once, before the main run, and not changed after.

The equivalence margin does not come from the pilot.  It is anchored to
the exposure scale, so it is known as soon as the flow is fixed, and the
pilot only sizes the sample that resolves it.  Keeping the anchor here
means a single place decides it for every consumer.
"""

from __future__ import annotations

# Non-overlapping seed bands (spec 1.2).  Pilot data never enters the
# main analysis, so its band is disjoint from the others.
SEED_BANDS: dict[str, tuple[int, int]] = {
    "exp1": (1, 999),
    "pilot": (1000, 1999),
    "exp2": (2000, 2999),
    "verify": (9000, 9999),
}

ENVS = ("E-fast", "E-slow", "E-outage")
FLOWS = ("F1", "F2", "F3")

# Stable environment, flow, and seed assignments for the confirmatory runs.
CONFIRMATORY_RUNS = tuple(
    (env, flow, seed)
    for seed, (env, flow) in enumerate(
        ((env, flow) for env in ENVS for flow in FLOWS),
        start=1,
    )
)

# The confirmatory family: three environments by three flows at the
# declared waiting-cost cell.  A2 - B1 in each is the only Holm-corrected
# comparison (spec 3.4).
NINE_CELLS = [f"{env} x {flow}" for env, flow, _ in CONFIRMATORY_RUNS]

# Main-narrative cells for the injection sweep.  The suspicion-blind
# flows leave the verify band nearly empty, so the cells that carry the
# verification story are the thick-middle ones; the sweep still takes the
# cell as an argument.
MAIN_CELLS = ["E-outage x F2", "E-slow x F2"]

# Equivalence margin, in basis points of mean exposure.  A per-payment
# edge below this cannot move an operator's decision: the smallest
# processing fee on a payment of this size is two orders of magnitude
# larger.  One value for every cell, so verdicts stay comparable.
EPS_BP = 10.0

# Mean exposure under the frozen flow (log-normal, median 30, sigma 1.0,
# clipped to [0.5, 2000]), measured on 2e6 draws.
MEAN_EXPOSURE_USD = 49.50

# The margin itself, in dollars per payment.
EPS = EPS_BP * 1e-4 * MEAN_EXPOSURE_USD


def eps_for(mean_exposure: float) -> float:
    """The margin at a measured mean exposure.  Callers that have the
    realised exposure of their own batch use this so the anchor tracks
    the batch rather than the declared average."""
    return EPS_BP * 1e-4 * mean_exposure


def in_band(seed: int, name: str) -> bool:
    """Whether a seed sits in the named band."""
    lo, hi = SEED_BANDS[name]
    return lo <= seed <= hi


def band_of(seed: int) -> str | None:
    """The band a seed belongs to, or None if it is outside every band."""
    for name, (lo, hi) in SEED_BANDS.items():
        if lo <= seed <= hi:
            return name
    return None
