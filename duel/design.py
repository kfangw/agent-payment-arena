"""Frozen design constants shared across the experiment: seed bands, the
nine confirmatory cells, the equivalence margin, and the main-narrative
cells.  Values the pilot decides (EPS, block lengths, axis point counts)
are set here once, before the main run, and not changed after.
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

# The confirmatory family: three environments by three flows at the
# declared waiting-cost cell.  A2 - B1 in each is the only Holm-corrected
# comparison (spec 3.4).
NINE_CELLS = [f"{e} x {f}" for e in ENVS for f in FLOWS]

# Main-narrative cells for the injection sweep (spec 2.1).  F1 may move to
# F2 once the flow numbers settle; the sweep takes the cell as an argument.
MAIN_CELLS = ["E-outage x F1", "E-slow x F1"]

# Equivalence margin in dollars per payment, anchored to 1 bp of mean
# exposure (spec 5.2).  None until the pilot fixes the mean exposure; the
# aggregator then requires it to be passed explicitly.
EPS: float | None = None


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
