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

# Main-narrative cells for the injection sweep (spec 2.1), on the settled
# F2 flow.  The sweep takes the cell as an argument, so this is the default.
MAIN_CELLS = ["E-outage x F2", "E-slow x F2"]

# Equivalence margin in dollars per payment, 1 bp of the mean exposure
# (spec 5.2).  With the flow settled, the mean exposure is fixed: 2M draws
# give $49.50, so 1 bp is $0.004950 per payment ($4.95 per 1000).  Frozen
# here so the verdict threshold stands before the data, and matched by the
# 1-bp anchor the injection sweep computes per cell.
EPS: float = 0.00495


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
