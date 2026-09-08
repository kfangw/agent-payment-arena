"""Experiment-matrix and seed-split guards."""

from __future__ import annotations

from dataclasses import dataclass

from arena.experiments.relationship.config import SeedSplit, SplitConfig
from arena.experiments.relationship.model import Phase


@dataclass(frozen=True)
class SplitGuard:
    """Fail closed when a command requests a seed from the wrong phase."""

    splits: SplitConfig

    def definition(self, phase: Phase) -> SeedSplit:
        """Return the configured split for a phase."""
        if phase is Phase.PILOT:
            return self.splits.pilot
        if phase is Phase.TUNING:
            return self.splits.tuning
        if phase is Phase.EVALUATION:
            return self.splits.evaluation
        raise AssertionError(f"unhandled phase: {phase!r}")

    def require_seed(self, phase: Phase, seed: int) -> None:
        """Raise when a seed does not belong to the requested phase."""
        split = self.definition(phase)
        if not split.contains(seed):
            raise ValueError(
                f"seed {seed} is outside {phase.value} range [{split.seed_start}, {split.seed_end}]"
            )

    def seeds(self, phase: Phase) -> range:
        """Return the complete inclusive seed range for a phase."""
        split = self.definition(phase)
        return range(split.seed_start, split.seed_end + 1)
