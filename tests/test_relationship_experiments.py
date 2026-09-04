"""Tests for protocol phase isolation."""

import pytest

from arena.experiments.relationship.config import ProtocolConfig, load_protocol
from arena.experiments.relationship.experiments import SplitGuard
from arena.experiments.relationship.model import Phase


def _guard() -> SplitGuard:
    return SplitGuard(load_protocol("configs/relationship/protocol-0.3.1.yaml").splits)


@pytest.mark.parametrize(
    ("phase", "first", "last", "count"),
    [
        (Phase.PILOT, 1_000, 1_029, 30),
        (Phase.TUNING, 10_000, 10_049, 50),
        (Phase.EVALUATION, 20_000, 20_099, 100),
    ],
)
def test_split_ranges_are_inclusive(phase: Phase, first: int, last: int, count: int) -> None:
    seeds = _guard().seeds(phase)

    assert seeds.start == first
    assert seeds.stop - 1 == last
    assert len(seeds) == count


def test_split_guard_rejects_cross_phase_seed() -> None:
    guard = _guard()

    guard.require_seed(Phase.PILOT, 1_000)
    with pytest.raises(ValueError, match="outside evaluation range"):
        guard.require_seed(Phase.EVALUATION, 1_000)


def test_overlapping_seed_ranges_are_rejected() -> None:
    raw = load_protocol("configs/relationship/protocol-0.3.1.yaml").model_dump(mode="json")
    raw["splits"]["tuning"]["seed_start"] = 1_020

    with pytest.raises(ValueError, match="seed ranges overlap"):
        ProtocolConfig.model_validate(raw)
