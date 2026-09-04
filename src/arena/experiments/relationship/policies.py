"""Policy interface for repeated relationship experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from arena.experiments.artifacts import JsonValue
from arena.experiments.relationship.model import (
    PolicyContext,
    RelationshipAction,
    RelationshipObservation,
)


class RelationshipPolicy(Protocol):
    """A buyer policy that can observe public relationship state only."""

    policy_id: str

    def reset(self, context: PolicyContext, rng: np.random.Generator) -> None:
        """Reset all private state before a relationship."""
        ...

    def act(self, observation: RelationshipObservation) -> RelationshipAction:
        """Choose one action from the public observation."""
        ...

    def snapshot(self) -> Mapping[str, JsonValue]:
        """Return every parameter that fixes policy behavior."""
        ...
