"""Policy interface for repeated relationship experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Protocol

import numpy as np

from arena.experiments.artifacts import JsonValue, to_json_value
from arena.experiments.relationship.model import (
    PolicyContext,
    RelationshipAction,
    RelationshipObservation,
    RelationshipParameters,
)
from arena.experiments.relationship.value import (
    DEFAULT_VALUE_SOLVER_CONFIG,
    ValueSolution,
    ValueSolverConfig,
    action_at_belief,
    solve_value,
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


@dataclass
class CompiledDepositExitPolicy:
    """A deterministic buyer policy compiled from one Bellman solution."""

    parameters: RelationshipParameters
    solution: ValueSolution
    solver_config: ValueSolverConfig
    policy_id: ClassVar[str] = "compiled_deposit_exit"
    _ready: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject action maps with disconnected regions or an invalid exit tail."""
        actions = tuple(region.action for region in self.solution.regions)
        if len(actions) != len(set(actions)):
            raise ValueError("compiled action regions must be connected")
        if not actions or actions[-1] is not RelationshipAction.EXIT:
            raise ValueError("compiled policy must end with an exit region")

    def reset(self, context: PolicyContext, rng: np.random.Generator) -> None:
        """Validate the public terms and reset the stateless policy."""
        del rng
        if context.parameters != self.parameters:
            raise ValueError("runtime parameters do not match compiled policy")
        self._ready = True

    def act(self, observation: RelationshipObservation) -> RelationshipAction:
        """Choose from the public posterior using the compiled value function."""
        if not self._ready:
            raise RuntimeError("policy must be reset before use")
        return action_at_belief(self.parameters, self.solution, observation.belief_bad)

    def snapshot(self) -> Mapping[str, JsonValue]:
        """Return the complete deterministic compilation record."""
        snapshot = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "parameters": asdict(self.parameters),
            "solver": asdict(self.solver_config),
            "solution": {
                "residual": self.solution.residual,
                "iterations": self.solution.iterations,
                "exit_boundary": self.solution.exit_boundary,
                "tie_order": [action.value for action in _SNAPSHOT_TIE_ORDER],
                "belief_grid": self.solution.beliefs,
                "value_grid": self.solution.values,
                "action_grid": [action.value for action in self.solution.actions],
                "regions": [
                    {
                        "action": region.action.value,
                        "lower": region.lower,
                        "upper": region.upper,
                    }
                    for region in self.solution.regions
                ],
            },
        }
        value = to_json_value(snapshot)
        if not isinstance(value, dict):
            raise AssertionError("policy snapshot must be a mapping")
        return value


def compile_deposit_exit_policy(
    parameters: RelationshipParameters,
    solver_config: ValueSolverConfig = DEFAULT_VALUE_SOLVER_CONFIG,
) -> CompiledDepositExitPolicy:
    """Compile fixed public terms into a read-only relationship policy."""
    return CompiledDepositExitPolicy(
        parameters, solve_value(parameters, solver_config), solver_config
    )


_SNAPSHOT_TIE_ORDER = (
    RelationshipAction.EXIT,
    RelationshipAction.DEFER,
    RelationshipAction.VERIFY,
    RelationshipAction.PAY,
)
