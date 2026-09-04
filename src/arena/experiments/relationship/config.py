"""Validated configuration for repeated relationship experiments."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arena.experiments.relationship.model import DataGrade, SellerMode


class FrozenModel(BaseModel):
    """Base class for immutable, closed experiment schemas."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DiscountingConfig(FrozenModel):
    """How relationship survival implements geometric discounting."""

    mechanism: Literal["realized_geometric_departure"]
    apply_time_weights_to_realized_path: Literal[False]
    seed_aggregation: Literal["ratio_of_sums"]


class EquivalenceMargin(FrozenModel):
    """The practical-equivalence threshold."""

    expression: str
    canonical_value: float = Field(gt=0)


class CanonicalParameters(FrozenModel):
    """The generated reference point shared by experiment cells."""

    v: float = Field(gt=0)
    u: float = Field(gt=0)
    alpha: float = Field(ge=0, lt=1)
    beta: float = Field(ge=0, lt=1)
    behavioral_cheat_rate: float = Field(ge=0, le=1)
    g: float = Field(ge=0)
    deposit_cap: float = Field(ge=0)
    prior_bad: float = Field(gt=0, lt=1)
    escrow_cost_rate: float = Field(ge=0)
    seller_margin: float = Field(ge=0)

    @model_validator(mode="after")
    def predicate_is_informative(self) -> Self:
        """Reject a predicate whose failure does not identify more bad sellers."""
        if 1 - self.beta <= self.alpha:
            raise ValueError("predicate must satisfy 1 - beta > alpha")
        return self


class OperatingPoint(FrozenModel):
    """One verification-cost ratio."""

    verification_cost_ratio: float = Field(gt=0)


class RelationshipLength(FrozenModel):
    """One geometric relationship-length setting."""

    continuation_probability: float = Field(ge=0, lt=1)
    expected_transactions: int = Field(gt=0)

    @model_validator(mode="after")
    def expectation_matches_probability(self) -> Self:
        """Keep the human-readable expected length consistent with survival."""
        expected = 1 / (1 - self.continuation_probability)
        if abs(expected - self.expected_transactions) > 1e-9:
            raise ValueError("expected_transactions must equal 1 / (1 - continuation_probability)")
        return self


class PolicyConfig(FrozenModel):
    """Stable policy identifiers and confirmatory eligibility."""

    tested: str
    eligible_primary_competitors: tuple[str, ...]
    diagnostic: tuple[str, ...]

    @model_validator(mode="after")
    def policy_ids_are_distinct(self) -> Self:
        """Reject duplicate or overlapping policy roles."""
        identifiers = (
            self.tested,
            *self.eligible_primary_competitors,
            *self.diagnostic,
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("policy identifiers must be unique across roles")
        return self


class SeedSplit(FrozenModel):
    """An inclusive seed range and its sample size."""

    seed_start: int = Field(ge=0)
    seed_end: int = Field(ge=0)
    relationships_per_seed_cell: int = Field(gt=0)

    @model_validator(mode="after")
    def range_is_ordered(self) -> Self:
        """Reject an empty inclusive range."""
        if self.seed_end < self.seed_start:
            raise ValueError("seed_end must be greater than or equal to seed_start")
        return self

    def contains(self, seed: int) -> bool:
        """Return whether a seed belongs to this split."""
        return self.seed_start <= seed <= self.seed_end


class SplitConfig(FrozenModel):
    """The three disjoint protocol phases."""

    pilot: SeedSplit
    tuning: SeedSplit
    evaluation: SeedSplit

    @model_validator(mode="after")
    def ranges_are_disjoint(self) -> Self:
        """Prevent information leakage through overlapping seeds."""
        ranges = (
            ("pilot", self.pilot),
            ("tuning", self.tuning),
            ("evaluation", self.evaluation),
        )
        for index, (left_name, left) in enumerate(ranges):
            for right_name, right in ranges[index + 1 :]:
                if max(left.seed_start, right.seed_start) <= min(left.seed_end, right.seed_end):
                    raise ValueError(f"{left_name} and {right_name} seed ranges overlap")
        return self


class InferenceConfig(FrozenModel):
    """Confirmatory resampling and multiplicity settings."""

    bootstrap_resamples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    simultaneous_primary_cells: int = Field(gt=0)
    primary_correction: Literal["bonferroni"]
    sensitivity_test: Literal["paired_sign_flip"]
    sensitivity_correction: Literal["holm"]


class AssumptionViolations(FrozenModel):
    """One-at-a-time robustness perturbations."""

    type_switch_probability: tuple[float, ...]
    verdict_delay: tuple[int, ...]
    partial_delivery_fraction: tuple[float, ...]
    verification_error_multiplier: tuple[float, ...]


class DataGradeLabels(FrozenModel):
    """Human-readable labels for input provenance."""

    G: Literal["generated"]
    P: Literal["parameterized"]
    M: Literal["measured"]


class ProtocolConfig(FrozenModel):
    """The complete frozen repeated-relationship protocol."""

    schema_version: int = Field(gt=0)
    protocol_version: str
    status: Literal["frozen"]
    frozen_at: date
    analysis_unit: Literal["relationship"]
    primary_outcome: Literal["geometric_departure_buyer_utility_per_transaction"]
    discounting: DiscountingConfig
    equivalence_margin: EquivalenceMargin
    canonical: CanonicalParameters
    operating_points: dict[str, OperatingPoint]
    relationship_lengths: dict[str, RelationshipLength]
    seller_modes: tuple[SellerMode, ...]
    policies: PolicyConfig
    splits: SplitConfig
    inference: InferenceConfig
    assumption_violations: AssumptionViolations
    data_grades: DataGradeLabels

    @model_validator(mode="after")
    def matrix_matches_confirmatory_count(self) -> Self:
        """Keep the declared primary-cell correction equal to the matrix size."""
        expected = (
            len(self.operating_points) * len(self.relationship_lengths) * len(self.seller_modes)
        )
        if self.inference.simultaneous_primary_cells != expected:
            raise ValueError(
                "simultaneous_primary_cells must equal operating points "
                "x relationship lengths x seller modes"
            )
        if set(self.operating_points) != {"H", "L"}:
            raise ValueError("operating points must be exactly H and L")
        if set(self.relationship_lengths) != {"short", "medium", "long"}:
            raise ValueError("relationship lengths must be short, medium, and long")
        if set(self.seller_modes) != {SellerMode.BEHAVIORAL, SellerMode.STRATEGIC}:
            raise ValueError("seller modes must contain behavioral and strategic")
        return self


def load_protocol(path: str | Path) -> ProtocolConfig:
    """Load and validate a frozen YAML protocol."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        raw: object = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("protocol root must be a mapping")
    return ProtocolConfig.model_validate(raw)


def grade_label(grade: DataGrade, labels: DataGradeLabels) -> str:
    """Return the configured label for an input grade."""
    if grade is DataGrade.GENERATED:
        return labels.G
    if grade is DataGrade.PARAMETERIZED:
        return labels.P
    if grade is DataGrade.MEASURED:
        return labels.M
    raise AssertionError(f"unhandled data grade: {grade!r}")
