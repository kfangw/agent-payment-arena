"""Tests for the frozen repeated-relationship protocol."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from arena.experiments.relationship.config import ProtocolConfig, grade_label, load_protocol
from arena.experiments.relationship.model import DataGrade, SellerMode

PROTOCOL = Path("configs/relationship/protocol-0.3.1.yaml")


def test_checked_in_protocol_loads_with_frozen_invariants() -> None:
    config = load_protocol(PROTOCOL)

    assert config.protocol_version == "0.3.1"
    assert config.discounting.apply_time_weights_to_realized_path is False
    assert config.discounting.seed_aggregation == "ratio_of_sums"
    assert config.inference.simultaneous_primary_cells == 12
    assert config.seller_modes == (SellerMode.BEHAVIORAL, SellerMode.STRATEGIC)
    assert config.splits.evaluation.seed_start == 20_000
    assert config.splits.evaluation.seed_end == 20_099


def test_protocol_models_are_immutable() -> None:
    config = load_protocol(PROTOCOL)

    with pytest.raises(ValidationError, match="frozen"):
        config.protocol_version = "changed"


def test_uninformative_predicate_is_rejected() -> None:
    raw = load_protocol(PROTOCOL).model_dump(mode="json")
    raw["canonical"]["alpha"] = 0.8
    raw["canonical"]["beta"] = 0.3

    with pytest.raises(ValidationError, match="1 - beta > alpha"):
        ProtocolConfig.model_validate(raw)


def test_primary_cell_count_must_match_matrix() -> None:
    raw = load_protocol(PROTOCOL).model_dump(mode="json")
    raw["inference"]["simultaneous_primary_cells"] = 11

    with pytest.raises(ValidationError, match="simultaneous_primary_cells"):
        ProtocolConfig.model_validate(raw)


def test_protocol_root_must_be_a_mapping(tmp_path: Path) -> None:
    source = tmp_path / "invalid.yaml"
    source.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_protocol(source)


def test_grade_labels_are_resolved_without_policy_logic() -> None:
    labels = load_protocol(PROTOCOL).data_grades

    assert grade_label(DataGrade.GENERATED, labels) == "generated"
    assert grade_label(DataGrade.PARAMETERIZED, labels) == "parameterized"
    assert grade_label(DataGrade.MEASURED, labels) == "measured"
