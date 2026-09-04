"""Tests for scenario construction and authorization ground truth."""

from dataclasses import asdict

from arena.scenarios import minimum_suite

MERCHANT = "0x" + "11" * 20
ATTACKER = "0x" + "22" * 20


def test_minimum_suite_pairs_benign_and_attack_cases() -> None:
    benign, attack = minimum_suite(MERCHANT, ATTACKER)

    assert benign.task == attack.task
    assert benign.ground_truth.authorizes(MERCHANT, 25)
    assert not attack.ground_truth.authorizes(ATTACKER, 25)
    assert attack.attack_family == "direct_prompt_injection"


def test_ground_truth_is_not_nested_in_agent_visible_resource() -> None:
    _, attack = minimum_suite(MERCHANT, ATTACKER)

    assert "ground_truth" not in asdict(attack.resource)
