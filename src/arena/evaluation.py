"""Repeated evaluation runner for the minimum offline suite."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from eth_account import Account

from arena.agents.constrained import SchemaConstrainedAgent
from arena.agents.protocol import EvaluationAgent
from arena.agents.scripted import ContentFollowingAgent, ScriptedAgent
from arena.delegator.model import Delegator, SigningDelegator
from arena.experiments.artifacts import git_revision
from arena.gateway.fake import FakeGateway
from arena.gateway.protocol import AcceptPolicy
from arena.gateway.schemas import Mandate
from arena.gateway.signatures import sign_mandate
from arena.loop import run_scenario
from arena.mcp_server.tools import PaymentTools
from arena.payment import PaymentAuthority
from arena.policies.payment import AlwaysVerifyPolicy, AskAbovePolicy
from arena.scenarios import Scenario, attack_catalog, minimum_suite
from arena.scoring import Metrics, score

MERCHANT = "0x" + "11" * 20
ATTACKER = "0x" + "22" * 20
TOKEN = "0x" + "33" * 20
DELEGATOR_KEY = "0x" + "01" * 32
AGENT_KEY = "0x" + "02" * 32


def run_mcp_demo(seed: int = 1) -> tuple[str, ...]:
    """Exercise fetch, pay, ask, and confirmed pay through the MCP tool surface."""
    scenario = minimum_suite(MERCHANT, ATTACKER)[0]
    now = datetime(2026, 1, 1, tzinfo=UTC)
    delegator = SigningDelegator(DELEGATOR_KEY, 1337)
    tools = PaymentTools(
        scenario.resource, _authority(scenario, AskAbovePolicy(20), seed, delegator), now
    )
    fetched = tools.fetch_resource()
    terms = cast(dict[str, object], fetched["paymentRequired"])
    nonce_key = "mcp-demo"
    first = tools.pay(str(terms["payTo"]), int(cast(str, terms["maxAmountRequired"])), nonce_key)
    confirmation = tools.ask_delegator(scenario.resource.amount, nonce_key)
    if confirmation is None:
        raise RuntimeError("demo delegator did not return a confirmation")
    second = tools.pay(
        scenario.resource.payee,
        scenario.resource.amount,
        nonce_key,
        confirmation,
    )
    return first.action.value, second.action.value


@dataclass(frozen=True)
class EvaluationRecord:
    """Metrics for one scenario, subject, policy, and repetition."""

    scenario_id: str
    agent_id: str
    policy_id: str
    repetition: int
    seed: int
    metrics: Metrics


@dataclass(frozen=True)
class EvaluationResult:
    """All records produced by one suite execution."""

    suite: str
    repetitions: int
    seed: int
    created_utc: str
    code_revision: str | None
    agent_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    suite_version: str
    records: tuple[EvaluationRecord, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible result mapping."""
        return asdict(self)


def load_result(path: Path) -> EvaluationResult:
    """Load an evaluation artifact written by the CLI."""
    raw = cast(dict[str, object], json.loads(path.read_text()))
    records_raw = cast(list[dict[str, object]], raw["records"])
    records = []
    for item in records_raw:
        metric_values = cast(dict[str, object], item["metrics"])
        metrics = Metrics(**metric_values)  # type: ignore[arg-type]
        records.append(
            EvaluationRecord(
                str(item["scenario_id"]),
                str(item["agent_id"]),
                str(item["policy_id"]),
                int(cast(int, item["repetition"])),
                int(cast(int, item["seed"])),
                metrics,
            )
        )
    return EvaluationResult(
        str(raw["suite"]),
        int(cast(int, raw["repetitions"])),
        int(cast(int, raw["seed"])),
        str(raw["created_utc"]),
        None if raw["code_revision"] is None else str(raw["code_revision"]),
        tuple(cast(list[str], raw["agent_ids"])),
        tuple(cast(list[str], raw["model_ids"])),
        tuple(cast(list[str], raw["scenario_ids"])),
        str(raw["suite_version"]),
        tuple(records),
    )


def run_minimum_suite(repetitions: int, seed: int = 1) -> EvaluationResult:
    """Run the offline minimum suite across baseline agents and policies."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    scenarios = minimum_suite(MERCHANT, ATTACKER)
    agents: tuple[EvaluationAgent, ...] = (
        ScriptedAgent(frozenset({MERCHANT}), 25),
        ContentFollowingAgent(),
    )
    return _run_suite("minimum", "1", scenarios, agents, repetitions, seed)


def run_attack_suite(
    repetitions: int,
    seed: int = 1,
    *,
    delegator_factory: Callable[[int], Delegator] | None = None,
) -> EvaluationResult:
    """Run the complete attack catalog with deterministic defense controls."""
    vulnerable = ContentFollowingAgent()
    agents: tuple[EvaluationAgent, ...] = (
        ScriptedAgent(frozenset({MERCHANT}), 25),
        vulnerable,
        SchemaConstrainedAgent(vulnerable, frozenset({MERCHANT}), 25),
    )
    return _run_suite(
        "attack-catalog",
        "1",
        attack_catalog(MERCHANT, ATTACKER),
        agents,
        repetitions,
        seed,
        delegator_factory,
    )


def _run_suite(
    suite: str,
    suite_version: str,
    scenarios: tuple[Scenario, ...],
    agents: tuple[EvaluationAgent, ...],
    repetitions: int,
    seed: int,
    delegator_factory: Callable[[int], Delegator] | None = None,
) -> EvaluationResult:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    policies: tuple[tuple[str, AcceptPolicy], ...] = (
        ("always-verify", AlwaysVerifyPolicy()),
        ("ask-above-20", AskAbovePolicy(20)),
    )
    records: list[EvaluationRecord] = []
    delegators: dict[tuple[int, str, str], Delegator] = {}
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for repetition in range(repetitions):
        repetition_seed = seed + repetition
        for scenario in scenarios:
            for agent in agents:
                for policy_id, policy in policies:
                    key = repetition, agent.agent_id, policy_id
                    delegator = delegators.setdefault(
                        key,
                        delegator_factory(repetition_seed)
                        if delegator_factory
                        else SigningDelegator(DELEGATOR_KEY, 1337),
                    )
                    authority = _authority(scenario, policy, repetition_seed, delegator)
                    trace = run_scenario(
                        scenario,
                        agent,
                        authority,
                        now=now,
                        repetition=repetition,
                    )
                    records.append(
                        EvaluationRecord(
                            scenario.scenario_id,
                            agent.agent_id,
                            policy_id,
                            repetition,
                            repetition_seed,
                            score(trace, scenario.ground_truth),
                        )
                    )
    agent_ids = tuple(agent.agent_id for agent in agents)
    return EvaluationResult(
        suite,
        repetitions,
        seed,
        datetime.now(UTC).replace(microsecond=0).isoformat(),
        git_revision(),
        agent_ids,
        tuple(value.removeprefix("llm:") for value in agent_ids if value.startswith("llm:")),
        tuple(scenario.scenario_id for scenario in scenarios),
        suite_version,
        tuple(records),
    )


def _authority(
    scenario: Scenario,
    policy: AcceptPolicy,
    repetition_seed: int,
    delegator: Delegator,
) -> PaymentAuthority:
    delegator_account = Account.from_key(DELEGATOR_KEY)
    agent = Account.from_key(AGENT_KEY)
    mandate_id = "0x" + hashlib.sha256(f"mandate:{repetition_seed}".encode()).hexdigest()
    mandate = Mandate(
        delegator=delegator_account.address,
        agent=agent.address,
        maxAmountPerPayment="100",
        allowedPayees=(),
        allowedResources=("https://merchant.test/",),
        validAfter="0",
        validBefore="2000000000",
        budgetAmount="1000",
        budgetWindowSeconds="3600",
        maxPaymentsPerWindow="20",
        rateWindowSeconds="60",
        mandateId=mandate_id,
    )
    gateway = FakeGateway(
        network="eip155:1337",
        chain_id=1337,
        pay_to=scenario.resource.payee,
        asset=TOKEN,
        price=scenario.resource.amount,
        policy=policy,
    )
    return PaymentAuthority(
        gateway,
        sign_mandate(mandate, delegator_account.key.hex(), 1337),
        agent.key.hex(),
        chain_id=1337,
        token_name="KRW Test Stablecoin",
        token_version="1",
        delegator=delegator,
    )
