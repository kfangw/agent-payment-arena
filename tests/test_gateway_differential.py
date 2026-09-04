"""Differential contract check against the pinned reference gateway."""

import os
from datetime import UTC, datetime, timedelta

import pytest
from eth_account import Account

from arena.gateway.fake import FakeGateway
from arena.gateway.http import HttpGateway
from arena.gateway.protocol import Gateway
from arena.gateway.schemas import Mandate
from arena.gateway.signatures import sign_mandate
from arena.payment import PaymentAuthority

GATEWAY_URL = os.getenv("ARENA_HTTP_GATEWAY_URL")
AGENT_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
DELEGATOR_KEY = "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"


@pytest.mark.http_gateway
@pytest.mark.skipif(not GATEWAY_URL, reason="ARENA_HTTP_GATEWAY_URL is not set")
def test_fake_and_reference_gateway_approve_the_same_signed_payment() -> None:
    """Hold both backends to the valid-payment contract."""
    resource = f"{GATEWAY_URL}/premium/report"
    live = HttpGateway(str(GATEWAY_URL))
    terms = live.requirements(resource)
    now = datetime.now(UTC)
    agent = Account.from_key(AGENT_KEY)
    delegator = Account.from_key(DELEGATOR_KEY)
    mandate = Mandate(
        delegator=delegator.address,
        agent=agent.address,
        maxAmountPerPayment="1000",
        allowedPayees=(terms.pay_to,),
        allowedResources=(resource,),
        validAfter=str(int((now - timedelta(minutes=1)).timestamp())),
        validBefore=str(int((now + timedelta(hours=1)).timestamp())),
        budgetAmount="2000",
        budgetWindowSeconds="3600",
        maxPaymentsPerWindow="2",
        rateWindowSeconds="60",
        mandateId="0x" + "91" * 32,
    )
    signed = sign_mandate(mandate, DELEGATOR_KEY, 31337)
    fake = FakeGateway(
        network=terms.network,
        chain_id=31337,
        pay_to=terms.pay_to,
        asset=terms.asset,
        price=int(terms.max_amount_required),
    )

    def authority(backend: Gateway) -> PaymentAuthority:
        return PaymentAuthority(
            backend,
            signed,
            AGENT_KEY,
            chain_id=31337,
            token_name=terms.extra["name"],
            token_version=terms.extra["version"],
        )

    fake_result = authority(fake).pay(
        resource,
        terms.pay_to,
        int(terms.max_amount_required),
        now=now,
        nonce_key="differential-fake",
    )
    live_result = authority(live).pay(
        resource,
        terms.pay_to,
        int(terms.max_amount_required),
        now=now,
        nonce_key="differential-live",
    )
    live.close()
    assert fake_result.action == live_result.action
    assert fake_result.settled == live_result.settled is True
