"""The autonomous inventory agent."""

from __future__ import annotations

import pytest
from tests.conftest import RecordingNotifier

from smart_shelf.agents.inventory_agent import InventoryAgent, RestockTask
from smart_shelf.agents.policies import AlertDecision, Urgency
from smart_shelf.core.config import Settings
from smart_shelf.core.observability import NoOpTracer
from smart_shelf.genai.schemas import (
    Discrepancy,
    DiscrepancyType,
    Severity,
    ShelfAuditResult,
)

pytestmark = pytest.mark.unit


def discrepancy(
    sku: str, severity: Severity, *, level: int = 0, action: str | None = None
) -> Discrepancy:
    return Discrepancy(
        discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
        severity=severity,
        sku=sku,
        product_name=f"Product {sku}",
        shelf_level=level,
        description=f"{sku} is missing",
        recommended_action=action,
    )


async def test_compliant_audit_sends_nothing(
    agent: InventoryAgent, notifier: RecordingNotifier
) -> None:
    outcome = await agent.run(ShelfAuditResult(compliance_score=1.0))
    assert outcome.alert_triggered is False
    assert notifier.sent == []
    assert outcome.notified_channels == []


async def test_non_compliant_audit_dispatches_an_alert(
    agent: InventoryAgent, notifier: RecordingNotifier
) -> None:
    result = ShelfAuditResult(
        store_id="STORE-1",
        shelf_id="BAY-3",
        compliance_score=0.4,
        summary="Half the shelf is empty.",
        discrepancies=[discrepancy("A", Severity.CRITICAL)],
    )
    outcome = await agent.run(result, audit_id="audit-7")

    assert outcome.alert_triggered is True
    assert outcome.notified_channels == ["recording"]
    assert len(notifier.sent) == 1

    sent = notifier.sent[0]
    assert sent.urgency == Urgency.IMMEDIATE.value
    assert sent.audit_id == "audit-7"
    assert sent.store_id == "STORE-1"
    assert "Half the shelf is empty." in sent.body
    assert sent.facts["compliance_score"] == "0.40"


async def test_alert_body_lists_the_policy_reasons(
    agent: InventoryAgent, notifier: RecordingNotifier
) -> None:
    await agent.run(ShelfAuditResult(compliance_score=0.1))
    assert "Reasons:" in notifier.sent[0].body


def test_plan_orders_the_worst_discrepancies_first(agent: InventoryAgent) -> None:
    result = ShelfAuditResult(
        discrepancies=[
            discrepancy("low", Severity.LOW),
            discrepancy("critical", Severity.CRITICAL),
            discrepancy("medium", Severity.MEDIUM),
            discrepancy("high", Severity.HIGH),
        ]
    )
    assert [task.sku for task in agent.plan(result)] == ["critical", "high", "medium", "low"]


def test_plan_is_capped(agent: InventoryAgent) -> None:
    result = ShelfAuditResult(
        discrepancies=[discrepancy(f"SKU-{i}", Severity.HIGH) for i in range(30)]
    )
    assert len(agent.plan(result, limit=5)) == 5


def test_task_prefers_the_recommended_action() -> None:
    task = RestockTask.from_discrepancy(discrepancy("A", Severity.HIGH, action="Restock 3 units."))
    assert task.action == "Restock 3 units."
    assert task.product_name == "Product A"


def test_task_falls_back_to_the_description() -> None:
    assert RestockTask.from_discrepancy(discrepancy("A", Severity.LOW)).action == "A is missing"


def test_task_names_an_unidentified_product() -> None:
    anonymous = Discrepancy(
        discrepancy_type=DiscrepancyType.DAMAGED_PACKAGING,
        severity=Severity.LOW,
        description="A carton is crushed.",
    )
    assert RestockTask.from_discrepancy(anonymous).product_name == "unidentified product"


async def test_delivery_failure_does_not_lose_the_decision(settings: Settings) -> None:
    failing = RecordingNotifier(delivered=False)
    agent = InventoryAgent(settings, notifier=failing, tracer=NoOpTracer())
    outcome = await agent.run(ShelfAuditResult(compliance_score=0.2))

    assert outcome.alert_triggered is True
    assert outcome.notified_channels == []
    assert len(failing.sent) == 1


def test_escalation_picks_the_highest_urgency() -> None:
    decisions = [
        AlertDecision(should_alert=True, urgency=Urgency.ROUTINE),
        AlertDecision(should_alert=True, urgency=Urgency.IMMEDIATE),
        AlertDecision(should_alert=False, urgency=Urgency.NONE),
    ]
    assert InventoryAgent.escalate(*decisions) is Urgency.IMMEDIATE
    assert InventoryAgent.escalate() is Urgency.NONE


async def test_the_run_is_traced(settings: Settings, notifier: RecordingNotifier) -> None:
    tracer = NoOpTracer()
    agent = InventoryAgent(settings, notifier=notifier, tracer=tracer)
    await agent.run(ShelfAuditResult(compliance_score=0.3), audit_id="a1")

    span = next(s for s in tracer.spans if s.name == "agent.run")
    assert span.metadata["audit_id"] == "a1"
    assert span.output["should_alert"] is True
    assert span.duration_ms is not None
