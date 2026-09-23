"""Alerting policy thresholds."""

from __future__ import annotations

import pytest

from smart_shelf.agents.policies import CompliancePolicy, Urgency
from smart_shelf.core.config import Settings
from smart_shelf.genai.schemas import (
    Discrepancy,
    DiscrepancyType,
    Severity,
    ShelfAuditResult,
)

pytestmark = pytest.mark.unit


def oos(sku: str, severity: Severity = Severity.HIGH) -> Discrepancy:
    return Discrepancy(
        discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
        severity=severity,
        sku=sku,
        description=f"{sku} is gone",
    )


def test_compliant_shelf_raises_no_alert(settings: Settings) -> None:
    decision = CompliancePolicy(settings).evaluate(ShelfAuditResult(compliance_score=1.0))
    assert decision.should_alert is False
    assert decision.urgency is Urgency.NONE
    assert "no action required" in decision.headline


def test_score_below_threshold_triggers_an_alert(settings: Settings) -> None:
    decision = CompliancePolicy(settings).evaluate(ShelfAuditResult(compliance_score=0.6))
    assert decision.should_alert is True
    assert decision.urgency is Urgency.ROUTINE
    assert "below the" in decision.reasons[0]


def test_score_exactly_at_the_threshold_is_compliant(settings: Settings) -> None:
    assert (
        not CompliancePolicy(settings)
        .evaluate(ShelfAuditResult(compliance_score=settings.compliance_alert_threshold))
        .should_alert
    )


def test_a_critical_discrepancy_escalates_immediately(settings: Settings) -> None:
    decision = CompliancePolicy(settings).evaluate(
        ShelfAuditResult(compliance_score=0.95, discrepancies=[oos("A", Severity.CRITICAL)])
    )
    assert decision.urgency is Urgency.IMMEDIATE
    assert decision.critical_skus == ["A"]


def test_high_severity_alone_is_urgent_not_immediate(settings: Settings) -> None:
    decision = CompliancePolicy(settings).evaluate(
        ShelfAuditResult(compliance_score=0.5, discrepancies=[oos("A"), oos("B")])
    )
    assert decision.urgency is Urgency.URGENT


def test_enough_stockouts_escalate_even_at_a_good_score(settings: Settings) -> None:
    decision = CompliancePolicy(settings).evaluate(
        ShelfAuditResult(
            compliance_score=0.99,
            discrepancies=[oos("A", Severity.LOW), oos("B", Severity.LOW), oos("C", Severity.LOW)],
        )
    )
    assert decision.should_alert is True
    assert decision.urgency is Urgency.IMMEDIATE
    assert any("out-of-stock" in reason for reason in decision.reasons)


def test_thresholds_are_configurable() -> None:
    strict = Settings(compliance_alert_threshold=0.99, critical_oos_threshold=1)
    assert CompliancePolicy(strict).evaluate(ShelfAuditResult(compliance_score=0.95)).should_alert
