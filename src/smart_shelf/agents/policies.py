"""Decision policy for the inventory agent.

The policy is deliberately explicit rather than model-driven: whether a store
gets paged at 7am is a business rule that has to be auditable, reproducible and
tunable per chain. The LLM decides *what is on the shelf*; this module decides
*what to do about it*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from smart_shelf.core.config import Settings
from smart_shelf.genai.schemas import DiscrepancyType, Severity, ShelfAuditResult


class Urgency(StrEnum):
    """How fast a human needs to act on the audit."""

    NONE = "none"
    ROUTINE = "routine"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


@dataclass(slots=True, frozen=True)
class AlertDecision:
    """The policy's verdict on a single audit."""

    should_alert: bool
    urgency: Urgency
    reasons: list[str] = field(default_factory=list)
    compliance_score: float = 1.0
    critical_skus: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        """One-line verdict for the alert title."""
        if not self.should_alert:
            return "Shelf compliant, no action required"
        return f"Shelf compliance {self.compliance_score:.0%}: {self.urgency.value} action needed"


class CompliancePolicy:
    """Turns an audit into an alerting decision using configured thresholds."""

    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.compliance_alert_threshold
        self._critical_oos = settings.critical_oos_threshold

    def evaluate(self, result: ShelfAuditResult) -> AlertDecision:
        """Apply the alerting rules to ``result``."""
        reasons: list[str] = []
        out_of_stock = result.discrepancies_of(DiscrepancyType.OUT_OF_STOCK)
        critical = [d for d in result.discrepancies if d.severity is Severity.CRITICAL]
        high = [d for d in result.discrepancies if d.severity is Severity.HIGH]

        if result.compliance_score < self._threshold:
            reasons.append(
                f"compliance score {result.compliance_score:.2f} is below the "
                f"{self._threshold:.2f} threshold"
            )
        if len(out_of_stock) >= self._critical_oos:
            reasons.append(
                f"{len(out_of_stock)} out-of-stock SKUs "
                f"(escalation threshold is {self._critical_oos})"
            )
        if critical:
            reasons.append(f"{len(critical)} critical-severity discrepancies")

        urgency = Urgency.NONE
        if reasons:
            if critical or len(out_of_stock) >= self._critical_oos:
                urgency = Urgency.IMMEDIATE
            elif high:
                urgency = Urgency.URGENT
            else:
                urgency = Urgency.ROUTINE

        return AlertDecision(
            should_alert=bool(reasons),
            urgency=urgency,
            reasons=reasons,
            compliance_score=result.compliance_score,
            critical_skus=sorted({d.sku for d in critical if d.sku}),
        )
