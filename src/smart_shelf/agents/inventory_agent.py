"""The autonomous inventory agent.

Given a structured audit, the agent: evaluates it against the compliance policy,
derives a prioritised restocking plan, and - when the policy says so - dispatches
an alert to every configured channel. It is deliberately a small, deterministic
state machine rather than an LLM loop, because its side effects reach real store
staff and must be explainable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_shelf.agents.notifiers import (
    DeliveryReceipt,
    Notification,
    Notifier,
    build_notifier,
)
from smart_shelf.agents.policies import AlertDecision, CompliancePolicy, Urgency
from smart_shelf.core.config import Settings
from smart_shelf.core.logging import get_logger
from smart_shelf.core.observability import Tracer, get_tracer
from smart_shelf.genai.schemas import Discrepancy, Severity, ShelfAuditResult

logger = get_logger(__name__)

_URGENCY_ORDER: dict[Urgency, int] = {
    Urgency.NONE: 0,
    Urgency.ROUTINE: 1,
    Urgency.URGENT: 2,
    Urgency.IMMEDIATE: 3,
}


@dataclass(slots=True, frozen=True)
class RestockTask:
    """One actionable instruction for store staff."""

    sku: str | None
    product_name: str
    shelf_level: int | None
    action: str
    severity: Severity
    priority: int

    @classmethod
    def from_discrepancy(cls, discrepancy: Discrepancy) -> RestockTask:
        """Derive an actionable task from a reported discrepancy."""
        return cls(
            sku=discrepancy.sku,
            product_name=discrepancy.product_name or discrepancy.sku or "unidentified product",
            shelf_level=discrepancy.shelf_level,
            action=discrepancy.recommended_action or discrepancy.description,
            severity=discrepancy.severity,
            priority=discrepancy.severity.rank,
        )


@dataclass(slots=True, frozen=True)
class AgentOutcome:
    """Everything the agent did, returned to the caller and persisted."""

    decision: AlertDecision
    tasks: list[RestockTask] = field(default_factory=list)
    receipts: list[DeliveryReceipt] = field(default_factory=list)

    @property
    def alert_triggered(self) -> bool:
        """Whether the policy decided this audit warrants an alert."""
        return self.decision.should_alert

    @property
    def notified_channels(self) -> list[str]:
        """Channels that accepted the alert."""
        return [receipt.channel for receipt in self.receipts if receipt.delivered]


class InventoryAgent:
    """Evaluates audits and triggers restocking alerts."""

    def __init__(
        self,
        settings: Settings,
        *,
        notifier: Notifier | None = None,
        policy: CompliancePolicy | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._settings = settings
        self._policy = policy or CompliancePolicy(settings)
        self._notifier = notifier or build_notifier(settings)
        self._tracer = tracer or get_tracer()

    def plan(self, result: ShelfAuditResult, *, limit: int = 20) -> list[RestockTask]:
        """Order discrepancies into a restocking work list, most severe first."""
        tasks = [RestockTask.from_discrepancy(d) for d in result.discrepancies]
        tasks.sort(key=lambda task: (-task.priority, task.shelf_level or 0, task.product_name))
        return tasks[:limit]

    def _build_notification(
        self,
        result: ShelfAuditResult,
        decision: AlertDecision,
        tasks: list[RestockTask],
        *,
        audit_id: str | None,
    ) -> Notification:
        top = tasks[:5]
        body_lines = [
            decision.headline,
            "",
            result.summary or "No summary produced.",
            "",
            "Reasons:",
            *(f"- {reason}" for reason in decision.reasons),
        ]
        if top:
            body_lines += ["", "Top actions:", *(f"- [{t.severity.value}] {t.action}" for t in top)]
        return Notification(
            title=f"Shelf restock required ({decision.urgency.value})",
            body="\n".join(body_lines),
            urgency=decision.urgency.value,
            store_id=result.store_id,
            shelf_id=result.shelf_id,
            audit_id=audit_id,
            facts={
                "compliance_score": f"{result.compliance_score:.2f}",
                "discrepancies": len(result.discrepancies),
                "empty_slots": result.empty_slot_count,
                "total_facings": result.total_facings,
            },
        )

    async def run(self, result: ShelfAuditResult, *, audit_id: str | None = None) -> AgentOutcome:
        """Evaluate one audit and act on it."""
        with self._tracer.span("agent.run", audit_id=audit_id) as span:
            decision = self._policy.evaluate(result)
            tasks = self.plan(result)
            receipts: list[DeliveryReceipt] = []

            if decision.should_alert:
                notification = self._build_notification(result, decision, tasks, audit_id=audit_id)
                receipt = await self._notifier.send(notification)
                receipts.append(receipt)
                logger.info(
                    "agent.alert_dispatched",
                    audit_id=audit_id,
                    urgency=decision.urgency.value,
                    delivered=receipt.delivered,
                    channel=receipt.channel,
                )
            else:
                logger.info(
                    "agent.no_alert",
                    audit_id=audit_id,
                    compliance_score=result.compliance_score,
                )

            span.set_output(
                should_alert=decision.should_alert,
                urgency=decision.urgency.value,
                tasks=len(tasks),
                delivered=all(r.delivered for r in receipts) if receipts else None,
            )
            return AgentOutcome(decision=decision, tasks=tasks, receipts=receipts)

    @staticmethod
    def escalate(*decisions: AlertDecision) -> Urgency:
        """Highest urgency across several decisions, for shelf-group rollups."""
        if not decisions:
            return Urgency.NONE
        return max((d.urgency for d in decisions), key=lambda u: _URGENCY_ORDER[u])
