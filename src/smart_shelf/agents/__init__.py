"""Autonomous inventory agent: compliance evaluation and restocking alerts."""

from __future__ import annotations

from smart_shelf.agents.inventory_agent import AgentOutcome, InventoryAgent, RestockTask
from smart_shelf.agents.notifiers import (
    CompositeNotifier,
    EmailWebhookNotifier,
    LoggingNotifier,
    Notification,
    Notifier,
    SlackWebhookNotifier,
    build_notifier,
)
from smart_shelf.agents.policies import AlertDecision, CompliancePolicy, Urgency

__all__ = [
    "AgentOutcome",
    "AlertDecision",
    "CompliancePolicy",
    "CompositeNotifier",
    "EmailWebhookNotifier",
    "InventoryAgent",
    "LoggingNotifier",
    "Notification",
    "Notifier",
    "RestockTask",
    "SlackWebhookNotifier",
    "Urgency",
    "build_notifier",
]
