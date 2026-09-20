"""Outbound alert channels.

Every channel implements :class:`Notifier`, so the agent fans out to Slack,
email or anything else without knowing the difference. Delivery failures are
recorded on the returned :class:`DeliveryReceipt` rather than raised - a store
alert that cannot be delivered must never lose the audit that produced it.

``notifications_dry_run`` is on by default, so a freshly cloned repository logs
alerts instead of posting them anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from smart_shelf.core.config import Settings
from smart_shelf.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class Notification:
    """A channel-agnostic alert payload."""

    title: str
    body: str
    urgency: str
    store_id: str | None = None
    shelf_id: str | None = None
    audit_id: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)

    def to_slack_blocks(self) -> dict[str, Any]:
        """Render as a Slack ``chat.postMessage`` / incoming-webhook body."""
        fact_lines = "\n".join(f"• *{k}*: {v}" for k, v in self.facts.items())
        return {
            "text": self.title,
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": self.title[:150]}},
                {"type": "section", "text": {"type": "mrkdwn", "text": self.body[:2900]}},
                *(
                    [{"type": "section", "text": {"type": "mrkdwn", "text": fact_lines[:2900]}}]
                    if fact_lines
                    else []
                ),
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"store `{self.store_id or 'n/a'}` · shelf "
                                f"`{self.shelf_id or 'n/a'}` · audit `{self.audit_id or 'n/a'}`"
                            ),
                        }
                    ],
                },
            ],
        }

    def to_email_payload(self, *, subject_prefix: str = "[Shelf Alert]") -> dict[str, Any]:
        """Render as a transactional-email provider body."""
        return {
            "subject": f"{subject_prefix} {self.title}"[:255],
            "body": self.body,
            "priority": self.urgency,
            "metadata": {
                "store_id": self.store_id,
                "shelf_id": self.shelf_id,
                "audit_id": self.audit_id,
                **self.facts,
            },
        }


@dataclass(slots=True, frozen=True)
class DeliveryReceipt:
    """Outcome of one delivery attempt."""

    channel: str
    delivered: bool
    detail: str = ""


@runtime_checkable
class Notifier(Protocol):
    """Delivers a :class:`Notification` to one channel."""

    channel: str

    async def send(
        self, notification: Notification
    ) -> DeliveryReceipt:  # pragma: no cover - protocol
        """Deliver ``notification`` and report the outcome."""
        ...


class LoggingNotifier:
    """Dry-run channel: writes the alert to the structured log and succeeds."""

    channel = "log"

    async def send(self, notification: Notification) -> DeliveryReceipt:
        """Log the alert and report it as delivered."""
        logger.warning(
            "alert.dry_run",
            title=notification.title,
            urgency=notification.urgency,
            store_id=notification.store_id,
            shelf_id=notification.shelf_id,
            audit_id=notification.audit_id,
            **notification.facts,
        )
        return DeliveryReceipt(channel=self.channel, delivered=True, detail="dry-run")


class _WebhookNotifier:
    """Shared POST-with-JSON plumbing for the webhook channels."""

    channel = "webhook"

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._client = client

    def _payload(self, notification: Notification) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    async def send(self, notification: Notification) -> DeliveryReceipt:
        payload = self._payload(notification)
        try:
            if self._client is not None:
                response = await self._client.post(self._url, json=payload, timeout=self._timeout)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self._url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("alert.delivery_failed", channel=self.channel, error=str(exc))
            return DeliveryReceipt(channel=self.channel, delivered=False, detail=str(exc))

        logger.info("alert.delivered", channel=self.channel, status=response.status_code)
        return DeliveryReceipt(
            channel=self.channel, delivered=True, detail=f"HTTP {response.status_code}"
        )


class SlackWebhookNotifier(_WebhookNotifier):
    """Posts Block Kit messages to a Slack incoming webhook."""

    channel = "slack"

    def _payload(self, notification: Notification) -> dict[str, Any]:
        return notification.to_slack_blocks()


class EmailWebhookNotifier(_WebhookNotifier):
    """Posts to a transactional-email webhook (SendGrid, SES relay, ...)."""

    channel = "email"

    def _payload(self, notification: Notification) -> dict[str, Any]:
        return notification.to_email_payload()


class CompositeNotifier:
    """Fans one notification out to several channels, collecting every receipt."""

    channel = "composite"

    def __init__(self, notifiers: list[Notifier]) -> None:
        self._notifiers = notifiers

    @property
    def channels(self) -> list[str]:
        """Names of the wrapped channels, in delivery order."""
        return [notifier.channel for notifier in self._notifiers]

    async def send(self, notification: Notification) -> DeliveryReceipt:
        """Deliver to every channel and collapse the receipts into one."""
        receipts = await self.send_all(notification)
        delivered = all(receipt.delivered for receipt in receipts) if receipts else False
        return DeliveryReceipt(
            channel=self.channel,
            delivered=delivered,
            detail=", ".join(f"{r.channel}={'ok' if r.delivered else 'failed'}" for r in receipts),
        )

    async def send_all(self, notification: Notification) -> list[DeliveryReceipt]:
        """Deliver sequentially so channel ordering stays predictable in logs."""
        return [await notifier.send(notification) for notifier in self._notifiers]


def build_notifier(settings: Settings, *, client: httpx.AsyncClient | None = None) -> Notifier:
    """Assemble the notifier fan-out described by configuration."""
    if settings.notifications_dry_run:
        return LoggingNotifier()

    notifiers: list[Notifier] = []
    if settings.slack_webhook_url:
        notifiers.append(
            SlackWebhookNotifier(
                settings.slack_webhook_url,
                timeout=settings.notification_timeout_seconds,
                client=client,
            )
        )
    if settings.email_webhook_url:
        notifiers.append(
            EmailWebhookNotifier(
                settings.email_webhook_url,
                timeout=settings.notification_timeout_seconds,
                client=client,
            )
        )
    if not notifiers:
        logger.warning("alert.no_channels_configured", hint="falling back to dry-run logging")
        return LoggingNotifier()
    return CompositeNotifier(notifiers)
