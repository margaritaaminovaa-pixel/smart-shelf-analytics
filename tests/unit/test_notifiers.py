"""Outbound alert channels."""

from __future__ import annotations

import httpx
import pytest

from smart_shelf.agents.notifiers import (
    CompositeNotifier,
    EmailWebhookNotifier,
    LoggingNotifier,
    Notification,
    Notifier,
    SlackWebhookNotifier,
    build_notifier,
)
from smart_shelf.core.config import Settings

pytestmark = pytest.mark.unit

NOTIFICATION = Notification(
    title="Shelf restock required (immediate)",
    body="Compliance 55%.",
    urgency="immediate",
    store_id="STORE-1",
    shelf_id="BAY-3",
    audit_id="abc-123",
    facts={"compliance_score": "0.55", "discrepancies": 4},
)


def transport(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_slack_payload_carries_blocks_and_context() -> None:
    payload = NOTIFICATION.to_slack_blocks()
    assert payload["text"] == NOTIFICATION.title
    assert payload["blocks"][0]["type"] == "header"
    context = payload["blocks"][-1]["elements"][0]["text"]
    assert "STORE-1" in context
    assert "abc-123" in context


def test_slack_payload_omits_the_facts_block_when_empty() -> None:
    bare = Notification(title="t", body="b", urgency="routine")
    types = [block["type"] for block in bare.to_slack_blocks()["blocks"]]
    assert types == ["header", "section", "context"]


def test_email_payload_prefixes_the_subject() -> None:
    payload = NOTIFICATION.to_email_payload()
    assert payload["subject"].startswith("[Shelf Alert] ")
    assert payload["metadata"]["store_id"] == "STORE-1"
    assert payload["priority"] == "immediate"


async def test_logging_notifier_always_succeeds() -> None:
    receipt = await LoggingNotifier().send(NOTIFICATION)
    assert receipt.delivered is True
    assert receipt.detail == "dry-run"


async def test_slack_notifier_posts_the_block_payload() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    async with transport(handler) as client:
        notifier = SlackWebhookNotifier("https://hooks.slack.test/abc", client=client)
        receipt = await notifier.send(NOTIFICATION)

    assert receipt.delivered is True
    assert receipt.detail == "HTTP 200"
    assert len(seen) == 1
    assert b"header" in seen[0].content


async def test_webhook_errors_are_reported_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with transport(handler) as client:
        receipt = await SlackWebhookNotifier("https://hooks.slack.test", client=client).send(
            NOTIFICATION
        )

    assert receipt.delivered is False
    assert "500" in receipt.detail


async def test_connection_errors_are_reported_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    async with transport(handler) as client:
        receipt = await EmailWebhookNotifier("https://mail.test/hook", client=client).send(
            NOTIFICATION
        )

    assert receipt.delivered is False
    assert "no route to host" in receipt.detail


async def test_composite_fans_out_to_every_channel() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200)

    async with transport(handler) as client:
        composite = CompositeNotifier(
            [
                SlackWebhookNotifier("https://slack.test", client=client),
                EmailWebhookNotifier("https://mail.test", client=client),
            ]
        )
        receipts = await composite.send_all(NOTIFICATION)

    assert composite.channels == ["slack", "email"]
    assert [r.channel for r in receipts] == ["slack", "email"]
    assert len(calls) == 2


async def test_composite_is_undelivered_when_any_channel_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if "slack" in str(request.url) else 503)

    async with transport(handler) as client:
        composite = CompositeNotifier(
            [
                SlackWebhookNotifier("https://slack.test", client=client),
                EmailWebhookNotifier("https://mail.test", client=client),
            ]
        )
        receipt = await composite.send(NOTIFICATION)

    assert receipt.delivered is False
    assert "slack=ok" in receipt.detail
    assert "email=failed" in receipt.detail


def test_dry_run_configuration_selects_the_logging_channel(settings: Settings) -> None:
    assert isinstance(build_notifier(settings), LoggingNotifier)


def test_configured_webhooks_build_a_composite() -> None:
    settings = Settings(
        notifications_dry_run=False,
        slack_webhook_url="https://hooks.slack.test/x",
        email_webhook_url="https://mail.test/hook",
    )
    notifier = build_notifier(settings)
    assert isinstance(notifier, CompositeNotifier)
    assert notifier.channels == ["slack", "email"]


def test_live_mode_without_channels_falls_back_to_logging() -> None:
    notifier = build_notifier(Settings(notifications_dry_run=False))
    assert isinstance(notifier, LoggingNotifier)


def test_every_channel_satisfies_the_protocol() -> None:
    assert isinstance(LoggingNotifier(), Notifier)
    assert isinstance(SlackWebhookNotifier("https://x.test"), Notifier)
    assert isinstance(CompositeNotifier([LoggingNotifier()]), Notifier)
