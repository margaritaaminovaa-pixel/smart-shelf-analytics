"""Synchronous bridge between Streamlit and the async audit pipeline.

Streamlit callbacks are synchronous, the pipeline is not. Each call opens its
own event loop, builds the object graph, runs one audit and tears it down.

The teardown matters: an ``aiosqlite`` connection is bound to the loop that
created it, so a cached connection would fail on the second interaction.
Reconnecting to local SQLite costs well under a millisecond.

Nothing here imports Streamlit, so it is testable without the ``ui`` extra.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from smart_shelf.agents.inventory_agent import InventoryAgent
from smart_shelf.agents.notifiers import DeliveryReceipt, Notification
from smart_shelf.core.config import DetectorBackend, Settings, get_settings
from smart_shelf.core.exceptions import SmartShelfError
from smart_shelf.core.observability import NoOpTracer, Span
from smart_shelf.cv.detector import build_detector
from smart_shelf.cv.preprocessing import PreparedImage, prepare_image
from smart_shelf.cv.shelves import DEFAULT_ROW_TOLERANCE
from smart_shelf.db.analytics import AnalyticsService, AnalyticsSummary, SkuOffender
from smart_shelf.db.models import AuditFilter, AuditRecord
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.engine import build_vlm_engine
from smart_shelf.genai.schemas import Planogram
from smart_shelf.services.audit_service import AuditOutcome, AuditService


class CapturingNotifier:
    """Records alerts in memory so the UI can display what would have been sent."""

    channel = "ui"

    def __init__(self) -> None:
        self.sent: list[Notification] = []

    async def send(self, notification: Notification) -> DeliveryReceipt:
        """Capture the alert and report it as delivered."""
        self.sent.append(notification)
        return DeliveryReceipt(channel=self.channel, delivered=True, detail="captured for display")


@dataclass(frozen=True, slots=True)
class AuditRunConfig:
    """Everything the sidebar can change about one run."""

    backend: DetectorBackend = DetectorBackend.HEURISTIC
    detection_confidence: float = 0.0
    row_merge_tolerance: float = DEFAULT_ROW_TOLERANCE
    compliance_alert_threshold: float = 0.80
    critical_oos_threshold: int = 3
    detection_max_items: int = 600
    preprocess_max_edge: int = 1600
    store_id: str | None = None
    shelf_id: str | None = None
    persist: bool = True

    def apply_to(self, settings: Settings) -> Settings:
        """Return ``settings`` with this run's overrides applied."""
        return settings.model_copy(
            update={
                "detector_backend": self.backend,
                "detection_confidence": self.detection_confidence,
                "row_merge_tolerance": self.row_merge_tolerance,
                "compliance_alert_threshold": self.compliance_alert_threshold,
                "critical_oos_threshold": self.critical_oos_threshold,
                "detection_max_items": self.detection_max_items,
                "preprocess_max_edge": self.preprocess_max_edge,
            }
        )


@dataclass(frozen=True, slots=True)
class AuditRun:
    """One completed audit, plus everything the UI wants to show about it."""

    outcome: AuditOutcome
    prepared: PreparedImage
    settings: Settings
    spans: list[Span] = field(default_factory=list)
    alerts: list[Notification] = field(default_factory=list)

    @property
    def stage_timings(self) -> list[tuple[str, float]]:
        """``(stage, milliseconds)`` for each traced stage, in execution order."""
        return [
            (span.name, round(span.duration_ms or 0.0, 2))
            for span in self.spans
            if span.duration_ms is not None
        ]


def resolve_settings(config: AuditRunConfig) -> Settings:
    """Build the effective settings for a run without touching the singleton."""
    return config.apply_to(get_settings())


async def _run_audit(
    image_bytes: bytes,
    *,
    settings: Settings,
    config: AuditRunConfig,
    planogram: Planogram | None,
    filename: str | None,
) -> AuditRun:
    tracer = NoOpTracer()
    notifier = CapturingNotifier()

    async with AuditRepository(settings.sqlite_path) as repository:
        service = AuditService(
            settings,
            detector=build_detector(settings),
            engine=build_vlm_engine(settings, tracer=tracer),
            repository=repository,
            agent=InventoryAgent(settings, notifier=notifier, tracer=tracer),
            tracer=tracer,
        )
        outcome = await service.run_audit(
            image_bytes,
            filename=filename,
            planogram=planogram,
            store_id=config.store_id,
            shelf_id=config.shelf_id,
            persist=config.persist,
        )

    # prepare_image is deterministic, so re-running it reproduces exactly the
    # pixel grid the detections were computed against.
    prepared = prepare_image(image_bytes, max_edge=settings.preprocess_max_edge)
    return AuditRun(
        outcome=outcome,
        prepared=prepared,
        settings=settings,
        spans=list(tracer.spans),
        alerts=list(notifier.sent),
    )


def run_audit(
    image_bytes: bytes,
    *,
    config: AuditRunConfig,
    planogram: Planogram | None = None,
    filename: str | None = None,
) -> AuditRun:
    """Run one audit end to end and return everything the UI needs.

    Raises:
        SmartShelfError: any pipeline failure, already carrying a user-facing
            message and an HTTP status the caller may ignore.
    """
    settings = resolve_settings(config)
    return asyncio.run(
        _run_audit(
            image_bytes,
            settings=settings,
            config=config,
            planogram=planogram,
            filename=filename,
        )
    )


async def _load_history(
    settings: Settings, query: AuditFilter
) -> tuple[list[AuditRecord], int, AnalyticsSummary, list[SkuOffender]]:
    async with AuditRepository(settings.sqlite_path) as repository:
        analytics = AnalyticsService(repository)
        records = await repository.list_audits(query)
        total = await repository.count(query)
        summary = await analytics.summary()
        offenders = await analytics.top_offenders(limit=10)
    return records, total, summary, offenders


def load_history(
    settings: Settings, query: AuditFilter | None = None
) -> tuple[list[AuditRecord], int, AnalyticsSummary, list[SkuOffender]]:
    """Read the audit history and its aggregates in one round trip."""
    return asyncio.run(_load_history(settings, query or AuditFilter()))


# -- sample data ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampleScenario:
    """One committed demo shelf, its planogram and its expected outcome."""

    name: str
    description: str
    store_id: str
    shelf_id: str
    image_path: Path
    planogram_path: Path
    empty_slots: int

    @property
    def title(self) -> str:
        """Human-readable name for a select box."""
        return self.name.removeprefix("shelf_").replace("_", " ").title()

    def read_image(self) -> bytes:
        """Load the scenario's shelf photograph."""
        return self.image_path.read_bytes()

    def read_planogram(self) -> Planogram:
        """Load the scenario's expected planogram."""
        return Planogram.model_validate_json(self.planogram_path.read_text(encoding="utf-8"))


def load_sample_scenarios(sample_dir: Path | None = None) -> list[SampleScenario]:
    """Read the sample-data manifest. Returns ``[]`` when it has not been generated."""
    root = sample_dir or get_settings().sample_data_dir
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return []
    try:
        payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    scenarios: list[SampleScenario] = []
    for entry in payload.get("scenarios", []):
        image = root / str(entry["image"])
        planogram = root / str(entry["planogram"])
        if not image.is_file() or not planogram.is_file():
            continue
        scenarios.append(
            SampleScenario(
                name=str(entry["name"]),
                description=str(entry["description"]),
                store_id=str(entry.get("store_id") or ""),
                shelf_id=str(entry.get("shelf_id") or ""),
                image_path=image,
                planogram_path=planogram,
                empty_slots=int(entry.get("empty_slots", 0)),
            )
        )
    return scenarios


__all__ = [
    "AuditRun",
    "AuditRunConfig",
    "CapturingNotifier",
    "SampleScenario",
    "SmartShelfError",
    "load_history",
    "load_sample_scenarios",
    "resolve_settings",
    "run_audit",
]
