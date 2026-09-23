"""Shared fixtures.

Every fixture here is offline and deterministic: the in-memory SQLite store, the
OpenCV heuristic detector and the mock VLM engine. No test touches the network,
a GPU or a real API key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import json
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from smart_shelf.agents.inventory_agent import InventoryAgent
from smart_shelf.agents.notifiers import DeliveryReceipt, Notification
from smart_shelf.api.app import create_app
from smart_shelf.api.dependencies import ApplicationContainer
from smart_shelf.core.config import (
    DetectorBackend,
    Environment,
    Settings,
    VLMBackend,
    reset_settings_cache,
)
from smart_shelf.core.observability import NoOpTracer, reset_tracer_cache
from smart_shelf.cv.detector import HeuristicShelfDetector
from smart_shelf.cv.preprocessing import PreparedImage, prepare_image
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.mock_engine import MockVLMEngine
from smart_shelf.genai.schemas import Planogram
from smart_shelf.services.audit_service import AuditService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample_data"
SCENARIOS = ("shelf_compliant", "shelf_minor_gaps", "shelf_critical_stockout")


@pytest.fixture(autouse=True)
def _isolated_caches() -> Iterator[None]:
    """Keep the settings and tracer singletons from leaking between tests."""
    reset_settings_cache()
    reset_tracer_cache()
    yield
    reset_settings_cache()
    reset_tracer_cache()


@pytest.fixture
def settings() -> Settings:
    """Offline test configuration."""
    return Settings(
        environment=Environment.TEST,
        detector_backend=DetectorBackend.HEURISTIC,
        vlm_backend=VLMBackend.MOCK,
        database_url="sqlite+aiosqlite:///:memory:",
        notifications_dry_run=True,
        log_level="WARNING",
        compliance_alert_threshold=0.80,
        critical_oos_threshold=3,
    )


@pytest.fixture
def tracer() -> NoOpTracer:
    return NoOpTracer()


# -- sample data --------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    if not (SAMPLE_DATA / "images" / "shelf_compliant.jpg").exists():
        pytest.skip("sample data missing; run `python scripts/generate_sample_data.py --force`")
    return SAMPLE_DATA


@pytest.fixture
def shelf_image_bytes(sample_data_dir: Path) -> bytes:
    """A fully stocked synthetic shelf."""
    return (sample_data_dir / "images" / "shelf_compliant.jpg").read_bytes()


@pytest.fixture
def stockout_image_bytes(sample_data_dir: Path) -> bytes:
    """A synthetic shelf with an almost empty bottom row."""
    return (sample_data_dir / "images" / "shelf_critical_stockout.jpg").read_bytes()


@pytest.fixture
def gaps_image_bytes(sample_data_dir: Path) -> bytes:
    """A synthetic shelf missing two facings from the middle row."""
    return (sample_data_dir / "images" / "shelf_minor_gaps.jpg").read_bytes()


@pytest.fixture
def planogram(sample_data_dir: Path) -> Planogram:
    raw = (sample_data_dir / "planograms" / "shelf_compliant.json").read_text(encoding="utf-8")
    return Planogram.model_validate(json.loads(raw))


@pytest.fixture
def prepared_image(shelf_image_bytes: bytes) -> PreparedImage:
    return prepare_image(shelf_image_bytes)


@pytest.fixture
def synthetic_image_bytes() -> bytes:
    """A tiny two-row shelf rendered in-process, for tests that need no fixtures."""
    canvas = np.full((240, 400, 3), 235, dtype=np.uint8)
    for row in range(2):
        top = 30 + row * 110
        cv2.rectangle(canvas, (10, top + 80), (390, top + 90), (170, 165, 155), -1)
        for slot in range(4):
            x = 20 + slot * 95
            colour = (40 + slot * 50, 90, 200 - slot * 30)
            cv2.rectangle(canvas, (x, top), (x + 70, top + 80), colour, -1)
            cv2.rectangle(canvas, (x, top), (x + 70, top + 80), (30, 30, 30), 2)
    success, buffer = cv2.imencode(".jpg", canvas)
    assert success
    return bytes(buffer.tobytes())


# -- collaborators ------------------------------------------------------------


class RecordingNotifier:
    """Captures notifications instead of sending them."""

    channel = "recording"

    def __init__(self, *, delivered: bool = True) -> None:
        self.sent: list[Notification] = []
        self._delivered = delivered

    async def send(self, notification: Notification) -> DeliveryReceipt:
        self.sent.append(notification)
        return DeliveryReceipt(channel=self.channel, delivered=self._delivered)


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
async def repository() -> AsyncIterator[AuditRepository]:
    repo = AuditRepository(":memory:")
    await repo.connect()
    try:
        yield repo
    finally:
        await repo.close()


@pytest.fixture
def agent(settings: Settings, notifier: RecordingNotifier, tracer: NoOpTracer) -> InventoryAgent:
    return InventoryAgent(settings, notifier=notifier, tracer=tracer)


@pytest.fixture
def audit_service(
    settings: Settings,
    repository: AuditRepository,
    agent: InventoryAgent,
    tracer: NoOpTracer,
) -> AuditService:
    return AuditService(
        settings,
        detector=HeuristicShelfDetector(),
        engine=MockVLMEngine(tracer=tracer),
        repository=repository,
        agent=agent,
        tracer=tracer,
    )


@pytest.fixture
async def container(
    settings: Settings,
    repository: AuditRepository,
    agent: InventoryAgent,
    tracer: NoOpTracer,
) -> ApplicationContainer:
    return await ApplicationContainer.create(
        settings,
        repository=repository,
        detector=HeuristicShelfDetector(),
        engine=MockVLMEngine(tracer=tracer),
        agent=agent,
    )


@pytest.fixture
def app(container: ApplicationContainer) -> FastAPI:
    application = create_app(container.settings, container=container)
    # httpx's ASGI transport does not run the lifespan, so wire the container in
    # directly; ownership stays with the fixture.
    application.state.container = container
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest.fixture
def upload_files(shelf_image_bytes: bytes) -> dict[str, Any]:
    return {"image": ("shelf_compliant.jpg", shelf_image_bytes, "image/jpeg")}
