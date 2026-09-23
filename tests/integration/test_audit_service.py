"""End-to-end pipeline orchestration, with real CV and the offline auditor."""

from __future__ import annotations

import pytest
from tests.conftest import RecordingNotifier

from smart_shelf.agents.inventory_agent import InventoryAgent
from smart_shelf.core.config import Settings
from smart_shelf.core.exceptions import DetectionError, InvalidImageError, VisionModelError
from smart_shelf.core.observability import NoOpTracer
from smart_shelf.cv.detector import HeuristicShelfDetector, ObjectDetector
from smart_shelf.cv.models import DetectionResult
from smart_shelf.cv.preprocessing import PreparedImage
from smart_shelf.db.models import AuditFilter
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.engine import VisionLanguageEngine, VLMRequest
from smart_shelf.genai.mock_engine import MockVLMEngine
from smart_shelf.genai.schemas import Planogram, ShelfAuditResult
from smart_shelf.services.audit_service import AuditService

pytestmark = pytest.mark.integration


class ExplodingDetector:
    """A detector whose backend is down."""

    name = "exploding"

    def detect(self, image: PreparedImage) -> DetectionResult:
        raise DetectionError("detector unavailable")


class ExplodingEngine:
    """A VLM whose provider is down."""

    name = "exploding"

    async def analyze(self, request: VLMRequest) -> ShelfAuditResult:
        raise VisionModelError("model unavailable")


def make_service(
    settings: Settings,
    repository: AuditRepository,
    *,
    detector: ObjectDetector | None = None,
    engine: VisionLanguageEngine | None = None,
    tracer: NoOpTracer | None = None,
) -> AuditService:
    """Assemble a service from fakes, defaulting to the offline stack."""
    tracer = tracer or NoOpTracer()
    return AuditService(
        settings,
        detector=detector or HeuristicShelfDetector(),
        engine=engine or MockVLMEngine(tracer),
        repository=repository,
        agent=InventoryAgent(settings, notifier=RecordingNotifier(), tracer=tracer),
        tracer=tracer,
    )


async def test_compliant_shelf_produces_a_clean_audit(
    audit_service: AuditService, shelf_image_bytes: bytes, planogram: Planogram
) -> None:
    outcome = await audit_service.run_audit(
        shelf_image_bytes, filename="shelf.jpg", planogram=planogram
    )

    assert outcome.detections.count == 18
    assert outcome.result.compliance_score == 1.0
    assert outcome.result.discrepancies == []
    assert outcome.agent.alert_triggered is False
    assert outcome.record.image_filename == "shelf.jpg"
    assert outcome.record.detector_backend == "opencv-heuristic"
    assert outcome.record.vlm_backend == "mock-vlm"
    assert outcome.record.duration_ms > 0


async def test_stockout_shelf_triggers_the_agent(
    audit_service: AuditService,
    stockout_image_bytes: bytes,
    planogram: Planogram,
    notifier: RecordingNotifier,
) -> None:
    outcome = await audit_service.run_audit(stockout_image_bytes, planogram=planogram)

    assert outcome.result.compliance_score < 0.8
    assert outcome.result.discrepancies
    assert outcome.agent.alert_triggered is True
    assert outcome.record.alert_triggered is True
    assert len(notifier.sent) == 1
    assert outcome.agent.tasks


async def test_the_audit_is_persisted(
    audit_service: AuditService, repository: AuditRepository, shelf_image_bytes: bytes
) -> None:
    outcome = await audit_service.run_audit(shelf_image_bytes)

    stored = await repository.get(outcome.record.audit_id)
    assert stored.result == outcome.result
    assert await repository.count() == 1


async def test_persistence_can_be_skipped(
    audit_service: AuditService, repository: AuditRepository, shelf_image_bytes: bytes
) -> None:
    await audit_service.run_audit(shelf_image_bytes, persist=False)
    assert await repository.count() == 0


async def test_identifiers_flow_into_the_record(
    audit_service: AuditService, shelf_image_bytes: bytes
) -> None:
    outcome = await audit_service.run_audit(
        shelf_image_bytes, store_id="STORE-77", shelf_id="BAY-2"
    )
    assert outcome.record.store_id == "STORE-77"
    assert outcome.record.shelf_id == "BAY-2"


async def test_oversized_uploads_are_rejected_before_decoding(
    settings: Settings, repository: AuditRepository, shelf_image_bytes: bytes
) -> None:
    service = make_service(settings.model_copy(update={"max_image_bytes": 1024}), repository)
    with pytest.raises(InvalidImageError, match="size limit"):
        await service.run_audit(shelf_image_bytes)


async def test_undecodable_uploads_are_rejected(audit_service: AuditService) -> None:
    with pytest.raises(InvalidImageError, match="not a decodable image"):
        await audit_service.run_audit(b"definitely not an image")


async def test_detector_failures_propagate(
    settings: Settings, repository: AuditRepository, shelf_image_bytes: bytes
) -> None:
    service = make_service(settings, repository, detector=ExplodingDetector())
    with pytest.raises(DetectionError, match="detector unavailable"):
        await service.run_audit(shelf_image_bytes)
    assert await repository.count() == 0


async def test_vlm_failures_propagate_without_persisting(
    settings: Settings, repository: AuditRepository, shelf_image_bytes: bytes
) -> None:
    service = make_service(settings, repository, engine=ExplodingEngine())
    with pytest.raises(VisionModelError, match="model unavailable"):
        await service.run_audit(shelf_image_bytes)
    assert await repository.count() == 0


async def test_history_returns_a_page_and_the_total(
    audit_service: AuditService, shelf_image_bytes: bytes, stockout_image_bytes: bytes
) -> None:
    await audit_service.run_audit(shelf_image_bytes, store_id="STORE-1")
    await audit_service.run_audit(stockout_image_bytes, store_id="STORE-2")

    records, total = await audit_service.history(AuditFilter(limit=1))
    assert len(records) == 1
    assert total == 2

    filtered, filtered_total = await audit_service.history(AuditFilter(store_id="STORE-2"))
    assert filtered_total == 1
    assert filtered[0].store_id == "STORE-2"


async def test_get_audit_reads_back_a_stored_record(
    audit_service: AuditService, shelf_image_bytes: bytes
) -> None:
    outcome = await audit_service.run_audit(shelf_image_bytes)
    assert (await audit_service.get_audit(outcome.record.audit_id)).result == outcome.result


async def test_the_pipeline_emits_one_span_per_stage(
    settings: Settings, repository: AuditRepository, shelf_image_bytes: bytes
) -> None:
    tracer = NoOpTracer()
    service = make_service(settings, repository, tracer=tracer)
    await service.run_audit(shelf_image_bytes)

    names = {span.name for span in tracer.spans}
    assert {"audit.run", "cv.preprocess", "cv.detect", "vlm.analyze", "agent.run"} <= names


async def test_reconciliation_runs_only_with_a_planogram(
    settings: Settings,
    repository: AuditRepository,
    shelf_image_bytes: bytes,
    planogram: Planogram,
) -> None:
    tracer = NoOpTracer()
    service = make_service(settings, repository, tracer=tracer)

    await service.run_audit(shelf_image_bytes)
    assert not [s for s in tracer.spans if s.name == "compliance.reconcile"]

    await service.run_audit(shelf_image_bytes, planogram=planogram)
    assert [s for s in tracer.spans if s.name == "compliance.reconcile"]


async def test_repeated_audits_of_one_image_share_a_digest(
    audit_service: AuditService, shelf_image_bytes: bytes
) -> None:
    first = await audit_service.run_audit(shelf_image_bytes)
    second = await audit_service.run_audit(shelf_image_bytes)

    assert first.record.image_sha256 == second.record.image_sha256
    assert first.record.audit_id != second.record.audit_id
