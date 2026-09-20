"""End-to-end shelf audit orchestration.

    bytes -> preprocess -> detect -> VLM -> reconcile -> agent -> persist

Every collaborator is injected, so the same service object drives the API, a
batch backfill job or a test with three fakes. The service owns the trace: one
``audit.run`` span per request, with a child span per stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from smart_shelf.agents.inventory_agent import AgentOutcome, InventoryAgent
from smart_shelf.core.config import Settings
from smart_shelf.core.exceptions import InvalidImageError
from smart_shelf.core.logging import get_logger
from smart_shelf.core.observability import Tracer, get_tracer
from smart_shelf.cv.detector import ObjectDetector
from smart_shelf.cv.models import DetectionResult
from smart_shelf.cv.preprocessing import prepare_image
from smart_shelf.db.models import AuditFilter, AuditRecord
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.engine import VisionLanguageEngine, VLMRequest
from smart_shelf.genai.schemas import Planogram, ShelfAuditResult
from smart_shelf.services.compliance import reconcile_with_planogram

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AuditOutcome:
    """Everything one audit produced."""

    record: AuditRecord
    detections: DetectionResult
    agent: AgentOutcome

    @property
    def result(self) -> ShelfAuditResult:
        """The structured audit this run produced."""
        return self.record.result


class AuditService:
    """Runs the shelf audit pipeline and persists the outcome."""

    def __init__(
        self,
        settings: Settings,
        *,
        detector: ObjectDetector,
        engine: VisionLanguageEngine,
        repository: AuditRepository,
        agent: InventoryAgent,
        tracer: Tracer | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._engine = engine
        self._repository = repository
        self._agent = agent
        self._tracer = tracer or get_tracer()

    async def run_audit(
        self,
        image_bytes: bytes,
        *,
        filename: str | None = None,
        planogram: Planogram | None = None,
        store_id: str | None = None,
        shelf_id: str | None = None,
        persist: bool = True,
    ) -> AuditOutcome:
        """Audit one shelf image.

        Raises:
            InvalidImageError: the payload is empty, oversized or undecodable.
            DetectionError: the detection backend failed.
            VisionModelError: the VLM failed or returned unusable output.
            PersistenceError: the audit could not be stored.
        """
        if len(image_bytes) > self._settings.max_image_bytes:
            raise InvalidImageError(
                "uploaded image exceeds the configured size limit",
                details={
                    "size_bytes": len(image_bytes),
                    "limit_bytes": self._settings.max_image_bytes,
                },
            )

        started = time.perf_counter()
        with self._tracer.span(
            "audit.run", store_id=store_id, shelf_id=shelf_id, filename=filename
        ) as root:
            with self._tracer.span("cv.preprocess", trace_id=root.trace_id):
                prepared = prepare_image(image_bytes, max_edge=self._settings.preprocess_max_edge)

            with self._tracer.span("cv.detect", trace_id=root.trace_id) as span:
                detections = self._detector.detect(prepared)
                span.set_output(detections=detections.count, rows=detections.geometry.row_count)

            audit = await self._engine.analyze(
                VLMRequest(
                    image=prepared,
                    detections=detections,
                    planogram=planogram,
                    store_id=store_id,
                    shelf_id=shelf_id,
                )
            )

            if planogram is not None and planogram.entries:
                with self._tracer.span("compliance.reconcile", trace_id=root.trace_id) as span:
                    audit = reconcile_with_planogram(audit, planogram)
                    span.set_output(compliance_score=audit.compliance_score)

            record = AuditRecord.from_result(
                audit,
                image_sha256=prepared.sha256,
                image_filename=filename,
                detector_backend=detections.backend,
                vlm_backend=getattr(self._engine, "name", "unknown"),
                detection_count=detections.count,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

            agent_outcome = await self._agent.run(audit, audit_id=record.audit_id)
            record = record.model_copy(update={"alert_triggered": agent_outcome.alert_triggered})

            if persist:
                await self._repository.save(record)

            root.set_output(
                audit_id=record.audit_id,
                compliance_score=record.compliance_score,
                discrepancies=record.discrepancy_count,
                alert_triggered=record.alert_triggered,
            )
            logger.info(
                "audit.completed",
                audit_id=record.audit_id,
                detections=detections.count,
                compliance_score=record.compliance_score,
                alert_triggered=record.alert_triggered,
                duration_ms=round(record.duration_ms, 2),
            )
            self._tracer.flush()
            return AuditOutcome(record=record, detections=detections, agent=agent_outcome)

    async def history(self, query: AuditFilter | None = None) -> tuple[list[AuditRecord], int]:
        """Return a page of audits plus the total matching the same filter."""
        query = query or AuditFilter()
        records = await self._repository.list_audits(query)
        total = await self._repository.count(query)
        return records, total

    async def get_audit(self, audit_id: str) -> AuditRecord:
        """Fetch one stored audit by id."""
        return await self._repository.get(audit_id)
