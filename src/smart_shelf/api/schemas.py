"""Request and response bodies for the public API.

Transport models are kept separate from the domain models so the wire format can
evolve (pagination envelopes, added links, deprecations) without forcing changes
on the VLM schemas or the persistence layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from smart_shelf.agents.policies import Urgency
from smart_shelf.db.models import AuditRecord
from smart_shelf.genai.schemas import Severity, ShelfAuditResult

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class DetectionSummary(BaseModel):
    """What the CV stage contributed, without shipping every bounding box."""

    backend: str
    detection_count: int = Field(ge=0)
    shelf_rows: int = Field(ge=0)
    mean_confidence: UnitFloat
    occupancy_ratio: UnitFloat
    duration_ms: float = Field(ge=0)


class RestockTaskResponse(BaseModel):
    """One prioritised action for store staff."""

    sku: str | None = None
    product_name: str
    shelf_level: int | None = None
    action: str
    severity: Severity


class AgentDecisionResponse(BaseModel):
    """What the autonomous agent decided and which channels it reached."""

    alert_triggered: bool
    urgency: Urgency
    reasons: list[str] = Field(default_factory=list)
    notified_channels: list[str] = Field(default_factory=list)
    tasks: list[RestockTaskResponse] = Field(default_factory=list)


class AuditUploadResponse(BaseModel):
    """``POST /api/v1/audit/upload`` response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audit_id": "6f1d5c3e-3a1e-4f0f-9f2a-9a7c6f1f7a11",
                "created_at": "2026-09-22T09:15:02.441Z",
                "compliance_score": 0.71,
                "alert_triggered": True,
                "detection": {
                    "backend": "opencv-heuristic",
                    "detection_count": 34,
                    "shelf_rows": 3,
                    "mean_confidence": 0.78,
                    "occupancy_ratio": 0.46,
                    "duration_ms": 61.2,
                },
                "agent": {
                    "alert_triggered": True,
                    "urgency": "immediate",
                    "reasons": ["compliance score 0.71 is below the 0.80 threshold"],
                    "notified_channels": ["log"],
                },
            }
        }
    )

    audit_id: str
    created_at: datetime
    compliance_score: UnitFloat
    alert_triggered: bool
    duration_ms: float = Field(ge=0)
    detection: DetectionSummary
    result: ShelfAuditResult
    agent: AgentDecisionResponse


class AuditSummaryResponse(BaseModel):
    """One row of the audit history listing."""

    audit_id: str
    created_at: datetime
    store_id: str | None = None
    shelf_id: str | None = None
    compliance_score: UnitFloat
    shelf_occupancy: UnitFloat
    discrepancy_count: int = Field(ge=0)
    product_count: int = Field(ge=0)
    empty_slot_count: int = Field(ge=0)
    max_severity: Severity | None = None
    alert_triggered: bool
    detector_backend: str
    vlm_backend: str
    summary: str = ""

    @classmethod
    def from_record(cls, record: AuditRecord) -> AuditSummaryResponse:
        """Project a stored record onto the listing row."""
        return cls(
            audit_id=record.audit_id,
            created_at=record.created_at,
            store_id=record.store_id,
            shelf_id=record.shelf_id,
            compliance_score=record.compliance_score,
            shelf_occupancy=record.shelf_occupancy,
            discrepancy_count=record.discrepancy_count,
            product_count=record.product_count,
            empty_slot_count=record.empty_slot_count,
            max_severity=record.max_severity,
            alert_triggered=record.alert_triggered,
            detector_backend=record.detector_backend,
            vlm_backend=record.vlm_backend,
            summary=record.summary,
        )


class AuditHistoryResponse(BaseModel):
    """``GET /api/v1/audit/history`` response."""

    items: list[AuditSummaryResponse] = Field(default_factory=list)
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ComponentHealth(BaseModel):
    """Health of one dependency."""

    name: str
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """``GET /health`` response."""

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    components: list[ComponentHealth] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    """``GET /api/v1/audit/analytics`` response."""

    total_audits: int = Field(ge=0)
    mean_compliance: UnitFloat
    alert_rate: UnitFloat
    out_of_stock_events: int = Field(ge=0)
    worst_store_id: str | None = None
    trend: list[dict[str, Any]] = Field(default_factory=list)
    top_offenders: list[dict[str, Any]] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    """Body of the canonical error envelope."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Every non-2xx response uses this shape."""

    error: ErrorDetail
    request_id: str | None = None
