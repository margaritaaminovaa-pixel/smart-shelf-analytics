"""Row and query models for the audit store.

:class:`AuditRecord` denormalises the handful of fields the history endpoint and
the analytics queries filter on, and keeps the full :class:`ShelfAuditResult` in
a JSON column. That keeps reads cheap without ever losing the original audit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Self
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from smart_shelf.genai.schemas import Severity, ShelfAuditResult

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class AuditRecord(BaseModel):
    """One persisted shelf audit."""

    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    store_id: str | None = None
    shelf_id: str | None = None
    image_sha256: str = Field(min_length=8, max_length=64)
    image_filename: str | None = Field(default=None, max_length=255)
    detector_backend: str
    vlm_backend: str
    detection_count: int = Field(ge=0)
    product_count: int = Field(ge=0)
    discrepancy_count: int = Field(ge=0)
    compliance_score: UnitFloat
    shelf_occupancy: UnitFloat
    empty_slot_count: int = Field(ge=0)
    max_severity: Severity | None = None
    alert_triggered: bool = False
    summary: str = ""
    duration_ms: float = Field(ge=0)
    result: ShelfAuditResult

    @model_validator(mode="after")
    def _normalize_created_at(self) -> Self:
        if self.created_at.tzinfo is None:
            object.__setattr__(self, "created_at", self.created_at.replace(tzinfo=UTC))
        return self

    @classmethod
    def from_result(
        cls,
        result: ShelfAuditResult,
        *,
        image_sha256: str,
        detector_backend: str,
        vlm_backend: str,
        detection_count: int,
        duration_ms: float,
        image_filename: str | None = None,
        alert_triggered: bool = False,
        audit_id: str | None = None,
    ) -> AuditRecord:
        """Build a persistable record from a completed audit."""
        return cls(
            audit_id=audit_id or str(uuid.uuid4()),
            store_id=result.store_id,
            shelf_id=result.shelf_id,
            image_sha256=image_sha256,
            image_filename=image_filename,
            detector_backend=detector_backend,
            vlm_backend=vlm_backend,
            detection_count=detection_count,
            product_count=len(result.products),
            discrepancy_count=len(result.discrepancies),
            compliance_score=result.compliance_score,
            shelf_occupancy=result.shelf_occupancy,
            empty_slot_count=result.empty_slot_count,
            max_severity=result.max_severity,
            alert_triggered=alert_triggered,
            summary=result.summary,
            duration_ms=duration_ms,
            result=result,
        )

    def to_row(self) -> dict[str, Any]:
        """Flatten into the column layout used by :class:`AuditRepository`."""
        return {
            "audit_id": self.audit_id,
            "created_at": self.created_at.isoformat(),
            "store_id": self.store_id,
            "shelf_id": self.shelf_id,
            "image_sha256": self.image_sha256,
            "image_filename": self.image_filename,
            "detector_backend": self.detector_backend,
            "vlm_backend": self.vlm_backend,
            "detection_count": self.detection_count,
            "product_count": self.product_count,
            "discrepancy_count": self.discrepancy_count,
            "compliance_score": self.compliance_score,
            "shelf_occupancy": self.shelf_occupancy,
            "empty_slot_count": self.empty_slot_count,
            "max_severity": self.max_severity.value if self.max_severity else None,
            "alert_triggered": int(self.alert_triggered),
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "payload": self.result.model_dump_json(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> AuditRecord:
        """Rehydrate from a database row."""
        return cls(
            audit_id=row["audit_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            store_id=row["store_id"],
            shelf_id=row["shelf_id"],
            image_sha256=row["image_sha256"],
            image_filename=row["image_filename"],
            detector_backend=row["detector_backend"],
            vlm_backend=row["vlm_backend"],
            detection_count=row["detection_count"],
            product_count=row["product_count"],
            discrepancy_count=row["discrepancy_count"],
            compliance_score=row["compliance_score"],
            shelf_occupancy=row["shelf_occupancy"],
            empty_slot_count=row["empty_slot_count"],
            max_severity=Severity(row["max_severity"]) if row["max_severity"] else None,
            alert_triggered=bool(row["alert_triggered"]),
            summary=row["summary"] or "",
            duration_ms=row["duration_ms"],
            result=ShelfAuditResult.model_validate_json(row["payload"]),
        )


class AuditFilter(BaseModel):
    """Query filter for the audit history endpoint."""

    model_config = ConfigDict(extra="forbid")

    store_id: str | None = Field(default=None, max_length=64)
    shelf_id: str | None = Field(default=None, max_length=64)
    min_compliance_score: UnitFloat | None = None
    max_compliance_score: UnitFloat | None = None
    alert_triggered: bool | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        low, high = self.min_compliance_score, self.max_compliance_score
        if low is not None and high is not None and low > high:
            msg = "min_compliance_score cannot exceed max_compliance_score"
            raise ValueError(msg)
        if self.since is not None and self.until is not None and self.since > self.until:
            msg = "since cannot be later than until"
            raise ValueError(msg)
        return self
