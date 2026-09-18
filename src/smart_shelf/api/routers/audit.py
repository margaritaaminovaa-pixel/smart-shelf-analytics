"""Shelf audit endpoints."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from pydantic import ValidationError

from smart_shelf.agents.policies import Urgency
from smart_shelf.api.dependencies import AnalyticsServiceDep, AuditServiceDep
from smart_shelf.api.errors import HTTP_422_UNPROCESSABLE, serializable_errors
from smart_shelf.api.schemas import (
    AgentDecisionResponse,
    AnalyticsResponse,
    AuditHistoryResponse,
    AuditSummaryResponse,
    AuditUploadResponse,
    DetectionSummary,
    ErrorResponse,
    RestockTaskResponse,
)
from smart_shelf.core.exceptions import InvalidImageError, InvalidRequestError
from smart_shelf.db.models import AuditFilter
from smart_shelf.genai.schemas import Planogram
from smart_shelf.services.audit_service import AuditOutcome

router = APIRouter(prefix="/audit", tags=["audit"])

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    HTTP_422_UNPROCESSABLE: {"model": ErrorResponse, "description": "Invalid input"},
    status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse, "description": "Upstream model failure"},
}


def _parse_planogram(raw: str | None, uploaded: bytes | None) -> Planogram | None:
    """Accept the planogram as an inline JSON form field or an uploaded file."""
    payload = raw if raw else (uploaded.decode("utf-8") if uploaded else None)
    if not payload or not payload.strip():
        return None
    try:
        return Planogram.model_validate(json.loads(payload))
    except json.JSONDecodeError as exc:
        raise InvalidImageError("planogram is not valid JSON", details={"cause": str(exc)}) from exc
    except ValidationError as exc:
        raise InvalidImageError(
            "planogram does not match the expected schema",
            details={"errors": serializable_errors(exc)},
        ) from exc


def _to_response(outcome: AuditOutcome) -> AuditUploadResponse:
    record = outcome.record
    return AuditUploadResponse(
        audit_id=record.audit_id,
        created_at=record.created_at,
        compliance_score=record.compliance_score,
        alert_triggered=record.alert_triggered,
        duration_ms=record.duration_ms,
        detection=DetectionSummary(
            backend=outcome.detections.backend,
            detection_count=outcome.detections.count,
            shelf_rows=outcome.detections.geometry.row_count,
            mean_confidence=round(outcome.detections.mean_confidence, 4),
            occupancy_ratio=round(outcome.detections.occupancy_ratio(), 4),
            duration_ms=outcome.detections.duration_ms,
        ),
        result=record.result,
        agent=AgentDecisionResponse(
            alert_triggered=outcome.agent.alert_triggered,
            urgency=outcome.agent.decision.urgency,
            reasons=outcome.agent.decision.reasons,
            notified_channels=outcome.agent.notified_channels,
            tasks=[
                RestockTaskResponse(
                    sku=task.sku,
                    product_name=task.product_name,
                    shelf_level=task.shelf_level,
                    action=task.action,
                    severity=task.severity,
                )
                for task in outcome.agent.tasks
            ],
        ),
    )


@router.post(
    "/upload",
    response_model=AuditUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ERROR_RESPONSES,
    summary="Audit a shelf image",
    description=(
        "Runs the full pipeline on one shelf photograph: preprocessing, object "
        "detection, multimodal structuring, planogram reconciliation and the "
        "autonomous restocking agent. Supply the expected planogram either as an "
        "inline JSON string in `planogram` or as an uploaded file in "
        "`planogram_file`; omit both to audit availability only."
    ),
)
async def upload_audit(
    service: AuditServiceDep,
    image: Annotated[UploadFile, File(description="Shelf photograph (JPEG, PNG or WebP).")],
    planogram: Annotated[
        str | None, Form(description="Expected planogram as an inline JSON object.")
    ] = None,
    planogram_file: Annotated[
        UploadFile | None, File(description="Expected planogram as an uploaded .json file.")
    ] = None,
    store_id: Annotated[str | None, Form(max_length=64)] = None,
    shelf_id: Annotated[str | None, Form(max_length=64)] = None,
) -> AuditUploadResponse:
    """Audit one shelf image and return the structured result."""
    image_bytes = await image.read()
    planogram_bytes = await planogram_file.read() if planogram_file is not None else None
    expected = _parse_planogram(planogram, planogram_bytes)

    outcome = await service.run_audit(
        image_bytes,
        filename=image.filename,
        planogram=expected,
        store_id=store_id or (expected.store_id if expected else None),
        shelf_id=shelf_id or (expected.shelf_id if expected else None),
    )
    return _to_response(outcome)


@router.get(
    "/history",
    response_model=AuditHistoryResponse,
    summary="List past audits",
    description=(
        "Returns audits newest first. Filter by compliance score to surface the "
        "shelves that need attention, e.g. `?max_compliance_score=0.8`."
    ),
)
async def audit_history(
    service: AuditServiceDep,
    store_id: Annotated[str | None, Query(max_length=64)] = None,
    shelf_id: Annotated[str | None, Query(max_length=64)] = None,
    min_compliance_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_compliance_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    alert_triggered: Annotated[bool | None, Query()] = None,
    since: Annotated[datetime | None, Query()] = None,
    until: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditHistoryResponse:
    """Page through the audit history."""
    try:
        query = AuditFilter(
            store_id=store_id,
            shelf_id=shelf_id,
            min_compliance_score=min_compliance_score,
            max_compliance_score=max_compliance_score,
            alert_triggered=alert_triggered,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    except ValidationError as exc:
        # Per-parameter bounds are enforced by FastAPI; this catches the
        # cross-parameter rules (inverted score or time ranges).
        raise InvalidRequestError(
            "query parameters are inconsistent", details={"errors": serializable_errors(exc)}
        ) from exc
    records, total = await service.history(query)
    return AuditHistoryResponse(
        items=[AuditSummaryResponse.from_record(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Aggregated compliance analytics",
    description="Headline KPIs, a daily compliance trend and the worst-offending SKUs.",
)
async def audit_analytics(
    analytics: AnalyticsServiceDep,
    trend_days: Annotated[int, Query(ge=1, le=365)] = 14,
    top_n: Annotated[int, Query(ge=1, le=50)] = 10,
) -> AnalyticsResponse:
    """Return dashboard-ready aggregates over the audit history."""
    summary = await analytics.summary(trend_days=trend_days)
    offenders = await analytics.top_offenders(limit=top_n)
    return AnalyticsResponse(
        total_audits=summary.total_audits,
        mean_compliance=summary.mean_compliance,
        alert_rate=summary.alert_rate,
        out_of_stock_events=summary.out_of_stock_events,
        worst_store_id=summary.worst_store_id,
        trend=[asdict(point) for point in summary.trend],
        top_offenders=[
            {**asdict(offender), "dominant_type": offender.dominant_type.value}
            for offender in offenders
        ],
    )


@router.get(
    "/{audit_id}",
    response_model=AuditUploadResponse,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    summary="Fetch one audit by id",
    response_model_exclude={"detection", "agent"},
)
async def get_audit(service: AuditServiceDep, audit_id: str) -> AuditUploadResponse:
    """Return a previously stored audit."""
    record = await service.get_audit(audit_id)
    return AuditUploadResponse(
        audit_id=record.audit_id,
        created_at=record.created_at,
        compliance_score=record.compliance_score,
        alert_triggered=record.alert_triggered,
        duration_ms=record.duration_ms,
        detection=DetectionSummary(
            backend=record.detector_backend,
            detection_count=record.detection_count,
            shelf_rows=0,
            mean_confidence=0.0,
            occupancy_ratio=record.shelf_occupancy,
            duration_ms=0.0,
        ),
        result=record.result,
        agent=AgentDecisionResponse(
            alert_triggered=record.alert_triggered,
            urgency=Urgency.NONE,
            reasons=[],
            notified_channels=[],
            tasks=[],
        ),
    )
