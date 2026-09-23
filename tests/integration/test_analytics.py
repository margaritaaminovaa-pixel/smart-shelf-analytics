"""Aggregations over the audit history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from smart_shelf.db.analytics import AnalyticsService
from smart_shelf.db.models import AuditRecord
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.schemas import (
    Discrepancy,
    DiscrepancyType,
    ProductItem,
    Severity,
    ShelfAuditResult,
)

pytestmark = pytest.mark.integration

BASE_TIME = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)


def record(
    *,
    score: float,
    store: str = "STORE-1",
    alert: bool = False,
    day_offset: int = 0,
    empty_slots: int = 0,
    failing: tuple[tuple[str, Severity], ...] = (),
) -> AuditRecord:
    result = ShelfAuditResult(
        store_id=store,
        shelf_id="BAY-1",
        compliance_score=score,
        empty_slot_count=empty_slots,
        products=[ProductItem(sku=sku, name=sku, facings=1) for sku, _ in failing],
        discrepancies=[
            Discrepancy(
                discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
                severity=severity,
                sku=sku,
                product_name=f"Product {sku}",
                description=f"{sku} missing",
            )
            for sku, severity in failing
        ],
    )
    built = AuditRecord.from_result(
        result,
        image_sha256="b" * 64,
        detector_backend="opencv-heuristic",
        vlm_backend="mock-vlm",
        detection_count=5,
        duration_ms=10.0,
        alert_triggered=alert,
    )
    return built.model_copy(update={"created_at": BASE_TIME + timedelta(days=day_offset)})


@pytest.fixture
def analytics(repository: AuditRepository) -> AnalyticsService:
    return AnalyticsService(repository)


async def test_summary_of_an_empty_store(analytics: AnalyticsService) -> None:
    summary = await analytics.summary()
    assert summary.total_audits == 0
    assert summary.mean_compliance == 0.0
    assert summary.alert_rate == 0.0
    assert summary.worst_store_id is None
    assert summary.trend == []


async def test_summary_aggregates_scores_and_alerts(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    await repository.save(record(score=1.0))
    await repository.save(record(score=0.5, alert=True, empty_slots=4))

    summary = await analytics.summary()
    assert summary.total_audits == 2
    assert summary.mean_compliance == 0.75
    assert summary.alert_rate == 0.5
    assert summary.out_of_stock_events == 4


async def test_summary_names_the_worst_store(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    await repository.save(record(score=0.95, store="STORE-GOOD"))
    await repository.save(record(score=0.30, store="STORE-BAD"))
    await repository.save(record(score=0.40, store="STORE-BAD", day_offset=1))

    assert (await analytics.summary()).worst_store_id == "STORE-BAD"


async def test_trend_is_one_point_per_day_oldest_first(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    await repository.save(record(score=0.4, day_offset=0))
    await repository.save(record(score=0.6, day_offset=0))
    await repository.save(record(score=0.9, day_offset=1, alert=True))

    trend = await analytics.compliance_trend(days=30)
    assert [point.day for point in trend] == ["2026-09-01", "2026-09-02"]
    assert trend[0].audit_count == 2
    assert trend[0].mean_compliance == 0.5
    assert trend[1].alert_count == 1


async def test_trend_window_is_bounded(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    for offset in range(6):
        await repository.save(record(score=0.5, day_offset=offset))

    assert len(await analytics.compliance_trend(days=3)) == 3


async def test_top_offenders_rank_by_frequency(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    await repository.save(
        record(score=0.4, failing=(("SKU-A", Severity.CRITICAL), ("SKU-B", Severity.LOW)))
    )
    await repository.save(record(score=0.4, day_offset=1, failing=(("SKU-A", Severity.CRITICAL),)))

    offenders = await analytics.top_offenders()
    assert [o.sku for o in offenders] == ["SKU-A", "SKU-B"]
    assert offenders[0].occurrences == 2
    assert offenders[0].critical_occurrences == 2
    assert offenders[0].product_name == "Product SKU-A"
    assert offenders[0].dominant_type is DiscrepancyType.OUT_OF_STOCK


async def test_top_offenders_are_capped(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    await repository.save(
        record(score=0.1, failing=tuple((f"SKU-{i}", Severity.HIGH) for i in range(8)))
    )
    assert len(await analytics.top_offenders(limit=3)) == 3


async def test_discrepancies_without_a_sku_are_skipped(
    repository: AuditRepository, analytics: AnalyticsService
) -> None:
    anonymous = ShelfAuditResult(
        compliance_score=0.2,
        discrepancies=[
            Discrepancy(
                discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
                severity=Severity.HIGH,
                description="a gap with no label",
            )
        ],
    )
    await repository.save(
        AuditRecord.from_result(
            anonymous,
            image_sha256="c" * 64,
            detector_backend="x",
            vlm_backend="y",
            detection_count=0,
            duration_ms=1.0,
        )
    )
    assert await analytics.top_offenders() == []
