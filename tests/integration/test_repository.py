"""Audit store reads, writes and filtering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from smart_shelf.core.exceptions import AuditNotFoundError, PersistenceError
from smart_shelf.db.models import AuditFilter, AuditRecord
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
    score: float = 0.9,
    store: str = "STORE-1",
    shelf: str = "BAY-1",
    alert: bool = False,
    day_offset: int = 0,
    skus: tuple[str, ...] = ("SKU-1",),
) -> AuditRecord:
    result = ShelfAuditResult(
        store_id=store,
        shelf_id=shelf,
        compliance_score=score,
        products=[ProductItem(sku=sku, name=sku, facings=2) for sku in skus],
        discrepancies=[
            Discrepancy(
                discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
                severity=Severity.CRITICAL,
                sku=sku,
                product_name=sku,
                description=f"{sku} missing",
            )
            for sku in skus
        ],
        summary="synthetic",
    )
    built = AuditRecord.from_result(
        result,
        image_sha256="a" * 64,
        detector_backend="opencv-heuristic",
        vlm_backend="mock-vlm",
        detection_count=12,
        duration_ms=42.0,
        alert_triggered=alert,
    )
    return built.model_copy(update={"created_at": BASE_TIME + timedelta(days=day_offset)})


async def test_save_and_fetch_roundtrip(repository: AuditRepository) -> None:
    saved = await repository.save(record(score=0.73))
    fetched = await repository.get(saved.audit_id)

    assert fetched.audit_id == saved.audit_id
    assert fetched.compliance_score == 0.73
    assert fetched.result == saved.result
    assert fetched.created_at == saved.created_at


async def test_missing_audit_raises(repository: AuditRepository) -> None:
    with pytest.raises(AuditNotFoundError, match="does not exist"):
        await repository.get("no-such-id")


async def test_saving_twice_replaces_the_row(repository: AuditRepository) -> None:
    first = await repository.save(record(score=0.5))
    await repository.save(first.model_copy(update={"compliance_score": 0.95}))

    assert await repository.count() == 1
    assert (await repository.get(first.audit_id)).compliance_score == 0.95


async def test_history_is_newest_first(repository: AuditRepository) -> None:
    for offset in (0, 2, 1):
        await repository.save(record(day_offset=offset))

    listed = await repository.list_audits()
    assert [r.created_at for r in listed] == sorted((r.created_at for r in listed), reverse=True)


async def test_filter_by_compliance_range(repository: AuditRepository) -> None:
    for score in (0.2, 0.55, 0.9):
        await repository.save(record(score=score))

    low = await repository.list_audits(AuditFilter(max_compliance_score=0.6))
    assert sorted(r.compliance_score for r in low) == [0.2, 0.55]

    band = await repository.list_audits(
        AuditFilter(min_compliance_score=0.5, max_compliance_score=0.95)
    )
    assert sorted(r.compliance_score for r in band) == [0.55, 0.9]


async def test_filter_by_store_and_shelf(repository: AuditRepository) -> None:
    await repository.save(record(store="STORE-1", shelf="BAY-1"))
    await repository.save(record(store="STORE-2", shelf="BAY-1"))
    await repository.save(record(store="STORE-2", shelf="BAY-9"))

    assert len(await repository.list_audits(AuditFilter(store_id="STORE-2"))) == 2
    assert len(await repository.list_audits(AuditFilter(shelf_id="BAY-9"))) == 1


async def test_filter_by_alert_flag(repository: AuditRepository) -> None:
    await repository.save(record(alert=True))
    await repository.save(record(alert=False))

    assert len(await repository.list_audits(AuditFilter(alert_triggered=True))) == 1


async def test_filter_by_time_window(repository: AuditRepository) -> None:
    for offset in range(4):
        await repository.save(record(day_offset=offset))

    window = await repository.list_audits(
        AuditFilter(since=BASE_TIME + timedelta(days=1), until=BASE_TIME + timedelta(days=2))
    )
    assert len(window) == 2


async def test_pagination(repository: AuditRepository) -> None:
    for offset in range(5):
        await repository.save(record(day_offset=offset))

    page = await repository.list_audits(AuditFilter(limit=2, offset=2))
    assert len(page) == 2
    assert await repository.count() == 5
    # count ignores pagination but honours the filter
    assert await repository.count(AuditFilter(limit=2, max_compliance_score=0.1)) == 0


async def test_delete_removes_the_row(repository: AuditRepository) -> None:
    saved = await repository.save(record())
    await repository.delete(saved.audit_id)

    assert await repository.count() == 0
    with pytest.raises(AuditNotFoundError):
        await repository.delete(saved.audit_id)


async def test_health_check_reports_a_live_connection(repository: AuditRepository) -> None:
    assert await repository.health_check() is True


async def test_using_a_closed_repository_fails_loudly() -> None:
    repo = AuditRepository(":memory:")
    with pytest.raises(PersistenceError, match="not connected"):
        await repo.get("anything")


async def test_repository_is_an_async_context_manager() -> None:
    async with AuditRepository(":memory:") as repo:
        await repo.save(record())
        assert await repo.count() == 1


async def test_connect_is_idempotent(repository: AuditRepository) -> None:
    same = await repository.connect()
    assert same is repository


async def test_file_backed_store_creates_its_directory(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    target = tmp_path / "nested" / "audits.db"
    async with AuditRepository(str(target)) as repo:
        await repo.save(record())
    assert target.exists()
