"""Pydantic contracts for the structured audit."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError
import pytest

from smart_shelf.genai.schemas import (
    Discrepancy,
    DiscrepancyType,
    Planogram,
    PlanogramEntry,
    ProductItem,
    Severity,
    ShelfAuditResult,
    StockState,
)

pytestmark = pytest.mark.unit


def discrepancy(severity: Severity, sku: str = "SKU-1") -> Discrepancy:
    return Discrepancy(
        discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
        severity=severity,
        sku=sku,
        description="missing",
    )


def test_severity_ranks_are_ordered() -> None:
    ranks = [s.rank for s in (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == 4


def test_extra_fields_are_rejected() -> None:
    # The VLM is schema-constrained; a stray field means the contract drifted.
    with pytest.raises(ValidationError):
        ProductItem(name="Cola", hallucinated_field="oops")


def test_product_defaults_are_conservative() -> None:
    item = ProductItem(name="Cola 330ml")
    assert item.facings == 1
    assert item.stock_state is StockState.IN_STOCK
    assert item.sku is None


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ProductItem(name="Cola", confidence=1.2)


def test_total_facings_sums_every_group() -> None:
    result = ShelfAuditResult(
        products=[
            ProductItem(name="A", facings=3),
            ProductItem(name="B", facings=2),
        ]
    )
    assert result.total_facings == 5


def test_max_severity_picks_the_worst() -> None:
    result = ShelfAuditResult(
        discrepancies=[
            discrepancy(Severity.LOW, "a"),
            discrepancy(Severity.CRITICAL, "b"),
            discrepancy(Severity.MEDIUM, "c"),
        ]
    )
    assert result.max_severity is Severity.CRITICAL


def test_max_severity_is_none_without_discrepancies() -> None:
    assert ShelfAuditResult().max_severity is None


def test_computed_fields_survive_serialisation() -> None:
    result = ShelfAuditResult(products=[ProductItem(name="A", facings=4)])
    assert result.model_dump()["total_facings"] == 4


def test_naive_timestamps_are_coerced_to_utc() -> None:
    result = ShelfAuditResult(captured_at=datetime(2026, 1, 1, 12, 0, 0))  # noqa: DTZ001
    assert result.captured_at.tzinfo is UTC


def test_discrepancies_can_be_filtered_by_type() -> None:
    result = ShelfAuditResult(
        discrepancies=[
            discrepancy(Severity.HIGH, "a"),
            Discrepancy(
                discrepancy_type=DiscrepancyType.MISSING_PRICE_TAG,
                severity=Severity.LOW,
                description="no tag",
            ),
        ]
    )
    assert len(result.discrepancies_of(DiscrepancyType.OUT_OF_STOCK)) == 1
    assert (
        len(result.discrepancies_of(DiscrepancyType.OUT_OF_STOCK, DiscrepancyType.LOW_STOCK)) == 1
    )


def test_planogram_entry_rejects_min_above_expected() -> None:
    with pytest.raises(ValidationError, match="min_facings cannot exceed"):
        PlanogramEntry(sku="S", name="N", shelf_level=0, expected_facings=2, min_facings=3)


def test_planogram_totals_and_index() -> None:
    planogram = Planogram(
        entries=[
            PlanogramEntry(sku="A", name="A", shelf_level=0, expected_facings=2),
            PlanogramEntry(sku="B", name="B", shelf_level=0, position_index=1, expected_facings=3),
        ]
    )
    assert planogram.total_expected_facings == 5
    assert set(planogram.by_sku()) == {"A", "B"}


def test_duplicate_skus_merge_their_expected_facings() -> None:
    planogram = Planogram(
        entries=[
            PlanogramEntry(sku="A", name="A", shelf_level=0, expected_facings=2, min_facings=1),
            PlanogramEntry(sku="A", name="A", shelf_level=1, expected_facings=3, min_facings=2),
        ]
    )
    merged = planogram.by_sku()["A"]
    assert merged.expected_facings == 5
    assert merged.min_facings == 3


def test_audit_result_roundtrips_through_json() -> None:
    original = ShelfAuditResult(
        store_id="S1",
        products=[ProductItem(sku="A", name="A", facings=2)],
        discrepancies=[discrepancy(Severity.HIGH)],
        compliance_score=0.5,
    )
    assert ShelfAuditResult.model_validate_json(original.model_dump_json()) == original
