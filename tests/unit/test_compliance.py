"""Deterministic planogram reconciliation."""

from __future__ import annotations

import pytest

from smart_shelf.genai.schemas import (
    Discrepancy,
    DiscrepancyType,
    Planogram,
    PlanogramEntry,
    ProductItem,
    Severity,
    ShelfAuditResult,
)
from smart_shelf.services.compliance import reconcile_with_planogram, score_compliance

pytestmark = pytest.mark.unit


def entry(sku: str, expected: int = 2, *, level: int = 0, minimum: int = 1) -> PlanogramEntry:
    return PlanogramEntry(
        sku=sku,
        name=f"Product {sku}",
        shelf_level=level,
        expected_facings=expected,
        min_facings=minimum,
    )


def plan(*entries: PlanogramEntry) -> Planogram:
    return Planogram(store_id="STORE-1", shelf_id="BAY-1", entries=list(entries))


def product(sku: str | None, facings: int, *, level: int = 0) -> ProductItem:
    return ProductItem(sku=sku, name=f"Product {sku}", facings=facings, shelf_level=level)


def test_empty_planogram_is_trivially_compliant() -> None:
    assert score_compliance([product("A", 3)], Planogram()) == 1.0


def test_full_shelf_scores_one() -> None:
    assert score_compliance([product("A", 2), product("B", 2)], plan(entry("A"), entry("B"))) == 1.0


def test_half_stocked_shelf_scores_one_half() -> None:
    assert score_compliance([product("A", 2)], plan(entry("A"), entry("B"))) == 0.5


def test_surplus_facings_never_exceed_full_marks() -> None:
    # Over-stocking A must not hide that B is gone.
    assert score_compliance([product("A", 10)], plan(entry("A"), entry("B"))) == 0.5


def test_products_without_a_sku_do_not_count() -> None:
    assert score_compliance([product(None, 5)], plan(entry("A"))) == 0.0


def test_missing_sku_becomes_a_critical_out_of_stock() -> None:
    result = reconcile_with_planogram(
        ShelfAuditResult(products=[product("A", 3)]),
        plan(entry("A", 3), entry("B", 3)),
    )
    missing = [d for d in result.discrepancies if d.sku == "B"]
    assert len(missing) == 1
    assert missing[0].discrepancy_type is DiscrepancyType.OUT_OF_STOCK
    assert missing[0].severity is Severity.CRITICAL
    assert result.compliance_score == 0.5


def test_partial_shortfall_below_minimum_is_low_stock() -> None:
    result = reconcile_with_planogram(
        ShelfAuditResult(products=[product("A", 1)]),
        plan(entry("A", 4, minimum=3)),
    )
    assert result.discrepancies[0].discrepancy_type is DiscrepancyType.LOW_STOCK
    assert result.discrepancies[0].severity is Severity.HIGH


def test_cosmetic_shortfall_is_an_incorrect_facing_count() -> None:
    result = reconcile_with_planogram(
        ShelfAuditResult(products=[product("A", 5)]),
        plan(entry("A", 6, minimum=1)),
    )
    assert result.discrepancies[0].discrepancy_type is DiscrepancyType.INCORRECT_FACINGS
    assert result.discrepancies[0].severity is Severity.LOW


def test_unlisted_product_is_reported_as_unexpected() -> None:
    result = reconcile_with_planogram(
        ShelfAuditResult(products=[product("A", 2), product("ROGUE", 1)]),
        plan(entry("A")),
    )
    rogue = [d for d in result.discrepancies if d.sku == "ROGUE"]
    assert len(rogue) == 1
    assert rogue[0].discrepancy_type is DiscrepancyType.UNEXPECTED_PRODUCT


def test_model_reported_discrepancies_are_not_duplicated() -> None:
    reported = Discrepancy(
        discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
        severity=Severity.HIGH,
        sku="B",
        description="The model already spotted this one.",
    )
    result = reconcile_with_planogram(
        ShelfAuditResult(products=[product("A", 2)], discrepancies=[reported]),
        plan(entry("A"), entry("B")),
    )
    assert [d.description for d in result.discrepancies] == [reported.description]


def test_reconciliation_backfills_shelf_identifiers() -> None:
    result = reconcile_with_planogram(ShelfAuditResult(), plan(entry("A")))
    assert result.store_id == "STORE-1"
    assert result.shelf_id == "BAY-1"


def test_reconciliation_overrides_an_optimistic_model_score() -> None:
    optimistic = ShelfAuditResult(products=[product("A", 1)], compliance_score=1.0)
    assert reconcile_with_planogram(optimistic, plan(entry("A", 4))).compliance_score == 0.25
