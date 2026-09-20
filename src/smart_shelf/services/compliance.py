"""Deterministic planogram reconciliation.

Models read shelves well but do not do stable arithmetic across runs. When an
expected planogram is supplied, the model's product list is treated as the
observation and everything scoreable is recomputed here: SKUs it failed to
mention, products the planogram does not list, and the compliance score itself.
The model's own ``compliance_score`` is used only for unconstrained audits.
"""

from __future__ import annotations

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


def _facings(count: int) -> str:
    """Render a facing count with correct grammar, e.g. ``1 facing``."""
    return f"{count} facing" if count == 1 else f"{count} facings"


def score_compliance(products: list[ProductItem], planogram: Planogram) -> float:
    """Fraction of expected facings actually present, clamped to ``[0, 1]``.

    Surplus facings never push the score above 1.0: over-stocking one SKU does
    not compensate for another being absent.
    """
    expected_total = planogram.total_expected_facings
    if expected_total <= 0:
        return 1.0

    observed: dict[str, int] = {}
    for item in products:
        if item.sku:
            observed[item.sku] = observed.get(item.sku, 0) + item.facings

    satisfied = sum(
        min(observed.get(sku, 0), entry.expected_facings)
        for sku, entry in planogram.by_sku().items()
    )
    return round(min(1.0, satisfied / expected_total), 4)


def _severity_for_shortfall(observed: int, entry: PlanogramEntry) -> Severity:
    if observed <= 0:
        return Severity.CRITICAL if entry.expected_facings >= 3 else Severity.HIGH
    if observed < entry.min_facings:
        return Severity.HIGH
    shortfall = (entry.expected_facings - observed) / entry.expected_facings
    return Severity.MEDIUM if shortfall >= 0.34 else Severity.LOW


def _discrepancy_key(discrepancy: Discrepancy) -> tuple[str | None, str]:
    return (discrepancy.sku, discrepancy.discrepancy_type.value)


def reconcile_with_planogram(result: ShelfAuditResult, planogram: Planogram) -> ShelfAuditResult:
    """Return a copy of ``result`` with the planogram-derived facts recomputed.

    Discrepancies the model already reported for a ``(sku, type)`` pair are kept
    as-is - its prose is usually better than a template - and only the gaps it
    missed are filled in.
    """
    observed: dict[str, int] = {}
    for item in result.products:
        if item.sku:
            observed[item.sku] = observed.get(item.sku, 0) + item.facings

    expected_index = planogram.by_sku()
    existing = {_discrepancy_key(d) for d in result.discrepancies}
    added: list[Discrepancy] = []

    for sku, entry in expected_index.items():
        count = observed.get(sku, 0)
        if count >= entry.expected_facings:
            continue
        kind = (
            DiscrepancyType.OUT_OF_STOCK
            if count == 0
            else DiscrepancyType.LOW_STOCK
            if count < entry.min_facings
            else DiscrepancyType.INCORRECT_FACINGS
        )
        if (sku, kind.value) in existing:
            continue
        added.append(
            Discrepancy(
                discrepancy_type=kind,
                severity=_severity_for_shortfall(count, entry),
                sku=sku,
                product_name=entry.name,
                shelf_level=entry.shelf_level,
                expected=_facings(entry.expected_facings),
                observed=_facings(count),
                description=(
                    f"Planogram reconciliation: {entry.name} ({sku}) has {count} of "
                    f"{entry.expected_facings} expected facings on shelf level "
                    f"{entry.shelf_level}."
                ),
                recommended_action=f"Restock {_facings(entry.expected_facings - count)} of {sku}.",
                confidence=0.95,
            )
        )

    for item in result.products:
        if not item.sku or item.sku in expected_index or item.facings <= 0:
            continue
        if (item.sku, DiscrepancyType.UNEXPECTED_PRODUCT.value) in existing:
            continue
        added.append(
            Discrepancy(
                discrepancy_type=DiscrepancyType.UNEXPECTED_PRODUCT,
                severity=Severity.MEDIUM,
                sku=item.sku,
                product_name=item.name,
                shelf_level=item.shelf_level,
                expected="not listed in the planogram",
                observed=f"{_facings(item.facings)} present",
                description=(
                    f"{item.name} ({item.sku}) occupies shelf level {item.shelf_level} but is "
                    "not part of the planogram for this shelf."
                ),
                recommended_action=f"Remove or relocate {item.sku} and restore the planogram.",
                confidence=0.85,
            )
        )

    merged = [*result.discrepancies, *added]
    out_of_stock = sum(1 for item in result.products if item.stock_state is StockState.OUT_OF_STOCK)
    empty_slots = max(
        result.empty_slot_count,
        out_of_stock + sum(1 for d in added if d.discrepancy_type is DiscrepancyType.OUT_OF_STOCK),
    )

    return result.model_copy(
        update={
            "discrepancies": merged,
            "compliance_score": score_compliance(result.products, planogram),
            "empty_slot_count": empty_slots,
            "store_id": result.store_id or planogram.store_id,
            "shelf_id": result.shelf_id or planogram.shelf_id,
            "category": result.category or planogram.category,
        }
    )
