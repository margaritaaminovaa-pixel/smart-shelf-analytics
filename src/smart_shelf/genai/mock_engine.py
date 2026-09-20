"""Deterministic offline auditor.

Same schemas and call signature as the production engine, but it derives
products and discrepancies from the detector's geometry using explicit rules
instead of calling a model. That makes the repository usable without an API key
and gives CI a golden baseline: the same image and planogram always produce the
same audit.

Product names come from the planogram when one is supplied, and are generic
placeholders otherwise. It never invents brand names.
"""

from __future__ import annotations

from itertools import pairwise

from smart_shelf.core.logging import get_logger
from smart_shelf.core.observability import Tracer, get_tracer
from smart_shelf.cv.models import Detection, DetectionResult
from smart_shelf.genai.engine import VLMRequest
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

logger = get_logger(__name__)

GAP_WIDTH_FACTOR = 1.5
"""A horizontal gap wider than this multiple of the median facing counts as empty."""


def _facings(count: int) -> str:
    """Render a facing count with correct grammar, e.g. ``1 facing``."""
    return f"{count} facing" if count == 1 else f"{count} facings"


def _stock_state(observed: int, entry: PlanogramEntry) -> StockState:
    if observed <= 0:
        return StockState.OUT_OF_STOCK
    if observed < max(1, entry.min_facings):
        return StockState.LOW_STOCK
    return StockState.IN_STOCK


def _severity_for(observed: int, entry: PlanogramEntry) -> Severity:
    if observed <= 0:
        return Severity.CRITICAL if entry.expected_facings >= 3 else Severity.HIGH
    shortfall = (entry.expected_facings - observed) / entry.expected_facings
    if shortfall >= 0.5:
        return Severity.HIGH
    if shortfall > 0:
        return Severity.MEDIUM
    return Severity.LOW


def _allocate(row: list[Detection], entries: list[PlanogramEntry]) -> dict[str, int]:
    """Split a row's detections across its planogram entries, left to right.

    Each entry receives detections in proportion to its expected facings, in
    planogram order, so a short row starves the right-most entries first - which
    is how facings actually disappear when staff pull stock forward.
    """
    allocation: dict[str, int] = {entry.sku: 0 for entry in entries}
    if not entries:
        return allocation

    available = len(row)
    total_expected = sum(entry.expected_facings for entry in entries) or 1
    for entry in sorted(entries, key=lambda e: e.position_index):
        share = round(available * entry.expected_facings / total_expected)
        allocation[entry.sku] = min(entry.expected_facings, max(0, share))
    return allocation


def _count_gaps(row: list[Detection]) -> int:
    """Count suspiciously wide horizontal gaps between neighbouring facings."""
    if len(row) < 2:
        return 0
    widths = sorted(det.box.width for det in row)
    median_width = widths[len(widths) // 2]
    if median_width <= 0:
        return 0
    ordered = sorted(row, key=lambda det: det.box.x1)
    gaps = 0
    for left, right in pairwise(ordered):
        if (right.box.x1 - left.box.x2) > median_width * GAP_WIDTH_FACTOR:
            gaps += 1
    return gaps


class MockVLMEngine:
    """Rule-based stand-in for a multimodal LLM."""

    name = "mock-vlm"

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tracer = tracer or get_tracer()

    async def analyze(self, request: VLMRequest) -> ShelfAuditResult:
        """Audit ``request`` with the deterministic rule set."""
        with self._tracer.span(
            "vlm.analyze", backend=self.name, detections=request.detections.count
        ) as span:
            result = (
                self._audit_against_planogram(request.detections, request.planogram)
                if request.planogram is not None and request.planogram.entries
                else self._audit_availability(request.detections)
            )
            merged = result.model_copy(
                update={
                    "store_id": request.store_id
                    or (request.planogram.store_id if request.planogram else None),
                    "shelf_id": request.shelf_id
                    or (request.planogram.shelf_id if request.planogram else None),
                    "category": request.planogram.category if request.planogram else None,
                }
            )
            span.set_output(
                products=len(merged.products),
                discrepancies=len(merged.discrepancies),
                compliance_score=merged.compliance_score,
            )
            logger.info(
                "vlm.completed",
                backend=self.name,
                products=len(merged.products),
                discrepancies=len(merged.discrepancies),
            )
            return merged

    # -- rule sets ------------------------------------------------------------

    def _audit_against_planogram(
        self, detections: DetectionResult, planogram: Planogram
    ) -> ShelfAuditResult:
        rows = detections.by_shelf_level()
        entries_by_row: dict[int, list[PlanogramEntry]] = {}
        for entry in planogram.entries:
            entries_by_row.setdefault(entry.shelf_level, []).append(entry)

        products: list[ProductItem] = []
        discrepancies: list[Discrepancy] = []
        satisfied_facings = 0

        for level, entries in sorted(entries_by_row.items()):
            row = rows.get(level, [])
            allocation = _allocate(row, entries)
            for entry in sorted(entries, key=lambda e: e.position_index):
                observed = allocation[entry.sku]
                satisfied_facings += min(observed, entry.expected_facings)
                state = _stock_state(observed, entry)
                products.append(
                    ProductItem(
                        sku=entry.sku,
                        name=entry.name,
                        brand=entry.brand,
                        category=entry.category or planogram.category,
                        facings=observed,
                        shelf_level=entry.shelf_level,
                        position_index=entry.position_index,
                        stock_state=state,
                        price_label_visible=observed > 0,
                        confidence=0.72 if observed else 0.6,
                    )
                )
                if observed >= entry.expected_facings:
                    continue
                kind = (
                    DiscrepancyType.OUT_OF_STOCK
                    if observed == 0
                    else DiscrepancyType.LOW_STOCK
                    if observed < entry.min_facings
                    else DiscrepancyType.INCORRECT_FACINGS
                )
                discrepancies.append(
                    Discrepancy(
                        discrepancy_type=kind,
                        severity=_severity_for(observed, entry),
                        sku=entry.sku,
                        product_name=entry.name,
                        shelf_level=entry.shelf_level,
                        expected=_facings(entry.expected_facings),
                        observed=_facings(observed),
                        description=(
                            f"{entry.name} ({entry.sku}) shows {observed} of "
                            f"{entry.expected_facings} expected facings on shelf level "
                            f"{entry.shelf_level}."
                        ),
                        recommended_action=(
                            f"Restock {_facings(entry.expected_facings - observed)} of {entry.sku}."
                        ),
                        confidence=0.7,
                    )
                )

        expected_total = planogram.total_expected_facings or 1
        score = round(min(1.0, satisfied_facings / expected_total), 4)
        empty_slots = sum(_count_gaps(row) for row in rows.values())
        empty_slots += sum(1 for p in products if p.stock_state is StockState.OUT_OF_STOCK)

        return ShelfAuditResult(
            products=products,
            discrepancies=discrepancies,
            compliance_score=score,
            shelf_occupancy=round(detections.occupancy_ratio(), 4),
            empty_slot_count=empty_slots,
            summary=(
                f"Detected {detections.count} facings across "
                f"{detections.geometry.row_count} shelf rows against a planogram of "
                f"{expected_total} expected facings. "
                f"{len(discrepancies)} deviations require attention."
            ),
        )

    def _audit_availability(self, detections: DetectionResult) -> ShelfAuditResult:
        rows = detections.by_shelf_level()
        products: list[ProductItem] = []
        discrepancies: list[Discrepancy] = []
        gaps_total = 0

        for level, row in rows.items():
            gaps = _count_gaps(row)
            gaps_total += gaps
            products.append(
                ProductItem(
                    sku=None,
                    name=f"Unidentified assortment (shelf level {level})",
                    facings=len(row),
                    shelf_level=level,
                    position_index=0,
                    stock_state=StockState.IN_STOCK if row else StockState.OUT_OF_STOCK,
                    price_label_visible=False,
                    confidence=round(min(0.9, detections.mean_confidence or 0.5), 4),
                )
            )
            if gaps:
                discrepancies.append(
                    Discrepancy(
                        discrepancy_type=DiscrepancyType.OUT_OF_STOCK,
                        severity=Severity.HIGH if gaps >= 2 else Severity.MEDIUM,
                        shelf_level=level,
                        expected="continuous product facing",
                        observed=f"{gaps} visible gap(s)",
                        description=(
                            f"Shelf level {level} shows {gaps} gap(s) wide enough to indicate "
                            "missing product."
                        ),
                        recommended_action=f"Inspect and refill shelf level {level}.",
                        confidence=0.55,
                    )
                )

        occupancy = detections.occupancy_ratio()
        score = round(max(0.0, 1.0 - 0.15 * gaps_total), 4) if detections.count else 0.0
        return ShelfAuditResult(
            products=products,
            discrepancies=discrepancies,
            compliance_score=score,
            shelf_occupancy=round(occupancy, 4),
            empty_slot_count=gaps_total,
            summary=(
                f"No planogram supplied. Detected {detections.count} facings across "
                f"{detections.geometry.row_count} rows with {gaps_total} visible gap(s)."
            ),
        )
