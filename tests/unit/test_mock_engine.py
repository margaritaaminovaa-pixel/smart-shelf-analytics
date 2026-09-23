"""The deterministic offline auditor."""

from __future__ import annotations

import pytest

from smart_shelf.core.observability import NoOpTracer
from smart_shelf.cv.detector import HeuristicShelfDetector
from smart_shelf.cv.preprocessing import PreparedImage, prepare_image
from smart_shelf.genai.engine import VLMRequest
from smart_shelf.genai.mock_engine import MockVLMEngine, _count_gaps
from smart_shelf.genai.schemas import DiscrepancyType, Planogram, StockState

pytestmark = pytest.mark.unit


def request_for(
    image: PreparedImage, planogram: Planogram | None = None, **kwargs: str
) -> VLMRequest:
    detections = HeuristicShelfDetector().detect(image)
    return VLMRequest(image=image, detections=detections, planogram=planogram, **kwargs)


async def test_compliant_shelf_scores_perfectly(
    prepared_image: PreparedImage, planogram: Planogram
) -> None:
    result = await MockVLMEngine(NoOpTracer()).analyze(request_for(prepared_image, planogram))
    assert result.compliance_score == 1.0
    assert result.discrepancies == []
    assert result.total_facings == planogram.total_expected_facings


async def test_stockout_shelf_reports_missing_facings(
    stockout_image_bytes: bytes, planogram: Planogram
) -> None:
    prepared = prepare_image(stockout_image_bytes)
    result = await MockVLMEngine(NoOpTracer()).analyze(request_for(prepared, planogram))

    assert result.compliance_score < 0.8
    assert result.discrepancies
    assert any(
        d.discrepancy_type in {DiscrepancyType.OUT_OF_STOCK, DiscrepancyType.LOW_STOCK}
        for d in result.discrepancies
    )
    assert any(p.stock_state is not StockState.IN_STOCK for p in result.products)


async def test_every_planogram_entry_becomes_a_product(
    prepared_image: PreparedImage, planogram: Planogram
) -> None:
    result = await MockVLMEngine(NoOpTracer()).analyze(request_for(prepared_image, planogram))
    assert {p.sku for p in result.products} == {e.sku for e in planogram.entries}


async def test_analysis_is_deterministic(
    prepared_image: PreparedImage, planogram: Planogram
) -> None:
    engine = MockVLMEngine(NoOpTracer())
    first = await engine.analyze(request_for(prepared_image, planogram))
    second = await engine.analyze(request_for(prepared_image, planogram))
    assert first.model_dump(exclude={"captured_at"}) == second.model_dump(exclude={"captured_at"})


async def test_availability_only_audit_without_a_planogram(
    prepared_image: PreparedImage,
) -> None:
    result = await MockVLMEngine(NoOpTracer()).analyze(request_for(prepared_image))
    assert result.products
    assert all(p.sku is None for p in result.products)
    assert 0.0 <= result.compliance_score <= 1.0


async def test_identifiers_are_carried_through(
    prepared_image: PreparedImage, planogram: Planogram
) -> None:
    result = await MockVLMEngine(NoOpTracer()).analyze(
        request_for(prepared_image, planogram, store_id="STORE-9", shelf_id="BAY-1")
    )
    assert result.store_id == "STORE-9"
    assert result.shelf_id == "BAY-1"


async def test_empty_detections_produce_a_zero_score(
    prepared_image: PreparedImage, planogram: Planogram
) -> None:
    detections = HeuristicShelfDetector(max_items=1).detect(prepared_image)
    empty = detections.model_copy(update={"detections": []})
    result = await MockVLMEngine(NoOpTracer()).analyze(
        VLMRequest(image=prepared_image, detections=empty, planogram=planogram)
    )
    assert result.compliance_score == 0.0
    assert all(p.stock_state is StockState.OUT_OF_STOCK for p in result.products)


def test_gap_counting_needs_at_least_two_facings(prepared_image: PreparedImage) -> None:
    rows = HeuristicShelfDetector().detect(prepared_image).by_shelf_level()
    assert _count_gaps([]) == 0
    assert _count_gaps(rows[0][:1]) == 0
    assert _count_gaps(rows[0]) == 0  # a fully stocked row has no wide gaps
