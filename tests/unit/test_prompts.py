"""Multimodal prompt construction."""

from __future__ import annotations

import pytest

from smart_shelf.cv.detector import HeuristicShelfDetector
from smart_shelf.cv.preprocessing import PreparedImage
from smart_shelf.genai.prompts import SYSTEM_PROMPT, build_user_prompt
from smart_shelf.genai.schemas import Planogram

pytestmark = pytest.mark.unit


def test_system_prompt_states_the_non_hallucination_rule() -> None:
    assert "Never invent a SKU" in SYSTEM_PROMPT
    assert "Row 0 is the" in SYSTEM_PROMPT


def test_prompt_reports_the_detector_evidence(prepared_image: PreparedImage) -> None:
    detections = HeuristicShelfDetector().detect(prepared_image)
    prompt = build_user_prompt(detections, None, store_id="STORE-1", shelf_id="BAY-3")

    assert "STORE-1" in prompt
    assert "BAY-3" in prompt
    assert f"total detections: {detections.count}" in prompt
    assert "per-row breakdown" in prompt
    assert "row 0:" in prompt


def test_prompt_without_a_planogram_asks_for_an_availability_audit(
    prepared_image: PreparedImage,
) -> None:
    detections = HeuristicShelfDetector().detect(prepared_image)
    prompt = build_user_prompt(detections, None)
    assert "None supplied" in prompt
    assert "Audit availability only" in prompt


def test_prompt_embeds_the_planogram_as_json(
    prepared_image: PreparedImage, planogram: Planogram
) -> None:
    detections = HeuristicShelfDetector().detect(prepared_image)
    prompt = build_user_prompt(detections, planogram)

    assert "```json" in prompt
    assert planogram.entries[0].sku in prompt
    assert f"total expected facings: {planogram.total_expected_facings}" in prompt


def test_prompt_handles_an_empty_detection_set(prepared_image: PreparedImage) -> None:
    detections = (
        HeuristicShelfDetector().detect(prepared_image).model_copy(update={"detections": []})
    )
    assert "no objects detected" in build_user_prompt(detections, None)


def test_an_empty_planogram_is_treated_as_absent(prepared_image: PreparedImage) -> None:
    detections = HeuristicShelfDetector().detect(prepared_image)
    assert "None supplied" in build_user_prompt(detections, Planogram())
