"""Generative-AI layer: structured shelf understanding via multimodal LLMs."""

from __future__ import annotations

from smart_shelf.genai.engine import VisionLanguageEngine, build_vlm_engine
from smart_shelf.genai.mock_engine import MockVLMEngine
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

__all__ = [
    "Discrepancy",
    "DiscrepancyType",
    "MockVLMEngine",
    "Planogram",
    "PlanogramEntry",
    "ProductItem",
    "Severity",
    "ShelfAuditResult",
    "StockState",
    "VisionLanguageEngine",
    "build_vlm_engine",
]
