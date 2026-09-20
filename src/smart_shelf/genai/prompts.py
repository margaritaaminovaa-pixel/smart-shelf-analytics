"""Prompt construction for the multimodal auditor.

The system prompt fixes the auditor persona and the rules it must not break; the
user prompt carries the CV evidence (row layout, box counts, occupancy) and the
expected planogram. Giving the model the detector's geometry rather than asking
it to count from scratch measurably reduces facing-count hallucination, because
the hard numeric work is done by a deterministic component.
"""

from __future__ import annotations

import json

from smart_shelf.cv.models import DetectionResult
from smart_shelf.genai.schemas import Planogram

SYSTEM_PROMPT = """\
You are a retail shelf auditor for a grocery chain. You are given one shelf
photograph, the output of an object detector run on that photograph, and
optionally the planogram the shelf is supposed to follow.

Rules:
1. Report only what is visible. Never invent a SKU, brand or price you cannot read.
2. Use the detector's shelf-row layout as ground truth for geometry. Row 0 is the
   topmost visible row.
3. `facings` is the number of identical units visible side by side in one group.
4. Raise a discrepancy for every planogram entry that is missing, under-stocked,
   misplaced, or has the wrong facing count, and for every product present that
   the planogram does not list.
5. Severity reflects lost sales: a fully out-of-stock fast-moving SKU is `critical`
   or `high`; a cosmetic facing-count deviation is `low`.
6. When no planogram is supplied, audit for availability only: gaps, empty slots
   and missing price labels.
7. Set `confidence` honestly. Low confidence is more useful than a confident guess.
8. `summary` is two sentences, written for a store manager, no markdown.
"""


def build_user_prompt(
    detections: DetectionResult,
    planogram: Planogram | None,
    *,
    store_id: str | None = None,
    shelf_id: str | None = None,
) -> str:
    """Render the textual half of the multimodal message."""
    rows = detections.by_shelf_level()
    row_lines = [
        f"  - row {level}: {len(items)} detected facings, "
        f"mean confidence {sum(i.confidence for i in items) / len(items):.2f}"
        for level, items in rows.items()
        if items
    ] or ["  - no objects detected"]

    sections: list[str] = [
        "## Shelf context",
        f"- store_id: {store_id or 'unknown'}",
        f"- shelf_id: {shelf_id or 'unknown'}",
        "",
        "## Detector evidence",
        f"- backend: {detections.backend}",
        f"- image size: {detections.image_width}x{detections.image_height}px",
        f"- total detections: {detections.count}",
        f"- shelf rows found: {detections.geometry.row_count}",
        f"- estimated shelf occupancy: {detections.occupancy_ratio():.2f}",
        "- per-row breakdown:",
        *row_lines,
    ]

    if planogram is not None and planogram.entries:
        payload = json.dumps(
            [entry.model_dump(mode="json") for entry in planogram.entries],
            indent=2,
            ensure_ascii=False,
        )
        sections += [
            "",
            "## Expected planogram",
            f"- total expected facings: {planogram.total_expected_facings}",
            "```json",
            payload,
            "```",
            "",
            "Compare the photograph against this planogram entry by entry and report"
            " every deviation.",
        ]
    else:
        sections += [
            "",
            "## Expected planogram",
            "None supplied. Audit availability only: gaps, empty slots, missing price"
            " labels and damaged packaging.",
        ]

    sections += [
        "",
        "Return the structured audit now.",
    ]
    return "\n".join(sections)
