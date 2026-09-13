#!/usr/bin/env python
"""Generate deterministic synthetic shelf images, planograms and fixtures.

Real retail photography cannot be committed to a public repository, so the
repository ships a generator instead of pixels-from-a-store. The images are
crude on purpose: flat-coloured cartons on lit shelves, with controllable gaps.
That is enough to exercise the contour detector, the row-clustering, the
planogram reconciliation and the alerting policy end to end, and every run
produces byte-identical output so CI can assert on it.

For realistic evaluation, point the pipeline at SKU110K or a similar public
dataset. See the "Data for testing and evaluation" section of README.md.

Usage:
    python scripts/generate_sample_data.py [--force] [--out data/sample_data]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SEED = 20260922
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
ROW_COUNT = 3
SLOTS_PER_ROW = 6

BACKDROP = (232, 230, 226)
SHELF_BOARD = (176, 170, 160)
SHELF_SHADOW = (140, 134, 126)

PALETTE: list[tuple[int, int, int]] = [
    (58, 62, 196),
    (52, 150, 235),
    (66, 176, 96),
    (196, 132, 46),
    (156, 72, 178),
    (48, 184, 196),
]

CATALOGUE: list[tuple[str, str, str]] = [
    ("SKU-1001", "Aurora Sparkling Water 500ml", "Aurora"),
    ("SKU-1002", "Nordfjell Cola 330ml", "Nordfjell"),
    ("SKU-1003", "Verdant Iced Tea Peach 500ml", "Verdant"),
    ("SKU-2001", "Granola Crunch Honey 400g", "MorningField"),
    ("SKU-2002", "Oat Rings Classic 350g", "MorningField"),
    ("SKU-2003", "Choco Pillows 375g", "CocoaLab"),
    ("SKU-3001", "Sea Salt Crisps 150g", "Tidebreak"),
    ("SKU-3002", "Paprika Crisps 150g", "Tidebreak"),
    ("SKU-3003", "Rye Crackers 200g", "Fjordbake"),
]


@dataclass(frozen=True, slots=True)
class Scenario:
    """One generated shelf: which slots hold product and what to call the file."""

    name: str
    description: str
    missing: dict[int, list[int]]
    """Shelf level -> slot indices left empty."""

    store_id: str = "STORE-0042"
    shelf_id: str = "AISLE-07-BAY-3"
    rows: int = ROW_COUNT
    slots: int = SLOTS_PER_ROW
    size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)
    dense: bool = False
    """Render without a clean backdrop, the way a real chiller photograph looks."""


SCENARIOS: list[Scenario] = [
    Scenario(
        name="shelf_compliant",
        description="Fully stocked shelf matching the planogram exactly.",
        missing={},
    ),
    Scenario(
        name="shelf_minor_gaps",
        description="Two facings pulled from the middle row - low severity.",
        missing={1: [2, 3]},
    ),
    Scenario(
        name="shelf_critical_stockout",
        description="Bottom row almost empty and the top row thinning out.",
        missing={0: [4, 5], 2: [0, 1, 2, 3, 4]},
    ),
    Scenario(
        name="shelf_dense_6row",
        description=(
            "High-resolution chiller bay: six rows of twelve tightly packed "
            "facings against a dark backdrop, with four gaps."
        ),
        missing={1: [5], 3: [0, 11], 4: [6]},
        shelf_id="AISLE-02-CHILLER-1",
        rows=6,
        slots=12,
        size=(1600, 1200),
        dense=True,
    ),
]


def _draw_carton(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    colour: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    """Draw one packaged product with a lid band, a label patch and a shadow."""
    jitter = int(rng.integers(-3, 4))
    x1, y1 = x + jitter, y
    x2, y2 = x1 + width, y1 + height

    cv2.rectangle(canvas, (x1 + 4, y2 - 6), (x2 + 6, y2 + 4), SHELF_SHADOW, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, -1)
    cv2.rectangle(
        canvas, (x1, y1), (x2, y1 + max(6, height // 6)), tuple(int(c * 0.72) for c in colour), -1
    )
    label_pad = max(4, width // 8)
    cv2.rectangle(
        canvas,
        (x1 + label_pad, y1 + height // 2 - height // 8),
        (x2 - label_pad, y1 + height // 2 + height // 8),
        (246, 246, 244),
        -1,
    )
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (38, 38, 42), 2)


def render_shelf(scenario: Scenario, *, rng: np.random.Generator) -> np.ndarray:
    """Render one synthetic shelf image for ``scenario``."""
    width, height = scenario.size
    rows, slots_per_row = scenario.rows, scenario.slots
    backdrop = (38, 36, 34) if scenario.dense else BACKDROP
    canvas = np.full((height, width, 3), backdrop, dtype=np.uint8)
    if scenario.dense:
        # A chiller has no flat backdrop: speckle it so the detector cannot
        # separate product from background by contrast alone.
        noise = rng.integers(-14, 15, size=canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    margin_x = max(20, width // 17)
    top = max(24, height // 11)
    row_height = (height - top - height // 13) // rows
    slot_width = (width - 2 * margin_x) // slots_per_row

    for level in range(rows):
        row_top = top + level * row_height
        board_y = row_top + row_height - 18
        cv2.rectangle(
            canvas,
            (margin_x - 20, board_y),
            (IMAGE_WIDTH - margin_x + 20, board_y + 14),
            SHELF_BOARD,
            -1,
        )
        cv2.rectangle(
            canvas,
            (margin_x - 20, board_y + 14),
            (IMAGE_WIDTH - margin_x + 20, board_y + 20),
            SHELF_SHADOW,
            -1,
        )

        empty_slots = set(scenario.missing.get(level, []))
        for slot in range(slots_per_row):
            if slot in empty_slots:
                continue
            product_height = row_height - max(12, row_height // 5)
            # Dense bays are merchandised shoulder to shoulder: no backdrop
            # shows between facings, which is what defeats contrast-based
            # column segmentation on real photographs.
            product_width = slot_width if scenario.dense else slot_width - 22
            x = margin_x + slot * slot_width + 11
            y = board_y - product_height
            colour = PALETTE[(level * slots_per_row + slot) % len(PALETTE)]
            _draw_carton(canvas, x, y, product_width, product_height, colour, rng)

    # A soft vertical light falloff, so CLAHE normalisation has something to do.
    gradient = np.linspace(1.06, 0.88, height, dtype=np.float32)[:, None, None]
    canvas = np.clip(canvas.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    cv2.putText(
        canvas,
        f"{scenario.store_id} / {scenario.shelf_id}",
        (margin_x - 18, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (90, 90, 96),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_planogram(scenario: Scenario) -> dict[str, Any]:
    """Expected layout for a scenario: every slot filled, regardless of the image."""
    entries: list[dict[str, Any]] = []
    categories = ["beverages", "cereals", "snacks", "confectionery", "household", "chilled"]
    for level in range(scenario.rows):
        for slot in range(0, scenario.slots, 2):
            sku, name, brand = CATALOGUE[(level * 3 + slot // 2) % len(CATALOGUE)]
            entries.append(
                {
                    "sku": f"{sku}-L{level}" if scenario.rows > ROW_COUNT else sku,
                    "name": name,
                    "brand": brand,
                    "category": categories[level % len(categories)],
                    "shelf_level": level,
                    "position_index": slot // 2,
                    "expected_facings": 2,
                    "min_facings": 1,
                }
            )
    return {
        "store_id": scenario.store_id,
        "shelf_id": scenario.shelf_id,
        "category": "ambient grocery",
        "entries": entries,
    }


def build_fixture(scenario: Scenario) -> dict[str, Any]:
    """A hand-written ShelfAuditResult used as a golden payload in tests and docs."""
    missing_total = sum(len(slots) for slots in scenario.missing.values())
    products: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []

    for entry in build_planogram(scenario)["entries"]:
        level = int(entry["shelf_level"])
        slot = int(entry["position_index"])
        empty = set(scenario.missing.get(level, []))
        observed = sum(0 if (slot * 2 + k) in empty else 1 for k in (0, 1))
        products.append(
            {
                "sku": entry["sku"],
                "name": entry["name"],
                "brand": entry["brand"],
                "category": entry["category"],
                "facings": observed,
                "shelf_level": level,
                "position_index": slot,
                "stock_state": "in_stock"
                if observed >= 2
                else ("out_of_stock" if observed == 0 else "low_stock"),
                "price_label_visible": observed > 0,
                "confidence": 0.86,
            }
        )
        if observed < 2:
            discrepancies.append(
                {
                    "discrepancy_type": "out_of_stock" if observed == 0 else "low_stock",
                    "severity": "critical" if observed == 0 else "medium",
                    "sku": entry["sku"],
                    "product_name": entry["name"],
                    "shelf_level": level,
                    "expected": "2 facings",
                    "observed": "1 facing" if observed == 1 else f"{observed} facings",
                    "description": (
                        f"{entry['name']} ({entry['sku']}) shows {observed} of 2 expected "
                        f"facings on shelf level {level}."
                    ),
                    "recommended_action": (
                        f"Restock {'1 facing' if 2 - observed == 1 else f'{2 - observed} facings'} "
                        f"of {entry['sku']}."
                    ),
                    "confidence": 0.9,
                }
            )

    expected_total = 2 * len(products)
    satisfied = sum(int(p["facings"]) for p in products)
    return {
        "store_id": scenario.store_id,
        "shelf_id": scenario.shelf_id,
        "category": "ambient grocery",
        "products": products,
        "discrepancies": discrepancies,
        "compliance_score": round(satisfied / expected_total, 4),
        "shelf_occupancy": round(satisfied / expected_total * 0.62, 4),
        "empty_slot_count": missing_total,
        "summary": (
            f"{scenario.description} {len(discrepancies)} planogram deviations detected across "
            f"{scenario.rows} shelf rows."
        ),
        "captured_at": "2026-09-22T08:30:00Z",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sample_data"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    images_dir = args.out / "images"
    planograms_dir = args.out / "planograms"
    fixtures_dir = args.out / "fixtures"
    for directory in (images_dir, planograms_dir, fixtures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    manifest: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        image_path = images_dir / f"{scenario.name}.jpg"
        planogram_path = planograms_dir / f"{scenario.name}.json"
        fixture_path = fixtures_dir / f"{scenario.name}_audit.json"

        if image_path.exists() and not args.force:
            print(f"skip   {image_path} (exists; use --force to overwrite)")
        else:
            image = render_shelf(scenario, rng=rng)
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print(f"write  {image_path}")

        write_json(planogram_path, build_planogram(scenario))
        write_json(fixture_path, build_fixture(scenario))
        print(f"write  {planogram_path}")
        print(f"write  {fixture_path}")

        manifest.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "store_id": scenario.store_id,
                "shelf_id": scenario.shelf_id,
                "image": str(image_path.relative_to(args.out)),
                "planogram": str(planogram_path.relative_to(args.out)),
                "expected_audit": str(fixture_path.relative_to(args.out)),
                "empty_slots": sum(len(v) for v in scenario.missing.values()),
            }
        )

    write_json(args.out / "manifest.json", {"seed": SEED, "scenarios": manifest})
    print(f"write  {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
