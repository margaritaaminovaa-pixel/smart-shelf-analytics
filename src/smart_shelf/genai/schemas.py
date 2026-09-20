"""Pydantic v2 contracts for structured shelf understanding.

These models are the single source of truth for three separate consumers: the
JSON schema handed to the multimodal LLM, the API response body, and the row
shapes persisted to the audit store. Field descriptions are intentionally
written as instructions - ``instructor`` forwards them to the model as the tool
schema, so they double as prompt engineering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]

_COMPUTED_FIELDS = frozenset({"total_facings", "max_severity"})
"""Read-only fields present in serialised output but never accepted as input."""


class StockState(StrEnum):
    """Observed availability of a planogram slot."""

    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    UNKNOWN = "unknown"


class DiscrepancyType(StrEnum):
    """Categories of planogram non-compliance the auditor can report."""

    OUT_OF_STOCK = "out_of_stock"
    LOW_STOCK = "low_stock"
    MISPLACED_PRODUCT = "misplaced_product"
    INCORRECT_FACINGS = "incorrect_facings"
    UNEXPECTED_PRODUCT = "unexpected_product"
    MISSING_PRICE_TAG = "missing_price_tag"
    DAMAGED_PACKAGING = "damaged_packaging"


class Severity(StrEnum):
    """Business impact of a discrepancy, ordered low to critical."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Ordinal position, low to critical."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class ProductItem(BaseModel):
    """A distinct product facing group identified on the shelf."""

    model_config = ConfigDict(extra="forbid")

    sku: str | None = Field(
        default=None,
        max_length=64,
        description="SKU or barcode if legible on the shelf label, else null.",
    )
    name: str = Field(
        min_length=1,
        max_length=160,
        description="Product name as printed on the packaging.",
    )
    brand: str | None = Field(default=None, max_length=96, description="Brand name if visible.")
    category: str | None = Field(
        default=None, max_length=96, description="Retail category, e.g. 'carbonated drinks'."
    )
    facings: int = Field(
        default=1, ge=0, le=999, description="Number of identical units visible side by side."
    )
    shelf_level: int = Field(
        default=0, ge=0, le=50, description="Shelf row index, 0 = topmost visible row."
    )
    position_index: int = Field(
        default=0, ge=0, le=999, description="Left-to-right slot within the shelf row."
    )
    stock_state: StockState = Field(
        default=StockState.IN_STOCK, description="Availability judged from visible depth and gaps."
    )
    price_label_visible: bool = Field(
        default=True, description="True when a readable price label sits under the facing."
    )
    confidence: UnitFloat = Field(default=0.8, description="Model confidence in this reading.")


class Discrepancy(BaseModel):
    """A single deviation between the observed shelf and the expected planogram."""

    model_config = ConfigDict(extra="forbid")

    discrepancy_type: DiscrepancyType = Field(description="Category of the deviation.")
    severity: Severity = Field(description="Business impact of the deviation.")
    sku: str | None = Field(default=None, max_length=64, description="Affected SKU, if known.")
    product_name: str | None = Field(default=None, max_length=160)
    shelf_level: int | None = Field(default=None, ge=0, le=50)
    expected: str | None = Field(
        default=None, max_length=240, description="What the planogram requires."
    )
    observed: str | None = Field(
        default=None, max_length=240, description="What is actually on the shelf."
    )
    description: str = Field(
        min_length=1, max_length=600, description="One-sentence explanation for the store manager."
    )
    recommended_action: str | None = Field(
        default=None, max_length=240, description="Concrete next step, e.g. 'restock 4 facings'."
    )
    confidence: UnitFloat = Field(default=0.8)


class ShelfAuditResult(BaseModel):
    """Complete structured audit of one shelf image.

    This is the object the VLM is asked to fill in, and the object the API
    returns. ``compliance_score`` is recomputed deterministically by the service
    layer whenever an expected planogram is supplied, so the model's own estimate
    is only used as a fallback for unconstrained audits.
    """

    model_config = ConfigDict(extra="forbid")

    store_id: str | None = Field(default=None, max_length=64)
    shelf_id: str | None = Field(default=None, max_length=64)
    category: str | None = Field(
        default=None, max_length=96, description="Dominant retail category on this shelf."
    )
    products: list[ProductItem] = Field(
        default_factory=list, description="Every distinct product facing group on the shelf."
    )
    discrepancies: list[Discrepancy] = Field(
        default_factory=list, description="Every planogram deviation found."
    )
    compliance_score: UnitFloat = Field(
        default=1.0, description="1.0 = fully compliant shelf, 0.0 = nothing matches."
    )
    shelf_occupancy: UnitFloat = Field(
        default=1.0, description="Fraction of shelf surface occupied by product."
    )
    empty_slot_count: int = Field(
        default=0, ge=0, le=999, description="Visible gaps where product should be."
    )
    summary: str = Field(
        default="", max_length=1200, description="Two-sentence summary for the store manager."
    )
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        """Allow this model's own serialised output to be re-validated.

        ``extra="forbid"`` keeps the LLM honest, but it would also reject the
        read-only computed fields that appear in ``model_dump``. Persistence and
        API round-trips depend on that working, so they are dropped on input.
        """
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in _COMPUTED_FIELDS}
        return data

    @model_validator(mode="after")
    def _normalize_captured_at(self) -> Self:
        if self.captured_at.tzinfo is None:
            object.__setattr__(self, "captured_at", self.captured_at.replace(tzinfo=UTC))
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_facings(self) -> int:
        """Total product units visible across every facing group."""
        return sum(item.facings for item in self.products)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_severity(self) -> Severity | None:
        """Highest severity among the reported discrepancies, if any."""
        if not self.discrepancies:
            return None
        return max((d.severity for d in self.discrepancies), key=lambda s: s.rank)

    def discrepancies_of(self, *types: DiscrepancyType) -> list[Discrepancy]:
        """Discrepancies matching any of ``types``."""
        wanted = set(types)
        return [d for d in self.discrepancies if d.discrepancy_type in wanted]


class PlanogramEntry(BaseModel):
    """One slot of the expected shelf layout."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)
    shelf_level: int = Field(ge=0, le=50)
    position_index: int = Field(default=0, ge=0, le=999)
    expected_facings: int = Field(default=1, ge=1, le=999)
    min_facings: int = Field(
        default=1, ge=0, le=999, description="Below this the slot counts as low stock."
    )
    brand: str | None = Field(default=None, max_length=96)
    category: str | None = Field(default=None, max_length=96)

    @model_validator(mode="after")
    def _validate_facings(self) -> Self:
        if self.min_facings > self.expected_facings:
            msg = "min_facings cannot exceed expected_facings"
            raise ValueError(msg)
        return self


class Planogram(BaseModel):
    """Expected shelf layout supplied alongside an audited image."""

    model_config = ConfigDict(extra="forbid")

    store_id: str | None = Field(default=None, max_length=64)
    shelf_id: str | None = Field(default=None, max_length=64)
    category: str | None = Field(default=None, max_length=96)
    entries: list[PlanogramEntry] = Field(default_factory=list)

    @property
    def total_expected_facings(self) -> int:
        """Facings the planogram requires across every slot."""
        return sum(entry.expected_facings for entry in self.entries)

    def by_sku(self) -> dict[str, PlanogramEntry]:
        """Index entries by SKU; later duplicates merge their expected facings."""
        index: dict[str, PlanogramEntry] = {}
        for entry in self.entries:
            existing = index.get(entry.sku)
            if existing is None:
                index[entry.sku] = entry
            else:
                index[entry.sku] = existing.model_copy(
                    update={
                        "expected_facings": existing.expected_facings + entry.expected_facings,
                        "min_facings": existing.min_facings + entry.min_facings,
                    }
                )
        return index
