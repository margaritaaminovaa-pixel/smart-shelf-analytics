"""Analytics over persisted audit history.

Aggregation runs in SQL wherever it can, because the interesting questions
("which SKU costs us the most availability?") are joins over thousands of rows,
not over the handful the API returns. The one exception is SKU attribution,
which reads the JSON payload - SQLite's ``json_each`` is available but keeping
it in Python makes the query portable to any backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_shelf.db.models import AuditFilter
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.schemas import DiscrepancyType, Severity


@dataclass(slots=True, frozen=True)
class ComplianceTrendPoint:
    """Average compliance for one calendar day."""

    day: str
    audit_count: int
    mean_compliance: float
    alert_count: int


@dataclass(slots=True, frozen=True)
class SkuOffender:
    """A SKU ranked by how often it breaks compliance."""

    sku: str
    product_name: str
    occurrences: int
    critical_occurrences: int
    dominant_type: DiscrepancyType


@dataclass(slots=True)
class _OffenderTally:
    """Mutable accumulator behind :meth:`AnalyticsService.top_offenders`."""

    name: str
    occurrences: int = 0
    critical_occurrences: int = 0
    types: dict[DiscrepancyType, int] = field(default_factory=dict)

    def add(self, kind: DiscrepancyType, severity: Severity) -> None:
        """Record one discrepancy."""
        self.occurrences += 1
        if severity is Severity.CRITICAL:
            self.critical_occurrences += 1
        self.types[kind] = self.types.get(kind, 0) + 1

    def to_offender(self, sku: str) -> SkuOffender:
        """Freeze into the public result type."""
        return SkuOffender(
            sku=sku,
            product_name=self.name,
            occurrences=self.occurrences,
            critical_occurrences=self.critical_occurrences,
            dominant_type=max(self.types.items(), key=lambda item: item[1])[0],
        )


@dataclass(slots=True, frozen=True)
class AnalyticsSummary:
    """Headline numbers for the operations dashboard."""

    total_audits: int
    mean_compliance: float
    alert_rate: float
    out_of_stock_events: int
    worst_store_id: str | None = None
    trend: list[ComplianceTrendPoint] = field(default_factory=list)


class AnalyticsService:
    """Read-only aggregations over :class:`AuditRepository`."""

    def __init__(self, repository: AuditRepository) -> None:
        self._repository = repository

    async def compliance_trend(self, *, days: int = 30) -> list[ComplianceTrendPoint]:
        """Daily mean compliance over the most recent ``days`` calendar days."""
        statement = """
            SELECT substr(created_at, 1, 10) AS day,
                   COUNT(*)                  AS audit_count,
                   AVG(compliance_score)     AS mean_compliance,
                   SUM(alert_triggered)      AS alert_count
            FROM audits
            GROUP BY day
            ORDER BY day DESC
            LIMIT :days
        """
        async with self._repository.connection.execute(statement, {"days": days}) as cursor:
            rows = await cursor.fetchall()
        return [
            ComplianceTrendPoint(
                day=str(row["day"]),
                audit_count=int(row["audit_count"]),
                mean_compliance=round(float(row["mean_compliance"] or 0.0), 4),
                alert_count=int(row["alert_count"] or 0),
            )
            for row in reversed(list(rows))
        ]

    async def top_offenders(self, *, limit: int = 10, sample: int = 500) -> list[SkuOffender]:
        """SKUs appearing most often in discrepancies across recent audits."""
        records = await self._repository.list_audits(AuditFilter(limit=min(sample, 500)))
        tally: dict[str, _OffenderTally] = {}
        for record in records:
            for discrepancy in record.result.discrepancies:
                if discrepancy.sku is None:
                    continue
                bucket = tally.setdefault(
                    discrepancy.sku,
                    _OffenderTally(name=discrepancy.product_name or discrepancy.sku),
                )
                bucket.add(discrepancy.discrepancy_type, discrepancy.severity)

        offenders = [bucket.to_offender(sku) for sku, bucket in tally.items()]
        offenders.sort(key=lambda o: (o.occurrences, o.critical_occurrences), reverse=True)
        return offenders[:limit]

    async def summary(self, *, trend_days: int = 14) -> AnalyticsSummary:
        """Headline KPIs plus a short compliance trend."""
        statement = """
            SELECT COUNT(*)              AS total,
                   AVG(compliance_score) AS mean_compliance,
                   SUM(alert_triggered)  AS alerts,
                   SUM(empty_slot_count) AS empty_slots
            FROM audits
        """
        async with self._repository.connection.execute(statement) as cursor:
            row = await cursor.fetchone()

        total = int(row["total"]) if row else 0
        alerts = int(row["alerts"] or 0) if row else 0

        worst_store: str | None = None
        async with self._repository.connection.execute(
            """
            SELECT store_id, AVG(compliance_score) AS mean_compliance
            FROM audits
            WHERE store_id IS NOT NULL
            GROUP BY store_id
            ORDER BY mean_compliance ASC
            LIMIT 1
            """
        ) as cursor:
            worst = await cursor.fetchone()
            if worst is not None:
                worst_store = str(worst["store_id"])

        return AnalyticsSummary(
            total_audits=total,
            mean_compliance=round(float(row["mean_compliance"] or 0.0), 4) if row else 0.0,
            alert_rate=round(alerts / total, 4) if total else 0.0,
            out_of_stock_events=int(row["empty_slots"] or 0) if row else 0,
            worst_store_id=worst_store,
            trend=await self.compliance_trend(days=trend_days),
        )
