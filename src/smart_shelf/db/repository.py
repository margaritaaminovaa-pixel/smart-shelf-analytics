"""Async audit history store backed by SQLite (``aiosqlite``).

SQLite keeps the reference deployment single-binary and dependency-free while
staying a genuine SQL store, so the analytics queries below are the same ones
you would run against Postgres or DuckDB in a larger installation. Every public
method is a coroutine; nothing in this module blocks the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiosqlite

from smart_shelf.core.exceptions import AuditNotFoundError, PersistenceError
from smart_shelf.core.logging import get_logger
from smart_shelf.db.models import AuditFilter, AuditRecord

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    audit_id          TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    store_id          TEXT,
    shelf_id          TEXT,
    image_sha256      TEXT NOT NULL,
    image_filename    TEXT,
    detector_backend  TEXT NOT NULL,
    vlm_backend       TEXT NOT NULL,
    detection_count   INTEGER NOT NULL,
    product_count     INTEGER NOT NULL,
    discrepancy_count INTEGER NOT NULL,
    compliance_score  REAL NOT NULL,
    shelf_occupancy   REAL NOT NULL,
    empty_slot_count  INTEGER NOT NULL,
    max_severity      TEXT,
    alert_triggered   INTEGER NOT NULL DEFAULT 0,
    summary           TEXT,
    duration_ms       REAL NOT NULL,
    payload           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audits_created_at ON audits (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audits_compliance ON audits (compliance_score);
CREATE INDEX IF NOT EXISTS idx_audits_store_shelf ON audits (store_id, shelf_id);
"""

_COLUMNS: tuple[str, ...] = (
    "audit_id",
    "created_at",
    "store_id",
    "shelf_id",
    "image_sha256",
    "image_filename",
    "detector_backend",
    "vlm_backend",
    "detection_count",
    "product_count",
    "discrepancy_count",
    "compliance_score",
    "shelf_occupancy",
    "empty_slot_count",
    "max_severity",
    "alert_triggered",
    "summary",
    "duration_ms",
    "payload",
)


class AuditRepository:
    """Connection-owning repository for :class:`AuditRecord` rows."""

    def __init__(self, database_path: str = ":memory:") -> None:
        self._path = database_path
        # Resolved eagerly: expanduser() touches the filesystem and connect() is
        # a coroutine that must not block the event loop.
        self._directory = (
            None if database_path == ":memory:" else Path(database_path).expanduser().parent
        )
        self._connection: aiosqlite.Connection | None = None

    # -- lifecycle ------------------------------------------------------------

    async def connect(self) -> Self:
        """Open the connection and apply the schema. Idempotent."""
        if self._connection is not None:
            return self
        try:
            if self._directory is not None:
                await asyncio.to_thread(self._directory.mkdir, parents=True, exist_ok=True)
            connection = await aiosqlite.connect(self._path)
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA foreign_keys=ON")
            await connection.executescript(SCHEMA)
            await connection.commit()
        except (aiosqlite.Error, OSError) as exc:
            raise PersistenceError(
                "failed to open the audit store", details={"path": self._path, "cause": str(exc)}
            ) from exc
        self._connection = connection
        logger.info("db.connected", path=self._path)
        return self

    async def close(self) -> None:
        """Close the connection. Idempotent."""
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None
        logger.info("db.closed", path=self._path)

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def connection(self) -> aiosqlite.Connection:
        """The open connection, or a :class:`PersistenceError` if there is none."""
        if self._connection is None:
            raise PersistenceError("repository is not connected; call connect() first")
        return self._connection

    # -- writes ---------------------------------------------------------------

    async def save(self, record: AuditRecord) -> AuditRecord:
        """Insert or replace one audit."""
        row = record.to_row()
        placeholders = ", ".join(f":{column}" for column in _COLUMNS)
        columns = ", ".join(_COLUMNS)
        statement = f"INSERT OR REPLACE INTO audits ({columns}) VALUES ({placeholders})"  # noqa: S608
        try:
            await self.connection.execute(statement, row)
            await self.connection.commit()
        except aiosqlite.Error as exc:
            raise PersistenceError(
                "failed to persist the audit", details={"audit_id": record.audit_id}
            ) from exc
        logger.info(
            "db.audit_saved",
            audit_id=record.audit_id,
            compliance_score=record.compliance_score,
        )
        return record

    async def delete(self, audit_id: str) -> None:
        """Remove one audit, raising when it does not exist."""
        cursor = await self.connection.execute("DELETE FROM audits WHERE audit_id = ?", (audit_id,))
        await self.connection.commit()
        if cursor.rowcount == 0:
            raise AuditNotFoundError(f"audit {audit_id} does not exist")

    # -- reads ----------------------------------------------------------------

    async def get(self, audit_id: str) -> AuditRecord:
        """Fetch one audit, raising :class:`AuditNotFoundError` when absent."""
        async with self.connection.execute(
            "SELECT * FROM audits WHERE audit_id = ?", (audit_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise AuditNotFoundError(f"audit {audit_id} does not exist")
        return AuditRecord.from_row(dict(row))

    @staticmethod
    def _build_where(query: AuditFilter) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if query.store_id is not None:
            clauses.append("store_id = :store_id")
            params["store_id"] = query.store_id
        if query.shelf_id is not None:
            clauses.append("shelf_id = :shelf_id")
            params["shelf_id"] = query.shelf_id
        if query.min_compliance_score is not None:
            clauses.append("compliance_score >= :min_score")
            params["min_score"] = query.min_compliance_score
        if query.max_compliance_score is not None:
            clauses.append("compliance_score <= :max_score")
            params["max_score"] = query.max_compliance_score
        if query.alert_triggered is not None:
            clauses.append("alert_triggered = :alert_triggered")
            params["alert_triggered"] = int(query.alert_triggered)
        if query.since is not None:
            clauses.append("created_at >= :since")
            params["since"] = query.since.isoformat()
        if query.until is not None:
            clauses.append("created_at <= :until")
            params["until"] = query.until.isoformat()
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    async def list_audits(self, query: AuditFilter | None = None) -> list[AuditRecord]:
        """Return audits newest first, narrowed by ``query``."""
        query = query or AuditFilter()
        where, params = self._build_where(query)
        params |= {"limit": query.limit, "offset": query.offset}
        # `where` is assembled from _build_where, which only ever emits fixed
        # clause strings; the values are bound parameters.
        statement = (
            f"SELECT * FROM audits{where} ORDER BY created_at DESC, audit_id DESC "  # noqa: S608
            "LIMIT :limit OFFSET :offset"
        )
        async with self.connection.execute(statement, params) as cursor:
            rows: Sequence[aiosqlite.Row] = list(await cursor.fetchall())
        return [AuditRecord.from_row(dict(row)) for row in rows]

    async def count(self, query: AuditFilter | None = None) -> int:
        """Total audits matching ``query``, ignoring limit and offset."""
        query = query or AuditFilter()
        where, params = self._build_where(query)
        async with self.connection.execute(
            f"SELECT COUNT(*) AS total FROM audits{where}",  # noqa: S608 - fixed clauses only
            params,
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["total"]) if row else 0

    async def health_check(self) -> bool:
        """Cheap liveness probe used by ``GET /health``."""
        try:
            async with self.connection.execute("SELECT 1") as cursor:
                await cursor.fetchone()
        except (aiosqlite.Error, PersistenceError):
            return False
        return True
