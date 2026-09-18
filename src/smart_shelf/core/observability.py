"""Langfuse-backed tracing with a zero-dependency fallback.

The rest of the codebase only ever sees the :class:`Tracer` protocol, so tracing
can be switched off (or Langfuse uninstalled entirely) without touching a single
call site. Spans are context managers; they record inputs, outputs, latency and
errors, and they never raise - a broken observability backend must not take
down an audit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import Any, Protocol, runtime_checkable
import uuid

from smart_shelf.core.config import Settings, get_settings
from smart_shelf.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Span:
    """A single unit of traced work."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    duration_ms: float | None = None
    error: str | None = None

    def set_output(self, **values: Any) -> None:
        """Attach result values that should be visible on the trace."""
        self.output.update(values)

    def set_metadata(self, **values: Any) -> None:
        """Attach contextual values to the span."""
        self.metadata.update(values)


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracing surface consumed by services, agents and engines."""

    def span(
        self, name: str, *, trace_id: str | None = None, **metadata: Any
    ) -> Any:  # pragma: no cover - protocol
        """Context manager yielding a :class:`Span`."""

    def flush(self) -> None:  # pragma: no cover - protocol
        """Push buffered spans to the backend."""


class NoOpTracer:
    """Records spans in memory only. Used in tests and when tracing is disabled."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, *, trace_id: str | None = None, **metadata: Any) -> Iterator[Span]:
        """Open a span and record it once the block exits."""
        current = Span(name=name, trace_id=trace_id or uuid.uuid4().hex, metadata=dict(metadata))
        try:
            yield current
        except Exception as exc:
            current.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            current.duration_ms = (time.perf_counter() - current.started_at) * 1000
            self.spans.append(current)
            logger.debug(
                "span.completed",
                span=current.name,
                trace_id=current.trace_id,
                duration_ms=round(current.duration_ms, 2),
                error=current.error,
            )

    def flush(self) -> None:
        """Nothing is buffered, so there is nothing to push."""
        return None


class LangfuseTracer:
    """Mirrors spans into Langfuse while keeping the in-memory behaviour."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self.spans: list[Span] = []
        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning("langfuse.unavailable", hint="pip install 'smart-shelf-analytics[obs]'")
            return

        public = settings.langfuse_public_key
        secret = settings.langfuse_secret_key
        if public is None or secret is None:
            logger.warning("langfuse.missing_credentials")
            return

        self._client = Langfuse(
            public_key=public.get_secret_value(),
            secret_key=secret.get_secret_value(),
            host=settings.langfuse_host,
        )

    @contextmanager
    def span(self, name: str, *, trace_id: str | None = None, **metadata: Any) -> Iterator[Span]:
        """Open a span, mirroring it into Langfuse when the client is live."""
        current = Span(name=name, trace_id=trace_id or uuid.uuid4().hex, metadata=dict(metadata))
        handle: Any | None = None
        if self._client is not None:
            try:
                handle = self._client.start_span(name=name, input=current.metadata)
            except Exception as exc:  # pragma: no cover - backend specific
                logger.warning("langfuse.span_start_failed", error=str(exc))
        try:
            yield current
        except Exception as exc:
            current.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            current.duration_ms = (time.perf_counter() - current.started_at) * 1000
            self.spans.append(current)
            if handle is not None:
                try:
                    handle.update(output=current.output, metadata=current.metadata)
                    handle.end()
                except Exception as exc:  # pragma: no cover - backend specific
                    logger.warning("langfuse.span_end_failed", error=str(exc))

    def flush(self) -> None:
        """Push buffered spans to Langfuse, swallowing backend failures."""
        if self._client is None:
            return
        try:  # pragma: no cover - backend specific
            self._client.flush()
        except Exception as exc:
            logger.warning("langfuse.flush_failed", error=str(exc))


def build_tracer(settings: Settings) -> Tracer:
    """Select a tracer implementation from configuration."""
    if settings.langfuse_enabled:
        return LangfuseTracer(settings)
    return NoOpTracer()


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    """Return the process-wide tracer singleton."""
    return build_tracer(get_settings())


def reset_tracer_cache() -> None:
    """Drop the cached tracer. Used by tests and by settings reloads."""
    get_tracer.cache_clear()
