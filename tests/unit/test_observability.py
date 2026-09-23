"""Tracing behaviour and the Langfuse fallback."""

from __future__ import annotations

import pytest

from smart_shelf.core.config import Settings
from smart_shelf.core.observability import (
    LangfuseTracer,
    NoOpTracer,
    Span,
    Tracer,
    build_tracer,
    get_tracer,
    reset_tracer_cache,
)

pytestmark = pytest.mark.unit


def test_noop_tracer_satisfies_the_protocol() -> None:
    assert isinstance(NoOpTracer(), Tracer)


def test_spans_record_metadata_output_and_duration() -> None:
    tracer = NoOpTracer()
    with tracer.span("cv.detect", backend="heuristic") as span:
        span.set_output(detections=12)
        span.set_metadata(rows=3)

    recorded = tracer.spans[0]
    assert recorded.name == "cv.detect"
    assert recorded.metadata == {"backend": "heuristic", "rows": 3}
    assert recorded.output == {"detections": 12}
    assert recorded.duration_ms is not None
    assert recorded.error is None


def test_child_spans_can_share_a_trace_id() -> None:
    tracer = NoOpTracer()
    with tracer.span("audit.run") as root, tracer.span("cv.detect", trace_id=root.trace_id):
        pass
    assert len({span.trace_id for span in tracer.spans}) == 1


def test_a_failing_span_records_the_error_and_reraises() -> None:
    tracer = NoOpTracer()
    with pytest.raises(RuntimeError, match="detector exploded"), tracer.span("cv.detect"):
        raise RuntimeError("detector exploded")

    assert tracer.spans[0].error == "RuntimeError: detector exploded"
    assert tracer.spans[0].duration_ms is not None


def test_flush_is_a_no_op() -> None:
    assert NoOpTracer().flush() is None


def test_disabled_configuration_yields_the_noop_tracer(settings: Settings) -> None:
    assert isinstance(build_tracer(settings), NoOpTracer)


def test_langfuse_without_the_package_degrades_gracefully() -> None:
    # langfuse is an optional extra; enabling it must never break an audit.
    tracer = build_tracer(Settings(langfuse_enabled=True))
    assert isinstance(tracer, LangfuseTracer)
    with tracer.span("vlm.analyze") as span:
        span.set_output(products=3)
    assert tracer.spans[0].output == {"products": 3}
    tracer.flush()


def test_tracer_is_cached_per_process() -> None:
    reset_tracer_cache()
    assert get_tracer() is get_tracer()
    reset_tracer_cache()


def test_spans_get_distinct_identifiers() -> None:
    first = Span(name="a", trace_id="t")
    second = Span(name="a", trace_id="t")
    assert first.span_id != second.span_id
