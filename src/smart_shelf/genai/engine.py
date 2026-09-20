"""Vision-language engine abstraction.

Services depend on :class:`VisionLanguageEngine`, never on a vendor SDK. Three
implementations are wired through :func:`build_vlm_engine`: OpenAI and Anthropic
(both via ``instructor`` for schema-enforced output) and a deterministic offline
engine used by tests, demos and CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from smart_shelf.core.config import Settings, VLMBackend
from smart_shelf.core.exceptions import ConfigurationError
from smart_shelf.core.observability import Tracer
from smart_shelf.cv.models import DetectionResult
from smart_shelf.cv.preprocessing import PreparedImage
from smart_shelf.genai.schemas import Planogram, ShelfAuditResult


@dataclass(slots=True, frozen=True)
class VLMRequest:
    """Everything the auditor model needs for one shelf image."""

    image: PreparedImage
    detections: DetectionResult
    planogram: Planogram | None = None
    store_id: str | None = None
    shelf_id: str | None = None


@runtime_checkable
class VisionLanguageEngine(Protocol):
    """Turns a shelf photograph plus CV evidence into a validated audit."""

    name: str

    async def analyze(  # pragma: no cover - protocol
        self, request: VLMRequest
    ) -> ShelfAuditResult:
        """Produce a validated audit for ``request``."""
        ...


def build_vlm_engine(settings: Settings, *, tracer: Tracer | None = None) -> VisionLanguageEngine:
    """Instantiate the engine named by ``settings.vlm_backend``.

    Args:
        settings: the configuration selecting the backend.
        tracer: tracer the engine should emit its ``vlm.analyze`` span to.
            Defaults to the process tracer. Callers that own a per-request or
            per-run tracer must pass it, or the span lands on the singleton and
            is never correlated with the rest of the pipeline.
    """
    from smart_shelf.genai.instructor_engine import InstructorVLMEngine
    from smart_shelf.genai.mock_engine import MockVLMEngine

    match settings.vlm_backend:
        case VLMBackend.MOCK:
            return MockVLMEngine(tracer)
        case VLMBackend.OPENAI:
            if settings.openai_api_key is None:
                raise ConfigurationError("SHELF_OPENAI_API_KEY is required for vlm_backend=openai")
            return InstructorVLMEngine(settings, tracer=tracer)
        case VLMBackend.ANTHROPIC:
            if settings.anthropic_api_key is None:
                raise ConfigurationError(
                    "SHELF_ANTHROPIC_API_KEY is required for vlm_backend=anthropic"
                )
            return InstructorVLMEngine(settings, tracer=tracer)
        case _:  # pragma: no cover - StrEnum is exhaustive
            raise ConfigurationError(f"unknown vlm_backend: {settings.vlm_backend}")
