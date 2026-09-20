"""Production VLM engine backed by ``instructor``.

``instructor`` patches the vendor client so the model is forced to emit an
object matching :class:`ShelfAuditResult`; validation failures are fed back to
the model as a repair turn rather than surfacing as a 500. The vendor SDKs are
imported lazily so the default install (and the whole test suite) stays free of
them.
"""

from __future__ import annotations

import time
from typing import Any

from smart_shelf.core.config import Settings, VLMBackend
from smart_shelf.core.exceptions import ConfigurationError, VisionModelError
from smart_shelf.core.logging import get_logger
from smart_shelf.core.observability import Tracer, get_tracer
from smart_shelf.cv.preprocessing import encode_jpeg, to_data_url
from smart_shelf.genai.engine import VLMRequest
from smart_shelf.genai.prompts import SYSTEM_PROMPT, build_user_prompt
from smart_shelf.genai.schemas import ShelfAuditResult

logger = get_logger(__name__)


class InstructorVLMEngine:
    """Schema-enforced multimodal auditor for OpenAI or Anthropic models."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._settings = settings
        self._tracer = tracer or get_tracer()
        self.name = f"instructor:{settings.vlm_backend.value}:{settings.vlm_model}"
        self._client = client if client is not None else self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> Any:
        try:
            import instructor
        except ImportError as exc:
            raise ConfigurationError(
                "instructor is not installed; install the 'llm' extra or set SHELF_VLM_BACKEND=mock"
            ) from exc

        if settings.vlm_backend is VLMBackend.OPENAI:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ConfigurationError(
                    "openai is not installed; install the 'llm' extra"
                ) from exc
            key = settings.openai_api_key
            if key is None:  # pragma: no cover - guarded by build_vlm_engine
                raise ConfigurationError("SHELF_OPENAI_API_KEY is required")
            return instructor.from_openai(
                AsyncOpenAI(api_key=key.get_secret_value(), timeout=settings.vlm_timeout_seconds)
            )

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ConfigurationError("anthropic is not installed; install the 'llm' extra") from exc
        key = settings.anthropic_api_key
        if key is None:  # pragma: no cover - guarded by build_vlm_engine
            raise ConfigurationError("SHELF_ANTHROPIC_API_KEY is required")
        return instructor.from_anthropic(
            AsyncAnthropic(api_key=key.get_secret_value(), timeout=settings.vlm_timeout_seconds)
        )

    def _build_messages(self, request: VLMRequest) -> list[dict[str, Any]]:
        text = build_user_prompt(
            request.detections,
            request.planogram,
            store_id=request.store_id,
            shelf_id=request.shelf_id,
        )
        if self._settings.vlm_backend is VLMBackend.OPENAI:
            image_part: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": to_data_url(request.image.pixels)},
            }
        else:
            import base64

            image_part = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(encode_jpeg(request.image.pixels)).decode("ascii"),
                },
            }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [image_part, {"type": "text", "text": text}]},
        ]

    async def analyze(self, request: VLMRequest) -> ShelfAuditResult:
        """Run the multimodal audit, returning a validated :class:`ShelfAuditResult`."""
        settings = self._settings
        with self._tracer.span(
            "vlm.analyze",
            backend=settings.vlm_backend.value,
            model=settings.vlm_model,
            detections=request.detections.count,
        ) as span:
            started = time.perf_counter()
            try:
                result: ShelfAuditResult = await self._client.chat.completions.create(
                    model=settings.vlm_model,
                    messages=self._build_messages(request),
                    response_model=ShelfAuditResult,
                    max_retries=settings.vlm_max_retries,
                    max_tokens=settings.vlm_max_tokens,
                    temperature=settings.vlm_temperature,
                )
            except Exception as exc:
                logger.error("vlm.failed", backend=self.name, error=str(exc))
                raise VisionModelError(
                    "vision-language model call failed",
                    details={"backend": self.name, "cause": str(exc)},
                ) from exc

            duration_ms = (time.perf_counter() - started) * 1000
            merged = result.model_copy(
                update={
                    "store_id": result.store_id or request.store_id,
                    "shelf_id": result.shelf_id or request.shelf_id,
                }
            )
            span.set_output(
                products=len(merged.products),
                discrepancies=len(merged.discrepancies),
                compliance_score=merged.compliance_score,
                duration_ms=round(duration_ms, 2),
            )
            logger.info(
                "vlm.completed",
                backend=self.name,
                products=len(merged.products),
                discrepancies=len(merged.discrepancies),
                duration_ms=round(duration_ms, 2),
            )
            return merged
