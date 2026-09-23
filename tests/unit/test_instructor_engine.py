"""The production VLM engine, driven through a fake instructor client."""

from __future__ import annotations

from typing import Any

import pytest

from smart_shelf.core.config import Settings, VLMBackend
from smart_shelf.core.exceptions import ConfigurationError, VisionModelError
from smart_shelf.core.observability import NoOpTracer
from smart_shelf.cv.detector import HeuristicShelfDetector
from smart_shelf.cv.preprocessing import PreparedImage
from smart_shelf.genai.engine import VLMRequest, build_vlm_engine
from smart_shelf.genai.instructor_engine import InstructorVLMEngine
from smart_shelf.genai.mock_engine import MockVLMEngine
from smart_shelf.genai.schemas import ProductItem, ShelfAuditResult

pytestmark = pytest.mark.unit

AUDIT = ShelfAuditResult(
    products=[ProductItem(sku="SKU-1", name="Cola 330ml", facings=3)],
    compliance_score=0.75,
    summary="One SKU short.",
)


class _Completions:
    def __init__(self, parent: _FakeClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> ShelfAuditResult:
        self._parent.calls.append(kwargs)
        if self._parent.error is not None:
            raise self._parent.error
        return AUDIT


class _Chat:
    def __init__(self, parent: _FakeClient) -> None:
        self.completions = _Completions(parent)


class _FakeClient:
    """Mimics the ``client.chat.completions.create`` surface instructor exposes."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error
        self.chat = _Chat(self)


@pytest.fixture
def anthropic_settings() -> Settings:
    return Settings(
        vlm_backend=VLMBackend.ANTHROPIC,
        anthropic_api_key="sk-ant-test",
        vlm_model="claude-sonnet-5",
        vlm_max_retries=3,
    )


@pytest.fixture
def openai_settings() -> Settings:
    return Settings(vlm_backend=VLMBackend.OPENAI, openai_api_key="sk-test", vlm_model="gpt-4o")


def request_for(image: PreparedImage, **kwargs: str) -> VLMRequest:
    return VLMRequest(image=image, detections=HeuristicShelfDetector().detect(image), **kwargs)


async def test_the_validated_audit_is_returned(
    anthropic_settings: Settings, prepared_image: PreparedImage
) -> None:
    client = _FakeClient()
    engine = InstructorVLMEngine(anthropic_settings, client=client, tracer=NoOpTracer())

    result = await engine.analyze(request_for(prepared_image))

    assert result.compliance_score == 0.75
    assert result.products[0].sku == "SKU-1"


async def test_model_settings_are_forwarded(
    anthropic_settings: Settings, prepared_image: PreparedImage
) -> None:
    client = _FakeClient()
    engine = InstructorVLMEngine(anthropic_settings, client=client, tracer=NoOpTracer())

    await engine.analyze(request_for(prepared_image))

    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["response_model"] is ShelfAuditResult
    assert call["max_retries"] == 3
    assert call["temperature"] == 0.0


async def test_anthropic_messages_carry_base64_image_blocks(
    anthropic_settings: Settings, prepared_image: PreparedImage
) -> None:
    client = _FakeClient()
    engine = InstructorVLMEngine(anthropic_settings, client=client, tracer=NoOpTracer())

    await engine.analyze(request_for(prepared_image))

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    image_block = messages[1]["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert image_block["source"]["data"]


async def test_openai_messages_carry_a_data_url(
    openai_settings: Settings, prepared_image: PreparedImage
) -> None:
    client = _FakeClient()
    engine = InstructorVLMEngine(openai_settings, client=client, tracer=NoOpTracer())

    await engine.analyze(request_for(prepared_image))

    image_block = client.calls[0]["messages"][1]["content"][0]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_identifiers_are_backfilled_from_the_request(
    anthropic_settings: Settings, prepared_image: PreparedImage
) -> None:
    engine = InstructorVLMEngine(anthropic_settings, client=_FakeClient(), tracer=NoOpTracer())

    result = await engine.analyze(request_for(prepared_image, store_id="STORE-5", shelf_id="BAY-2"))

    assert result.store_id == "STORE-5"
    assert result.shelf_id == "BAY-2"


async def test_provider_failures_become_vision_model_errors(
    anthropic_settings: Settings, prepared_image: PreparedImage
) -> None:
    client = _FakeClient(error=RuntimeError("429 rate limited"))
    engine = InstructorVLMEngine(anthropic_settings, client=client, tracer=NoOpTracer())

    with pytest.raises(VisionModelError, match="vision-language model call failed") as caught:
        await engine.analyze(request_for(prepared_image))

    assert "429 rate limited" in caught.value.details["cause"]


async def test_the_call_is_traced(
    anthropic_settings: Settings, prepared_image: PreparedImage
) -> None:
    tracer = NoOpTracer()
    engine = InstructorVLMEngine(anthropic_settings, client=_FakeClient(), tracer=tracer)

    await engine.analyze(request_for(prepared_image))

    span = next(s for s in tracer.spans if s.name == "vlm.analyze")
    assert span.metadata["model"] == "claude-sonnet-5"
    assert span.output["compliance_score"] == 0.75


def test_the_engine_name_identifies_the_backend(anthropic_settings: Settings) -> None:
    engine = InstructorVLMEngine(anthropic_settings, client=_FakeClient(), tracer=NoOpTracer())
    assert engine.name == "instructor:anthropic:claude-sonnet-5"


def test_the_factory_selects_the_mock_engine(settings: Settings) -> None:
    assert isinstance(build_vlm_engine(settings), MockVLMEngine)


@pytest.mark.parametrize("backend", [VLMBackend.OPENAI, VLMBackend.ANTHROPIC])
def test_the_factory_demands_an_api_key(backend: VLMBackend) -> None:
    with pytest.raises(ConfigurationError, match="API_KEY is required"):
        build_vlm_engine(Settings(vlm_backend=backend))
