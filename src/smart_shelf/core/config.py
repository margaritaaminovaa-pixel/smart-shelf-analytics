"""Typed application configuration.

Settings are loaded from the process environment and an optional ``.env`` file,
validated once at import boundaries and then shared through
:func:`get_settings`. Nothing in the codebase reads ``os.environ`` directly.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample_data"


class Environment(StrEnum):
    """Deployment environment, used to tune logging and error verbosity."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class DetectorBackend(StrEnum):
    """Object detection implementation to bind at runtime."""

    YOLO = "yolo"
    """Ultralytics YOLOv8/v11 weights. Requires the ``cv`` optional extra."""

    HEURISTIC = "heuristic"
    """Pure-OpenCV contour pipeline. No weights, deterministic, always available."""

    HYBRID = "hybrid"
    """YOLO first, falling back to OpenCV when weights or detections are missing."""


class VLMBackend(StrEnum):
    """Vision-language model implementation to bind at runtime."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"
    """Deterministic offline engine so the pipeline runs with no API keys."""


class Settings(BaseSettings):
    """Root settings object. One instance per process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHELF_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    # -- Application ----------------------------------------------------------
    app_name: str = "smart-shelf-analytics"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # -- Logging --------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = Field(
        default=False,
        description="Emit newline-delimited JSON logs. Forced on outside local dev.",
    )

    # -- Computer vision ------------------------------------------------------
    detector_backend: DetectorBackend = DetectorBackend.HEURISTIC
    yolo_weights: str = Field(
        default="hf:chistopat/sku110k-yolo11-object-detector/weights/sku110k-yolo11-s640.pt",
        description="Ultralytics weights name or path, used when backend is 'yolo'.",
    )
    yolo_weights_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "smart-shelf-analytics" / "weights",
        description=(
            "Where auto-downloaded YOLO weights are cached. Kept outside the "
            "package so it survives reinstalls and works from a wheel."
        ),
    )
    detection_confidence: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Tuned for the SKU-110K checkpoint, where 0.25 drops real facings. "
            "The OpenCV score floor is 0.4, so this does not filter it."
        ),
    )
    detection_max_items: int = Field(
        default=600,
        ge=1,
        le=5000,
        description=(
            "A full supermarket aisle runs to several hundred facings; the old "
            "200 cap truncated them."
        ),
    )
    hybrid_min_detections: int = Field(
        default=4,
        ge=0,
        description=(
            "Below this many YOLO detections the hybrid backend also runs the "
            "OpenCV pipeline and keeps whichever found more."
        ),
    )
    row_merge_tolerance: float = Field(
        default=0.6,
        ge=0.05,
        le=3.0,
        description=(
            "Vertical merging tolerance for shelf-row clustering, as a multiple of "
            "the median facing height. Lower splits rows, higher merges them."
        ),
    )
    max_image_bytes: int = Field(
        default=15 * 1024 * 1024, ge=1024, description="Upload size ceiling."
    )
    preprocess_max_edge: int = Field(
        default=1600,
        ge=128,
        le=8192,
        description=(
            "Longest edge after resizing. Measured on a six-row dense bay, "
            "dropping to 1280 costs about seven points of recall."
        ),
    )

    # -- Vision language model ------------------------------------------------
    vlm_backend: VLMBackend = VLMBackend.MOCK
    vlm_model: str = "claude-sonnet-5"
    vlm_max_tokens: int = Field(default=4096, ge=256, le=32_000)
    vlm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    vlm_max_retries: int = Field(default=2, ge=0, le=10)
    vlm_timeout_seconds: float = Field(default=60.0, gt=0)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # -- Observability (Langfuse) ---------------------------------------------
    langfuse_enabled: bool = False
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    # -- Persistence ----------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/audits.db",
        description="Async SQLite DSN. Use 'sqlite+aiosqlite:///:memory:' for tests.",
    )

    # -- Autonomous agent -----------------------------------------------------
    compliance_alert_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Audits scoring below this trigger a restocking alert.",
    )
    critical_oos_threshold: int = Field(
        default=3,
        ge=1,
        description="Out-of-stock discrepancy count that escalates to CRITICAL.",
    )
    slack_webhook_url: str | None = None
    email_webhook_url: str | None = None
    notification_timeout_seconds: float = Field(default=10.0, gt=0)
    notifications_dry_run: bool = Field(
        default=True,
        description="Log alerts instead of posting them. Default-safe for demos.",
    )

    # -- Sample data ----------------------------------------------------------
    sample_data_dir: Path = DEFAULT_SAMPLE_DATA_DIR

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        if not value.startswith("sqlite"):
            msg = "database_url must be a SQLite DSN, e.g. sqlite+aiosqlite:///./x.db"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _enforce_production_invariants(self) -> Settings:
        if self.environment is Environment.PRODUCTION:
            if self.debug:
                msg = "debug must be disabled in production"
                raise ValueError(msg)
            if self.vlm_backend is VLMBackend.MOCK:
                msg = "vlm_backend 'mock' is not allowed in production"
                raise ValueError(msg)
        return self

    @property
    def sqlite_path(self) -> str:
        """Filesystem path (or ``:memory:``) extracted from :attr:`database_url`."""
        _, _, tail = self.database_url.partition(":///")
        return tail or ":memory:"

    @property
    def use_json_logs(self) -> bool:
        """Whether logs are emitted as newline-delimited JSON."""
        return self.log_json or self.environment in {
            Environment.STAGING,
            Environment.PRODUCTION,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Tests use this after patching the environment."""
    get_settings.cache_clear()
