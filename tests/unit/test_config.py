"""Settings validation and environment invariants."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from smart_shelf.core.config import (
    DetectorBackend,
    Environment,
    Settings,
    VLMBackend,
    get_settings,
    reset_settings_cache,
)

pytestmark = pytest.mark.unit


def test_defaults_are_offline_safe() -> None:
    settings = Settings(_env_file=None)
    assert settings.detector_backend is DetectorBackend.HEURISTIC
    assert settings.vlm_backend is VLMBackend.MOCK
    assert settings.notifications_dry_run is True


def test_sqlite_path_is_extracted_from_the_dsn() -> None:
    assert Settings(database_url="sqlite+aiosqlite:///./data/audits.db").sqlite_path == (
        "./data/audits.db"
    )
    assert Settings(database_url="sqlite+aiosqlite:///:memory:").sqlite_path == ":memory:"


def test_non_sqlite_dsn_is_rejected() -> None:
    with pytest.raises(ValidationError, match="SQLite DSN"):
        Settings(database_url="postgresql+asyncpg://localhost/shelf")


def test_production_refuses_the_mock_vlm() -> None:
    with pytest.raises(ValidationError, match="not allowed in production"):
        Settings(environment=Environment.PRODUCTION, vlm_backend=VLMBackend.MOCK)


def test_production_refuses_debug() -> None:
    with pytest.raises(ValidationError, match="debug must be disabled"):
        Settings(
            environment=Environment.PRODUCTION,
            debug=True,
            vlm_backend=VLMBackend.ANTHROPIC,
        )


@pytest.mark.parametrize(
    ("environment", "log_json", "expected"),
    [
        (Environment.LOCAL, False, False),
        (Environment.LOCAL, True, True),
        (Environment.STAGING, False, True),
        (Environment.TEST, False, False),
    ],
)
def test_json_logs_are_forced_outside_local(
    environment: Environment, log_json: bool, expected: bool
) -> None:
    settings = Settings(environment=environment, log_json=log_json, vlm_backend=VLMBackend.MOCK)
    assert settings.use_json_logs is expected


def test_confidence_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(detection_confidence=1.5)


def test_settings_are_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_settings_cache()
    monkeypatch.setenv("SHELF_LOG_LEVEL", "ERROR")
    first = get_settings()
    monkeypatch.setenv("SHELF_LOG_LEVEL", "DEBUG")
    assert get_settings() is first
    reset_settings_cache()
    assert get_settings().log_level == "DEBUG"
