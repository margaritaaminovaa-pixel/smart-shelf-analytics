"""The synchronous bridge between Streamlit and the async pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_shelf.agents.notifiers import Notification
from smart_shelf.core.config import DetectorBackend, reset_settings_cache
from smart_shelf.core.exceptions import InvalidImageError
from smart_shelf.genai.schemas import Planogram
from smart_shelf.ui.runner import (
    AuditRunConfig,
    CapturingNotifier,
    load_history,
    load_sample_scenarios,
    resolve_settings,
    run_audit,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the settings singleton at a throwaway audit store."""
    target = tmp_path / "audits.db"
    monkeypatch.setenv("SHELF_DATABASE_URL", f"sqlite+aiosqlite:///{target}")
    monkeypatch.setenv("SHELF_NOTIFICATIONS_DRY_RUN", "true")
    reset_settings_cache()
    return target


async def test_capturing_notifier_records_instead_of_sending() -> None:
    notifier = CapturingNotifier()
    receipt = await notifier.send(Notification(title="t", body="b", urgency="routine"))

    assert receipt.delivered is True
    assert receipt.channel == "ui"
    assert [n.title for n in notifier.sent] == ["t"]


def test_config_overrides_are_applied_to_settings() -> None:
    settings = resolve_settings(
        AuditRunConfig(
            backend=DetectorBackend.YOLO,
            detection_confidence=0.42,
            row_merge_tolerance=1.25,
            compliance_alert_threshold=0.5,
            critical_oos_threshold=7,
        )
    )
    assert settings.detector_backend is DetectorBackend.YOLO
    assert settings.detection_confidence == 0.42
    assert settings.row_merge_tolerance == 1.25
    assert settings.compliance_alert_threshold == 0.5
    assert settings.critical_oos_threshold == 7


def test_run_returns_detections_traces_and_alerts(
    isolated_db: Path, stockout_image_bytes: bytes, planogram: Planogram
) -> None:
    run = run_audit(
        stockout_image_bytes,
        config=AuditRunConfig(persist=False),
        planogram=planogram,
        filename="stockout.jpg",
    )

    assert run.outcome.detections.count == 11
    assert run.outcome.result.compliance_score < 0.8
    assert run.outcome.agent.alert_triggered is True
    assert run.alerts and run.alerts[0].urgency == "immediate"
    assert {name for name, _ in run.stage_timings} >= {
        "audit.run",
        "cv.detect",
        "vlm.analyze",
        "agent.run",
    }


def test_prepared_image_matches_the_detection_grid(
    isolated_db: Path, shelf_image_bytes: bytes
) -> None:
    # The overlay is drawn on a re-prepared image, so it must be pixel-identical
    # to the one the detector saw or the boxes would be offset.
    run = run_audit(shelf_image_bytes, config=AuditRunConfig(persist=False))
    assert run.prepared.width == run.outcome.detections.image_width
    assert run.prepared.height == run.outcome.detections.image_height


def test_confidence_override_prunes_detections(isolated_db: Path, shelf_image_bytes: bytes) -> None:
    permissive = run_audit(shelf_image_bytes, config=AuditRunConfig(persist=False))
    strict = run_audit(
        shelf_image_bytes,
        config=AuditRunConfig(detection_confidence=0.93, persist=False),
    )
    assert strict.outcome.detections.count < permissive.outcome.detections.count


def test_tolerance_override_merges_shelf_rows(isolated_db: Path, shelf_image_bytes: bytes) -> None:
    split = run_audit(
        shelf_image_bytes, config=AuditRunConfig(row_merge_tolerance=0.6, persist=False)
    )
    merged = run_audit(
        shelf_image_bytes, config=AuditRunConfig(row_merge_tolerance=2.0, persist=False)
    )
    assert split.outcome.detections.geometry.row_count == 3
    assert merged.outcome.detections.geometry.row_count == 1


def test_alert_threshold_override_changes_the_verdict(
    isolated_db: Path, gaps_image_bytes: bytes, planogram: Planogram
) -> None:
    # The gapped shelf scores 0.83: compliant under a lenient policy, not under
    # a strict one. The threshold is exclusive, so a perfect shelf can never
    # alert no matter how high it is set.
    lenient = run_audit(
        gaps_image_bytes,
        config=AuditRunConfig(compliance_alert_threshold=0.5, persist=False),
        planogram=planogram,
    )
    strict = run_audit(
        gaps_image_bytes,
        config=AuditRunConfig(compliance_alert_threshold=0.95, persist=False),
        planogram=planogram,
    )
    assert lenient.outcome.result.compliance_score == pytest.approx(0.8333, abs=1e-3)
    assert lenient.outcome.agent.alert_triggered is False
    assert strict.outcome.agent.alert_triggered is True


def test_persist_false_leaves_the_store_empty(isolated_db: Path, shelf_image_bytes: bytes) -> None:
    run_audit(shelf_image_bytes, config=AuditRunConfig(persist=False))
    _, total, _, _ = load_history(resolve_settings(AuditRunConfig()))
    assert total == 0


def test_persist_true_writes_one_row(isolated_db: Path, shelf_image_bytes: bytes) -> None:
    run = run_audit(shelf_image_bytes, config=AuditRunConfig(persist=True, store_id="S1"))
    records, total, summary, _ = load_history(resolve_settings(AuditRunConfig()))

    assert total == 1
    assert records[0].audit_id == run.outcome.record.audit_id
    assert records[0].store_id == "S1"
    assert summary.total_audits == 1


def test_repeated_runs_do_not_share_an_event_loop(
    isolated_db: Path, shelf_image_bytes: bytes
) -> None:
    # aiosqlite connections are loop-bound; a cached connection would blow up
    # on the second asyncio.run, so this is the regression guard for that.
    for _ in range(3):
        run_audit(shelf_image_bytes, config=AuditRunConfig(persist=True))
    _, total, _, _ = load_history(resolve_settings(AuditRunConfig()))
    assert total == 3


def test_a_bad_upload_raises_a_domain_error(isolated_db: Path) -> None:
    with pytest.raises(InvalidImageError):
        run_audit(b"not an image", config=AuditRunConfig(persist=False))


def test_sample_scenarios_load_from_the_manifest(sample_data_dir: Path) -> None:
    scenarios = load_sample_scenarios(sample_data_dir)

    assert [s.name for s in scenarios] == [
        "shelf_compliant",
        "shelf_minor_gaps",
        "shelf_critical_stockout",
        "shelf_dense_6row",
    ]
    assert [s.title for s in scenarios] == [
        "Compliant",
        "Minor Gaps",
        "Critical Stockout",
        "Dense 6Row",
    ]
    first = scenarios[0]
    assert first.read_image()[:2] == b"\xff\xd8"
    assert first.read_planogram().total_expected_facings == 18
    assert first.store_id == "STORE-0042"


def test_missing_manifest_yields_no_scenarios(tmp_path: Path) -> None:
    assert load_sample_scenarios(tmp_path) == []


def test_corrupt_manifest_yields_no_scenarios(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{ not json", encoding="utf-8")
    assert load_sample_scenarios(tmp_path) == []


def test_manifest_entries_with_missing_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "name": "ghost",
                        "description": "points at files that do not exist",
                        "image": "images/ghost.jpg",
                        "planogram": "planograms/ghost.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_sample_scenarios(tmp_path) == []
