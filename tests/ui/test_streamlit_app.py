"""End-to-end checks of the Streamlit front end.

Driven through Streamlit's own ``AppTest`` harness, which executes the real
script in-process and exposes the rendered element tree. That catches the
failure mode a screenshot never would: a widget change that raises somewhere
down the pipeline.

Skipped when the optional ``ui`` extra is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit", reason="requires the 'ui' extra")

from streamlit.testing.v1 import AppTest

from smart_shelf.core.config import reset_settings_cache

pytestmark = pytest.mark.ui

APP = Path(__file__).resolve().parents[2] / "app.py"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """A freshly started app pointed at a throwaway audit store."""
    monkeypatch.setenv("SHELF_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'audits.db'}")
    monkeypatch.setenv("SHELF_NOTIFICATIONS_DRY_RUN", "true")
    # A path-like value keeps the YOLO backend from downloading a checkpoint,
    # so the tests stay offline and deterministic.
    monkeypatch.setenv("SHELF_YOLO_WEIGHTS", str(tmp_path / "no-such-model.pt"))
    reset_settings_cache()
    instance = AppTest.from_file(str(APP), default_timeout=120)
    instance.run()
    return instance


def metrics(at: AppTest) -> dict[str, Any]:
    return {m.label: m.value for m in at.metric}


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_starts_and_audits_the_default_sample(app: AppTest) -> None:
    assert_clean(app)
    values = metrics(app)
    assert values["Compliance score"] == "100%"
    assert values["Detections"] == "18"
    assert values["Restocking alert"] == "CLEAR"


def test_every_sidebar_control_is_present(app: AppTest) -> None:
    keys = {w.key for w in app.sidebar.slider} | {w.key for w in app.sidebar.selectbox}
    keys |= {w.key for w in app.sidebar.toggle} | {w.key for w in app.sidebar.button}
    assert {
        "sample",
        "planogram_mode",
        "backend",
        "confidence",
        "tolerance",
        "alert_threshold",
        "show_bands",
        "show_labels",
        "live",
        "save",
    } <= keys


def test_both_images_are_rendered_side_by_side(app: AppTest) -> None:
    # One original, one annotated.
    assert len(app.image) == 2


def test_switching_to_the_stockout_sample_raises_an_alert(app: AppTest) -> None:
    app.sidebar.selectbox(key="sample").select_index(2).run()

    assert_clean(app)
    values = metrics(app)
    assert values["Restocking alert"] == "RAISED"
    assert values["Detections"] == "11"
    assert values["Compliance score"] == "50%"


def test_raising_confidence_prunes_detections(app: AppTest) -> None:
    app.sidebar.slider(key="confidence").set_value(0.93).run()

    assert_clean(app)
    assert int(metrics(app)["Detections"]) < 18


def test_raising_merging_tolerance_collapses_shelf_rows(app: AppTest) -> None:
    before = next(m.delta for m in app.metric if m.label == "Detections")
    assert before == "3 shelf rows"

    app.sidebar.slider(key="tolerance").set_value(2.0).run()

    assert_clean(app)
    after = next(m.delta for m in app.metric if m.label == "Detections")
    assert after == "1 shelf rows"


def test_tightening_the_policy_flips_the_verdict(app: AppTest) -> None:
    app.sidebar.selectbox(key="sample").select_index(1).run()
    assert metrics(app)["Restocking alert"] == "CLEAR"

    app.sidebar.slider(key="alert_threshold").set_value(0.95).run()

    assert_clean(app)
    assert metrics(app)["Restocking alert"] == "RAISED"


def test_dropping_the_planogram_switches_to_an_availability_audit(app: AppTest) -> None:
    app.sidebar.selectbox(key="planogram_mode").set_value("None (availability only)").run()

    assert_clean(app)
    # Without a planogram the mock auditor reports one group per shelf row.
    assert metrics(app)["Detections"] == "18"


def test_selecting_yolo_with_unusable_weights_fails_gracefully(app: AppTest) -> None:
    app.sidebar.selectbox(key="backend").set_value("YOLO (ultralytics)").run()

    # An unreachable checkpoint must surface as guidance, never a traceback.
    assert not app.exception, [e.value for e in app.exception]
    assert app.error, "expected a rendered error for the unusable weights"
    assert "weights not found" in " ".join(str(e.value) for e in app.error).lower()
    assert any("heuristic" in str(i.value).lower() for i in app.info)


def test_switching_to_upload_shows_the_landing_state(app: AppTest) -> None:
    app.sidebar.radio(key="source_mode").set_value("Upload a photo").run()

    assert_clean(app)
    assert not app.metric  # nothing audited yet
    assert any("upload a photo" in str(m.value).lower() for m in app.markdown)


def test_live_preview_off_does_not_audit_on_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELF_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'a.db'}")
    reset_settings_cache()
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["live"] = False
    at.run()

    assert_clean(at)
    assert not at.metric
    assert any("Live preview is off" in str(i.value) for i in at.info)


def test_saving_writes_a_row_that_the_history_tab_reports(app: AppTest) -> None:
    assert any("No audits saved yet" in str(i.value) for i in app.info)

    app.sidebar.button(key="save").click().run()

    assert_clean(app)
    values = metrics(app)
    assert values["Audits stored"] == "1"
    assert values["Mean compliance"] == "100%"


def test_structured_json_is_exposed(app: AppTest) -> None:
    assert app.json, "expected the ShelfAuditResult to be rendered"
    payload = app.json[0].value
    assert "compliance_score" in payload
    assert "products" in payload


def test_dense_six_row_sample_renders_every_row(app: AppTest) -> None:
    app.sidebar.selectbox(key="sample").select_index(3).run()

    assert_clean(app)
    values = metrics(app)
    # The bay holds 72 slots with four gaps; the OpenCV backend should find the
    # bulk of them and, critically, resolve all six shelf rows.
    assert int(values["Detections"]) >= 55
    rows = next(m.delta for m in app.metric if m.label == "Detections")
    assert rows == "6 shelf rows"


def test_density_controls_are_exposed(app: AppTest) -> None:
    keys = {w.key for w in app.sidebar.select_slider}
    assert {"max_edge", "max_items"} <= keys


def test_the_analysis_resolution_changes_the_result(app: AppTest) -> None:
    # Dense segmentation is resolution sensitive, and not monotonically: 960
    # over-segments this sample while 1280 under-segments it. The control just
    # has to reach the detector.
    app.sidebar.selectbox(key="sample").select_index(3).run()
    at_default = int(metrics(app)["Detections"])

    app.sidebar.select_slider(key="max_edge").set_value(1280).run()

    assert_clean(app)
    assert int(metrics(app)["Detections"]) != at_default


def test_the_detection_cap_truncates_the_result(app: AppTest) -> None:
    app.sidebar.selectbox(key="sample").select_index(3).run()
    app.sidebar.select_slider(key="max_items").set_value(100).run()

    assert_clean(app)
    assert int(metrics(app)["Detections"]) <= 100


def test_the_hybrid_backend_is_selectable(app: AppTest) -> None:
    # Weights are unreachable in tests, so hybrid must degrade to OpenCV and
    # still produce a result rather than an error.
    app.sidebar.selectbox(key="backend").set_value("Hybrid (YOLO + OpenCV)").run()

    assert_clean(app)
    assert int(metrics(app)["Detections"]) > 0
