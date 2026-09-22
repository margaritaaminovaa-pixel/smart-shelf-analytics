"""Streamlit front end for Smart Shelf Analytics.

Run it with::

    streamlit run app.py                 # repository-root shortcut
    python -m smart_shelf.ui             # installed package
    make ui

The app calls the pipeline in-process through :mod:`smart_shelf.ui.runner`
rather than over HTTP, so it needs no running API and shows exactly the objects
the service layer produces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

import streamlit as st

from smart_shelf import __version__
from smart_shelf.core.config import DetectorBackend, Settings
from smart_shelf.core.exceptions import DetectionError, SmartShelfError
from smart_shelf.db.models import AuditFilter
from smart_shelf.genai.schemas import Planogram, Severity
from smart_shelf.ui.annotate import annotate_detections, row_color_hex, to_rgb
from smart_shelf.ui.runner import (
    AuditRun,
    AuditRunConfig,
    SampleScenario,
    load_history,
    load_sample_scenarios,
    resolve_settings,
    run_audit,
)

PAGE_TITLE = "Smart Shelf Analytics"
PAGE_ICON = "🛒"

SEVERITY_ICON: dict[Severity, str] = {
    Severity.LOW: "🟢",
    Severity.MEDIUM: "🟡",
    Severity.HIGH: "🟠",
    Severity.CRITICAL: "🔴",
}

CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
  .ssa-title { font-size: 2.05rem; font-weight: 700; letter-spacing: -0.02em;
               margin-bottom: 0.15rem; }
  .ssa-sub   { color: #6b7280; font-size: 1.0rem; margin-bottom: 0.2rem; }
  .ssa-pipe  { color: #9ca3af; font-size: 0.82rem; font-family: ui-monospace,
               SFMono-Regular, Menlo, monospace; margin-bottom: 1.1rem; }
  .ssa-legend { display: flex; flex-wrap: wrap; gap: 0.55rem; margin: 0.5rem 0 0.2rem; }
  .ssa-chip  { display: inline-flex; align-items: center; gap: 0.4rem;
               padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem;
               background: rgba(128,128,128,0.12); }
  .ssa-dot   { width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; }
  .ssa-empty { border: 1px dashed rgba(128,128,128,0.4); border-radius: 0.75rem;
               padding: 2.2rem; text-align: center; color: #6b7280; }
  div[data-testid="stMetric"] { background: rgba(128,128,128,0.07);
               border: 1px solid rgba(128,128,128,0.16); border-radius: 0.7rem;
               padding: 0.85rem 1rem; }
</style>
"""


@dataclass(frozen=True, slots=True)
class SidebarState:
    """Everything the sidebar resolved for this rerun."""

    image_bytes: bytes | None
    filename: str | None
    planogram: Planogram | None
    config: AuditRunConfig
    show_bands: bool
    show_labels: bool
    live: bool
    save_clicked: bool
    scenario: SampleScenario | None

    def signature(self) -> str:
        """Stable key for the inputs that change the audit result."""
        digest = hashlib.sha256(self.image_bytes or b"").hexdigest()
        planogram = self.planogram.model_dump_json() if self.planogram else ""
        parts = (
            digest,
            self.config.backend.value,
            f"{self.config.detection_confidence:.4f}",
            f"{self.config.row_merge_tolerance:.4f}",
            f"{self.config.compliance_alert_threshold:.4f}",
            str(self.config.critical_oos_threshold),
            str(self.config.detection_max_items),
            str(self.config.preprocess_max_edge),
            self.config.store_id or "",
            self.config.shelf_id or "",
            hashlib.sha256(planogram.encode()).hexdigest(),
        )
        return "|".join(parts)


# -- sidebar -------------------------------------------------------------------


def _render_sidebar() -> SidebarState:
    """Draw every control and return the resolved run configuration."""
    sidebar = st.sidebar
    sidebar.markdown(f"### {PAGE_ICON} Controls")
    sidebar.caption(f"smart-shelf-analytics v{__version__}")

    # -- image source ----------------------------------------------------------
    sidebar.markdown("#### 1 · Shelf image")
    scenarios = load_sample_scenarios()
    options = ["Sample shelf", "Upload a photo"] if scenarios else ["Upload a photo"]
    source = sidebar.radio(
        "Source", options, horizontal=True, label_visibility="collapsed", key="source_mode"
    )

    image_bytes: bytes | None = None
    filename: str | None = None
    scenario: SampleScenario | None = None

    if source == "Sample shelf" and scenarios:
        # Options are the scenario *names*, not the objects: widget values are
        # compared and formatted by Streamlit, and a dataclass round-trips
        # poorly through that. Names are unique by construction.
        by_name = {item.name: item for item in scenarios}
        chosen = sidebar.selectbox(
            "Sample scenario",
            list(by_name),
            format_func=lambda name: by_name[name].title,
            help="Committed synthetic shelves - one click, no upload needed.",
            key="sample",
        )
        scenario = by_name.get(chosen) if chosen else None
        if scenario is not None:
            sidebar.caption(scenario.description)
            image_bytes = scenario.read_image()
            filename = scenario.image_path.name
    else:
        if not scenarios:
            sidebar.info(
                "No sample data found. Run `make sample-data` to enable the "
                "one-click demo shelves.",
                icon=":material/info:",
            )
        upload = sidebar.file_uploader(
            "Shelf photograph", type=["jpg", "jpeg", "png", "webp"], key="upload"
        )
        if upload is not None:
            image_bytes = upload.getvalue()
            filename = upload.name

    # -- planogram -------------------------------------------------------------
    sidebar.markdown("#### 2 · Expected planogram")
    planogram_choices = (
        ["Matching sample", "Upload JSON", "None (availability only)"]
        if scenario is not None
        else ["Upload JSON", "None (availability only)"]
    )
    planogram_mode = sidebar.selectbox(
        "Planogram",
        planogram_choices,
        label_visibility="collapsed",
        help="Without a planogram the audit reports gaps and empty slots only.",
        key="planogram_mode",
    )

    planogram: Planogram | None = None
    if planogram_mode == "Matching sample" and scenario is not None:
        planogram = scenario.read_planogram()
        sidebar.caption(f"{planogram.total_expected_facings} expected facings")
    elif planogram_mode == "Upload JSON":
        uploaded = sidebar.file_uploader("Planogram JSON", type=["json"])
        if uploaded is not None:
            try:
                planogram = Planogram.model_validate_json(uploaded.getvalue().decode("utf-8"))
                sidebar.caption(f"{planogram.total_expected_facings} expected facings")
            except (ValueError, UnicodeDecodeError) as exc:
                sidebar.error(f"Could not read that planogram: {exc}", icon=":material/error:")

    # -- detection -------------------------------------------------------------
    sidebar.markdown("#### 3 · Detection")
    backend_label = sidebar.selectbox(
        "Backend",
        ["OpenCV (heuristic)", "YOLO (ultralytics)", "Hybrid (YOLO + OpenCV)"],
        help=(
            "OpenCV needs no weights and always works. YOLO uses the SKU-110K "
            "checkpoint, fetched on first use. Hybrid runs YOLO and falls back "
            "to OpenCV when it finds too little."
        ),
        key="backend",
    )
    backend = {
        "YOLO (ultralytics)": DetectorBackend.YOLO,
        "Hybrid (YOLO + OpenCV)": DetectorBackend.HYBRID,
    }.get(backend_label, DetectorBackend.HEURISTIC)
    confidence = sidebar.slider(
        "Detection confidence",
        0.0,
        0.99,
        0.0,
        0.01,
        help=(
            "Drops facings scoring below this. The OpenCV score is a "
            "rectangularity heuristic, not a calibrated probability - it only "
            "starts pruning above roughly 0.80."
        ),
        key="confidence",
    )
    tolerance = sidebar.slider(
        "Vertical merging tolerance",
        0.10,
        2.00,
        0.60,
        0.05,
        help=(
            "How far apart two facings can sit vertically and still count as one "
            "shelf row, as a multiple of the median facing height. Raise it to "
            "merge rows, lower it to split them."
        ),
        key="tolerance",
    )

    max_edge = sidebar.select_slider(
        "Analysis resolution",
        options=[960, 1280, 1600, 2048, 2560],
        value=1600,
        help=(
            "Longest edge the image is scaled to before detection. Dense bays "
            "are sensitive to it: on the six-row sample the count plateaus at "
            "1600, and downscaling further distorts it in both directions."
        ),
        key="max_edge",
    )
    max_items = sidebar.select_slider(
        "Max detections",
        options=[100, 200, 400, 600, 1000, 2000],
        value=600,
        help="A full aisle can exceed several hundred facings; a low cap truncates them.",
        key="max_items",
    )

    # -- alerting --------------------------------------------------------------
    sidebar.markdown("#### 4 · Alerting policy")
    alert_threshold = sidebar.slider(
        "Compliance alert threshold",
        0.0,
        1.0,
        0.80,
        0.05,
        help="Audits scoring below this raise a restocking alert.",
        key="alert_threshold",
    )
    critical_oos = sidebar.number_input(
        "Out-of-stock escalation count",
        min_value=1,
        max_value=50,
        value=3,
        help="This many out-of-stock SKUs escalates the alert to 'immediate'.",
        key="oos_threshold",
    )

    # -- display ---------------------------------------------------------------
    sidebar.markdown("#### 5 · Display")
    show_bands = sidebar.toggle("Shelf row bands", value=True, key="show_bands")
    show_labels = sidebar.toggle(
        "Box labels",
        value=True,
        help="Turn off on dense shelves, where per-box chips overlap.",
        key="show_labels",
    )
    live = sidebar.toggle(
        "Live preview",
        value=True,
        help="Re-audit automatically whenever a control changes.",
        key="live",
    )
    save_clicked = sidebar.button(
        "Run & save to history", type="primary", width="stretch", icon=":material/save:", key="save"
    )

    store_id = scenario.store_id if scenario else None
    shelf_id = scenario.shelf_id if scenario else None

    return SidebarState(
        image_bytes=image_bytes,
        filename=filename,
        planogram=planogram,
        config=AuditRunConfig(
            backend=backend,
            detection_confidence=confidence,
            row_merge_tolerance=tolerance,
            compliance_alert_threshold=alert_threshold,
            critical_oos_threshold=int(critical_oos),
            detection_max_items=int(max_items),
            preprocess_max_edge=int(max_edge),
            store_id=store_id or None,
            shelf_id=shelf_id or None,
        ),
        show_bands=show_bands,
        show_labels=show_labels,
        live=live,
        save_clicked=save_clicked,
        scenario=scenario,
    )


# -- main view -----------------------------------------------------------------


def _render_header() -> None:
    st.markdown(f'<div class="ssa-title">{PAGE_ICON} {PAGE_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ssa-sub">Photograph a shelf, get planogram compliance, '
        "structured audit data and an automated restocking decision.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ssa-pipe">preprocess → detect → multimodal LLM → reconcile '
        "→ agent → persist</div>",
        unsafe_allow_html=True,
    )


def _render_metrics(run: AuditRun) -> None:
    """The four headline numbers."""
    record = run.outcome.record
    result = run.outcome.result
    decision = run.outcome.agent.decision
    threshold = run.settings.compliance_alert_threshold

    first, second, third, fourth = st.columns(4)
    gap = result.compliance_score - threshold
    first.metric(
        "Compliance score",
        f"{result.compliance_score:.0%}",
        delta=f"{gap:+.0%} vs threshold",
        delta_color="normal",
        help=f"Alert threshold is {threshold:.0%}.",
    )
    second.metric(
        "Detections",
        record.detection_count,
        delta=f"{run.outcome.detections.geometry.row_count} shelf rows",
        delta_color="off",
        help="Product facings located by the detector.",
    )
    third.metric(
        "Occupancy",
        f"{result.shelf_occupancy:.0%}",
        delta=f"{result.total_facings} facings",
        delta_color="off",
        help="Share of the image surface covered by product.",
    )
    if decision.should_alert:
        fourth.metric(
            "Restocking alert",
            "RAISED",
            delta=decision.urgency.value,
            delta_color="inverse",
            help="The autonomous agent decided this shelf needs attention.",
        )
    else:
        fourth.metric(
            "Restocking alert",
            "CLEAR",
            delta="no action",
            delta_color="off",
            help="The shelf satisfies the configured policy.",
        )


def _render_legend(run: AuditRun) -> None:
    rows = run.outcome.detections.by_shelf_level()
    if not rows:
        return
    chips = "".join(
        f'<span class="ssa-chip"><span class="ssa-dot" '
        f'style="background:{row_color_hex(level)}"></span>'
        f"Row {level} · {len(items)} facings</span>"
        for level, items in rows.items()
    )
    st.markdown(f'<div class="ssa-legend">{chips}</div>', unsafe_allow_html=True)


def _render_images(run: AuditRun, *, show_bands: bool, show_labels: bool) -> None:
    """Original photo beside the annotated one."""
    original, annotated = st.columns(2, gap="medium")
    prepared = run.prepared

    with original:
        st.markdown("##### Original")
        st.image(to_rgb(prepared.pixels), width="stretch")
        st.caption(
            f"{prepared.original_width}x{prepared.original_height}px "
            f"-> analysed at {prepared.width}x{prepared.height}px, "
            f"sha256 {prepared.sha256[:12]}"
        )

    with annotated:
        st.markdown("##### Detected facings")
        st.image(
            annotate_detections(
                prepared.pixels,
                run.outcome.detections,
                show_bands=show_bands,
                show_labels=show_labels,
            ),
            width="stretch",
        )
        st.caption(
            f"{run.outcome.detections.backend} · "
            f"{run.outcome.detections.count} facings in "
            f"{run.outcome.detections.geometry.row_count} rows · "
            f"{run.outcome.detections.duration_ms:.0f}ms"
        )
        _render_legend(run)


def _render_discrepancies(run: AuditRun) -> None:
    discrepancies = run.outcome.result.discrepancies
    if not discrepancies:
        st.success("No planogram deviations found on this shelf.", icon=":material/check_circle:")
        return
    st.dataframe(
        [
            {
                "": SEVERITY_ICON.get(d.severity, ""),
                "Severity": d.severity.value,
                "Type": d.discrepancy_type.value.replace("_", " "),
                "SKU": d.sku or "-",
                "Product": d.product_name or "-",
                "Row": d.shelf_level if d.shelf_level is not None else "-",
                "Expected": d.expected or "-",
                "Observed": d.observed or "-",
                "Confidence": round(d.confidence, 2),
            }
            for d in sorted(discrepancies, key=lambda d: -d.severity.rank)
        ],
        width="stretch",
        hide_index=True,
    )


def _render_restock_plan(run: AuditRun) -> None:
    tasks = run.outcome.agent.tasks
    if not tasks:
        st.info(
            "The agent produced no restocking tasks for this shelf.", icon=":material/checklist:"
        )
        return
    for index, task in enumerate(tasks, start=1):
        icon = SEVERITY_ICON.get(task.severity, "")
        st.markdown(
            f"**{index}. {icon} {task.product_name}**"
            f"{f' · `{task.sku}`' if task.sku else ''}"
            f"{f' · row {task.shelf_level}' if task.shelf_level is not None else ''}"
        )
        st.caption(task.action)


def _render_agent_log(run: AuditRun) -> None:
    decision = run.outcome.agent.decision
    st.markdown(f"**Verdict:** {decision.headline}")
    if decision.reasons:
        for reason in decision.reasons:
            st.markdown(f"- {reason}")
    else:
        st.caption("No policy rule fired.")

    if not run.alerts:
        st.info(
            "No alert dispatched, so no channel payload was produced.",
            icon=":material/notifications_off:",
        )
        return

    st.divider()
    for notification in run.alerts:
        st.markdown(f"**Dispatched:** `{notification.urgency}` · {notification.title}")
        st.code(notification.body, language="text")
        with st.expander("Slack Block Kit payload"):
            st.json(notification.to_slack_blocks())
        with st.expander("Email webhook payload"):
            st.json(notification.to_email_payload())


def _render_trace(run: AuditRun) -> None:
    timings = run.stage_timings
    if not timings:
        st.caption("No spans recorded.")
        return
    st.bar_chart(
        {"milliseconds": dict(timings)},
        horizontal=True,
        height=220,
    )
    st.dataframe(
        [{"Stage": name, "Duration (ms)": ms} for name, ms in timings],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Spans come from the same tracer interface that ships audits to Langfuse "
        "when `SHELF_LANGFUSE_ENABLED=true`."
    )


def _render_details(run: AuditRun) -> None:
    record = run.outcome.record
    tabs = st.tabs(
        [
            f"Discrepancies ({record.discrepancy_count})",
            f"Restock plan ({len(run.outcome.agent.tasks)})",
            "Agent log",
            "Structured JSON",
            "Pipeline trace",
        ]
    )
    with tabs[0]:
        _render_discrepancies(run)
    with tabs[1]:
        _render_restock_plan(run)
    with tabs[2]:
        _render_agent_log(run)
    with tabs[3]:
        st.caption("Exactly the `ShelfAuditResult` the API returns and the store persists.")
        st.json(record.result.model_dump(mode="json"))
        st.download_button(
            "Download audit JSON",
            data=record.result.model_dump_json(indent=2),
            file_name=f"audit-{record.audit_id}.json",
            mime="application/json",
            icon=":material/download:",
        )
    with tabs[4]:
        _render_trace(run)


def _render_landing(state: SidebarState) -> None:
    st.markdown(
        '<div class="ssa-empty">'
        "<h4>Pick a sample shelf or upload a photo</h4>"
        "<p>The sidebar has three synthetic shelves committed to the repository - "
        "a compliant bay, one with minor gaps, and one with a critical stockout. "
        "Selecting one runs the whole pipeline immediately.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if state.image_bytes:
        st.image(state.image_bytes, caption=state.filename, width=520)


# -- history tab ---------------------------------------------------------------


def _render_history(settings: Settings) -> None:
    try:
        records, total, summary, offenders = load_history(settings, AuditFilter(limit=200))
    except SmartShelfError as exc:
        st.error(f"Could not read the audit store: {exc.message}", icon=":material/error:")
        return

    if total == 0:
        st.info(
            "No audits saved yet. Use **Run & save to history** in the sidebar to persist one.",
            icon=":material/folder_open:",
        )
        return

    first, second, third, fourth = st.columns(4)
    first.metric("Audits stored", summary.total_audits)
    second.metric("Mean compliance", f"{summary.mean_compliance:.0%}")
    third.metric("Alert rate", f"{summary.alert_rate:.0%}")
    fourth.metric("Empty slots seen", summary.out_of_stock_events)

    if summary.trend:
        st.markdown("##### Compliance trend")
        st.line_chart(
            {"mean compliance": {p.day: p.mean_compliance for p in summary.trend}},
            height=260,
        )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("##### Recent audits")
        st.dataframe(
            [
                {
                    "When": r.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Store": r.store_id or "-",
                    "Shelf": r.shelf_id or "-",
                    "Compliance": round(r.compliance_score, 3),
                    "Discrepancies": r.discrepancy_count,
                    "Alert": "yes" if r.alert_triggered else "-",
                    "Detector": r.detector_backend,
                }
                for r in records
            ],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.markdown("##### Worst-offending SKUs")
        if offenders:
            st.dataframe(
                [
                    {
                        "SKU": o.sku,
                        "Product": o.product_name,
                        "Times": o.occurrences,
                        "Critical": o.critical_occurrences,
                    }
                    for o in offenders
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No SKU-attributed discrepancies yet.")


# -- orchestration -------------------------------------------------------------


def _execute(state: SidebarState, *, persist: bool) -> AuditRun | None:
    """Run the pipeline, translating domain failures into UI feedback."""
    if state.image_bytes is None:
        return None
    config = replace(state.config, persist=persist)
    try:
        with st.spinner("Running the audit pipeline..."):
            return run_audit(
                state.image_bytes,
                config=config,
                planogram=state.planogram,
                filename=state.filename,
            )
    except DetectionError as exc:
        st.error(f"Detection failed: {exc.message}", icon=":material/error:")
        if config.backend is DetectorBackend.YOLO:
            st.info(
                "YOLO downloads its checkpoint on first use, so this usually "
                "means no network or an unreadable `SHELF_YOLO_WEIGHTS` path. "
                "Switch the sidebar back to **OpenCV (heuristic)**, which needs "
                "no weights at all.",
                icon=":material/lightbulb:",
            )
        return None
    except SmartShelfError as exc:
        st.error(f"{exc.code}: {exc.message}", icon=":material/error:")
        if exc.details:
            st.json(exc.details)
        return None


def _resolve_run(state: SidebarState) -> AuditRun | None:
    """Reuse the memoised run unless an input that affects it changed."""
    if state.image_bytes is None:
        return None

    if state.save_clicked:
        run = _execute(state, persist=True)
        if run is not None:
            st.session_state["run"] = run
            st.session_state["signature"] = state.signature()
            st.toast("Audit saved to history", icon=":material/save:")
        return run

    cached: AuditRun | None = st.session_state.get("run")
    if cached is not None and st.session_state.get("signature") == state.signature():
        return cached

    if not state.live and cached is not None:
        st.warning(
            "Live preview is off - the view below is from the previous settings. "
            "Use **Run & save to history** to re-run.",
            icon=":material/pause:",
        )
        return cached
    if not state.live and cached is None:
        st.info(
            "Live preview is off. Use **Run & save to history** to run once.",
            icon=":material/pause:",
        )
        return None

    run = _execute(state, persist=False)
    if run is not None:
        st.session_state["run"] = run
        st.session_state["signature"] = state.signature()
    return run


def main() -> None:
    """Entry point executed by Streamlit on every rerun."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    _render_header()

    state = _render_sidebar()
    audit_tab, history_tab = st.tabs(["Live audit", "History & analytics"])

    with audit_tab:
        run = _resolve_run(state)
        if run is None:
            _render_landing(state)
        else:
            _render_metrics(run)
            st.divider()
            _render_images(run, show_bands=state.show_bands, show_labels=state.show_labels)
            st.divider()
            _render_details(run)

    with history_tab:
        _render_history(resolve_settings(state.config))


def _describe_engine(settings: Settings) -> dict[str, Any]:  # pragma: no cover - debug aid
    """Small helper used when troubleshooting which backends are bound."""
    return {"detector": settings.detector_backend.value, "vlm": settings.vlm_backend.value}


# Streamlit re-executes the script it was given on every interaction, but
# `sys.modules` survives. Rendering must therefore happen in an explicit call,
# never as an import side effect: importing this module a second time is a
# no-op and would leave the page blank. This guard fires when Streamlit runs
# THIS file (`python -m smart_shelf.ui`); the repository-root `app.py` imports
# `main` and calls it itself.
if __name__ == "__main__":
    main()
