"""Streamlit front end.

``app`` is intentionally not imported here: it pulls in Streamlit, which is an
optional extra. Import :mod:`smart_shelf.ui.annotate` and
:mod:`smart_shelf.ui.runner` freely; import ``app`` only under Streamlit.
"""

from __future__ import annotations

from smart_shelf.ui.annotate import annotate_detections, row_color, row_color_hex, to_rgb
from smart_shelf.ui.runner import (
    AuditRun,
    AuditRunConfig,
    SampleScenario,
    load_history,
    load_sample_scenarios,
    run_audit,
)

__all__ = [
    "AuditRun",
    "AuditRunConfig",
    "SampleScenario",
    "annotate_detections",
    "load_history",
    "load_sample_scenarios",
    "row_color",
    "row_color_hex",
    "run_audit",
    "to_rgb",
]
