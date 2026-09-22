"""Repository-root shortcut for the Streamlit front end.

    streamlit run app.py

The real application lives in :mod:`smart_shelf.ui.app`; this file only makes
``src/`` importable for a fresh clone that has not been installed yet, checks
that Streamlit is actually driving, and then hands over.
"""

from __future__ import annotations

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import streamlit.runtime
except ImportError:  # pragma: no cover - depends on the install profile
    raise SystemExit(
        "The Streamlit front end needs the optional 'ui' extra.\n\n"
        "    uv sync --locked --extra ui\n"
        "    streamlit run app.py\n"
    ) from None

if not streamlit.runtime.exists():
    # `python app.py` would import Streamlit with no script runner attached and
    # then exit silently, so say what to do instead.
    raise SystemExit(
        "This file must be launched by Streamlit, not by Python directly.\n\n"
        "    streamlit run app.py\n"
        "    python -m smart_shelf.ui      # equivalent\n"
        "    make ui\n"
    )

from smart_shelf.ui.app import main

# Called explicitly rather than relying on an import side effect: Streamlit
# re-runs this script on every interaction, but the import above is cached
# after the first run, so a module-level render would only ever paint once.
main()
