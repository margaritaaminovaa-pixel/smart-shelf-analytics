"""Launch the Streamlit front end: ``python -m smart_shelf.ui``."""

from __future__ import annotations

from pathlib import Path
import sys

HINT = (
    "The Streamlit front end needs the optional 'ui' extra.\n\n"
    "    uv sync --locked --extra ui      # or: pip install 'smart-shelf-analytics[ui]'\n"
)


def main() -> None:
    """Hand the packaged app over to Streamlit's CLI."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:  # pragma: no cover - depends on the install profile
        raise SystemExit(HINT) from None

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
