"""Run the API with uvicorn: ``python -m smart_shelf``."""

from __future__ import annotations

import argparse

import uvicorn

from smart_shelf.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="smart-shelf", description="Run the analytics API.")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container entrypoint
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    uvicorn.run(
        "smart_shelf.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=None if args.reload else args.workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
