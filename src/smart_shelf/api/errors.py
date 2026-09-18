"""HTTP error translation.

All handlers emit the same envelope::

    {"error": {"code": "...", "message": "...", "details": {...}}, "request_id": "..."}

so clients parse one shape regardless of whether the failure came from
validation, the domain, or an unhandled bug.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from smart_shelf.core.exceptions import SmartShelfError
from smart_shelf.core.logging import get_logger

logger = get_logger(__name__)

_UNSAFE_ERROR_KEYS = frozenset({"ctx", "input", "url"})
"""Keys that carry exception objects or raw request payloads."""

HTTP_422_UNPROCESSABLE = 422
"""Spelled numerically: Starlette renamed its 422 constant mid-1.x."""


def serializable_errors(exc: ValidationError | RequestValidationError) -> list[dict[str, Any]]:
    """Render Pydantic errors as JSON-safe dicts.

    ``ValidationError.errors()`` embeds the original exception object under
    ``ctx`` and the raw input under ``input``; neither survives JSON encoding,
    and the raw input may contain an entire uploaded payload.
    """
    if isinstance(exc, ValidationError):
        raw: Sequence[Any] = exc.errors(
            include_url=False, include_context=False, include_input=False
        )
    else:
        # FastAPI's RequestValidationError.errors() takes no keyword arguments.
        raw = exc.errors()
    return [
        {key: value for key, value in dict(error).items() if key not in _UNSAFE_ERROR_KEYS}
        for error in raw
    ]


def error_body(
    code: str, message: str, details: dict[str, Any] | None = None, request_id: str | None = None
) -> dict[str, Any]:
    """Build the canonical error envelope."""
    return {
        "error": {"code": code, "message": message, "details": details or {}},
        "request_id": request_id,
    }


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return str(value) if value else None


async def smart_shelf_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map a domain error onto its declared status code."""
    assert isinstance(exc, SmartShelfError)
    logger.warning(
        "request.domain_error",
        code=exc.code,
        message=exc.message,
        path=request.url.path,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.code, exc.message, exc.details, _request_id(request)),
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Flatten FastAPI validation failures into the shared envelope."""
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE,
        content=error_body(
            "validation_error",
            "request payload failed validation",
            {"errors": serializable_errors(exc)},
            _request_id(request),
        ),
    )


async def http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Wrap Starlette HTTP errors (404, 405, ...) in the shared envelope."""
    assert isinstance(exc, StarletteHTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(f"http_{exc.status_code}", str(exc.detail), None, _request_id(request)),
        headers=getattr(exc, "headers", None),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. Logs with a stack trace and never leaks internals."""
    logger.exception("request.unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body(
            "internal_error", "an unexpected error occurred", None, _request_id(request)
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to ``app``."""
    app.add_exception_handler(SmartShelfError, smart_shelf_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
