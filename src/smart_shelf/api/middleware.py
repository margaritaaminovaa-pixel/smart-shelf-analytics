"""Cross-cutting HTTP middleware: request identity and access logging."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

from smart_shelf.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns every request an id, binds it to the log context and times it.

    The id is echoed back in ``X-Request-ID`` and embedded in error envelopes,
    so a support ticket screenshot is enough to find the exact trace.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Tag, time and log one request."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(
            request_id=request_id, path=request.url.path, method=request.method
        )
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request.failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"
        logger.info(
            "request.completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        structlog.contextvars.clear_contextvars()
        return response
