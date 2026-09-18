"""FastAPI application factory.

``create_app`` is a factory rather than a module-level singleton so tests can
build an isolated app per case with injected fakes, and so multiple
configurations can coexist in one process.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smart_shelf import __version__
from smart_shelf.api.dependencies import ApplicationContainer
from smart_shelf.api.errors import register_exception_handlers
from smart_shelf.api.middleware import RequestContextMiddleware
from smart_shelf.api.routers import audit_router, health_router
from smart_shelf.core.config import Settings, get_settings
from smart_shelf.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """\
Automated retail shelf auditing.

Upload a shelf photograph and the service runs object detection, structures the
scene with a multimodal LLM against strict Pydantic schemas, reconciles it with
the expected planogram, and lets an autonomous agent decide whether the shelf
needs restocking.
"""


def _lifespan(
    settings: Settings, container: ApplicationContainer | None
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the startup/shutdown handler for one application instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or await ApplicationContainer.create(settings)
        logger.info(
            "app.started",
            environment=settings.environment.value,
            detector=getattr(app.state.container.detector, "name", "unknown"),
            vlm=getattr(app.state.container.engine, "name", "unknown"),
        )
        try:
            yield
        finally:
            if container is None:
                await app.state.container.aclose()
            logger.info("app.stopped")

    return lifespan


def create_app(
    settings: Settings | None = None,
    *,
    container: ApplicationContainer | None = None,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: configuration override; defaults to the process settings.
        container: a pre-built object graph. When supplied the app does not own
            it and will not close it on shutdown - this is how tests inject fakes.
    """
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings, force=True)

    app = FastAPI(
        title="Smart Shelf Analytics",
        description=DESCRIPTION,
        version=__version__,
        debug=settings.debug,
        lifespan=_lifespan(settings, container),
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        contact={"name": "Retail Analytics Platform"},
        license_info={"name": "MIT"},
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment.value != "production" else [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(audit_router, prefix=settings.api_v1_prefix)
    return app


app = create_app
"""Alias kept so ``uvicorn smart_shelf.api.app:app --factory`` works."""
