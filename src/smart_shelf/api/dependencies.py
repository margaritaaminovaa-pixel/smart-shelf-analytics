"""Composition root and FastAPI dependency providers.

The object graph is built once per process in :class:`ApplicationContainer` and
handed to request handlers through ``Depends``. Tests override
``app.state.container`` with fakes, so no handler ever constructs a collaborator
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Annotated, Self

from fastapi import Depends, Request

from smart_shelf.agents.inventory_agent import InventoryAgent
from smart_shelf.core.config import Settings, get_settings
from smart_shelf.core.observability import Tracer, get_tracer
from smart_shelf.cv.detector import ObjectDetector, build_detector
from smart_shelf.db.analytics import AnalyticsService
from smart_shelf.db.repository import AuditRepository
from smart_shelf.genai.engine import VisionLanguageEngine, build_vlm_engine
from smart_shelf.services.audit_service import AuditService


@dataclass(slots=True)
class ApplicationContainer:
    """Owns every long-lived collaborator for the lifetime of the process."""

    settings: Settings
    repository: AuditRepository
    detector: ObjectDetector
    engine: VisionLanguageEngine
    agent: InventoryAgent
    audit_service: AuditService
    analytics: AnalyticsService
    tracer: Tracer

    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        *,
        repository: AuditRepository | None = None,
        detector: ObjectDetector | None = None,
        engine: VisionLanguageEngine | None = None,
        agent: InventoryAgent | None = None,
    ) -> Self:
        """Build and connect the object graph. Any part can be injected."""
        settings = settings or get_settings()
        tracer = get_tracer()

        repository = repository or AuditRepository(settings.sqlite_path)
        await repository.connect()

        detector = detector or build_detector(settings)
        engine = engine or build_vlm_engine(settings, tracer=tracer)
        agent = agent or InventoryAgent(settings, tracer=tracer)

        return cls(
            settings=settings,
            repository=repository,
            detector=detector,
            engine=engine,
            agent=agent,
            audit_service=AuditService(
                settings,
                detector=detector,
                engine=engine,
                repository=repository,
                agent=agent,
                tracer=tracer,
            ),
            analytics=AnalyticsService(repository),
            tracer=tracer,
        )

    async def aclose(self) -> None:
        """Flush traces and release the database connection."""
        self.tracer.flush()
        await self.repository.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def get_container(request: Request) -> ApplicationContainer:
    """Fetch the container attached to the running application."""
    container: ApplicationContainer = request.app.state.container
    return container


def get_audit_service(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> AuditService:
    """Provide the audit pipeline service."""
    return container.audit_service


def get_analytics_service(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> AnalyticsService:
    """Provide the analytics query service."""
    return container.analytics


def get_repository(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> AuditRepository:
    """Provide the audit store."""
    return container.repository


def get_app_settings(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> Settings:
    """Provide the active configuration."""
    return container.settings


ContainerDep = Annotated[ApplicationContainer, Depends(get_container)]
AuditServiceDep = Annotated[AuditService, Depends(get_audit_service)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
RepositoryDep = Annotated[AuditRepository, Depends(get_repository)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
