"""The generated OpenAPI document is part of the public contract."""

from __future__ import annotations

from httpx import AsyncClient
import pytest

from smart_shelf import __version__

pytestmark = pytest.mark.api


async def test_openapi_document_is_served(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["title"] == "Smart Shelf Analytics"
    assert document["info"]["version"] == __version__


async def test_every_endpoint_is_documented(client: AsyncClient) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]

    assert "/health" in paths
    assert "/api/v1/audit/upload" in paths
    assert "/api/v1/audit/history" in paths
    assert "/api/v1/audit/analytics" in paths
    assert "/api/v1/audit/{audit_id}" in paths


async def test_upload_declares_a_multipart_body_and_error_responses(
    client: AsyncClient,
) -> None:
    upload = (await client.get("/openapi.json")).json()["paths"]["/api/v1/audit/upload"]["post"]

    assert "multipart/form-data" in upload["requestBody"]["content"]
    assert set(upload["responses"]) >= {"201", "422", "502"}
    assert upload["tags"] == ["audit"]


async def test_domain_schemas_are_exported(client: AsyncClient) -> None:
    schemas = (await client.get("/openapi.json")).json()["components"]["schemas"]

    assert {"ShelfAuditResult", "ProductItem", "Discrepancy", "ErrorResponse"} <= set(schemas)
    assert "total_facings" in schemas["ShelfAuditResult"]["properties"]


async def test_interactive_docs_are_available(client: AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/redoc")).status_code == 200
