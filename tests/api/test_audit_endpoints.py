"""POST /api/v1/audit/upload, GET /audit/history and GET /audit/analytics."""

from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
import pytest

from smart_shelf.genai.schemas import Planogram

pytestmark = pytest.mark.api

PREFIX = "/api/v1/audit"


def image_part(payload: bytes, name: str = "shelf.jpg") -> dict[str, Any]:
    return {"image": (name, payload, "image/jpeg")}


async def test_upload_returns_a_structured_audit(
    client: AsyncClient, shelf_image_bytes: bytes
) -> None:
    response = await client.post(f"{PREFIX}/upload", files=image_part(shelf_image_bytes))

    assert response.status_code == 201
    body = response.json()
    assert body["audit_id"]
    assert 0.0 <= body["compliance_score"] <= 1.0
    assert body["detection"]["backend"] == "opencv-heuristic"
    assert body["detection"]["detection_count"] == 18
    assert body["detection"]["shelf_rows"] == 3
    assert body["result"]["products"]
    assert "total_facings" in body["result"]
    assert body["agent"]["urgency"] in {"none", "routine", "urgent", "immediate"}


async def test_upload_with_an_inline_planogram_scores_compliance(
    client: AsyncClient, shelf_image_bytes: bytes, planogram: Planogram
) -> None:
    response = await client.post(
        f"{PREFIX}/upload",
        files=image_part(shelf_image_bytes),
        data={"planogram": planogram.model_dump_json(), "store_id": "STORE-1"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["compliance_score"] == 1.0
    assert body["result"]["discrepancies"] == []
    assert body["result"]["store_id"] == "STORE-1"
    assert body["alert_triggered"] is False


async def test_upload_with_an_uploaded_planogram_file(
    client: AsyncClient, shelf_image_bytes: bytes, planogram: Planogram
) -> None:
    files = image_part(shelf_image_bytes)
    files["planogram_file"] = (
        "planogram.json",
        planogram.model_dump_json().encode(),
        "application/json",
    )

    response = await client.post(f"{PREFIX}/upload", files=files)

    assert response.status_code == 201
    assert response.json()["compliance_score"] == 1.0


async def test_stockout_upload_raises_an_alert_with_restock_tasks(
    client: AsyncClient, stockout_image_bytes: bytes, planogram: Planogram
) -> None:
    response = await client.post(
        f"{PREFIX}/upload",
        files=image_part(stockout_image_bytes, "stockout.jpg"),
        data={"planogram": planogram.model_dump_json()},
    )

    body = response.json()
    assert response.status_code == 201
    assert body["compliance_score"] < 0.8
    assert body["alert_triggered"] is True
    assert body["agent"]["reasons"]
    assert body["agent"]["tasks"]
    assert body["agent"]["tasks"][0]["severity"] in {"critical", "high"}


async def test_upload_rejects_a_non_image(client: AsyncClient) -> None:
    response = await client.post(
        f"{PREFIX}/upload", files={"image": ("notes.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


async def test_upload_rejects_malformed_planogram_json(
    client: AsyncClient, shelf_image_bytes: bytes
) -> None:
    response = await client.post(
        f"{PREFIX}/upload",
        files=image_part(shelf_image_bytes),
        data={"planogram": "{not json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"
    assert "not valid JSON" in response.json()["error"]["message"]


async def test_upload_rejects_a_planogram_that_breaks_the_schema(
    client: AsyncClient, shelf_image_bytes: bytes
) -> None:
    response = await client.post(
        f"{PREFIX}/upload",
        files=image_part(shelf_image_bytes),
        data={"planogram": json.dumps({"entries": [{"sku": "A"}]})},
    )

    assert response.status_code == 422
    assert "does not match the expected schema" in response.json()["error"]["message"]


async def test_upload_requires_an_image(client: AsyncClient) -> None:
    response = await client.post(f"{PREFIX}/upload", data={"store_id": "STORE-1"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_an_empty_planogram_field_is_treated_as_absent(
    client: AsyncClient, shelf_image_bytes: bytes
) -> None:
    response = await client.post(
        f"{PREFIX}/upload", files=image_part(shelf_image_bytes), data={"planogram": "   "}
    )
    assert response.status_code == 201


async def test_history_is_empty_before_any_upload(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/history")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


async def test_history_lists_previous_audits(
    client: AsyncClient, shelf_image_bytes: bytes, stockout_image_bytes: bytes
) -> None:
    await client.post(
        f"{PREFIX}/upload", files=image_part(shelf_image_bytes), data={"store_id": "STORE-1"}
    )
    await client.post(
        f"{PREFIX}/upload",
        files=image_part(stockout_image_bytes, "stockout.jpg"),
        data={"store_id": "STORE-2"},
    )

    body = (await client.get(f"{PREFIX}/history")).json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert {item["store_id"] for item in body["items"]} == {"STORE-1", "STORE-2"}
    assert "summary" in body["items"][0]


async def test_history_filters_by_compliance_score(
    client: AsyncClient, shelf_image_bytes: bytes, stockout_image_bytes: bytes, planogram: Planogram
) -> None:
    payload = {"planogram": planogram.model_dump_json()}
    await client.post(f"{PREFIX}/upload", files=image_part(shelf_image_bytes), data=payload)
    await client.post(
        f"{PREFIX}/upload", files=image_part(stockout_image_bytes, "s.jpg"), data=payload
    )

    failing = (await client.get(f"{PREFIX}/history", params={"max_compliance_score": 0.8})).json()
    assert failing["total"] == 1
    assert failing["items"][0]["compliance_score"] < 0.8

    passing = (await client.get(f"{PREFIX}/history", params={"min_compliance_score": 0.9})).json()
    assert passing["total"] == 1


async def test_history_filters_by_alert_flag(
    client: AsyncClient, stockout_image_bytes: bytes, planogram: Planogram
) -> None:
    await client.post(
        f"{PREFIX}/upload",
        files=image_part(stockout_image_bytes),
        data={"planogram": planogram.model_dump_json()},
    )

    alerted = (await client.get(f"{PREFIX}/history", params={"alert_triggered": True})).json()
    assert alerted["total"] == 1


async def test_history_paginates(client: AsyncClient, shelf_image_bytes: bytes) -> None:
    for index in range(3):
        await client.post(
            f"{PREFIX}/upload",
            files=image_part(shelf_image_bytes, f"shelf-{index}.jpg"),
            data={"store_id": f"STORE-{index}"},
        )

    page = (await client.get(f"{PREFIX}/history", params={"limit": 2, "offset": 1})).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["offset"] == 1


async def test_history_rejects_an_out_of_range_score(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/history", params={"min_compliance_score": 5})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_history_rejects_an_inverted_time_window(client: AsyncClient) -> None:
    response = await client.get(
        f"{PREFIX}/history",
        params={"since": "2026-09-10T00:00:00Z", "until": "2026-09-01T00:00:00Z"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_history_rejects_an_inverted_score_range(client: AsyncClient) -> None:
    response = await client.get(
        f"{PREFIX}/history", params={"min_compliance_score": 0.9, "max_compliance_score": 0.1}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_fetching_one_audit_by_id(client: AsyncClient, shelf_image_bytes: bytes) -> None:
    created = (await client.post(f"{PREFIX}/upload", files=image_part(shelf_image_bytes))).json()

    response = await client.get(f"{PREFIX}/{created['audit_id']}")

    assert response.status_code == 200
    assert response.json()["audit_id"] == created["audit_id"]
    assert response.json()["result"] == created["result"]


async def test_fetching_an_unknown_audit_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "audit_not_found"


async def test_analytics_of_an_empty_store(client: AsyncClient) -> None:
    body = (await client.get(f"{PREFIX}/analytics")).json()

    assert body["total_audits"] == 0
    assert body["mean_compliance"] == 0.0
    assert body["trend"] == []
    assert body["top_offenders"] == []


async def test_analytics_aggregates_uploaded_audits(
    client: AsyncClient, stockout_image_bytes: bytes, planogram: Planogram
) -> None:
    await client.post(
        f"{PREFIX}/upload",
        files=image_part(stockout_image_bytes),
        data={"planogram": planogram.model_dump_json(), "store_id": "STORE-1"},
    )

    body = (await client.get(f"{PREFIX}/analytics", params={"top_n": 3})).json()

    assert body["total_audits"] == 1
    assert body["alert_rate"] == 1.0
    assert body["worst_store_id"] == "STORE-1"
    assert len(body["trend"]) == 1
    assert len(body["top_offenders"]) <= 3
    assert body["top_offenders"][0]["dominant_type"]
