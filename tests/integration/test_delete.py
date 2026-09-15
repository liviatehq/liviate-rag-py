"""client.delete() against a mock vector-store data plane."""

from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer


def _mock_exchange(mock_server: HTTPServer) -> None:
    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_json(
        {
            "token": "scoped-jwt",
            "access": "rw",
            "expires_at": 9999999999,
            "collection_name": "testtenant__docs",
            "qdrant_url": mock_server.url_for("/").rstrip("/"),
        }
    )


async def test_delete_by_ids(async_client, mock_server: HTTPServer):
    _mock_exchange(mock_server)
    mock_server.expect_request("/collections/testtenant__docs/points/delete", method="POST").respond_with_json(
        {"result": {"status": "acknowledged"}}
    )

    await async_client.delete("docs", ids=["a", "b"])


async def test_delete_by_filter(async_client, mock_server: HTTPServer):
    _mock_exchange(mock_server)
    mock_server.expect_request("/collections/testtenant__docs/points/delete", method="POST").respond_with_json(
        {"result": {"status": "acknowledged"}}
    )

    await async_client.delete("docs", filter={"must": [{"key": "metadata.tag", "match": {"value": "x"}}]})


async def test_delete_requires_exactly_one_of_ids_or_filter(async_client):
    with pytest.raises(ValueError):
        await async_client.delete("docs")

    with pytest.raises(ValueError):
        await async_client.delete("docs", ids=["a"], filter={"must": []})


def test_delete_sync_client(sync_client, mock_server: HTTPServer):
    _mock_exchange(mock_server)
    mock_server.expect_request("/collections/testtenant__docs/points/delete", method="POST").respond_with_json(
        {"result": {"status": "acknowledged"}}
    )

    sync_client.delete("docs", ids=["a"])
