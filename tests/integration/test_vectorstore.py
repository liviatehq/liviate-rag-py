"""Focused tests for VectorStoreClient's token-exchange behavior: concurrent
cache-miss dedup and defensive parsing of the exchange response.
"""

from __future__ import annotations

import asyncio

import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

from liviate_rag._vectorstore import VectorStoreClient
from liviate_rag.exceptions import LiviateError


async def test_concurrent_searches_share_one_token_exchange(mock_server: HTTPServer, base_url: str):
    exchange_calls = []

    def handle_exchange(request: Request) -> Response:
        exchange_calls.append(1)
        return Response(
            '{"token": "t", "access": "r", "expires_at": 9999999999, '
            '"collection_name": "ns__docs", "qdrant_url": "%s"}' % base_url,
            content_type="application/json",
        )

    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_handler(
        handle_exchange
    )
    mock_server.expect_request("/collections/ns__docs/points/query", method="POST").respond_with_json(
        {"result": {"points": []}}
    )

    client = VectorStoreClient(
        "test-key", timeout=10.0, exchange_url=f"{base_url}/api/tenancy/vectordb/exchange-token/"
    )
    try:
        await asyncio.gather(*[client.search("docs", [0.1, 0.2], top_k=3, filter=None) for _ in range(5)])
    finally:
        await client.close()

    assert len(exchange_calls) == 1, "concurrent cache misses for the same key should share one exchange call"


async def test_missing_exchange_field_raises_without_leaking_field_name(mock_server: HTTPServer, base_url: str):
    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_json(
        {"token": "t", "access": "r", "expires_at": 9999999999, "collection_name": "ns__docs"}
        # "qdrant_url" deliberately missing
    )

    client = VectorStoreClient(
        "test-key", timeout=10.0, exchange_url=f"{base_url}/api/tenancy/vectordb/exchange-token/"
    )
    try:
        with pytest.raises(LiviateError) as exc_info:
            await client.search("docs", [0.1, 0.2], top_k=3, filter=None)
        assert "qdrant" not in str(exc_info.value).lower()
        assert "data-plane URL" in str(exc_info.value)
    finally:
        await client.close()


async def test_exchange_request_sends_collection_field(mock_server: HTTPServer, base_url: str):
    # The exchange endpoint's current field name is "collection" (the older "name" is kept as a
    # permanent server-side alias, but this client should send the current name -- see
    # _vectorstore module docstring).
    captured_form = {}

    def handle_exchange(request: Request) -> Response:
        captured_form.update(request.form)
        return Response(
            '{"token": "t", "access": "r", "expires_at": 9999999999, '
            '"collection_name": "ns__docs", "qdrant_url": "%s"}' % base_url,
            content_type="application/json",
        )

    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_handler(
        handle_exchange
    )
    mock_server.expect_request("/collections/ns__docs/points/query", method="POST").respond_with_json(
        {"result": {"points": []}}
    )

    client = VectorStoreClient(
        "test-key", timeout=10.0, exchange_url=f"{base_url}/api/tenancy/vectordb/exchange-token/"
    )
    try:
        await client.search("docs", [0.1, 0.2], top_k=3, filter=None)
    finally:
        await client.close()

    assert captured_form.get("collection") == "docs"
    assert "name" not in captured_form


async def test_exchange_resolves_physical_collection_when_present(mock_server: HTTPServer, base_url: str):
    # A response carrying only the new "physical_collection" field (no "collection_name" at all)
    # must still resolve the real collection name correctly.
    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_json(
        {
            "token": "t", "access": "r", "expires_at": 9999999999,
            "physical_collection": "ns__docs", "qdrant_url": base_url,
        }
    )
    mock_server.expect_request("/collections/ns__docs/points/query", method="POST").respond_with_json(
        {"result": {"points": []}}
    )

    client = VectorStoreClient(
        "test-key", timeout=10.0, exchange_url=f"{base_url}/api/tenancy/vectordb/exchange-token/"
    )
    try:
        # Would 404 against a URL built from the wrong collection name if resolution failed.
        await client.search("docs", [0.1, 0.2], top_k=3, filter=None)
    finally:
        await client.close()
