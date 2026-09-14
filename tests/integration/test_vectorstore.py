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
