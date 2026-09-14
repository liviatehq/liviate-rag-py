"""Shared test fixtures: a local mock LiteLLM-compatible server so the
integration suite never depends on a live Liviate backend."""

from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer

from liviate_rag import AsyncRAGClient, RAGClient


@pytest.fixture
def mock_server():
    with HTTPServer() as server:
        yield server


@pytest.fixture
def base_url(mock_server: HTTPServer) -> str:
    return mock_server.url_for("/").rstrip("/")


@pytest.fixture
async def async_client(base_url: str):
    # Points the vector-store's token-exchange call at the SAME local mock server as everything
    # else -- the mock exchange response's own "qdrant_url" field then points data-plane calls
    # back at it too, same as the real exchange endpoint does. See _vectorstore.py.
    client = AsyncRAGClient(
        api_key="test-key", base_url=base_url,
        _exchange_url=f"{base_url}/api/tenancy/vectordb/exchange-token/",
    )
    yield client
    await client.close()


@pytest.fixture
def sync_client(base_url: str):
    client = RAGClient(
        api_key="test-key", base_url=base_url,
        _exchange_url=f"{base_url}/api/tenancy/vectordb/exchange-token/",
    )
    yield client
    client.close()
