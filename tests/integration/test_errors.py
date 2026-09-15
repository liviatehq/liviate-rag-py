"""Regression tests: every backend failure should surface as this package's
own exception hierarchy, not a raw httpx/openai exception -- see
liviate_rag/_http.py's raise_for_status() and translate_openai_errors().
"""

from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer

from liviate_rag.exceptions import APIError, RateLimitError


async def test_non_429_error_wraps_as_api_error(async_client, mock_server: HTTPServer):
    mock_server.expect_request("/v1/rerank", method="POST").respond_with_json(
        {"error": "collection not found"}, status=404,
    )

    with pytest.raises(APIError) as exc_info:
        await async_client.rerank("query", ["doc a"])
    assert exc_info.value.status_code == 404


async def test_embed_429_wraps_as_rate_limit_error(async_client, mock_server: HTTPServer):
    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {"error": {"message": "rate limited"}}, status=429, headers={"Retry-After": "3"},
    )

    with pytest.raises(RateLimitError) as exc_info:
        await async_client.embed(["hello"])
    assert exc_info.value.retry_after == 3.0
