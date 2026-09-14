"""Integration test for embed() against a mock OpenAI-compatible server."""

from __future__ import annotations

from pytest_httpserver import HTTPServer


async def test_embed_returns_vectors_and_usage(async_client, mock_server: HTTPServer):
    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "liviate/embedding",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )

    result = await async_client.embed(["hello world"])

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert result.usage.tokens == 4
    assert result.timing.embed_ms >= 0


def test_embed_sync_client(sync_client, mock_server: HTTPServer):
    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
            "model": "liviate/embedding",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    )

    result = sync_client.embed(["hi"])

    assert result.vectors == [[0.5]]
    assert result.usage.tokens == 2
