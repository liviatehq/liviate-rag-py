"""Integration test for rerank() against a mock server.

Uses the assumed Cohere-shaped response (see liviate_rag/_rerank.py's
module docstring) -- update this fixture alongside _parse_rerank_response
once the real LiteLLM fork rerank endpoint shape is confirmed.
"""

from __future__ import annotations

from pytest_httpserver import HTTPServer


async def test_rerank_returns_best_first(async_client, mock_server: HTTPServer):
    mock_server.expect_request("/v1/rerank", method="POST").respond_with_json(
        {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
    )

    result = await async_client.rerank("query", ["doc a", "doc b"])

    assert [d.text for d in result.ranked] == ["doc b", "doc a"]
    assert result.ranked[0].score == 0.9
    assert result.usage.rerank_docs == 2
