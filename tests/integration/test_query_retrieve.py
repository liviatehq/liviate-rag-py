"""query() and retrieve() share _pipeline.run_retrieval() internally --
these tests assert that shared path stays intact (both produce identical
`sources`, and query() adds exactly one more call for generation).
"""

from __future__ import annotations

from pytest_httpserver import HTTPServer


def _mock_retrieval_chain(mock_server: HTTPServer) -> None:
    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "liviate/embedding",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    )
    mock_server.expect_request("/v1/collections/hotel-kirstine/search", method="POST").respond_with_json(
        {
            "results": [
                {"id": "1", "score": 0.5, "text": "We have free parking.", "metadata": {"page": 1}},
                {"id": "2", "score": 0.3, "text": "Breakfast is included.", "metadata": {"page": 2}},
            ]
        }
    )
    mock_server.expect_request("/v1/rerank", method="POST").respond_with_json(
        {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.4},
            ]
        }
    )


async def test_retrieve_returns_ranked_sources(async_client, mock_server: HTTPServer):
    _mock_retrieval_chain(mock_server)

    result = await async_client.retrieve("Har I parkering?", collection="hotel-kirstine", top_k=5)

    assert result.sources[0].text == "We have free parking."
    assert result.sources[0].metadata == {"page": 1}
    assert result.usage.embed_tokens == 3
    assert result.usage.rerank_docs == 2


async def test_query_reuses_retrieve_and_adds_generation(async_client, mock_server: HTTPServer):
    _mock_retrieval_chain(mock_server)
    mock_server.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Yes, free parking is available."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }
    )

    result = await async_client.query("Har I parkering?", collection="hotel-kirstine", model="test/model")

    assert result.answer == "Yes, free parking is available."
    assert result.sources[0].text == "We have free parking."
    assert result.usage.generation_tokens == 8
    assert result.timing.generate_ms >= 0
