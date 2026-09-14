"""Shared retrieval pipeline used by both query() and retrieve().

Also owns vector_search(), the one place that talks to the managed vector
database. NOTE: the search endpoint shape below is an assumption (agreed as
a reasonable starting point, not yet verified against the live backend) —
isolated here so a real spec lands as a one-function change. Never reference
the underlying vector-store technology by name in this module, or anywhere
else in the package — see project brief.
"""

from __future__ import annotations

import time

import httpx
from openai import AsyncOpenAI

from ._embed import DEFAULT_EMBED_MODEL, embed
from ._http import raise_for_status
from ._rerank import DEFAULT_RERANK_MODEL, rerank
from .types import RankedDocument, RetrieveResult, Timing, Usage


async def vector_search(
    http: httpx.AsyncClient,
    collection: str,
    vector: list[float],
    top_k: int,
    filter: dict | None,
) -> list[RankedDocument]:
    body: dict = {"vector": vector, "top_k": top_k}
    if filter:
        body["filter"] = filter
    response = await http.post(f"/v1/collections/{collection}/search", json=body)
    await raise_for_status(response)
    results = response.json()["results"]
    return [
        RankedDocument(text=r["text"], score=r["score"], index=i, metadata=r.get("metadata", {}))
        for i, r in enumerate(results)
    ]


async def run_retrieval(
    *,
    http: httpx.AsyncClient,
    embed_client: AsyncOpenAI,
    query: str,
    collection: str,
    top_k: int,
    filter: dict | None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str | None = DEFAULT_RERANK_MODEL,
) -> RetrieveResult:
    """embed(query) -> vector_search(collection) -> optional rerank().

    Shared by query() and retrieve() (both clients) so the two methods
    can't drift out of sync with each other.
    """
    embed_result = await embed(embed_client, [query], embed_model)
    embed_vector = embed_result.vectors[0]

    retrieve_start = time.perf_counter()
    candidates = await vector_search(http, collection, embed_vector, top_k, filter)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

    usage = Usage(embed_tokens=embed_result.usage.tokens)
    timing = Timing(embed_ms=embed_result.timing.embed_ms, retrieve_ms=retrieve_ms)

    if rerank_model is None or not candidates:
        return RetrieveResult(sources=candidates, usage=usage, timing=timing)

    rerank_result = await rerank(http, query, [c.text for c in candidates], rerank_model)
    usage.rerank_docs = rerank_result.usage.rerank_docs
    timing.rerank_ms = rerank_result.timing.rerank_ms

    sources = [
        RankedDocument(text=r.text, score=r.score, index=r.index, metadata=candidates[r.index].metadata)
        for r in rerank_result.ranked
    ]
    return RetrieveResult(sources=sources, usage=usage, timing=timing)
