"""Shared retrieval pipeline used by both query() and retrieve().

Also owns vector_search(), the one place that talks to the managed vector
database — via VectorStoreClient (see _vectorstore.py), never the plain
`http` client used for /v1/ingest* (that one carries `api_key` straight
through; the vector store needs a different, short-lived credential that
_vectorstore.py obtains transparently). Never reference the underlying
vector-store technology by name in this module, or anywhere else in the
package — see project brief.
"""

from __future__ import annotations

import time

import httpx
from openai import AsyncOpenAI

from ._embed import DEFAULT_EMBED_MODEL, embed
from ._rerank import DEFAULT_RERANK_MODEL, rerank
from ._vectorstore import VectorStoreClient
from .exceptions import LiviateError
from .types import RankedDocument, RetrieveResult, Timing, Usage


async def vector_search(
    vectorstore: VectorStoreClient,
    collection: str,
    vector: list[float],
    top_k: int,
    filter: dict | None,
) -> list[RankedDocument]:
    points = await vectorstore.search(collection, vector, top_k, filter)
    results = []
    for i, p in enumerate(points):
        payload = p.get("payload") or {}
        if "text" not in payload:
            raise LiviateError(
                f"Point {p.get('id')!r} in collection {collection!r} has no 'text' in its "
                "payload -- was it written outside of ingest()? retrieve()/query() can't rank "
                "or return a source with no text content."
            )
        results.append(
            RankedDocument(text=payload["text"], score=p["score"], index=i, metadata=payload.get("metadata", {}))
        )
    return results


async def run_retrieval(
    *,
    vectorstore: VectorStoreClient,
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
    candidates = await vector_search(vectorstore, collection, embed_vector, top_k, filter)
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
