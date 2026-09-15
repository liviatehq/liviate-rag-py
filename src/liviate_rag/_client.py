"""RAGClient — synchronous facade over AsyncRAGClient.

AsyncRAGClient owns all business logic; this class bridges each call
through a single persistent background event loop rather than a fresh
asyncio.run() per call. httpx.AsyncClient's (and AsyncOpenAI's) connection
pool is loop-bound: constructing the client once and then driving it from a
new event loop on every call is unsupported and can raise "Event loop is
closed" / "attached to a different loop" errors on the second call.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from typing import Any

from ._async_client import AsyncRAGClient
from ._http import DEFAULT_BASE_URL
from .types import EmbedResult, IngestJob, IngestResult, QueryResult, RerankResult, RetrieveResult


class _BackgroundLoop:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


class RAGClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        exchange_url: str | None = None,
    ):
        self._loop = _BackgroundLoop()
        self._async: AsyncRAGClient = self._loop.run(
            _make_async_client(api_key, base_url, timeout, exchange_url)
        )

    def close(self) -> None:
        self._loop.run(self._async.close())
        self._loop.close()

    def __enter__(self) -> "RAGClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- ingest ------------------------------------------------------

    def ingest(
        self,
        source,
        collection: str,
        *,
        source_type: str = "auto",
        wait: bool = True,
        metadata: dict | None = None,
        timeout: float | None = None,
        embed_model: str = "liviate/embedding",
    ) -> IngestResult | IngestJob:
        outcome = self._loop.run(
            self._async.ingest(
                source, collection, source_type=source_type, wait=wait, metadata=metadata,
                timeout=timeout, embed_model=embed_model,
            )
        )
        return outcome if wait else self._wrap_job(outcome)

    def ingest_site(
        self,
        source: str,
        collection: str,
        *,
        max_pages: int = 50,
        wait: bool = False,
        metadata: dict | None = None,
        timeout: float | None = None,
    ) -> IngestResult | IngestJob:
        outcome = self._loop.run(
            self._async.ingest_site(source, collection, max_pages=max_pages, wait=wait, metadata=metadata, timeout=timeout)
        )
        return outcome if wait else self._wrap_job(outcome)

    def _wrap_job(self, async_job) -> IngestJob:
        return IngestJob(
            job_ids=async_job.job_ids,
            collection=async_job.collection,
            _poll_fn=lambda: self._loop.run(async_job.result()),
        )

    # -- embed / rerank ------------------------------------------------

    def embed(self, texts: list[str], model: str = "liviate/embedding") -> EmbedResult:
        return self._loop.run(self._async.embed(texts, model))

    def rerank(self, query: str, documents: list[str], model: str = "liviate/rerank") -> RerankResult:
        return self._loop.run(self._async.rerank(query, documents, model))

    # -- delete ------------------------------------------------

    def delete(self, collection: str, *, ids: list[str] | None = None, filter: dict | None = None) -> None:
        self._loop.run(self._async.delete(collection, ids=ids, filter=filter))

    # -- retrieve / query ------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection: str,
        *,
        top_k: int = 5,
        filter: dict | None = None,
        embed_model: str = "liviate/embedding",
        rerank_model: str | None = "liviate/rerank",
    ) -> RetrieveResult:
        """``embed_model`` must match whatever model the collection was
        ingested with (see ``ingest(..., embed_model=...)``)."""
        return self._loop.run(
            self._async.retrieve(
                query, collection, top_k=top_k, filter=filter, embed_model=embed_model, rerank_model=rerank_model,
            )
        )

    def query(
        self,
        query: str,
        collection: str,
        *,
        model: str,
        top_k: int = 5,
        filter: dict | None = None,
        embed_model: str = "liviate/embedding",
        rerank_model: str | None = "liviate/rerank",
        stream: bool = False,
    ) -> QueryResult | Iterator[str]:
        if stream:
            return self._sync_stream(query, collection, model, top_k, filter, embed_model, rerank_model)
        return self._loop.run(
            self._async.query(
                query, collection, model=model, top_k=top_k, filter=filter,
                embed_model=embed_model, rerank_model=rerank_model, stream=False,
            )
        )

    def _sync_stream(
        self, query: str, collection: str, model: str, top_k: int, filter: dict | None,
        embed_model: str, rerank_model: str | None,
    ) -> Iterator[str]:
        async_gen = self._loop.run(
            self._async.query(
                query, collection, model=model, top_k=top_k, filter=filter,
                embed_model=embed_model, rerank_model=rerank_model, stream=True,
            )
        )
        while True:
            try:
                yield self._loop.run(async_gen.__anext__())
            except StopAsyncIteration:
                break


async def _make_async_client(
    api_key: str | None, base_url: str, timeout: float, exchange_url: str | None,
) -> AsyncRAGClient:
    return AsyncRAGClient(api_key, base_url=base_url, timeout=timeout, exchange_url=exchange_url)
