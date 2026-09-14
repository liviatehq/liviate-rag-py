"""Direct access to the managed vector database's own data-plane API.

This SDK holds exactly one long-lived secret: the caller's ``api_key``. It is never presented to
the vector store directly -- the vector store validates a different, short-lived credential that
this module obtains by exchanging ``api_key`` with Liviate's own token-exchange endpoint, then
caches in memory (per collection, per access level) until shortly before it expires. This is the
only place in the whole client where that exchange happens; every other module (embed, rerank,
generate) sends ``api_key`` straight through as a normal bearer token.

The exchange response also carries the vector store's real, tenant-namespaced collection name
(distinct from the logical ``collection`` the caller passes -- e.g. "docs-kb" maps to something
like "<tenant>__docs-kb" on the actual store) and its real data-plane URL. Both are used as-is
rather than guessed/hardcoded here, so this module never encodes an assumption about either
naming scheme or deployment topology.

Never reference the underlying vector-store technology by name in this module, or anywhere else
in the package -- see project brief.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ._http import raise_for_status

EXCHANGE_URL = "https://console.liviate.com/api/tenancy/vectordb/exchange-token/"

# Refresh this many seconds before the exchanged credential's real expiry, so an in-flight
# request can never race a credential that expires mid-call.
_REFRESH_MARGIN_S = 30.0


class _CachedGrant:
    __slots__ = ("token", "data_plane_url", "real_collection_name", "_expires_at_monotonic")

    def __init__(self, token: str, data_plane_url: str, real_collection_name: str, ttl_seconds: float):
        self.token = token
        self.data_plane_url = data_plane_url.rstrip("/")
        self.real_collection_name = real_collection_name
        self._expires_at_monotonic = time.monotonic() + ttl_seconds

    @property
    def usable(self) -> bool:
        return time.monotonic() < (self._expires_at_monotonic - _REFRESH_MARGIN_S)


class VectorStoreClient:
    """One instance per RAGClient/AsyncRAGClient -- owns the exchanged-grant cache and the httpx
    clients used for the exchange call and direct data-plane calls. A read ('r') and a write
    ('rw') grant for the same collection are cached separately, since they're genuinely different
    credentials."""

    def __init__(self, api_key: str, timeout: float, *, exchange_url: str = EXCHANGE_URL):
        self._api_key = api_key
        self._exchange_url = exchange_url
        self._exchange_http = httpx.AsyncClient(timeout=timeout)
        self._timeout = timeout
        self._data_http_by_base: dict[str, httpx.AsyncClient] = {}
        self._cache: dict[tuple[str, str], _CachedGrant] = {}

    async def close(self) -> None:
        await self._exchange_http.aclose()
        for client in self._data_http_by_base.values():
            await client.aclose()

    def _data_http_for(self, base_url: str) -> httpx.AsyncClient:
        client = self._data_http_by_base.get(base_url)
        if client is None:
            client = httpx.AsyncClient(base_url=base_url, timeout=self._timeout)
            self._data_http_by_base[base_url] = client
        return client

    async def _grant_for(self, collection: str, access: str, vector_size: int | None = None) -> _CachedGrant:
        key = (collection, access)
        cached = self._cache.get(key)
        if cached is not None and cached.usable:
            return cached

        data: dict[str, Any] = {"name": collection, "access": access}
        if vector_size is not None:
            # Lets the exchange endpoint auto-create the collection on first write -- an ingest
            # caller already knows its embedding dimension (it just computed the vectors), so
            # there's no need to require a separate "create the collection first" step through
            # the console UI before a customer's very first ingest() can succeed.
            data["vector_size"] = str(vector_size)
        response = await self._exchange_http.post(
            self._exchange_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            data=data,
        )
        await raise_for_status(response)
        body = response.json()
        # expires_at is a Unix timestamp (server clock), not a duration -- convert to a
        # monotonic-relative TTL once here so _CachedGrant never has to compare against
        # wall-clock time (which can jump; monotonic can't).
        ttl_seconds = max(0.0, float(body["expires_at"]) - time.time())
        grant = _CachedGrant(
            token=body["token"],
            data_plane_url=body["qdrant_url"],
            real_collection_name=body["collection_name"],
            ttl_seconds=ttl_seconds,
        )
        self._cache[key] = grant
        return grant

    async def search(
        self, collection: str, vector: list[float], top_k: int, filter: dict | None,
    ) -> list[dict[str, Any]]:
        grant = await self._grant_for(collection, "r")
        body: dict[str, Any] = {"query": vector, "limit": top_k, "with_payload": True}
        if filter:
            body["filter"] = filter
        response = await self._data_http_for(grant.data_plane_url).post(
            f"/collections/{grant.real_collection_name}/points/query",
            headers={"Authorization": f"Bearer {grant.token}"},
            json=body,
        )
        await raise_for_status(response)
        return response.json()["result"]["points"]

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """points: ``[{"id": ..., "vector": [...], "payload": {"text": ..., "metadata": {...}}}]``
        -- ``id`` must be a UUID string or unsigned integer (the data-plane's own requirement)."""
        vector_size = len(points[0]["vector"]) if points else None
        grant = await self._grant_for(collection, "rw", vector_size=vector_size)
        response = await self._data_http_for(grant.data_plane_url).put(
            f"/collections/{grant.real_collection_name}/points",
            headers={"Authorization": f"Bearer {grant.token}"},
            json={"points": points},
        )
        await raise_for_status(response)
