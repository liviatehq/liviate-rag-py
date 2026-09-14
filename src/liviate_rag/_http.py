"""Shared HTTP transport.

Liviate's own endpoints (ingest, vector search, rerank) go through a plain
httpx client. ``/v1/embeddings`` and chat completions go through the
``openai`` package's client pointed at the LiteLLM-fronted gateway, per the
brief's instruction to reuse an existing OpenAI-compatible client rather
than hand-rolling those request shapes.
"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from .exceptions import RateLimitError

DEFAULT_BASE_URL = "https://api.liviate.com"


def build_async_httpx_client(api_key: str, base_url: str, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def build_async_openai_client(api_key: str, base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=f"{base_url.rstrip('/')}/v1")


async def raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RateLimitError(
            f"Rate limited by Liviate API ({response.request.url}).",
            retry_after=float(retry_after) if retry_after else None,
        )
    response.raise_for_status()
