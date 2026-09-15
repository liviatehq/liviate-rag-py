"""Shared HTTP transport.

Liviate's own endpoints (ingest, vector search, rerank) go through a plain
httpx client. ``/v1/embeddings`` and chat completions go through the
``openai`` package's client pointed at the LiteLLM-fronted gateway, per the
brief's instruction to reuse an existing OpenAI-compatible client rather
than hand-rolling those request shapes.
"""

from __future__ import annotations

from contextlib import contextmanager

import httpx
import openai
from openai import AsyncOpenAI

from .exceptions import APIError, RateLimitError

DEFAULT_BASE_URL = "https://liviate.com"


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
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text[:500]
        raise APIError(
            f"Liviate API returned {response.status_code} for {response.request.url}: {body}",
            status_code=response.status_code,
        ) from exc


@contextmanager
def translate_openai_errors():
    """embed() and generate() go through the openai client instead of the plain httpx one
    above, so they raise openai's own exception hierarchy by default. Wraps calls made inside
    this block so those surface as this package's own LiviateError/RateLimitError too --
    otherwise `except LiviateError` around an SDK call silently misses everything that happens
    to route through the openai client."""
    try:
        yield
    except openai.RateLimitError as exc:
        retry_after = None
        header = getattr(exc.response, "headers", {}).get("Retry-After") if exc.response else None
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = None
        raise RateLimitError(str(exc), retry_after=retry_after) from exc
    except openai.APIStatusError as exc:
        raise APIError(str(exc), status_code=exc.status_code) from exc
    except openai.APIError as exc:
        raise APIError(str(exc)) from exc
