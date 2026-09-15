"""embed() implementation — thin wrapper over an OpenAI-compatible client.

Reuses the `openai` package's async client against the LiteLLM-fronted
embedding backend rather than hand-rolling the /v1/embeddings request shape.
"""

from __future__ import annotations

import time

from openai import AsyncOpenAI

from ._http import translate_openai_errors
from .types import EmbedResult, EmbedUsage, Timing

DEFAULT_EMBED_MODEL = "liviate/embedding"


async def embed(client: AsyncOpenAI, texts: list[str], model: str) -> EmbedResult:
    start = time.perf_counter()
    with translate_openai_errors():
        response = await client.embeddings.create(model=model, input=texts)
    elapsed_ms = (time.perf_counter() - start) * 1000

    vectors = [item.embedding for item in response.data]
    tokens = response.usage.total_tokens if response.usage else 0

    return EmbedResult(
        vectors=vectors,
        usage=EmbedUsage(tokens=tokens),
        timing=Timing(embed_ms=elapsed_ms),
    )
