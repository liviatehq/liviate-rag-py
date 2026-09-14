"""generate() — the final step of query().

Generation routes through customer-selected models on the separate
Inference product rather than a Liviate-bundled default (see
RAGClient.query()'s required ``model`` kwarg) — it calls chat completions
through the same OpenAI-compatible gateway used for embeddings.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .types import RankedDocument, Timing, Usage

_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. If the "
    "context doesn't contain the answer, say so."
)


def _build_messages(query: str, sources: list[RankedDocument]) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] {s.text}" for i, s in enumerate(sources))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


async def generate(
    client: AsyncOpenAI, query: str, sources: list[RankedDocument], model: str,
) -> tuple[str, Usage, Timing]:
    start = time.perf_counter()
    response = await client.chat.completions.create(model=model, messages=_build_messages(query, sources))
    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = response.choices[0].message.content or ""
    tokens = response.usage.completion_tokens if response.usage else 0
    return answer, Usage(generation_tokens=tokens), Timing(generate_ms=elapsed_ms)


async def generate_stream(
    client: AsyncOpenAI, query: str, sources: list[RankedDocument], model: str,
) -> AsyncIterator[str]:
    stream = await client.chat.completions.create(
        model=model, messages=_build_messages(query, sources), stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
