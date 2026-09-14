# liviate-rag

Client SDK for Liviate's RAG stack — ingest, embed, retrieve, rerank, and
(optionally) generate — in a few lines of Python.

```bash
pip install liviate-rag
```

## Quickstart

```python
from liviate_rag import RAGClient

client = RAGClient(api_key="...")  # or set LIVIATE_API_KEY

client.ingest("handbook.pdf", collection="hotel-kirstine")

result = client.query("Har I parkering?", collection="hotel-kirstine", model="anthropic/claude-sonnet-5")
print(result.answer)
print(result.sources)
print(result.usage)
print(result.timing)
```

Want ranked context only, and to call your own LLM?

```python
context = client.retrieve("Har I parkering?", collection="hotel-kirstine", top_k=5)
```

An `AsyncRAGClient` with the same method surface is available for async
codebases:

```python
from liviate_rag import AsyncRAGClient

async with AsyncRAGClient() as client:
    result = await client.query("...", collection="...", model="...")
```

## Ingest

`ingest()` handles a single file, URL, text string, or file-like stream —
detecting which one automatically. See the source docstrings in
`_ingest/detect.py` and `_ingest/handlers.py` for the exact detection order
and supported file types (`.pdf`, `.docx`, `.md`, `.txt`, `.csv`, `.json`,
`.html`).

Whole-site crawling is a separate, explicit call:

```python
client.ingest_site("https://example.com", collection="x", max_pages=200)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit tests/integration   # no network/credentials required
LIVIATE_E2E=1 pytest tests/e2e         # real environment, run manually
```

## Status

This is a v1 scaffold. A few request/response shapes are built against
reasonable assumptions pending confirmation from the live backend — see the
"NOTE"/"ASSUMPTION" comments in `_rerank.py`, `_pipeline.py`, and
`_ingest/handlers.py` before relying on them in production.
