"""liviate-rag quickstart.

Run with:

    LIVIATE_API_KEY=sk-... python examples/quickstart.py

The key needs:
  - the `liviate/embedding` and `liviate/rerank` models on its allow-list (or an empty
    models list, which means "all models")
  - Vector Database scope enabled, with the "quickstart" collection allowed (or an empty
    collections list, which means "all collections")
  - a chat model on its allow-list if you want the generate() step -- this example uses
    deepinfra/deepseek-ai/DeepSeek-V4-Flash

Create a key with this shape from the console: Inference -> API Keys -> Generate key.
The key's own error message lists exactly which models it's allowed to use if this one
isn't available to you (PermissionDeniedError: "key not allowed to access model").
"""

from __future__ import annotations

import os

from liviate_rag import RAGClient

COLLECTION = "quickstart"
GENERATION_MODEL = "deepinfra/deepseek-ai/DeepSeek-V4-Flash"


def main() -> None:
    # api_key= can also just be omitted -- RAGClient reads LIVIATE_API_KEY from the
    # environment automatically if you don't pass one explicitly.
    with RAGClient(api_key=os.environ.get("LIVIATE_API_KEY")) as client:
        print(f"Ingesting into collection {COLLECTION!r}...")
        ingest_result = client.ingest(
            source="Liviate offers a managed vector database, embedding, and reranking as separately billed products.",
            collection=COLLECTION,
            source_type="text",
        )
        print(f"  -> {ingest_result.chunks_created} chunk(s) created ({ingest_result.source_type})")

        # A collection is created automatically the first time you ingest() into a name
        # that doesn't exist yet -- no separate "create the collection first" step needed.

        print("\nretrieve() -- ranked context only, no generation:")
        retrieval = client.retrieve("What does Liviate offer?", collection=COLLECTION, top_k=3)
        for i, source in enumerate(retrieval.sources, 1):
            print(f"  {i}. (score={source.score:.3f}) {source.text}")

        print(f"\nquery() -- retrieval + generation via {GENERATION_MODEL}:")
        result = client.query("What does Liviate offer?", collection=COLLECTION, model=GENERATION_MODEL)
        print(f"  answer: {result.answer}")
        print(f"  usage:  {result.usage}")
        print(f"  timing: {result.timing}")


if __name__ == "__main__":
    main()
