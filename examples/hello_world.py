"""The smallest possible liviate-rag program -- just proves your install
and API key work. No collection, no model string, nothing else to set up.

Run with:

    LIVIATE_API_KEY=sk-... python examples/hello_world.py

For the full ingest -> retrieve -> query pipeline, see examples/quickstart.py.
"""

from liviate_rag import RAGClient

with RAGClient() as client:
    result = client.embed(["Hello, world!"])
    print(f"Embedded 1 text into a {len(result.vectors[0])}-dimensional vector.")
    print(f"Tokens used: {result.usage.tokens}")
