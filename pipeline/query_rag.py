"""
query_rag.py
-------------
Quick sanity check on retrieval quality before wiring the RAG index
into the full classify -> retrieve -> guardrails -> output pipeline.

Run: python query_rag.py
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "apple_support_history"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

print(f"Collection loaded: {collection.count()} entries.\n")

# A handful of test queries spanning different intents -- edit these to
# match real examples from your own golden set for a more realistic check.
test_queries = [
    "My battery is draining so fast since I updated to the newest iOS",
    "I can't log into my Apple ID, it says my password is wrong",
    "My phone won't connect to WiFi anymore",
    "I was charged twice for the same app, I want a refund",
]

for query in test_queries:
    print(f"{'=' * 70}")
    print(f"QUERY: {query}")
    print(f"{'=' * 70}")
    results = collection.query(query_texts=[query], n_results=3)

    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1
    ):
        print(f"\n  [{rank}] similarity_distance={dist:.3f}  dm_redirect={meta['is_dm_redirect']}")
        print(f"      Matched historical tweet: {doc[:120]}")
        print(f"      Apple's reply: {meta['apple_reply_text'][:150]}")
    print()

print("\nReview above: do the retrieved historical tweets actually resemble the query?")
print("If matches look irrelevant, consider a different embedding model or better")
print("text preprocessing before wiring this into the full pipeline.")
