"""
build_rag_index.py
--------------------
Builds a local, persistent ChromaDB vector index of historical Apple
support (customer_text -> apple_reply_text) pairs, for grounding the
agent's replies in real precedent.

CRITICAL: golden set rows are excluded from this index (via
golden_excluded_ids.csv) so the agent can never retrieve the answer to
a question it's being evaluated on.

Uses a local, free, CPU-only embedding model (no API key, no cost).

Run: python build_rag_index.py
Requires: pip install chromadb sentence-transformers polars
Output: a persistent index in ./chroma_db (a local folder, not a file)
"""

import polars as pl
import chromadb
from chromadb.utils import embedding_functions

PAIRS_FILE = "apple_support_pairs.csv"
EXCLUDED_IDS_FILE = "golden_excluded_ids.csv"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "apple_support_history"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # free, local, CPU-friendly
BATCH_SIZE = 500

print("Step 1: Loading historical pairs and golden-set exclusion list...")
pairs_df = pl.read_csv(PAIRS_FILE)
excluded_df = pl.read_csv(EXCLUDED_IDS_FILE)

before = pairs_df.height
pairs_df = pairs_df.join(excluded_df, on="customer_tweet_id", how="anti")
after = pairs_df.height
print(f"   Excluded {before - after} golden-set rows from the RAG index (leakage prevention).")
print(f"   Indexing {after} remaining historical pairs.")

print(f"\nStep 2: Loading local embedding model ({EMBEDDING_MODEL})...")
print("   (First run downloads the model, ~130MB, then it's cached locally.)")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

print("\nStep 3: Initializing persistent ChromaDB collection...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Fresh start each time this script runs, so re-running doesn't duplicate entries
try:
    client.delete_collection(COLLECTION_NAME)
    print("   (Cleared existing collection to rebuild fresh.)")
except Exception:
    pass

collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

print(f"\nStep 4: Indexing {after} pairs in batches of {BATCH_SIZE}...")
rows = pairs_df.to_dicts()

for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i : i + BATCH_SIZE]
    collection.add(
        ids=[str(r["customer_tweet_id"]) for r in batch],
        documents=[r["customer_text"] for r in batch],  # what we embed and search against
        metadatas=[
            {
                "apple_reply_text": r["apple_reply_text"],
                "is_dm_redirect": bool(r["is_dm_redirect"]),
                "thread_depth": int(r["thread_depth"]),
            }
            for r in batch
        ],
    )
    print(f"   Indexed {min(i + BATCH_SIZE, len(rows))}/{len(rows)}...", end="\r")

print(f"\n\nDone. Index built at '{CHROMA_DIR}' with {collection.count()} entries.")
print("Run query_rag.py to test retrieval quality before wiring this into the full pipeline.")
