"""
store.py — Save and query chunks + embeddings in ChromaDB.

WHAT IS CHROMADB?
-----------------
ChromaDB is a vector database. A vector database does one thing really well:
given a query vector, find the N most similar vectors stored in the DB.

Under the hood it uses an algorithm called HNSW (Hierarchical Navigable
Small World graphs) — a graph-based approximate nearest neighbor search.
You don't need to know the internals, but you should know:

  - It's APPROXIMATE — not guaranteed to find the absolute closest vector,
    but finds something very close, very fast. (ANN = Approximate Nearest Neighbor)
  - It's persistent — ChromaDB writes to disk, so your embeddings survive restarts
  - Collections — ChromaDB organizes data into "collections" (like tables in SQL)

WHAT DOES CHROMADB STORE PER CHUNK?
  - id: the chunk_id string (must be unique)
  - embedding: the vector (list of 384 floats for BGE-small)
  - document: the raw text (so you can return it in results)
  - metadata: dict of { source, page } — for citations

WHY NOT JUST USE A REGULAR DATABASE?
SQL/MongoDB stores exact data. If you search for "revenue growth",
it looks for those exact words. It can't find "sales increased" because
the words are different even though the meaning is similar.

Vector databases compare meanings, not exact strings. That's the whole point.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from pathlib import Path


def get_collection(
    persist_dir: str = "data/processed/chroma_db",
    collection_name: str = "finance_aid",
):
    """
    Get or create a ChromaDB collection.

    PersistentClient means data is saved to disk. If you restart the app,
    your embeddings are still there — you don't re-embed every time.

    This is important: embedding 500 chunks takes ~30 seconds on CPU.
    You don't want to do that on every app restart.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    # get_or_create_collection: if it exists, return it; if not, create it.
    # cosine distance is the right metric for normalized embeddings.
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def store_chunks(
    chunks: list[dict],
    persist_dir: str = "data/processed/chroma_db",
    collection_name: str = "finance_aid",
) -> None:
    """
    Store embedded chunks into ChromaDB.

    ChromaDB's add() takes parallel lists:
      - ids: list of unique string IDs
      - embeddings: list of vectors
      - documents: list of raw text strings
      - metadatas: list of dicts (source, page)

    We add in batches of 100 to avoid memory issues with large document sets.
    """
    collection = get_collection(persist_dir, collection_name)

    # Filter out chunks that don't have embeddings yet
    embedded = [c for c in chunks if "embedding" in c]
    if len(embedded) < len(chunks):
        logger.warning(f"{len(chunks) - len(embedded)} chunks missing embeddings, skipping them.")

    # Check for existing IDs to avoid duplicates
    existing_ids = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_chunks = [c for c in embedded if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already exist in the collection. Nothing new to add.")
        return

    logger.info(f"Storing {len(new_chunks)} new chunks into ChromaDB (skipping {len(embedded) - len(new_chunks)} duplicates)...")

    batch_size = 100
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )
        logger.debug(f"Stored batch {i // batch_size + 1} ({len(batch)} chunks)")

    logger.info(f"Done. Collection now has {collection.count()} total chunks.")


def query_collection(
    query_embedding: list[float],
    top_k: int = 20,
    persist_dir: str = "data/processed/chroma_db",
    collection_name: str = "finance_aid",
    source_filter: str = None,
) -> list[dict]:
    """
    Find the top-K most similar chunks to a query embedding.

    Returns a list of dicts with: text, source, page, chunk_id, score

    The score is the cosine DISTANCE (0 = identical, 1 = opposite).
    We convert it to similarity: similarity = 1 - distance.

    source_filter: optionally restrict search to a specific PDF.
    Useful when the user uploads multiple documents and asks about one.
    """
    collection = get_collection(persist_dir, collection_name)

    total = collection.count()
    if total == 0:
        logger.warning(
            "ChromaDB collection is empty. No documents have been ingested yet. "
            "Run: python run_day1.py  (with a real PDF in data/raw/)"
        )
        return []

    where_clause = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),  # can't request more than what's stored
        include=["documents", "metadatas", "distances"],
        where=where_clause,
    )

    # ChromaDB returns nested lists (one per query), so we unpack [0]
    chunks = []
    for doc, meta, dist, chunk_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        chunks.append({
            "chunk_id": chunk_id,
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - dist, 4),  # convert distance to similarity
        })

    return chunks
