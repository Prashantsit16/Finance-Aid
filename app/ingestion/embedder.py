"""
embedder.py — Convert text chunks into vectors (embeddings).

THIS IS THE HEART OF RAG. Understand this well.

WHAT IS AN EMBEDDING?
---------------------
An embedding is a list of numbers (a vector) that represents the
MEANING of a piece of text in a high-dimensional space.

Example:
  "The company's revenue grew 12% YoY" → [0.21, -0.83, 0.44, ..., 0.12]
                                            (384 numbers for BGE-small)

Key property: texts with similar MEANING will have similar vectors.
Similar = the numbers point in roughly the same direction in space.
We measure this with cosine similarity.

COSINE SIMILARITY:
  - 1.0 = identical meaning
  - 0.0 = completely unrelated
  - -1.0 = opposite meaning (rare in practice)

When you search "what was EBITDA?", we embed that query into a vector,
then find all chunks whose vectors are closest to it. Those are your results.

WHY BGE-SMALL?
--------------
BAAI/bge-small-en-v1.5 is from Beijing Academy of AI.
- Small: 33M parameters, ~130MB download
- Fast: runs on CPU, ~2s for 100 chunks
- Good: ranks near top on MTEB benchmark (the standard embedding evaluation)
- Free: no API, runs locally

Alternatives you should know:
  - all-MiniLM-L6-v2: lighter but lower quality
  - bge-large-en-v1.5: better but 4x slower
  - text-embedding-3-small (OpenAI): great but paid API, data goes to OpenAI

For a financial/legal domain: BGE-small is the right tradeoff.

BATCH SIZE:
We embed chunks in batches of 32. Why not all at once?
- Memory: 500 chunks * 1800 chars would eat your RAM
- If it crashes mid-way, you lose everything. Batching lets you retry.
"""

from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np


# Load once, reuse. Loading a model is expensive (~2s). Encoding is fast.
# This is a module-level singleton — common pattern for ML models.
_model = None


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    """
    Lazy-load the embedding model.
    First call downloads + loads it. Subsequent calls reuse it.

    Why lazy loading?
    - Don't load the model if this module is imported but not used
    - App starts faster
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {model_name}")
        logger.info("First run will download the model (~130MB). This is a one-time download.")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded and ready.")
    return _model


def embed_chunks(
    chunks: list[dict],
    batch_size: int = 32,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[dict]:
    """
    Generate embeddings for all chunks. Returns chunks with embeddings added.

    BGE models need a specific prefix for query vs document embedding:
    - For DOCUMENTS being stored: add "Represent this sentence: " prefix
    - For QUERIES at search time: add "Represent this question for retrieval: "

    This asymmetric prompting is a BGE-specific trick that improves retrieval quality.
    Most other models don't need this.
    """
    model = get_embedding_model(model_name)

    # Prepare texts with BGE document prefix
    texts = [f"Represent this sentence: {chunk['text']}" for chunk in chunks]

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

    # encode() returns a numpy array of shape (num_chunks, embedding_dim)
    # normalize_embeddings=True makes cosine similarity = dot product (faster search)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # important for cosine similarity
        convert_to_numpy=True,
    )

    # Attach embeddings to chunk dicts
    enriched_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched_chunks.append({
            **chunk,
            "embedding": embedding.tolist()  # convert numpy array to list for JSON/ChromaDB compat
        })

    logger.info(
        f"Done. Embedding shape: ({len(chunks)}, {len(embeddings[0])}). "
        f"Each chunk is now a {len(embeddings[0])}-dimensional vector."
    )

    return enriched_chunks


def embed_query(
    query: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[float]:
    """
    Embed a single query at search time.

    Note the different prefix vs documents — this is the BGE asymmetric encoding.
    The model was trained to align these two different prefixes in vector space.
    """
    model = get_embedding_model(model_name)

    query_with_prefix = f"Represent this question for retrieval: {query}"
    embedding = model.encode(
        [query_with_prefix],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding[0].tolist()
