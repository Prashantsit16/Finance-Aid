"""
retrieval/pipeline.py — The full Day 2 retrieval pipeline in one function.

This wires dense.py + bm25.py + fusion.py + reranker.py together.

Flow:
  Query
    |
    +----------------+
    |                |
  Dense            BM25
  (semantic)     (keyword)
    |                |
    +-------+--------+
            |
      RRF Fusion (20 candidates)
            |
      Cross-Encoder Reranking
            |
        Top-5 chunks

After this, Day 3 will take these Top-5 chunks and pass them to the LLM
with a prompt to generate a grounded answer.
"""

from loguru import logger
from app.retrieval.dense import dense_search
from app.retrieval.bm25 import build_bm25_index_from_chroma, BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.core.config import settings


# BM25 index is built once at module import time (or lazily on first call)
# and reused for all subsequent queries.
_bm25_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Lazy-load BM25 index from ChromaDB."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = build_bm25_index_from_chroma()
    return _bm25_index


def retrieve(
    query: str,
    top_k_final: int = None,
    source_filter: str = None,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query: natural language question
        top_k_final: how many chunks to return after reranking (default 5)
        source_filter: restrict to a specific PDF filename

    Returns:
        List of top_k_final chunks, each with:
          - chunk_id, text, source, page
          - rerank_score (from cross-encoder)
          - found_in (["dense"] or ["bm25"] or ["dense", "bm25"])
    """
    top_k_final = top_k_final or settings.TOP_K_RERANK

    logger.info(f"Starting hybrid retrieval for: '{query}'")

    # --- Stage 1: Dense retrieval ---
    logger.info("Stage 1: Dense search...")
    dense_results = dense_search(
        query=query,
        top_k=settings.TOP_K_DENSE,
        source_filter=source_filter,
    )
    logger.info(f"  Dense: {len(dense_results)} candidates")

    # --- Stage 2: BM25 retrieval ---
    logger.info("Stage 2: BM25 search...")
    bm25_index = get_bm25_index()
    bm25_results = bm25_index.search(query=query, top_k=settings.TOP_K_BM25)

    # Apply source filter to BM25 results manually (BM25 has no built-in filter)
    if source_filter:
        bm25_results = [r for r in bm25_results if r["source"] == source_filter]

    logger.info(f"  BM25: {len(bm25_results)} candidates")

    # --- Stage 3: Reciprocal Rank Fusion ---
    logger.info("Stage 3: Fusing results with RRF...")
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=settings.TOP_K_DENSE,  # pass more candidates to reranker than we need
    )
    logger.info(f"  Fused: {len(fused)} unique candidates")

    # --- Stage 4: Cross-encoder reranking ---
    logger.info(f"Stage 4: Reranking top {len(fused)} candidates...")
    final_chunks = rerank(
        query=query,
        candidates=fused,
        top_k=top_k_final,
    )

    logger.info(f"Retrieval complete. Returning {len(final_chunks)} chunks.")
    return final_chunks
