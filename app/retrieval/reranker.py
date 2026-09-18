"""
reranker.py — Cross-encoder reranking of fused retrieval candidates.

This is the LAST step in retrieval and often the most impactful one.

THE PROBLEM WITH EVERYTHING WE BUILT SO FAR:
Dense search asks: "is chunk X similar to query Y?"
BM25 asks: "does chunk X contain words from query Y?"

Neither is the right question.

The right question is: "does chunk X ACTUALLY ANSWER query Y?"

That's what the cross-encoder does.

BI-ENCODER vs CROSS-ENCODER:
------------------------------
BI-ENCODER (what we used in Day 1 for dense search):
  - Encodes query and document SEPARATELY into vectors
  - Similarity = dot product of the two vectors
  - Fast: you pre-compute document vectors and store them
  - Good for first-stage retrieval over thousands of chunks

CROSS-ENCODER (what we use now):
  - Sees query AND document TOGETHER: "[CLS] query [SEP] document [SEP]"
  - Full attention between every query token and every document token
  - Produces a single relevance SCORE (0 to 1)
  - Slow: can't pre-compute, must run at query time
  - Much more accurate than bi-encoder

WHY WE USE BOTH:
  - Cross-encoder is too slow to run over 10,000 chunks
  - Bi-encoder + BM25 narrows it down to top 20 candidates fast
  - Cross-encoder runs over just those 20 → picks the actual best 5
  - This two-stage pattern is how production systems work

THE MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset
  - MS MARCO is a massive dataset of real web queries + relevant passages
  - MiniLM-L-6 = 6-layer MiniLM, small and fast, good quality
  - Produces a logit score (not a probability, but higher = more relevant)

WHAT "RERANKING" MEANS:
  Input: 20 candidates from fusion (in fusion order)
  Output: 5 best candidates re-ordered by cross-encoder relevance score

The cross-encoder often completely changes the order. A chunk ranked #8
by dense+BM25 might jump to #1 after reranking because it actually answers
the question directly. This happens more than you'd think.
"""

from sentence_transformers import CrossEncoder
from loguru import logger

# Load once, reuse (same lazy singleton pattern as embedder.py)
_cross_encoder = None

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder() -> CrossEncoder:
    """
    Lazy-load the cross-encoder model.
    Downloads ~120MB on first use. Then cached locally.
    """
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates using the cross-encoder.

    Steps:
    1. Build (query, chunk_text) pairs for the cross-encoder
    2. Score all pairs — cross-encoder gives each pair a relevance logit
    3. Sort by score descending
    4. Return top_k

    Why logits instead of probabilities?
    The model returns raw logits (unbounded numbers, can be negative).
    We don't need them to sum to 1 or be between 0-1.
    We just need them to be orderable — higher = more relevant.
    """
    if not candidates:
        logger.warning("Reranker received empty candidate list.")
        return []

    model = get_cross_encoder()

    # Build input pairs: [(query, chunk_text), ...]
    pairs = [(query, chunk["text"]) for chunk in candidates]

    logger.info(f"Cross-encoder scoring {len(pairs)} candidate pairs...")

    # predict() returns an array of scores, one per pair
    scores = model.predict(pairs, show_progress_bar=False)

    # Attach scores to candidates
    scored = []
    for chunk, score in zip(candidates, scores):
        scored.append({
            **chunk,
            "rerank_score": round(float(score), 4),
        })

    # Sort by rerank score descending (highest = most relevant)
    reranked = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

    top = reranked[:top_k]

    # Log the rank changes — this is interesting to see
    logger.info(f"Reranking complete. Top {top_k} chunks selected.")
    for i, chunk in enumerate(top, 1):
        old_rank = chunk.get("fused_rank", "?")
        logger.debug(
            f"  Final #{i} | Was fused #{old_rank} | "
            f"Score: {chunk['rerank_score']} | "
            f"Source: {chunk['source']} p.{chunk['page']}"
        )

    return top
