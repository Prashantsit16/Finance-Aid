"""
fusion.py — Combine dense and BM25 results into one ranked list.

THE PROBLEM THIS SOLVES:
Dense search returns 20 chunks with cosine similarity scores (0 to 1).
BM25 returns 20 chunks with BM25 scores (could be 0.5, 3.2, 7.8 — no fixed range).

You can't just merge these two lists by score because the scales are completely
different. A BM25 score of 3.2 vs a cosine similarity of 0.87 — which is "better"?
There's no way to know.

SOLUTION: RECIPROCAL RANK FUSION (RRF)
---------------------------------------
Instead of using scores, RRF uses RANKS.

Rank 1 is best, rank 2 is second best, etc.
Doesn't matter that one list uses cosine similarity and the other uses BM25 —
both produce a ranked list, and ranks ARE comparable.

THE FORMULA:
For each chunk, RRF score = sum of: 1 / (k + rank_in_each_list)

k is a constant (default 60) that dampens the effect of very high ranks.
It prevents a chunk ranked #1 in one list from dominating everything.

EXAMPLE:
Chunk A: rank 1 in dense, rank 3 in BM25
  RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

Chunk B: rank 2 in dense, NOT in BM25 (only appeared in dense)
  RRF = 1/(60+2) + 0 = 0.01613

Chunk C: rank 10 in dense, rank 1 in BM25
  RRF = 1/(60+10) + 1/(60+1) = 0.01429 + 0.01639 = 0.03068

Final ranking: A > C > B

WHY RRF IS GOOD:
  - No normalization needed (works with any scoring functions)
  - Naturally boosts chunks that appear in BOTH lists (consensus)
  - The k=60 constant is from the original RRF paper and works well in practice
  - Simple to implement and understand

Papers: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
Learning Methods" — Cormack, Clarke, Buettcher (2009)
"""

from loguru import logger


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion.

    Returns top_k chunks sorted by RRF score (highest first).
    Each returned chunk has an 'rrf_score' and a 'sources' field
    showing which retriever(s) found it.

    The 'sources' field is useful for debugging — if a chunk appears
    in both retrievers, it's a strong signal that it's actually relevant.
    """

    # Build a map of chunk_id -> chunk data + accumulated RRF score
    fused: dict[str, dict] = {}

    # Process dense results
    for result in dense_results:
        cid = result["chunk_id"]
        rrf_contribution = 1.0 / (k + result["rank"])

        if cid not in fused:
            fused[cid] = {
                **result,
                "rrf_score": 0.0,
                "found_in": [],
                # Remove retriever-specific fields that don't apply to fused result
            }

        fused[cid]["rrf_score"] += rrf_contribution
        fused[cid]["found_in"].append("dense")

    # Process BM25 results
    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_contribution = 1.0 / (k + result["rank"])

        if cid not in fused:
            fused[cid] = {
                **result,
                "rrf_score": 0.0,
                "found_in": [],
            }

        fused[cid]["rrf_score"] += rrf_contribution
        fused[cid]["found_in"].append("bm25")

    # Sort by RRF score descending
    sorted_chunks = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Count how many chunks appeared in both — good debug signal
    both = sum(1 for c in sorted_chunks if len(c["found_in"]) == 2)
    logger.info(
        f"RRF fusion: {len(dense_results)} dense + {len(bm25_results)} BM25 "
        f"-> {len(sorted_chunks)} unique chunks ({both} appeared in both)"
    )

    # Return top_k after fusion (these go to the reranker next)
    top_chunks = sorted_chunks[:top_k]

    # Renumber ranks for the next stage
    for i, chunk in enumerate(top_chunks, start=1):
        chunk["fused_rank"] = i

    return top_chunks
