"""
run_day2.py — Test Day 2 hybrid retrieval pipeline.

Make sure you ran run_day1.py first (data needs to be in ChromaDB).

Run with:
    python run_day2.py
    python run_day2.py --query "What is EBITDA?"

This script runs the SAME query through three retrievers and shows you
the difference in results — so you can see WHY hybrid + reranking matters.
"""

import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")


def print_results(title: str, results: list[dict], show_score_key: str = "score"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if not results:
        print("  No results.")
        return
    for i, r in enumerate(results, 1):
        score = r.get(show_score_key, r.get("rrf_score", r.get("score", "?")))
        found_in = r.get("found_in", [r.get("retriever", "?")])
        print(f"\n  #{i} | Score: {score} | Page: {r['page']} | Source: {r['source']}")
        print(f"       Found by: {found_in}")
        preview = r["text"][:250].replace("\n", " ")
        print(f"       Text: {preview}...")


def run_comparison(query: str):
    print(f"\n{'#'*60}")
    print(f"  QUERY: \"{query}\"")
    print(f"{'#'*60}")

    # ---- Dense only ----
    from app.retrieval.dense import dense_search
    dense_results = dense_search(query, top_k=5)
    print_results("DENSE ONLY (semantic / vector search)", dense_results, "score")

    # ---- BM25 only ----
    from app.retrieval.bm25 import build_bm25_index_from_chroma
    bm25_index = build_bm25_index_from_chroma()
    bm25_results = bm25_index.search(query, top_k=5)
    print_results("BM25 ONLY (keyword search)", bm25_results, "score")

    # ---- Hybrid + Rerank ----
    from app.retrieval.pipeline import retrieve
    hybrid_results = retrieve(query, top_k_final=5)
    print_results("HYBRID + RERANK (final pipeline)", hybrid_results, "rerank_score")


def print_what_you_learned():
    print("""
============================================================
WHAT YOU JUST BUILT (Day 2 concepts)
============================================================

1. DENSE RETRIEVAL (recap)
   Embeds query -> finds closest chunk vectors in ChromaDB.
   Great for meaning, bad for exact keywords.

2. BM25
   Keyword search. Scores chunks by: how often do query words
   appear (TF) weighted by how rare those words are (IDF).
   Built fresh from ChromaDB data on startup (~1 sec for 1000 chunks).

3. RECIPROCAL RANK FUSION (RRF)
   Merges two ranked lists using 1/(k + rank) formula.
   Uses ranks, not scores — so incompatible scoring systems are fine.
   Boosts chunks that appear in BOTH lists (consensus signal).

4. CROSS-ENCODER RERANKING
   Takes top-20 fused candidates, scores each (query, chunk) pair
   together with full attention. Much more accurate than bi-encoder.
   Too slow for 10k chunks, perfect for 20 candidates.

   The key insight: bi-encoder embeds query and doc SEPARATELY.
   Cross-encoder sees them TOGETHER — it can see how the query words
   relate to specific words in the chunk. That's why it's better.

Interview questions you can now answer:
  - What is the difference between dense and sparse retrieval?
  - What is BM25 and how does it score documents?
  - What is Reciprocal Rank Fusion and why not just add scores?
  - What is a cross-encoder vs bi-encoder?
  - Why do we do two-stage retrieval (retrieve then rerank)?
============================================================
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    queries = [args.query] if args.query else [
        "What is the main topic of this document?",
        "What are the key points mentioned?",
    ]

    for q in queries:
        run_comparison(q)

    print_what_you_learned()
