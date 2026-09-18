"""
dense.py — Dense (semantic) retrieval from ChromaDB.

This is basically a clean wrapper around what we built in Day 1.
We separate it into its own file because Day 2 runs MULTIPLE retrievers
in parallel and then merges them. Having each retriever isolated makes
the code clean and easy to swap later.

WHAT DENSE RETRIEVAL DOES:
  - Embeds the query into a 384-dim vector
  - Asks ChromaDB: "which stored vectors are closest to this?"
  - Returns top-K chunks sorted by cosine similarity

WHERE IT WINS:
  - Conceptual / paraphrased questions
  - "What are the main risks?" — even if the document says "key challenges"
  - Understanding intent, not just keywords

WHERE IT FAILS:
  - Exact terminology: "Section 4.2" — won't match unless those exact words
    are near the embedding centroid
  - Specific numbers, codes, IDs: "CIN L85110KA1981PLC013115"
  - Rare proper nouns, acronyms the model hasn't seen much

That's why Day 2 exists — BM25 covers what dense misses.
"""

from app.ingestion.embedder import embed_query
from app.ingestion.store import query_collection
from app.core.config import settings


def dense_search(
    query: str,
    top_k: int = None,
    source_filter: str = None,
) -> list[dict]:
    """
    Run dense retrieval for a query.

    Returns list of dicts:
    [
        {
            "chunk_id": "annual_report_12_03",
            "text": "...",
            "source": "annual_report.pdf",
            "page": 12,
            "score": 0.87,       # cosine similarity (higher = better)
            "rank": 1,           # rank in THIS retriever's results
            "retriever": "dense" # tag so fusion knows where this came from
        },
        ...
    ]

    The "rank" field is critical for Day 2's Reciprocal Rank Fusion.
    RRF cares about rank order, not the actual score value.
    """
    top_k = top_k or settings.TOP_K_DENSE

    query_vec = embed_query(query)

    raw_results = query_collection(
        query_embedding=query_vec,
        top_k=top_k,
        source_filter=source_filter,
    )

    # Add rank and retriever tag to each result
    results = []
    for rank, chunk in enumerate(raw_results, start=1):
        results.append({
            **chunk,
            "rank": rank,
            "retriever": "dense",
        })

    return results
