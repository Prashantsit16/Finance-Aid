"""
bm25.py — Keyword-based retrieval using BM25.

WHAT IS BM25?
-------------
BM25 = Best Match 25. It's an improvement on TF-IDF from the 1990s.
It's still the gold standard for keyword search — even Google uses variants of it.

The name "25" is just because it was the 25th iteration of the algorithm
the researchers were working on. Nothing magical about the number.

HOW IT SCORES:
BM25 scores a document for a query based on two things:

1. TERM FREQUENCY (TF): How often does the query word appear in the chunk?
   BUT — BM25 caps this. If "revenue" appears 10 times vs 2 times,
   the 10-time chunk isn't 5x more relevant. Returns diminish.
   This is the key improvement over basic TF-IDF.

2. INVERSE DOCUMENT FREQUENCY (IDF): How rare is this word across ALL chunks?
   "the" appears everywhere → low IDF, doesn't help ranking
   "EBITDA" appears in few chunks → high IDF, very helpful for ranking

   Formula (simplified):
   score = sum over query terms of: IDF(term) * TF(term, chunk) / (TF + k1 * (1 - b + b * chunk_len / avg_len))

   k1 and b are tuning parameters (default k1=1.5, b=0.75). You don't need to
   tune these for most cases.

WHERE BM25 WINS (dense search loses):
  - Exact section references: "Section 4.2", "Clause 7(b)"
  - Specific numbers: "₹420 crore", "12.3%"
  - Rare proper nouns: company names, CIN numbers, regulatory codes
  - Acronyms: "SEBI", "EBITDA", "CRR"

WHERE BM25 LOSES (dense search wins):
  - Paraphrased questions: query says "profit" but doc says "earnings"
  - Conceptual questions: "what are the risks?" (no single keyword)
  - Synonyms: "terminated" vs "dismissed" in legal docs

IMPORTANT LIMITATION:
BM25 doesn't persist. There's no BM25 "database" on disk.
We rebuild the BM25 index from all chunks in ChromaDB every time the app
starts. For 500-1000 chunks this takes < 1 second. Totally fine.

At scale (millions of docs) you'd use Elasticsearch which has BM25 built in.
For this project, rank_bm25 library is perfect.
"""

from rank_bm25 import BM25Okapi
from loguru import logger
import re


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.

    BM25 works on TOKENS (individual words). How you tokenize matters.

    We:
      - Lowercase (so "Revenue" and "revenue" match)
      - Split on whitespace and punctuation
      - Keep tokens with 2+ characters (removes stray punctuation)

    We do NOT use a fancy NLP tokenizer here because:
      1. Speed — we're tokenizing thousands of chunks
      2. BM25 doesn't need lemmatization or stemming for financial text
      3. "EBITDA" should stay as "ebitda" not get split further
    """
    text = text.lower()
    tokens = re.split(r"[\s\.,;:!?()\[\]{}\"\'\-\/\\]+", text)
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """
    A BM25 index built from a list of chunks.

    This is a class (not just functions) because we build the index once
    and then query it multiple times. Rebuilding on every query would be wasteful.

    Usage:
        index = BM25Index(chunks)       # build once
        results = index.search(query)   # query many times
    """

    def __init__(self, chunks: list[dict]):
        """
        Build the BM25 index from chunks.

        chunks: list of dicts with at least 'text' and 'chunk_id'

        BM25Okapi is the standard variant of BM25.
        (Okapi refers to the Okapi IR system at City University London,
        where BM25 was developed in the early 90s.)
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        self.chunks = chunks

        # Tokenize every chunk's text
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

        # Build the BM25 index — this is fast even for 1000 chunks
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Score all chunks against the query and return top-K.

        Returns same format as dense_search for easy merging in fusion.py.

        Note: BM25 scores are NOT cosine similarity. They're raw BM25 scores.
        You can't directly compare BM25 scores with dense scores.
        That's why RRF (Reciprocal Rank Fusion) uses RANKS not scores.
        """
        query_tokens = tokenize(query)

        if not query_tokens:
            logger.warning("BM25 query tokenized to empty list. Returning no results.")
            return []

        # get_scores returns a score for EVERY chunk in the corpus
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-K chunks sorted by score descending
        # argsort gives ascending order, so we reverse with [::-1]
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = scores[idx]

            # Skip chunks with zero score — they have none of the query terms
            if score == 0:
                break

            results.append({
                **self.chunks[idx],
                "score": round(float(score), 4),
                "rank": rank,
                "retriever": "bm25",
            })

        return results


def build_bm25_index_from_chroma(
    persist_dir: str = "data/processed/chroma_db",
    collection_name: str = "finance_aid",
) -> BM25Index:
    """
    Load all chunks from ChromaDB and build a BM25 index over them.

    We call this once at app startup and reuse the index.
    This is the standard pattern — BM25 lives in memory, ChromaDB on disk.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)

    # Fetch all stored chunks (no limit — we want ALL for BM25 corpus)
    total = collection.count()
    if total == 0:
        raise ValueError("ChromaDB collection is empty. Run ingestion first.")

    logger.info(f"Loading {total} chunks from ChromaDB to build BM25 index...")

    # ChromaDB's get() can fetch all at once (fine for thousands of chunks)
    results = collection.get(include=["documents", "metadatas"])

    chunks = []
    for chunk_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        chunks.append({
            "chunk_id": chunk_id,
            "text": doc,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
        })

    logger.info(f"Loaded {len(chunks)} chunks. Building BM25 index...")
    return BM25Index(chunks)
