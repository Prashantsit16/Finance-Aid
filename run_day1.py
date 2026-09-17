"""
run_day1.py — The script you actually run to test Day 1.

This script:
1. Downloads a sample financial PDF (Infosys Annual Report — public domain)
2. Runs the full ingestion pipeline on it
3. Lets you search it with natural language queries
4. Prints results so you can see the RAG pipeline working

Run this with:
    python run_day1.py

Or to just search (after ingestion is done):
    python run_day1.py --search-only --query "What was the revenue?"
"""

import sys
import urllib.request
from pathlib import Path
from loguru import logger

# Configure logger to be readable (not JSON, not overly verbose)
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")


SAMPLE_PDF_URL = "https://www.africau.edu/images/default/sample.pdf"
SAMPLE_PDF_PATH = Path("data/raw/sample_report.pdf")


def download_sample_pdf():
    """Download a sample PDF if none exists in data/raw/."""
    if SAMPLE_PDF_PATH.exists():
        logger.info(f"Sample PDF already exists: {SAMPLE_PDF_PATH}")
        return

    SAMPLE_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading sample PDF from {SAMPLE_PDF_URL}...")
    urllib.request.urlretrieve(SAMPLE_PDF_URL, SAMPLE_PDF_PATH)
    logger.info(f"Downloaded to {SAMPLE_PDF_PATH}")


def run_ingestion():
    from app.ingestion.pipeline import ingest_pdf

    download_sample_pdf()

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING INGESTION PIPELINE")
    logger.info("=" * 60 + "\n")

    summary = ingest_pdf(str(SAMPLE_PDF_PATH))

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key:<25} {value}")
    print("=" * 60 + "\n")

    return summary


def run_search(queries: list[str] = None):
    from app.ingestion.pipeline import search

    if queries is None:
        queries = [
            "What is the main topic of this document?",
            "What are the key sections?",
        ]

    print("\n" + "=" * 60)
    print("RUNNING SEARCH QUERIES")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 40)

        results = search(query, top_k=3)

        if not results:
            print("  No results found. Did ingestion run first?")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n  Result #{i} | Score: {result['score']} | Page: {result['page']} | Source: {result['source']}")
            print(f"  Chunk ID: {result['chunk_id']}")

            # Print first 300 chars of the chunk text
            preview = result["text"][:300].replace("\n", " ")
            print(f"  Text: {preview}...")

    print("\n" + "=" * 60)
    print("Day 1 complete! Your PDF is now a searchable vector database.")
    print("=" * 60)


def print_what_you_learned():
    print("""
============================================================
WHAT YOU JUST BUILT (Day 1 concepts)
============================================================

1. PDF EXTRACTION
   Your PDF was opened with PyMuPDF (fitz). It reads the
   internal PDF structure and reconstructs text from the
   coordinates where characters are drawn on the page.

2. CLEANING
   Headers, footers, page numbers and broken hyphenation
   were removed. Clean text = better embeddings.

3. CHUNKING WITH OVERLAP
   The text was split into ~450-token chunks with 50-token
   overlap. Why overlap? So no sentence gets cut in half
   between two chunks. Context stays intact.

4. EMBEDDINGS
   Each chunk was converted to a 384-dimensional vector
   by BAAI/bge-small-en-v1.5. Similar text = similar vectors.
   This is what makes semantic search possible.

5. CHROMADB
   All vectors + text + metadata stored on disk. When you
   query, ChromaDB uses HNSW to find the nearest vectors
   in ~milliseconds. You can restart the app and the data
   is still there.

6. SEARCH
   Your query was embedded with the same model (different
   prefix for queries vs documents — BGE asymmetric encoding).
   ChromaDB found the chunks with highest cosine similarity.

Interview questions you can now answer:
  - What is an embedding?
  - Why chunk documents?
  - What is cosine similarity?
  - What does ChromaDB store?
  - What is approximate nearest neighbor search?
============================================================
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finance-Aid Day 1 Demo")
    parser.add_argument("--search-only", action="store_true", help="Skip ingestion, just search")
    parser.add_argument("--query", type=str, nargs="+", help="Custom query to search")
    args = parser.parse_args()

    if not args.search_only:
        run_ingestion()

    queries = args.query if args.query else None
    run_search(queries)

    print_what_you_learned()
