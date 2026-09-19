"""
run_day1.py — Test the Day 1 ingestion pipeline.

Run this with:
    python run_day1.py

Or to just search (after ingestion is done):
    python run_day1.py --search-only --query "What was the revenue?"

Before running:
    Drop any PDF into:  data/raw/
    (annual report, legal contract, textbook, anything)
"""

import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")

RAW_DIR = Path("data/raw")


def find_pdf() -> Path:
    """
    Find the first PDF in data/raw/.
    If none found, print a clear error and exit.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = list(RAW_DIR.glob("*.pdf"))

    if not pdfs:
        print("""
============================================================
  NO PDF FOUND in data/raw/
============================================================
  Drop any PDF into:  data/raw/
  Then re-run:        python run_day1.py

  Suggested free sources:
    Infosys AR:  https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-23.pdf
    Or any PDF from your laptop works too.
============================================================
""")
        sys.exit(1)

    pdf = pdfs[0]
    logger.info(f"Found PDF: {pdf.name}")
    if len(pdfs) > 1:
        logger.info(f"  (Found {len(pdfs)} PDFs, using first: {pdf.name}. Others: {[p.name for p in pdfs[1:]]})")
    return pdf


def run_ingestion():
    from app.ingestion.pipeline import ingest_pdf

    pdf_path = find_pdf()

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING INGESTION PIPELINE")
    logger.info("=" * 60 + "\n")

    summary = ingest_pdf(str(pdf_path))

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
