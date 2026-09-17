"""
pipeline.py — Wire all Day 1 components together into one callable function.

This is the entry point for document ingestion.
You call ingest_pdf() and the whole pipeline runs.

The pipeline:
  PDF file
    |
    v  extractor.py
  Raw text per page [{page, text, source}]
    |
    v  cleaner.py
  Cleaned text per page (headers/footers removed, whitespace normalized)
    |
    v  chunker.py
  Chunks with metadata [{chunk_id, text, source, page}]
    |
    v  embedder.py
  Chunks + embeddings [{..., embedding: [384 floats]}]
    |
    v  store.py
  Saved to ChromaDB (persistent on disk)

After this runs, the document is queryable.
"""

from pathlib import Path
from loguru import logger

from app.ingestion.extractor import extract_pdf
from app.ingestion.cleaner import clean_document
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_chunks
from app.ingestion.store import store_chunks, query_collection
from app.ingestion.embedder import embed_query


def ingest_pdf(
    pdf_path: str,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    persist_dir: str = "data/processed/chroma_db",
    collection_name: str = "finance_aid",
) -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Returns a summary dict with stats about what was processed.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Starting ingestion: {pdf_path.name}")
    logger.info("=" * 50)

    # Step 1: Extract
    logger.info("Step 1/4: Extracting text from PDF...")
    pages = extract_pdf(str(pdf_path))
    logger.info(f"  -> {len(pages)} pages extracted")

    # Step 2: Clean
    logger.info("Step 2/4: Cleaning extracted text...")
    clean_pages = clean_document(pages)
    logger.info(f"  -> {len(clean_pages)} pages after cleaning")

    # Step 3: Chunk
    logger.info("Step 3/4: Chunking into retrievable pieces...")
    chunks = chunk_pages(clean_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"  -> {len(chunks)} chunks created")

    # Step 4: Embed + Store
    logger.info("Step 4/4: Generating embeddings and storing in ChromaDB...")
    embedded_chunks = embed_chunks(chunks)
    store_chunks(embedded_chunks, persist_dir=persist_dir, collection_name=collection_name)

    summary = {
        "file": pdf_path.name,
        "pages_extracted": len(pages),
        "pages_after_cleaning": len(clean_pages),
        "chunks_created": len(chunks),
        "chunks_embedded": len(embedded_chunks),
    }

    logger.info("=" * 50)
    logger.info(f"Ingestion complete: {summary}")
    return summary


def search(
    query: str,
    top_k: int = 5,
    persist_dir: str = "data/processed/chroma_db",
    collection_name: str = "finance_aid",
    source_filter: str = None,
) -> list[dict]:
    """
    Search the vector store with a natural language query.
    This is the simplest possible retrieval — pure dense search.

    Day 2 will replace this with hybrid (dense + BM25) + reranking.
    But for Day 1, this is enough to validate the pipeline works.
    """
    logger.info(f"Searching for: '{query}'")

    query_vec = embed_query(query)
    results = query_collection(
        query_embedding=query_vec,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        source_filter=source_filter,
    )

    logger.info(f"Found {len(results)} results")
    return results


def ingest_directory(
    directory: str,
    **kwargs,
) -> list[dict]:
    """
    Ingest all PDFs in a directory.
    Useful for batch processing.
    """
    directory = Path(directory)
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs to ingest: {[f.name for f in pdf_files]}")

    summaries = []
    for pdf_file in pdf_files:
        try:
            summary = ingest_pdf(str(pdf_file), **kwargs)
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_file.name}: {e}")

    return summaries
