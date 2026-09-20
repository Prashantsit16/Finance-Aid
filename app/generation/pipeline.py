"""
generation/pipeline.py — Full end-to-end RAG pipeline.

This is the function that ties ALL three days together:
  - Day 1: ChromaDB storage (already done at ingestion time)
  - Day 2: Hybrid retrieval (dense + BM25 + RRF + cross-encoder)
  - Day 3: LLM generation with Pydantic structured output

Call answer_query(query) and you get back a fully structured RAGResponse
with the answer, confidence, and citations. That's the complete RAG system.

Day 4 wraps this in a FastAPI endpoint.
Day 5 evaluates the quality of these responses with RAGAS.
"""

from loguru import logger
from app.retrieval.pipeline import retrieve
from app.generation.llm import generate_answer
from app.models.schemas import RAGResponse


def answer_query(
    query: str,
    top_k: int = 5,
    source_filter: str = None,
    model: str = None,
) -> RAGResponse:
    """
    The complete RAG pipeline: query in → structured answer out.

    Args:
        query:         the user's natural language question
        top_k:         how many chunks to pass to the LLM (default 5)
        source_filter: restrict retrieval to a specific PDF filename
        model:         which Ollama model to use (default from config)

    Returns:
        RAGResponse with answer, citations, confidence, answer_found
    """
    logger.info(f"Starting RAG pipeline for: '{query}'")

    # ── Stage 1+2: Retrieval (Day 2's pipeline) ────────────────────────────
    logger.info("Retrieving relevant chunks...")
    chunks = retrieve(
        query=query,
        top_k_final=top_k,
        source_filter=source_filter,
    )
    logger.info(f"Retrieved {len(chunks)} chunks for generation.")

    # ── Stage 3: Generation (Day 3) ────────────────────────────────────────
    logger.info("Generating answer with LLM...")
    response = generate_answer(
        query=query,
        chunks=chunks,
        model=model,
    )

    logger.info(
        f"Answer generated. found={response.answer_found}, "
        f"confidence={response.confidence}, "
        f"citations={len(response.citations)}"
    )

    return response
