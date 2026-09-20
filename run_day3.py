"""
run_day3.py — Test the full RAG pipeline end-to-end.

This is the first time you see:
  Question → Retrieval → LLM → Structured Answer with Citations

Prerequisites:
  1. Ollama installed and running  →  ollama serve
  2. Model pulled                  →  ollama pull mistral
  3. Day 1 ingestion done          →  python run_day1.py (with a real PDF)

Run with:
    python run_day3.py
    python run_day3.py --query "What was the total revenue?"
    python run_day3.py --no-llm   (test retrieval only, skips LLM call)
"""

import sys
import json
from loguru import logger

logger.remove()
logger.add(sys.stderr,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
           level="INFO")


def check_prerequisites():
    """Check ChromaDB has data. Ollama check happens when we call it."""
    import chromadb
    client = chromadb.PersistentClient(path="data/processed/chroma_db")
    try:
        col = client.get_collection("finance_aid")
        count = col.count()
        if count == 0:
            raise ValueError("empty")
        print(f"\n  ChromaDB: {count} chunks ready.\n")
    except Exception:
        print("""
============================================================
  Run ingestion first: python run_day1.py
============================================================""")
        sys.exit(1)


def print_response(response):
    """Pretty-print a RAGResponse."""
    print("\n" + "=" * 65)
    print("  RAG RESPONSE")
    print("=" * 65)
    print(f"\n  Query:       {response.query}")
    print(f"  Found:       {response.answer_found}")
    print(f"  Confidence:  {response.confidence}")
    print(f"  Model:       {response.model_used}")
    print(f"  Chunks used: {response.retrieval_chunks_used}")

    print(f"\n  Answer:\n")
    # wrap the answer at 60 chars for readability
    words = response.answer.split()
    line, lines = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > 60:
            lines.append("  " + " ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))

    if response.citations:
        print(f"\n  Citations ({len(response.citations)}):")
        for i, c in enumerate(response.citations, 1):
            print(f"\n    [{i}] {c.source}  |  Page {c.page}")
            excerpt = c.excerpt[:150].replace("\n", " ")
            print(f"        \"{excerpt}...\"")
    else:
        print("\n  No citations returned.")

    print("\n" + "=" * 65)


def run_retrieval_only(query: str):
    """Run Day 2 retrieval only — no LLM. Useful for testing without Ollama."""
    from app.retrieval.pipeline import retrieve

    print(f"\n  [Retrieval only] Query: '{query}'")
    chunks = retrieve(query, top_k_final=5)
    print(f"\n  Top {len(chunks)} chunks retrieved:\n")

    for i, c in enumerate(chunks, 1):
        print(f"  #{i} | Page {c['page']} | {c['source']} | Rerank: {c.get('rerank_score', '?')}")
        preview = c["text"][:200].replace("\n", " ")
        print(f"       {preview}...\n")


def run_full_rag(query: str):
    """Run the complete Day 1 + Day 2 + Day 3 pipeline."""
    from app.generation.pipeline import answer_query

    print(f"\n  Running full RAG pipeline for: '{query}'")
    response = answer_query(query)
    print_response(response)
    return response


def print_what_you_learned():
    print("""
============================================================
WHAT YOU JUST BUILT (Day 3 concepts)
============================================================

1. PYDANTIC STRUCTURED OUTPUT
   LLMs return strings. Pydantic validates that string (parsed as JSON)
   matches your schema. If a field is missing or wrong type, you get a
   clear ValidationError — not a random KeyError somewhere downstream.
   answer_found=False lets the LLM say "I don't know" gracefully.

2. PROMPT ENGINEERING FOR RAG
   System prompt = rules (only use context, output JSON, never hallucinate).
   Context block = retrieved chunks tagged with [SOURCE | PAGE | ID].
   User message = the actual question + output format reminder.
   Showing the exact JSON schema example in the prompt dramatically
   improves how reliably the model follows it.

3. LAYERED JSON PARSING
   LLMs don't always output clean JSON. Three strategies in order:
   (1) Direct json.loads() — works when model follows instructions
   (2) Extract from markdown code fences ```json ... ```
   (3) Find first { and last } and parse that substring
   If all fail → graceful fallback response, never a 500 crash.

4. TEMPERATURE = 0
   For factual QA: always use temperature=0. Zero randomness means
   deterministic outputs — same question → same answer. You don't want
   the model to "creatively" interpret financial data differently each run.

5. CIRCUIT BREAKER PATTERN
   parse_llm_response() NEVER raises an exception. All parsing failures
   are caught and returned as a structured fallback RAGResponse.
   The API endpoint always gets back a valid object. This is the
   difference between a demo that crashes and a system you can ship.

6. OLLAMA
   Local LLM runner. Pulls and serves open-source models (Mistral, Llama3).
   REST API at localhost:11434, mostly compatible with OpenAI's chat format.
   Financial documents should stay local — don't send them to external APIs.

Interview questions you can now answer:
  - What is prompt engineering in a RAG context?
  - Why use Pydantic for LLM output?
  - How do you prevent hallucinations in a RAG system?
  - What does "grounded" mean in the context of RAG?
  - What is temperature and why set it to 0 for QA?
  - What is the circuit breaker pattern?
============================================================
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finance-Aid Day 3 Demo")
    parser.add_argument("--query", type=str, default=None,
                        help="Question to ask the RAG system")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM call, show retrieval results only")
    args = parser.parse_args()

    check_prerequisites()

    queries = [args.query] if args.query else [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
    ]

    if args.no_llm:
        for q in queries:
            run_retrieval_only(q)
    else:
        for q in queries:
            run_full_rag(q)

    print_what_you_learned()
