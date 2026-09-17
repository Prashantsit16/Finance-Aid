"""
chunker.py — Split cleaned text into chunks the embedding model can handle.

This is the most important design decision in the whole RAG pipeline.
Get chunking wrong and everything downstream suffers.

WHY CHUNK AT ALL?
-----------------
Embedding models have a max token limit (~512 tokens for most models).
A single page of a financial report can be 800-1200 tokens.
You can't embed the whole page as one unit.

Also: retrieval granularity matters. If someone asks "what was the EBITDA?",
you want to retrieve the 2-3 sentences around that number, not the entire
chapter. Smaller chunks = more precise retrieval.

WHY OVERLAP?
------------
Imagine chunking "The company reported EBITDA of 420 crore in Q3."
Without overlap:
  Chunk A: "The company reported EBITDA"
  Chunk B: "of 420 crore in Q3."

Neither chunk makes sense alone. With overlap, Chunk B repeats the
tail of Chunk A, so each chunk has enough context to be self-contained.

CHUNK SIZE INTUITION:
  Too small (< 100 tokens): chunks lose context, retrieval gets noisy
  Too large (> 600 tokens): embedding quality drops, retrieval is too broad
  Sweet spot: 300-512 tokens with 50-100 token overlap

We use CHARACTERS as a proxy for tokens here (simpler, no tokenizer needed).
Rough rule: 1 token ≈ 4 characters in English.
So chunk_size=512 chars ≈ 128 tokens (safe for BGE-small which allows 512 tokens).

Wait — we want bigger chunks. Let's use 1800 chars ≈ 450 tokens with 200 char overlap.
"""

from loguru import logger


def chunk_text(
    text: str,
    chunk_size: int = 1800,    # characters, ~450 tokens
    chunk_overlap: int = 200,  # characters, ~50 tokens
) -> list[str]:
    """
    Split text into overlapping chunks.

    Algorithm:
    - Start at position 0
    - Take chunk_size characters
    - Move forward by (chunk_size - chunk_overlap) characters
    - Repeat until end of text

    We also try to break at sentence boundaries (". ") rather than
    mid-sentence. This is important for chunk quality.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap  # how far we advance each time

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary near the end of this chunk.
        # Look backwards from `end` for a period followed by space.
        if end < len(text):
            # Search for ". " or ".\n" in the last 20% of the chunk
            search_from = end - int(chunk_size * 0.2)
            boundary = -1

            for i in range(end, search_from, -1):
                if i < len(text) and text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1] in " \n"):
                    boundary = i + 1
                    break

            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip tiny chunks
            chunks.append(chunk)

        start += step

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Chunk all pages and attach metadata to each chunk.

    This metadata is critical. When you retrieve chunk #47 later,
    you need to know:
      - Which file it came from (for citation)
      - Which page it was on (for citation)
      - Its unique ID (so ChromaDB can store and retrieve it)

    The chunk_id format: "{source_stem}_{page}_{index}"
    Example: "annual_report_2023_12_03" means:
      - file: annual_report_2023.pdf
      - page: 12
      - 4th chunk on that page (index 03)

    This ID format is readable and tells you exactly where to look
    in the original document.
    """
    all_chunks = []

    for page in pages:
        source = page["source"]
        page_num = page["page"]
        text = page["text"]

        # Get chunks from this page's text
        page_chunks = chunk_text(text, chunk_size, chunk_overlap)

        # Attach metadata to each chunk
        source_stem = source.replace(".pdf", "").replace(" ", "_")

        for idx, chunk_text_content in enumerate(page_chunks):
            chunk_id = f"{source_stem}_{page_num}_{idx:02d}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_content,
                "source": source,
                "page": page_num,
            })

    logger.info(
        f"Created {len(all_chunks)} chunks from {len(pages)} pages. "
        f"Avg {len(all_chunks) / max(len(pages), 1):.1f} chunks/page."
    )

    return all_chunks
