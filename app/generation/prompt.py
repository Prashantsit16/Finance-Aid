"""
prompt.py — Build the prompt that gets sent to the LLM.

PROMPT ENGINEERING IN RAG — what it means:
-------------------------------------------
In a plain chatbot, you just send the user message.
In RAG, you send: system instructions + retrieved context + user query.

The structure matters a lot. A poorly written prompt gets vague, hallucinated,
or incorrectly formatted answers. A well-written prompt gets grounded, structured,
and citable answers.

THE PROMPT HAS THREE PARTS:
  1. SYSTEM PROMPT  — tells the LLM who it is and what rules to follow
  2. CONTEXT BLOCK  — the retrieved chunks from ChromaDB (the "knowledge")
  3. USER QUERY     — what the user actually asked

WHY FORMAT CHUNKS WITH [SOURCE] TAGS?
When we embed chunks like:

  [SOURCE: annual_report.pdf | PAGE: 12]
  The company reported EBITDA of ₹420 crore...

The LLM can refer back to specific sources when generating citations.
Without these tags, it has no way to know which text came from which page.

WHY EXPLICIT OUTPUT FORMAT INSTRUCTIONS?
LLMs are probabilistic — they don't always follow instructions perfectly.
But if you show them the EXACT JSON schema you expect, with an example,
they follow it much more reliably. This is called "few-shot prompting" when
you show examples, and "schema prompting" when you show the output structure.

We instruct the LLM to output ONLY JSON (no preamble, no explanation).
This lets us parse it directly with json.loads() → Pydantic.
"""

from app.models.schemas import RAGResponse, Citation


# The JSON schema we expect the LLM to output — shown as an example in the prompt.
# This is the single most important thing for getting structured output reliably.
OUTPUT_SCHEMA_EXAMPLE = '''
{
  "answer": "The company reported EBITDA of ₹420 crore in FY23, representing a 12% year-on-year growth.",
  "answer_found": true,
  "confidence": "high",
  "citations": [
    {
      "source": "annual_report_2023.pdf",
      "page": 47,
      "excerpt": "The company reported EBITDA of ₹420 crore in FY23...",
      "chunk_id": "annual_report_2023_47_02"
    }
  ]
}
'''

SYSTEM_PROMPT = """You are a financial and legal document analyst.
Your job is to answer questions strictly based on the document context provided below.

RULES (follow these exactly):
1. Only use information from the provided context. Do NOT use external knowledge.
2. If the answer is not in the context, set answer_found to false and say so clearly.
3. Always cite which document and page your answer comes from.
4. Never guess or hallucinate numbers, dates, or facts.
5. If multiple chunks say different things, mention the discrepancy.
6. Be concise. Answer the question directly, then cite sources.

OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object. No explanation before or after.
No markdown code fences. No "Here is the answer:" preamble. Pure JSON only.

JSON schema:
{
  "answer": "<your answer here>",
  "answer_found": true or false,
  "confidence": "high" or "medium" or "low",
  "citations": [
    {
      "source": "<filename>",
      "page": <page number as integer>,
      "excerpt": "<exact text from the context that supports this answer>",
      "chunk_id": "<chunk_id from the context>"
    }
  ]
}

Example of a correct response:
""" + OUTPUT_SCHEMA_EXAMPLE


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM.
    
    Each chunk gets a clear header with source + page + chunk_id.
    The LLM uses these headers to construct accurate citations.
    
    Why include chunk_id?
    So the LLM can reference it in its citation output, which lets us 
    cross-reference the citation back to the exact stored chunk for verification.
    """
    if not chunks:
        return "No relevant context was found in the documents."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[CHUNK {i} | SOURCE: {chunk['source']} | "
            f"PAGE: {chunk['page']} | ID: {chunk['chunk_id']}]"
        )
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build the full prompt for the LLM.
    
    Returns: (system_prompt, user_message)
    
    We return two separate strings because most LLMs distinguish between
    the system role (instructions) and the user role (the actual query).
    In Ollama / OpenAI API format: these go into separate message objects.
    
    The user message combines the context + the question in a clear structure
    that mirrors how a human analyst would receive a briefing:
      "Here are the documents. Now answer this question."
    """
    context = format_context(chunks)
    
    user_message = f"""DOCUMENT CONTEXT:
{context}

---

QUESTION: {query}

Remember: respond with ONLY valid JSON matching the schema. No other text."""

    return SYSTEM_PROMPT, user_message
