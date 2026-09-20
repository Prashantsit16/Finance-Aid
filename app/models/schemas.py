"""
schemas.py — Pydantic models for the entire Finance-Aid system.

WHY PYDANTIC?
-------------
When an LLM generates text, it returns an unstructured string.
Your frontend expects structured data: answer, sources, page numbers.

Without Pydantic:
  response = llm.generate(prompt)
  # Now what? Parse the string manually? Split on newlines? Hope for the best?
  # This breaks constantly. LLMs don't follow instructions 100% of the time.

With Pydantic:
  response = RAGResponse(**json.loads(llm_output))
  # If the LLM output doesn't match the schema, you catch a clear ValidationError.
  # You can then retry, fall back, or return a structured error — not a crash.

Pydantic is also self-documenting. The schema IS the contract between
your backend and frontend. FastAPI uses these same models for API validation.
One class → input validation + output serialization + API docs, all for free.

DESIGN DECISIONS:
  - answer_found: bool lets us distinguish "I don't know" from an empty string
  - confidence: enum keeps it controlled ("high"/"medium"/"low", not free-form)
  - citations: list so multiple sources can be returned
  - excerpt: the actual chunk text used — lets user verify the answer
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class Citation(BaseModel):
    """
    A single source reference for the answer.
    
    When the LLM answers "The EBITDA was ₹420 crore", the citation tells you:
      - Which file that came from
      - Which page
      - The actual text that was used (so you can verify)
    
    This is what makes RAG trustworthy over a plain LLM.
    """
    source: str = Field(description="PDF filename the chunk came from")
    page: int = Field(description="Page number in the original document")
    excerpt: str = Field(description="The actual text excerpt used to generate this answer")
    chunk_id: str = Field(default="", description="Internal chunk identifier")


class RAGResponse(BaseModel):
    """
    The complete, structured response from the RAG pipeline.
    
    This is what the API returns. Pydantic validates every field.
    If the LLM forgets to include 'citations', you get a clear error, not a KeyError.
    
    answer_found=False is the graceful fallback:
      Instead of hallucinating, the LLM says "I couldn't find this in the document."
      That's the right behaviour. Better to say "I don't know" than to make something up.
    """
    query: str = Field(description="The original question that was asked")
    answer: str = Field(description="The generated answer, grounded in the retrieved context")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source references used to generate the answer"
    )
    answer_found: bool = Field(
        default=True,
        description="False if the answer was not found in the retrieved documents"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="How confident the answer is based on source quality"
    )
    model_used: str = Field(default="", description="Which LLM model generated the answer")
    retrieval_chunks_used: int = Field(default=0, description="How many chunks were passed to the LLM")


class IngestResponse(BaseModel):
    """Response from the /upload endpoint."""
    filename: str
    pages_extracted: int
    chunks_created: int
    chunks_embedded: int
    status: str = "success"


class QueryRequest(BaseModel):
    """Request body for the /query endpoint (Day 4)."""
    query: str = Field(min_length=3, max_length=1000)
    source_filter: str | None = Field(default=None, description="Restrict search to specific PDF")
    top_k: int = Field(default=5, ge=1, le=20)
