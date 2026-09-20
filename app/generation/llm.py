"""
llm.py — LLM integration, JSON parsing, and Pydantic validation.

This is the most fragile part of the whole system because we depend on
an external model to output structured JSON. It often doesn't. This file
is about making that reliable.

THE PROBLEM WITH LLM OUTPUT:
  You ask for JSON. The LLM gives you:
    "Sure! Here is the answer: ```json\n{ ... }\n```"
  Or:
    "Based on the context provided, { ... }"
  Or just:
    "{ ... }" (the good case)
  
  Your json.loads() call fails on the first two cases.

THE SOLUTION — LAYERED PARSING:
  1. Try direct JSON parse
  2. Try to extract JSON from inside markdown code fences (```json ... ```)
  3. Try to find the first { and last } and parse that substring
  4. If all fail: return a structured fallback response, never crash

OLLAMA:
  Ollama is a local LLM runner. You install it, pull a model (mistral, llama3),
  and it runs a REST API at localhost:11434. 
  
  Why Ollama over OpenAI API?
    - Free, no API key, data stays local
    - Works offline
    - You can swap models without changing code
    - Privacy — financial documents should not be sent to external APIs

  The request format is identical to OpenAI's chat completions API (mostly),
  so if you later want to switch to OpenAI, it's a one-line URL change.

TEMPERATURE=0:
  Temperature controls randomness. 0 = deterministic (same input → same output).
  For financial/legal QA you want ZERO randomness. You don't want the model
  to creatively interpret "what was the EBITDA?" in different ways each time.
"""

import json
import re
from loguru import logger
import requests
from pydantic import ValidationError

from app.models.schemas import RAGResponse, Citation
from app.generation.prompt import build_prompt
from app.core.config import settings


# ── Ollama client ─────────────────────────────────────────────────────────────

def call_ollama(
    system_prompt: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.0,
) -> str:
    """
    Call Ollama's local chat completions API.
    
    Returns the raw text response from the model.
    
    The Ollama API format (messages) is compatible with OpenAI's API.
    If you have an OpenAI key and want to use GPT-4 instead, you'd change:
      - base_url to https://api.openai.com
      - Add Authorization header
      - model to "gpt-4o"
    Everything else stays the same.
    """
    model = model or settings.LLM_MODEL
    url = f"{settings.LLM_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,          # get the full response at once
        "options": {
            "temperature": temperature,
            "num_predict": 1024,  # max tokens to generate
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {settings.LLM_BASE_URL}. "
            f"Is Ollama running? Start it with: ollama serve\n"
            f"Then pull a model: ollama pull mistral"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Ollama timed out after 120s. The model may be loading. Try again."
        )
    except KeyError:
        raise ValueError(f"Unexpected Ollama response format: {response.text[:200]}")


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json_from_response(raw: str) -> dict:
    """
    Try multiple strategies to extract valid JSON from LLM output.
    
    LLMs don't always output clean JSON even when asked to. This function
    tries increasingly aggressive parsing strategies.
    
    Strategy 1: Direct parse (works when the model follows instructions perfectly)
    Strategy 2: Extract from markdown code fences (```json...```)
    Strategy 3: Find first { and last } and parse that substring
    Strategy 4: Raise a clear error with the raw output for debugging
    """
    raw = raw.strip()

    # Strategy 1: direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown code fences
    # Matches ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { and last } — handles "Here is the answer: {...}"
    brace_start = raw.find("{")
    brace_end   = raw.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(raw[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # All strategies failed
    logger.error(f"Could not extract JSON from LLM response. Raw output:\n{raw[:500]}")
    raise ValueError(f"LLM did not return valid JSON. Raw: {raw[:300]}")


# ── Pydantic validation + fallback ────────────────────────────────────────────

def parse_llm_response(raw: str, query: str, chunks: list[dict], model: str) -> RAGResponse:
    """
    Parse the raw LLM text into a validated RAGResponse.
    
    If parsing succeeds: return the Pydantic model with full citations.
    If parsing fails for ANY reason: return a graceful fallback response.
    
    NEVER raise an exception from this function. The caller (the API endpoint)
    should always get back a valid RAGResponse, even if something went wrong.
    That's the contract.
    
    This is the "circuit breaker" pattern — we contain failures here so they
    don't propagate to the user as a 500 error.
    """
    try:
        data = extract_json_from_response(raw)

        # Pydantic validates the structure and types
        # If the LLM output is missing a field or has wrong type, ValidationError is raised here
        response = RAGResponse(
            query=query,
            answer=data.get("answer", ""),
            answer_found=data.get("answer_found", True),
            confidence=data.get("confidence", "medium"),
            citations=[
                Citation(
                    source=c.get("source", ""),
                    page=int(c.get("page", 0)),
                    excerpt=c.get("excerpt", ""),
                    chunk_id=c.get("chunk_id", ""),
                )
                for c in data.get("citations", [])
            ],
            model_used=model,
            retrieval_chunks_used=len(chunks),
        )
        return response

    except (ValueError, ValidationError, KeyError, TypeError) as e:
        # Parsing failed — return graceful fallback
        logger.warning(f"Failed to parse LLM output into RAGResponse: {e}. Returning fallback.")
        return RAGResponse(
            query=query,
            answer=(
                "I was able to retrieve relevant context but encountered an error "
                "formatting the response. The most relevant retrieved text is: "
                + (chunks[0]["text"][:400] if chunks else "No context available.")
            ),
            answer_found=bool(chunks),
            confidence="low",
            citations=[
                Citation(
                    source=c["source"],
                    page=c["page"],
                    excerpt=c["text"][:200],
                    chunk_id=c["chunk_id"],
                )
                for c in chunks[:2]
            ],
            model_used=model + " (fallback)",
            retrieval_chunks_used=len(chunks),
        )


# ── Main generation function ──────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = None,
) -> RAGResponse:
    """
    Main function: takes a query + retrieved chunks → returns a RAGResponse.
    
    Flow:
      1. Build the prompt (system + user message)
      2. Call Ollama with temperature=0
      3. Extract JSON from the raw text response
      4. Validate with Pydantic
      5. Return RAGResponse (or fallback if anything failed)
    
    If Ollama is not running, returns a clear error response instead of crashing.
    """
    model = model or settings.LLM_MODEL

    if not chunks:
        return RAGResponse(
            query=query,
            answer="No relevant content was found in the documents to answer this question.",
            answer_found=False,
            confidence="low",
            citations=[],
            model_used=model,
            retrieval_chunks_used=0,
        )

    system_prompt, user_message = build_prompt(query, chunks)

    logger.info(f"Calling LLM ({model}) with {len(chunks)} context chunks...")

    try:
        raw_response = call_ollama(system_prompt, user_message, model)
        logger.debug(f"Raw LLM response (first 300 chars): {raw_response[:300]}")
    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.error(f"LLM call failed: {e}")
        return RAGResponse(
            query=query,
            answer=f"LLM unavailable: {str(e)}",
            answer_found=False,
            confidence="low",
            citations=[],
            model_used=model,
            retrieval_chunks_used=len(chunks),
        )

    return parse_llm_response(raw_response, query, chunks, model)
