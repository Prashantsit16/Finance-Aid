# Day 3 — LLM Generation + Structured Output
**Date:** Sept 20, 2026
**Status:** Done

---

## What I built today

- [x] `app/models/schemas.py` — Pydantic models: Citation, RAGResponse, QueryRequest, IngestResponse
- [x] `app/generation/prompt.py` — Prompt builder: system prompt + context formatter + user message
- [x] `app/generation/llm.py` — Ollama client, layered JSON parsing, Pydantic validation, graceful fallback
- [x] `app/generation/pipeline.py` — Single answer_query() wiring retrieval + generation
- [x] `run_day3.py` — End-to-end demo with --no-llm flag for testing without Ollama

---

## The Full Pipeline (all 3 days connected)

```
Question
   |
   v  Day 2: retrieval/pipeline.py
Dense + BM25 → RRF → Cross-Encoder → Top-5 chunks
   |
   v  Day 3: generation/prompt.py
System prompt + [SOURCE|PAGE|ID] context block + question
   |
   v  Day 3: generation/llm.py
Ollama (Mistral/Llama3) @ localhost:11434, temp=0
   |
   v  Day 3: generation/llm.py (parsing)
extract_json → json.loads → Pydantic validation
   |
   v
RAGResponse { answer, citations[], confidence, answer_found }
```

---

## Key decisions I made and why

**Why Pydantic for output?**
LLMs return strings. You need structured data. Without Pydantic you parse
manually — fragile and breaks constantly. Pydantic gives you: schema validation,
clear error messages on failure, and the same models reused in FastAPI for API
validation and documentation. One class does three jobs.

**Why show the JSON schema in the prompt?**
LLMs follow explicit examples better than abstract instructions. Showing the
exact schema (with an example response) in the system prompt dramatically
reduces malformed output. This is schema prompting — more reliable than
just saying "output JSON".

**Why temperature=0?**
Temperature controls randomness. For financial/legal QA you want deterministic
answers. "What was the EBITDA?" should return the same answer every time,
not vary based on model sampling. Zero temperature = greedy decoding.

**Why Ollama over OpenAI API?**
Financial documents are confidential. You don't want annual reports sent to
external APIs. Ollama runs everything locally — model, inference, data.
The REST format is mostly OpenAI-compatible so switching later is trivial.

**Why `answer_found=False` instead of empty string?**
Empty string is ambiguous — could mean "no answer" or "answer is blank text".
A boolean field is explicit. Frontend can check `answer_found` and show a
"not found in document" message instead of showing blank content.

---

## Layered JSON parsing — why it works

LLMs don't always output clean JSON even when asked. Three fallback strategies:

1. **Direct parse**: `json.loads(raw)` — works most of the time
2. **Markdown fence**: extract content inside ` ```json ... ``` `  
   LLMs sometimes wrap output in code fences even when told not to
3. **Brace extraction**: find first `{` and last `}`, parse that substring  
   Handles "Here is the answer: { ... }" preamble text

If all fail → graceful fallback response using raw chunk text. Never a crash.

---

## Interview Q&A

**Q: What is prompt engineering in a RAG context?**
Structuring the prompt into three parts: system prompt (rules + output schema),
context block (retrieved chunks tagged with source/page metadata), and user query.
The key insight is that the context block is dynamically generated per query —
it's not static. The LLM sees different information every time based on what
retrieval found. Prompt engineering in RAG means making the model reliably
use ONLY that dynamic context, not its training knowledge.

**Q: Why use Pydantic for LLM output instead of just parsing JSON manually?**
Three reasons: (1) Validation — Pydantic checks types, required fields, and
enum values automatically. If the model outputs confidence="very high", Pydantic
rejects it (not in Literal["high","medium","low"]). (2) Error clarity — you get
a ValidationError with the exact field that failed, not a KeyError deep in business
logic. (3) FastAPI integration — the same Pydantic model is used as the API
response type, giving you free OpenAPI docs and response serialization.

**Q: How do you prevent hallucinations in a RAG system?**
Four layers: (1) Prompt rule: "Only use information from the provided context." 
(2) answer_found field: when the model determines the answer isn't in context,
it sets this to false and says so — instead of making something up.
(3) Citations: every claim must link to a source chunk with page number. 
(4) RAGAS evaluation (Day 5): faithfulness metric measures whether each claim
in the answer is supported by the retrieved context. Automated detection.

**Q: What is temperature and why set it to 0 for QA?**
Temperature controls how random the model sampling is. High temperature (1.0+):
creative, varied output — good for writing, brainstorming. Temperature 0:
greedy decoding, deterministic — same input always produces the same output.
For financial/legal QA you want zero. You don't want the model to "creatively"
interpret numbers or dates differently each time you ask.

**Q: What is the circuit breaker pattern?**
A design pattern where you wrap a potentially failing operation in a handler
that returns a safe fallback instead of crashing. In our llm.py,
parse_llm_response() catches ALL parsing failures (JSON, Pydantic, KeyError)
and returns a valid RAGResponse with low confidence and raw chunk text.
The caller (API endpoint) always gets a valid object. No 500 errors from
parsing failures. The "circuit breaks" before the fault propagates.

---

## What is NOT perfect yet

- No retry logic if LLM returns bad JSON (could retry 1-2 times before fallback)
- Context window management — if 5 chunks total > 8k tokens, Mistral truncates
- No streaming — response arrives all at once after 5-30 seconds
- Ollama must be running manually — no auto-start

All of these are real production concerns that would come up in interviews.
