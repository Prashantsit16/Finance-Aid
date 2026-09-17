# Finance-Aid — Financial & Legal Document Intelligence RAG

> I built this to actually understand how RAG systems work end to end — not just call an API and call it a day.
> If you are interviewing me and you ask about retrieval, embeddings, or vector search, this repo is my answer.

---

## What this is

A document question-answering system that can read financial reports and legal contracts (PDFs), and answer questions about them with citations — telling you exactly which page and section the answer came from.

You upload a PDF. You ask a question. It finds the relevant parts, passes them to an LLM, and gives you a grounded answer with source references. No hallucinations (or at least, we measure for them).

**Real use cases this handles:**
- "What was the company's EBITDA in FY23?"
- "What does clause 4.2 say about termination?"
- "Summarize the key risks mentioned in this annual report."

---

## Why I built it this way

The first instinct when building something like this is to just stuff the whole PDF into the LLM prompt. That breaks immediately — PDFs are long, context windows have limits, and you end up paying a lot for very bad answers.

So the real problem is: **how do you find the right 3-5 paragraphs out of a 200-page document?**

That's what this entire system is about. Everything else — the API, the UI, the evaluation — is just plumbing around that core retrieval problem.

---

## Architecture

```
USER
 |
 v
Streamlit Frontend
 |  (HTTP / REST)
 v
FastAPI Backend
 |
 +----------------------------------+
 |                                  |
 v                                  v
Document Pipeline              Query Pipeline
PDF extraction                 Embed query (HF)
  |                              |
Cleaning + Chunking           Hybrid Search
  |                           +----+----+
Metadata tagging              |         |
  |                        Dense       BM25
HF Embeddings            (semantic) (keywords)
  |                           +----+----+
ChromaDB (store)               Fusion
                                  |
                          Cross-Encoder Reranking
                                  |
                             Top-K Chunks
                                  |
                                 LLM
                                  |
                        Pydantic Structured Output
                                  |
                          Answer + Citations

RAGAS Evaluation runs across the full pipeline
  -> Faithfulness / Answer Relevancy / Context Precision

Everything packaged in Docker.
No Kubernetes. No agents. No MCP. No fine-tuning.
```

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| PDF parsing | PyMuPDF / pdfplumber | Handles real-world messy PDFs |
| Embeddings | HuggingFace (BGE/SBERT) | Open-source, no API cost |
| Vector store | ChromaDB | Simple, local, persistent |
| Keyword search | BM25 (rank_bm25) | Catches exact terminology dense search misses |
| Reranking | Cross-Encoder | Reorders candidates by actual relevance |
| LLM | Llama 3 / Mistral | Open-source generation |
| Output schema | Pydantic | Structured answers with citations |
| Backend | FastAPI | Clean API layer |
| Frontend | Streamlit | Fast to build, easy to demo |
| Evaluation | RAGAS | Measures retrieval + generation quality |
| Deployment | Docker + docker-compose | Reproducible, portable |

---

## Project Structure

```
Finance-Aid/
├── app/
│   ├── api/              # FastAPI routes (upload, query, evaluate)
│   ├── core/             # Config, constants, shared utilities
│   ├── ingestion/        # PDF extraction, cleaning, chunking
│   ├── retrieval/        # Dense search, BM25, hybrid fusion, reranking
│   ├── generation/       # LLM prompting, Pydantic response models
│   ├── evaluation/       # RAGAS scoring logic
│   └── models/           # Pydantic schemas
├── data/
│   ├── raw/              # Original PDFs go here
│   └── processed/        # Chunked + embedded data
├── tests/                # Unit + integration tests
├── notebooks/            # Exploration, debugging, experiments
├── docker/               # Dockerfile + compose configs
├── requirements.txt
└── README.md
```

---

## 5-Day Build Log

### Day 1 — Document Ingestion + RAG Foundation
**Date:** Sept 18, 2026
**Status:** In progress

Goal: Get a PDF to vector database pipeline working end to end.

What I implemented:
- PDF text extraction using PyMuPDF / pdfplumber
- Text cleaning — stripping headers, footers, page artifacts
- Chunking with overlap so context does not get cut mid-sentence
- Each chunk gets tagged: { text, source, page, chunk_id }
- Generate embeddings with a HuggingFace model
- Store in ChromaDB

By end of day, the system goes: PDF -> vector DB -> query -> top relevant chunks

**Interview concepts this day covers:**
- What is an embedding and why does it work?
- Why do we chunk instead of using the full document?
- Why not just put the entire PDF in the LLM context?
- What does a vector database actually store?
- How does cosine similarity retrieval work?

---

### Day 2 — Hybrid Retrieval + Reranking
**Date:** Sept 19, 2026
**Status:** Upcoming

Dense search (vector similarity) is good at understanding meaning.
BM25 is good at exact keyword matching.

If someone searches "Section 4.2 EBITDA 420 crore", dense search might miss it.
If someone asks "what is EBITDA", keyword search is useless.
Hybrid does both.

Then cross-encoder reranking — instead of asking "which chunks are similar to this query?", it asks "does this chunk actually answer this question?" Much better signal.

Implementing: Dense retrieval + BM25 + reciprocal rank fusion + cross-encoder reranker

---

### Day 3 — LLM Generation + Structured Output
**Date:** Sept 20, 2026
**Status:** Upcoming

Take the top-K chunks, build a prompt, get a grounded answer.
Pydantic enforces the response shape so the frontend always gets answer + citations[] — no free-form text parsing.

Graceful fallback when the answer is not in the document.

---

### Day 4 — FastAPI Backend + Streamlit Frontend
**Date:** Sept 21, 2026
**Status:** Upcoming

Wire everything into a proper API:
- POST /upload — ingest a PDF
- POST /query — ask a question
- GET /evaluate — run RAGAS

Build the Streamlit UI: upload panel, chat interface, citation display.

---

### Day 5 — RAGAS Evaluation + Docker
**Date:** Sept 22, 2026
**Status:** Upcoming

RAGAS evaluation metrics:
- Faithfulness — does the answer actually come from the retrieved chunks?
- Answer Relevancy — does the answer address what was asked?
- Context Precision — are the retrieved chunks actually relevant?

Then containerize everything with Docker so it runs the same anywhere.

---

## Key Concepts (Interview Quick Reference)

**What is RAG?**
Retrieval-Augmented Generation. You retrieve relevant context from a document store and pass it to an LLM. The LLM generates an answer grounded in that context rather than making things up from training data.

**Why chunking?**
LLMs have token limits. A 200-page PDF is roughly 150k tokens. You only need the 3-5 paragraphs that answer the question. Chunking lets you retrieve just those paragraphs.

**Why hybrid retrieval?**
Dense search captures semantic meaning. BM25 captures exact keyword matches. Financial documents have a lot of specific terminology ("EBITDA", "Section 4.2", ticker symbols) where exact match matters.

**What does a cross-encoder do?**
A bi-encoder (standard embedding model) embeds query and document independently, then compares. A cross-encoder sees both together and scores relevance jointly — much more accurate, but slower, so we only run it on the top candidates from the first retrieval pass.

**What is RAGAS?**
An evaluation framework for RAG systems. It measures whether your retrieved chunks are actually relevant and whether your generated answers are faithful to those chunks — without needing human-labeled ground truth for every question.

---

## Setup

```bash
git clone https://github.com/Prashantsit16/Finance-Aid.git
cd Finance-Aid
pip install -r requirements.txt

# Drop your PDFs in data/raw/
# Then run ingestion
python -m app.ingestion.pipeline

# Start the API
uvicorn app.api.main:app --reload

# Start the UI (separate terminal)
streamlit run app/frontend/app.py
```

**Or with Docker:**
```bash
docker-compose up --build
```

---

## What I want to add after Day 5

- Multi-document comparison ("Compare the risk sections of these two reports")
- Fine-tuned reranker on financial domain data
- Async ingestion queue for large document batches
- Auth layer so multiple users can have separate document stores

---

*This is a learning project. Every decision in this repo was made consciously and I can explain the tradeoff behind it.*
