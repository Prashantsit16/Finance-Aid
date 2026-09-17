# Day 1 — Document Ingestion + RAG Foundation
**Date:** Sept 18, 2026
**Status:** Done

---

## What I built today

- [x] PDF extraction with PyMuPDF + pdfplumber fallback
- [x] Text cleaning pipeline (headers, footers, hyphenation, whitespace)
- [x] Chunking with overlap (1800 chars, 200 overlap)
- [x] Metadata per chunk: { chunk_id, text, source, page }
- [x] HuggingFace embedding with BAAI/bge-small-en-v1.5
- [x] ChromaDB persistent vector store
- [x] End-to-end: PDF -> query -> top relevant chunks working

Files written:
  app/ingestion/extractor.py
  app/ingestion/cleaner.py
  app/ingestion/chunker.py
  app/ingestion/embedder.py
  app/ingestion/store.py
  app/ingestion/pipeline.py
  run_day1.py

---

## Key decisions I made and why

**Why PyMuPDF as primary extractor?**
Faster than pdfplumber, handles most real PDFs well. We added pdfplumber
as a fallback specifically for pages that return too little text — which
usually means the layout was complex (tables, multi-column).

**Why 1800 char chunks with 200 char overlap?**
1800 chars is roughly 450 tokens, safely within BGE-small's 512 token limit.
Overlap of 200 chars (~50 tokens) ensures that a sentence split across
a chunk boundary still appears complete in at least one chunk.

**Why BGE-small over other models?**
Open source (no API cost), runs on CPU, 384 dimensions (not too big),
and ranks well on MTEB. For financial text specifically, the BGE models
are known to perform well on domain-specific terminology.

**Why not just use a regular database?**
SQL stores exact data and searches for exact words. Vector databases compare
meaning. "Revenue increased" and "sales grew" mean the same thing but share
no words — only a vector database would match them to a query about revenue.

**Why normalize embeddings?**
When embeddings are L2-normalized (unit length), cosine similarity equals
the dot product. Dot product is faster to compute than full cosine similarity
for large-scale search. ChromaDB handles this internally.

---

## What surprised me

PDFs are NOT text files. They're canvas-based documents where text is drawn
at X,Y coordinates. When you extract text you're reconstructing it from those
coordinates — which is why headers/footers repeat and why lines sometimes
break mid-word.

BGE has asymmetric prompting. You add a different prefix to documents vs
queries. I didn't expect that. Most embedding model tutorials don't mention it
but it matters for retrieval quality.

ChromaDB stores embeddings on disk automatically. I expected I would have to
manage serialization myself, but PersistentClient handles that. You restart
the app and the vectors are still there.

---

## What is NOT good enough yet

- Chunking is character-based, not token-based. A tokenizer would be more accurate.
- No handling of tables in PDFs (numbers in tables won't extract well)
- No deduplication if you re-ingest the same file (handled by chunk_id check)
- Search is pure dense only — Day 2 fixes this with BM25 + reranking

---

## Interview Q&A

**Q: What is an embedding?**
A list of numbers that represents the meaning of text in high-dimensional space.
Texts with similar meaning have similar vectors (point in similar directions).
BGE-small produces 384-dimensional embeddings. You can think of each dimension
as capturing some abstract feature of meaning.

**Q: Why do we chunk documents instead of using the full document?**
Two reasons. First: embedding models have token limits (512 for BGE-small) —
you can't embed a 200-page document as one unit. Second: retrieval granularity.
If someone asks about EBITDA, you want to retrieve the 2-3 sentences about EBITDA,
not the entire financial report. Chunking gives you that precision.

**Q: What does ChromaDB store per chunk?**
Four things: a unique string ID, the embedding vector (384 floats), the raw text,
and a metadata dict (source filename, page number). When you query, it returns the
closest vectors along with their text and metadata — which becomes your citation.

**Q: How does cosine similarity retrieval work?**
Both the query and each chunk are represented as vectors. Cosine similarity
measures the angle between them. A smaller angle (similarity close to 1.0)
means the meanings are close. ChromaDB uses HNSW (approximate nearest neighbor)
to find the closest vectors without comparing against every single chunk.

**Q: Why not just put the entire PDF in the LLM context?**
Token limits: GPT-4 allows ~128k tokens, but a 200-page report is ~150k tokens.
Cost: LLM APIs charge per token — feeding the whole document is expensive.
Quality: LLMs with very long contexts tend to "lose" information in the middle
(the "lost in the middle" problem). Retrieval finds the relevant parts first.

---

## What Day 2 builds on top of this

Today's search is pure dense (semantic) search. It works great when the
question is conceptual ("what are the risks?") but struggles with exact
terminology ("Section 4.2", specific numbers, proper nouns).

Day 2 adds BM25 keyword search running in parallel, then fuses both results,
then re-ranks the top candidates with a cross-encoder. That three-stage
pipeline is what makes retrieval actually production-grade.
