# Day 2 — Hybrid Retrieval + Reranking
**Date:** Sept 19, 2026
**Status:** Done

---

## What I built today

- [x] `app/retrieval/dense.py` — clean wrapper around Day 1 vector search
- [x] `app/retrieval/bm25.py` — BM25 keyword index built from ChromaDB chunks
- [x] `app/retrieval/fusion.py` — Reciprocal Rank Fusion (RRF) to merge both
- [x] `app/retrieval/pipeline.py` — full 4-stage retrieval pipeline
- [x] `app/retrieval/reranker.py` — cross-encoder reranking (ms-marco-MiniLM)
- [x] `run_day2.py` — comparison demo showing dense vs BM25 vs hybrid

---

## The 4-Stage Retrieval Pipeline

```
Query
  |
  +------------------+
  |                  |
Dense (top 20)    BM25 (top 20)
  |                  |
  +--------+---------+
           |
     RRF Fusion -> top 20 unique candidates
           |
     Cross-Encoder -> top 5 final chunks
```

---

## Key decisions I made and why

**Why RRF over score normalization?**
Dense scores are cosine similarities (0-1). BM25 scores are unbounded positive
numbers. You can't average them or weight them without knowing their distributions.
RRF avoids this entirely by using rank position, which is comparable across systems.

**Why k=60 in RRF?**
From the original 2009 paper. It's a smoothing constant that prevents rank 1
from being astronomically more valuable than rank 2. Works well empirically.
You could tune this but 60 is the standard starting point.

**Why cross-encoder/ms-marco-MiniLM-L-6-v2?**
Trained on MS MARCO — 1M+ real web queries with human-judged relevant passages.
MiniLM-L-6 is the smallest variant that still has good quality. Fast on CPU.
The "-L-6" means 6 transformer layers (vs 12 for full BERT).

**Why run reranker only over 20 candidates?**
Cross-encoder runs a full transformer forward pass for EVERY (query, chunk) pair.
Over 1000 chunks that's 1000 forward passes — way too slow at query time.
Over 20 candidates it's ~200ms. Totally acceptable.

---

## What surprised me

BM25 doesn't persist. Every time the app starts you rebuild the index from
ChromaDB. I expected it to be stored somewhere, but it's built in memory each
time. For 1000 chunks this is < 1 second, so it's fine.

Cross-encoder scores are raw logits — they can be negative. A score of -3.5
doesn't mean "this chunk is wrong", it just means it's less relevant than a chunk
scoring 0.2 or 1.8. The ordering is what matters, not the absolute values.

The rank changes after reranking are often dramatic. Chunk ranked #7 by fusion
can jump to #1 after cross-encoder scoring. This is because fusion is still
comparing vectors/keywords, not actually asking "does this answer the question?"

---

## What is NOT good enough yet

- No answer generation yet — we have the right chunks but no LLM response
- BM25 index is rebuilt on every fresh Python process (not persisted to disk)
- No query expansion or reformulation
- Cross-encoder takes ~200ms per query on CPU — fine for demo, slow for production

---

## Interview Q&A

**Q: What is the difference between dense and sparse retrieval?**
Dense retrieval encodes text as continuous vectors and finds similar vectors
using approximate nearest neighbor search. It captures semantic meaning —
"revenue grew" matches "sales increased". Sparse retrieval (BM25) works on
exact term matching weighted by term frequency and rarity. It's better at
specific terms, numbers, and acronyms that may be underrepresented in the
embedding model's training data.

**Q: What is BM25 and how does it score?**
Best Match 25. It scores a document for a query based on:
- Term Frequency: how often query words appear in the document, with
  diminishing returns (capped via saturation function)
- Inverse Document Frequency: query words that appear in fewer documents
  get higher weight (they're more discriminative)
- Document length normalization: shorter documents don't get unfair advantage

k1 and b are tuning parameters (default 1.5 and 0.75) from the original paper.

**Q: What is Reciprocal Rank Fusion and why not just add scores?**
RRF merges ranked lists using 1/(k + rank) for each item in each list.
We don't add scores because different retrievers produce scores on incompatible
scales — BM25 scores have no upper bound while cosine similarity is between
0 and 1. You'd need to normalize them first and normalization introduces
assumptions about score distributions. RRF avoids all of this by using rank
positions, which are directly comparable.

**Q: What is a cross-encoder vs bi-encoder? When do you use which?**
Bi-encoder: encodes query and document independently into separate vectors.
Similarity = dot product. Fast because document vectors are pre-computed.
Used for first-stage retrieval over large corpora.

Cross-encoder: takes (query, document) concatenated as input. Full attention
between all query and document tokens. Produces a single relevance score.
Slow (no pre-computation), but much more accurate. Used for reranking
a small set of candidates from first-stage retrieval.

**Q: Why two-stage retrieval?**
Accuracy vs speed tradeoff. You can't run a cross-encoder over 100,000 chunks
at query time — too slow. Bi-encoder + BM25 can retrieve top-20 relevant
candidates in milliseconds. Cross-encoder then precisely scores those 20.
You get speed from stage 1 and accuracy from stage 2.
