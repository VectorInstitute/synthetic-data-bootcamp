"""Retrieval primitives for Phase 9: BM25, a real semantic embedding retriever,
reciprocal rank fusion, and the ranking metrics used to score them.

The semantic retriever is `fastembed` (ONNX Runtime, no torch) with
`minishlab/potion-retrieval-32M`, a static embedding model distilled from a larger
transformer (Model2Vec): embedding is a token lookup plus pooling, not a transformer
forward pass, so it's genuinely semantic (trained to put related text nearby in
vector space) without the per-query inference cost that comes with a full encoder.

Two things were tried first and ruled out:
- `sentence-transformers` (pulls in torch): installing it timed out after 3 minutes
  in this sandboxed environment with no prior download cache.
- `fastembed` with the default `BAAI/bge-small-en-v1.5` transformer model: it
  installs fine (ONNX Runtime alone, ~20MB), but measured at ~9 texts/second on this
  machine's 16 CPU cores -- fine for the ~1,300 synthetic chunks, but the real
  6-company sample alone chunks into ~60,000 paragraphs, which would take roughly
  two hours to embed. `potion-retrieval-32M` measured at ~1,900 texts/second on the
  same corpus (no transformer layers to run at inference time), bringing that down
  to well under a minute, with comparable semantic separation in a spot check
  (query-answer cosine 0.62 vs. query-unrelated-text cosine 0.03).

Calibration note: `fastembed==0.9.0` declares `huggingface-hub<2.0` but imports an
internal module (`_snapshot_download`) that huggingface-hub removed in its 1.x
release, so installing fastembed alone (which resolved huggingface-hub 1.33.0) broke
on import. Fixed by pinning `huggingface-hub==0.36.2` in requirements.txt.
"""

import math
import re

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")
SEMANTIC_MODEL_NAME = "minishlab/potion-retrieval-32M"

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Load the embedding model once and reuse it across every retriever built."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=SEMANTIC_MODEL_NAME)
    return _model


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Bm25Retriever:
    """Thin wrapper around rank_bm25.BM25Okapi, keyed by chunk_id."""

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query: str) -> list[str]:
        """Return chunk_ids sorted best-first for `query`."""
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [self.chunk_ids[i] for i in order]

    def top_score(self, query: str) -> float:
        """Return the best (unbounded) BM25 score for `query`, or 0.0 for an empty corpus."""
        scores = self._bm25.get_scores(tokenize(query))
        return float(max(scores)) if len(scores) else 0.0


class SemanticRetriever:
    """Cosine similarity over real sentence embeddings; see module docstring."""

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = chunk_ids
        embeddings = np.array(list(_get_model().embed(texts, batch_size=256)), dtype=np.float32)
        self.doc_matrix = _normalize(embeddings)

    def rank(self, query: str) -> list[str]:
        """Return chunk_ids sorted best-first for `query`."""
        q_vec = np.array(next(iter(_get_model().embed([query]))), dtype=np.float32)
        q_vec = _normalize(q_vec[np.newaxis, :])[0]
        scores = self.doc_matrix @ q_vec
        order = np.argsort(-scores)
        return [self.chunk_ids[i] for i in order]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Combine several best-first chunk_id rankings into one via RRF (standard k=60)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda c: -scores[c])


def recall_at_k(ranking: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if set(ranking[:k]) & gold_ids else 0.0


def reciprocal_rank(ranking: list[str], gold_ids: set[str]) -> float:
    for i, chunk_id in enumerate(ranking):
        if chunk_id in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranking: list[str], gold_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k (one gold chunk_id set per question)."""
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranking[:k]) if c in gold_ids)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
