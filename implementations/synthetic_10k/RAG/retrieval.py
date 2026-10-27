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

Hosted embeddings: a model name starting with `text-embedding-` (for example
`text-embedding-3-small`) is embedded through an OpenAI-compatible endpoint instead,
such as Vector's proxy on the Coder workspace. It reads the same variables as
src.step08_generate_filings: SYNBENCH_BASE_URL (default
https://proxy.vectorinstitute.ai/v1) and the key from OPENAI_API_KEY or SYNBENCH_API_KEY.
Vectors are cached on disk (one SQLite file per model under data/embedding_cache/), so
a chunk is only ever embedded once across runs and corpora. Set RAG_EMBEDDING_MODEL to
pick the model SemanticRetriever uses by default; it stays potion-retrieval-32M if unset.
"""

import hashlib
import math
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")
POTION_MODEL_NAME = "minishlab/potion-retrieval-32M"
SEMANTIC_MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL") or POTION_MODEL_NAME
DEFAULT_BASE_URL = "https://proxy.vectorinstitute.ai/v1"
CACHE_DIR = Path(__file__).resolve().parents[1] / "dylan_pipeline" / "data" / "embedding_cache"
# text-embedding-3 models accept up to 8,191 tokens per input; a few real-filing table
# chunks run longer, so inputs are cut to a character budget well under that.
MAX_INPUT_CHARS = 20_000


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (see module docstring)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts, batch_size=256)), dtype=np.float32)


class OpenAIEmbedder:
    """Embeddings from an OpenAI-compatible endpoint, cached on disk by text hash."""

    def __init__(self, model_name: str, batch_size: int = 256, workers: int = 4) -> None:
        from openai import OpenAI  # noqa: PLC0415

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SYNBENCH_API_KEY")
        if not api_key:
            raise SystemExit(f"Set OPENAI_API_KEY or SYNBENCH_API_KEY to embed with {model_name}.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.workers = workers
        self._client = OpenAI(base_url=os.environ.get("SYNBENCH_BASE_URL", DEFAULT_BASE_URL), api_key=api_key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model_name)}.sqlite")
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _request(self, batch: list[str]) -> list[np.ndarray]:
        for attempt in range(5):
            try:
                response = self._client.embeddings.create(model=self.model_name, input=batch)
                return [np.array(d.embedding, dtype=np.float32) for d in sorted(response.data, key=lambda d: d.index)]
            except Exception:  # noqa: BLE001 -- proxy errors vary; retry then re-raise
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        return []

    def embed(self, texts: list[str]) -> np.ndarray:
        inputs = [(t[:MAX_INPUT_CHARS] or " ") for t in texts]
        keys = [self._key(t) for t in inputs]
        found: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(keys))
        for i in range(0, len(unique), 500):
            part = unique[i : i + 500]
            rows = self._db.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})",
                part,
            ).fetchall()
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})

        missing = list(dict.fromkeys(t for t, k in zip(inputs, keys, strict=True) if k not in found))
        batches = [missing[i : i + self.batch_size] for i in range(0, len(missing), self.batch_size)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for batch, vectors in zip(batches, pool.map(self._request, batches), strict=True):
                new = {self._key(t): v for t, v in zip(batch, vectors, strict=True)}
                self._db.executemany(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                    [(k, v.tobytes()) for k, v in new.items()],
                )
                self._db.commit()
                found.update(new)
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)


_embedders: dict[str, FastEmbedEmbedder | OpenAIEmbedder] = {}


def get_embedder(model_name: str = SEMANTIC_MODEL_NAME) -> FastEmbedEmbedder | OpenAIEmbedder:
    """Load each embedding model once and reuse it across every retriever built."""
    if model_name not in _embedders:
        if model_name.startswith("text-embedding-"):
            _embedders[model_name] = OpenAIEmbedder(model_name)
        else:
            _embedders[model_name] = FastEmbedEmbedder(model_name)
    return _embedders[model_name]


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

    def __init__(self, chunk_ids: list[str], texts: list[str], model_name: str = SEMANTIC_MODEL_NAME) -> None:
        self.chunk_ids = chunk_ids
        self.embedder = get_embedder(model_name)
        self.doc_matrix = _normalize(self.embedder.embed(texts))

    def rank(self, query: str) -> list[str]:
        """Return chunk_ids sorted best-first for `query`."""
        q_vec = _normalize(self.embedder.embed([query]))[0]
        scores = self.doc_matrix @ q_vec
        order = np.argsort(-scores)
        return [self.chunk_ids[i] for i in order]

    def rank_many(self, queries: list[str]) -> list[list[str]]:
        """Return one best-first ranking per query, embedding all queries in one batch."""
        q_matrix = _normalize(self.embedder.embed(queries))
        return [[self.chunk_ids[i] for i in np.argsort(-(self.doc_matrix @ q))] for q in q_matrix]


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
