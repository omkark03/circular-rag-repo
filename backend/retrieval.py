"""Retrieval quality layer: hybrid search + cross-encoder reranking.

Pipeline (each stage degrades gracefully if its dependency is missing):

  1. Vector search (ChromaDB)          — semantic similarity
  2. BM25 keyword search (rank-bm25)   — exact terms, circular numbers, clauses
  3. Reciprocal Rank Fusion            — merge both rankings
  4. Cross-encoder rerank              — scores (query, chunk) pairs jointly,
                                         far more accurate than embeddings alone
  5. Return TOP_K best chunks

This is what makes retrieval fast AND accurate without fine-tuning the LLM:
all repository knowledge is pre-embedded at upload time ("trained" into the
index), and the reranker sharpens what reaches Phi-3.
"""
import re
import threading

import config
import ingest

_lock = threading.Lock()
_bm25 = None            # (BM25Okapi, [chunk dicts]) cache
_bm25_version = -1      # invalidated when the collection count changes
_reranker = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u0900-\u097F]+", text.lower())


def invalidate():
    """Call after ingest/reindex so BM25 rebuilds lazily."""
    global _bm25_version
    _bm25_version = -1


def _get_bm25():
    """Build (or reuse) a BM25 index over every chunk in the collection."""
    global _bm25, _bm25_version
    col = ingest.get_collection()
    count = col.count()
    if count == 0:
        return None
    with _lock:
        if _bm25 is not None and _bm25_version == count:
            return _bm25
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return None  # dependency missing -> vector-only
        data = col.get(include=["documents", "metadatas"])
        chunks = [{"id": i, "text": doc, "meta": meta}
                  for i, (doc, meta) in enumerate(zip(data["documents"],
                                                      data["metadatas"]))]
        bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        _bm25 = (bm25, chunks)
        _bm25_version = count
        return _bm25


def _vector_candidates(question: str, k: int) -> list[dict]:
    col = ingest.get_collection()
    if col.count() == 0:
        return []
    q_emb = ingest.get_embedder().encode([question]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=min(k, col.count()))
    return [{"text": d, "meta": m, "vec_score": 1 - dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                  res["distances"][0])]


def _bm25_candidates(question: str, k: int) -> list[dict]:
    idx = _get_bm25()
    if idx is None:
        return []
    bm25, chunks = idx
    scores = bm25.get_scores(_tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"text": chunks[i]["text"], "meta": chunks[i]["meta"],
             "bm25_score": float(scores[i])} for i in top if scores[i] > 0]


def _rrf_fuse(*ranked_lists, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion keyed on chunk identity."""
    fused: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            key = f"{item['meta'].get('doc_id')}:{item['meta'].get('chunk')}"
            if key not in fused:
                fused[key] = {**item, "rrf": 0.0}
            fused[key]["rrf"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


def _get_reranker():
    global _reranker
    if _reranker is None and config.RERANK_ENABLED:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(config.RERANK_MODEL, device=config.EMBED_DEVICE)
        except Exception:
            _reranker = False  # unavailable -> skip reranking
    return _reranker or None


def _rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    rr = _get_reranker()
    if not rr or len(candidates) <= 1:
        return candidates[:top_k]
    scores = rr.predict([(question, c["text"]) for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def _filter_relevant(candidates: list[dict], score_key: str,
                     min_score: float, max_gap: float | None) -> list[dict]:
    """Drop candidates that aren't actually relevant, using an absolute
    floor and (optionally) a floor relative to this query's best score.
    Returns [] if nothing clears the bar — callers should treat that the
    same as an empty repository, not force-cite the closest-available junk."""
    if not candidates:
        return []
    scored = [c for c in candidates if score_key in c]
    if not scored:
        return candidates  # scoring unavailable for these -> can't filter safely
    best = max(c[score_key] for c in scored)
    return [c for c in scored
            if c[score_key] >= min_score
            and (max_gap is None or (best - c[score_key]) <= max_gap)]


def retrieve(question: str, top_k: int = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    n = config.TOP_K_CANDIDATES

    vec = _vector_candidates(question, n)
    if not vec:
        return []

    if config.HYBRID_ENABLED:
        bm = _bm25_candidates(question, n)
        candidates = _rrf_fuse(vec, bm) if bm else vec
    else:
        candidates = vec

    final = _rerank(question, candidates[:n], top_k)

    if _get_reranker():
        final = _filter_relevant(final, "rerank_score",
                                 config.RERANK_MIN_SCORE, config.RERANK_MAX_GAP)
    else:
        # reranker unavailable -> fall back to an absolute vector-similarity
        # floor so retrieval still refuses to cite obviously-unrelated chunks
        final = _filter_relevant(final, "vec_score", config.VECTOR_MIN_SCORE, None)

    # normalize a display score for the UI
    for c in final:
        c["score"] = round(c.get("rerank_score", c.get("rrf", c.get("vec_score", 0))), 3)
    return final
