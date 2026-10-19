"""Embeddings and semantic search.

Hybrid retrieval: dense (Gemini embeddings, cosine) + sparse (BM25), merged with
reciprocal rank fusion. Dense finds meaning ("performance drops when feed changes"),
BM25 catches exact model names ("PF-DS") that embeddings tend to blur.
"""
import hashlib
import json
import re

import numpy as np
from google.genai import types
from rank_bm25 import BM25Okapi

import config


# ---------------------------------------------------------------- embeddings
# Every embedded text is cached on disk (keyed by model + task type + text). Re-running ingest, or
# repeating a demo question, costs zero API calls, which matters with a daily request quota.
_cache: dict | None = None


def _cache_file():
    return config.INDEX_DIR / "embed_cache.json"


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        p = _cache_file()
        _cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _cache


def _save_cache():
    config.INDEX_DIR.mkdir(exist_ok=True)
    _cache_file().write_text(json.dumps(_cache), encoding="utf-8")


def _key(text: str, task_type: str) -> str:
    raw = f"{config.EMBED_MODEL}|{config.EMBED_DIM}|{task_type}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def embed_texts(texts: list[str], task_type: str, role: str = "agent",
                batch_size: int = 20) -> np.ndarray:
    """Embed texts -> (n, dim) array of L2-normalised vectors.

    task_type: "RETRIEVAL_DOCUMENT" for things you index, "RETRIEVAL_QUERY" for questions.
    role:      which API key / rate-limit budget to use ("ingest" or "agent").
    - Cached: only texts never embedded before hit the API.
    - Rate-limited: config.api_call keeps us under the requests/min and tokens/min budgets.
    - Normalised by us because reduced-dimension outputs are not unit length.
    """
    cache = _load_cache()
    keys = [_key(t, task_type) for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]

    for start in range(0, len(todo), batch_size):
        idxs = todo[start:start + batch_size]
        batch = [texts[i] for i in idxs]
        est_tokens = int(sum(len(t) for t in batch) / 3)   # pessimistic: ~3 characters per token
        resp = config.api_call(
            role, "embed",
            lambda b=batch: config.get_client(role).models.embed_content(
                model=config.EMBED_MODEL,
                contents=b,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBED_DIM,
                ),
            ),
            requests=len(batch), tokens=est_tokens,
        )
        for i, e in zip(idxs, resp.embeddings):
            cache[keys[i]] = [round(float(x), 6) for x in e.values]
        _save_cache()   # saved per batch, so a crash or quota error loses nothing
        if len(todo) > 1:
            print(f"  embedded {min(start + batch_size, len(todo))}/{len(todo)}")

    arr = np.array([cache[k] for k in keys], dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


# ---------------------------------------------------------------- tokenizing
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; 'pf-ds' yields 'pf-ds', 'pf' and 'ds'."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "-" in tok:
            out.extend(tok.split("-"))
    return out


# ---------------------------------------------------------------- the index
class BrochureIndex:
    def __init__(self):
        self.chunks = json.loads((config.INDEX_DIR / "brochure_chunks.json").read_text(encoding="utf-8"))
        self.vectors = np.load(config.INDEX_DIR / "brochure_vectors.npy")
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in self.chunks])

    def products(self) -> list[str]:
        return sorted({c["product"] for c in self.chunks})

    def match_product(self, name: str) -> set[str]:
        """Products whose name contains every word of `name` (forgiving filter)."""
        words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
        want = words(name)
        return {p for p in self.products() if want <= words(p)}

    def search(self, query: str, k: int = config.TOP_K, product: str | None = None,
               per_product: int | None = None) -> dict:
        """Return {"results": [...], "best_dense": float}.

        product      -> restrict to matching brochure(s)
        per_product  -> cap hits per brochure (use 1 to compare several products)
        """
        if product:
            allowed_names = self.match_product(product)
            allowed = np.array([i for i, c in enumerate(self.chunks) if c["product"] in allowed_names])
        else:
            allowed = np.arange(len(self.chunks))
        if len(allowed) == 0:
            return {"results": [], "best_dense": 0.0}

        qv = embed_texts([query], "RETRIEVAL_QUERY")[0]
        dense = self.vectors[allowed] @ qv                           # cosine (vectors are unit length)
        sparse = self.bm25.get_scores(tokenize(query))[allowed]

        fused = np.zeros(len(allowed))
        for rank, j in enumerate(np.argsort(-dense)):
            fused[j] += 1.0 / (config.RRF_K + rank + 1)
        for rank, j in enumerate(np.argsort(-sparse)):
            if sparse[j] > 0:                                        # ignore chunks with no keyword overlap
                fused[j] += 1.0 / (config.RRF_K + rank + 1)

        results, per_count = [], {}
        for j in np.argsort(-fused):
            c = self.chunks[allowed[j]]
            if per_product and per_count.get(c["product"], 0) >= per_product:
                continue
            per_count[c["product"]] = per_count.get(c["product"], 0) + 1
            results.append({
                "source": c["source"],
                "product": c["product"],
                "page": c["page"] if c["page"] is not None else "unknown",
                "score": round(float(dense[j]), 3),
                "text": c["text"],
            })
            if len(results) >= k:
                break
        return {"results": results, "best_dense": float(dense.max())}