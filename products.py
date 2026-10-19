"""Brochure Q&A tool.

This file does NOT write answers. It returns evidence (chunks with brochure + page),
or an explicit "no confident match" signal. The model writes the answer from the
evidence, and the system prompt forbids using anything else for product facts.
"""
import config
from retrieval import BrochureIndex

_index: BrochureIndex | None = None


def get_index() -> BrochureIndex:
    global _index
    if _index is None:
        _index = BrochureIndex()
    return _index


def product_names() -> list[str]:
    return get_index().products()


def search_brochures(query: str, product: str | None = None, diverse: bool = False) -> dict:
    idx = get_index()

    if product and not idx.match_product(product):
        return {
            "status": "unknown_product",
            "message": f"No brochure matches '{product}'.",
            "available_products": idx.products(),
        }

    # diverse=True -> at most one chunk per brochure, so the model can compare products
    out = idx.search(query, k=config.TOP_K + 1 if diverse else config.TOP_K,
                     product=product, per_product=1 if diverse else None)

    if not out["results"] or out["best_dense"] < config.MIN_DENSE_SCORE:
        return {
            "status": "no_confident_match",
            "message": "The brochures do not appear to contain information on this.",
            "best_score": round(out["best_dense"], 3),
        }
    return {"status": "ok", "results": out["results"]}