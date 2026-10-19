"""Historical service-cost estimates.

The LLM never does arithmetic here. Records were extracted once (ingest.py) into
index/service_records.json; this module matches comparable past visits, computes
statistics in plain Python, and returns caveats for the model to pass on.
"""
import json
import re
import statistics

import numpy as np

import config
from retrieval import embed_texts

# Heuristic similarity weights (they sum to 1). Tunable, and not a trained model.
W_EQUIP, W_SERVICE, W_COUNTRY = 0.40, 0.40, 0.20
MIN_COMPARABLE = 0.55

REQUIRED = ("equipment", "service_type", "country")

_cache: tuple[list[dict], np.ndarray] | None = None


def _load() -> tuple[list[dict], np.ndarray]:
    global _cache
    if _cache is None:
        records = json.loads((config.INDEX_DIR / "service_records.json").read_text(encoding="utf-8"))
        vectors = np.load(config.INDEX_DIR / "service_vectors.npy")
        _cache = (records, vectors)
    return _cache


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _equipment_match(query: str, rec: dict) -> tuple[float, str]:
    q, model, family = _norm(query), _norm(rec.get("equipment_model")), _norm(rec.get("equipment_family"))
    if q and model and (q in model or model in q):
        return 1.0, "exact"
    if q and family and (family in q or q in family):
        return 0.5, "same_family"
    return 0.0, "none"


def _total(rec: dict) -> tuple[float | None, bool]:
    """(total, was_derived). Use the stated total; derive only if all three parts exist."""
    if rec.get("total_cost") is not None:
        return float(rec["total_cost"]), False
    parts = [rec.get("parts_cost"), rec.get("labor_cost"), rec.get("travel_cost")]
    if all(p is not None for p in parts):
        return float(sum(parts)), True
    return None, False


def estimate_service_cost(equipment: str | None = None, service_type: str | None = None,
                          country: str | None = None, quantity: float | None = None) -> dict:
    # 1) Validate inputs in code, so asking for missing info doesn't depend on the prompt alone.
    given = {"equipment": equipment, "service_type": service_type, "country": country}
    missing = [f for f in REQUIRED if not (given[f] and str(given[f]).strip())]
    if missing:
        return {"status": "missing_fields", "missing": missing,
                "message": "Ask the user for these before estimating."}

    records, vectors = _load()

    # 2) Score every past visit: rule-based equipment + country, embedding-based service type.
    qv = embed_texts([service_type], "RETRIEVAL_QUERY")[0]
    sims = vectors @ qv
    scored = []
    for rec, sim in zip(records, sims):
        equip_score, equip_label = _equipment_match(equipment, rec)
        service_score = min(max((float(sim) - 0.5) / 0.4, 0.0), 1.0)   # map cosine ~0.5-0.9 -> 0-1
        same_country = _norm(country) == _norm(rec.get("country"))
        score = W_EQUIP * equip_score + W_SERVICE * service_score + W_COUNTRY * (1.0 if same_country else 0.0)
        total, derived = _total(rec)
        scored.append({
            "report_id": rec["report_id"],
            "score": round(score, 3),
            "equipment_model": rec.get("equipment_model"),
            "equipment_match": equip_label,
            "service_type": rec.get("service_type"),
            "country": rec.get("country"),
            "same_country": same_country,
            "date": rec.get("date"),
            "quantity": rec.get("quantity"),
            "quantity_unit": rec.get("quantity_unit"),
            "total_cost": total,
            "total_derived": derived,
            "currency": rec.get("currency"),
        })
    scored.sort(key=lambda r: -r["score"])
    comparable = [r for r in scored if r["score"] >= MIN_COMPARABLE]

    # 3) Nothing comparable -> say so, show the closest cases for transparency.
    if not comparable:
        return {"status": "no_comparable_cases",
                "message": "No past service report is a close match. Do not invent a figure.",
                "closest_cases": scored[:3]}

    # 4) Statistics per currency (never mix currencies).
    with_total = [r for r in comparable if r["total_cost"] is not None and r["currency"]]
    by_currency: dict[str, dict] = {}
    for cur in sorted({r["currency"] for r in with_total}):
        vals = [r["total_cost"] for r in with_total if r["currency"] == cur]
        by_currency[cur] = {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals)}

    n = len(with_total)
    # Confidence rests on the best-supported single currency: figures are never pooled across currencies.
    best_n = max((c["n"] for c in by_currency.values()), default=0)
    confidence = "high" if best_n >= 5 else "medium" if best_n >= 3 else "low" if best_n >= 1 else "none"

    # 5) Caveats the model must pass on.
    caveats = []
    if n < len(comparable):
        caveats.append(f"{len(comparable) - n} comparable report(s) had no usable total cost and were left out of the statistics.")
    other_country = [r for r in comparable if not r["same_country"]]
    if other_country:
        caveats.append(f"{len(other_country)} of {len(comparable)} comparable cases are from other countries (travel and local costs may differ).")
    not_exact = [r for r in comparable if r["equipment_match"] != "exact"]
    if not_exact:
        caveats.append(f"{len(not_exact)} of {len(comparable)} comparable cases are a different model in the same "
                       f"product family, not {equipment}.")
    if any(r["total_derived"] for r in with_total):
        caveats.append("Some totals were derived by summing parts, labor and travel because no total was stated.")
    if len(by_currency) > 1:
        caveats.append("Cases use different currencies; they are NOT converted. Report each separately.")
    if quantity:
        qtys = [r["quantity"] for r in comparable if r["quantity"]]
        if qtys and not (0.5 <= statistics.median(qtys) / float(quantity) <= 2.0):
            caveats.append(f"Past visits typically involved a different quantity (median {statistics.median(qtys):g}) than requested ({quantity:g}); totals are not scaled.")
    if best_n <= 2:
        caveats.append("Very few comparable cases per currency: treat each figure as a rough indication only.")
    used_ids = {r["report_id"] for r in with_total}

    return {
        "status": "ok",
        "request": {**given, "quantity": quantity},
        "n_comparable": len(comparable),
        "n_with_total": n,
        "confidence": confidence,
        "totals_by_currency": by_currency,
        "used_report_ids": [r["report_id"] for r in with_total],   # exactly the reports behind the statistics
        "cases_used": with_total[:8],                              # details for the closest few
        "excluded_no_total": [r["report_id"] for r in comparable if r["report_id"] not in used_ids],
        "caveats": caveats,
    }