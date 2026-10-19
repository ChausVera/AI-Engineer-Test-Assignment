"""Reads the source files and builds the indexes the agent searches. Run once.

    python ingest.py --inspect        # no API calls: check page markers / chunk counts first
    python ingest.py                  # build everything
    python ingest.py --only services  # or: brochures
    python ingest.py --force          # redo extraction even if cached

Outputs in index/:
    brochure_chunks.json, brochure_vectors.npy
    service_records.json, service_vectors.npy
    extractions/<report>.json          (per-report cache so a rate-limit failure doesn't lose work)
"""
import argparse
import json
import re
import time
from typing import Optional

import numpy as np
from google.genai import types
from pydantic import BaseModel

import config
from retrieval import embed_texts

# ======================================================================
# Brochures: page-aware chunking
# ======================================================================
# ADJUST THIS after looking at one .md file (run --inspect). It matches lines like
# "--- Page 3 ---", "<!-- page 3 -->", "## Page 3", "[Page 3]".
PAGE_RE = re.compile(r"(?im)^[\s#>*\[<!\-=]*page\s*(\d+)\b[^\n]*$")
MAX_CHARS = 1200   # ~300 tokens: small enough to be specific, big enough to keep context
MIN_CHARS = 40


def product_name(path) -> str:
    return config.PRODUCT_NAMES.get(path.stem, re.sub(r"[_\-]+", " ", path.stem).strip())


def split_pages(text: str) -> list[tuple[Optional[int], str]]:
    pages, current, buf = [], None, []
    for line in text.splitlines():
        m = PAGE_RE.match(line)
        if m:
            if buf:
                pages.append((current, "\n".join(buf)))
            current, buf = int(m.group(1)), []
        else:
            buf.append(line)
    if buf:
        pages.append((current, "\n".join(buf)))
    return pages


def _split_long(p: str) -> list[str]:
    parts: list[str] = []
    for s in re.split(r"(?<=[.!?])\s+", p):
        parts.extend(s[i:i + MAX_CHARS] for i in range(0, len(s), MAX_CHARS))
    out, cur = [], ""
    for s in parts:
        if cur and len(cur) + len(s) + 1 > MAX_CHARS:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        out.append(cur)
    return out


def pack_paragraphs(text: str) -> list[str]:
    pieces: list[str] = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if p:
            pieces.extend([p] if len(p) <= MAX_CHARS else _split_long(p))
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > MAX_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= MIN_CHARS]


def chunk_brochures() -> list[dict]:
    chunks = []
    for f in sorted(config.BROCHURE_DIR.glob("*.md")):
        product = product_name(f)
        for page, page_text in split_pages(f.read_text(encoding="utf-8", errors="ignore")):
            for piece in pack_paragraphs(page_text):
                # The header is embedded WITH the text so "Larox PF" queries land on the right brochure.
                chunks.append({
                    "id": len(chunks),
                    "source": f.name,
                    "product": product,
                    "page": page,
                    "text": piece,
                    "embed_text": f"Product: {product} | Page {page}\n{piece}",
                })
    return chunks


def build_brochure_index(inspect_only: bool = False):
    chunks = chunk_brochures()
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        by_file.setdefault(c["source"], []).append(c)
    for name, cs in by_file.items():
        pages = sorted({c["page"] for c in cs if c["page"] is not None})
        gaps = [n for n in range(pages[0], pages[-1] + 1) if n not in pages] if pages else []
        no_page = sum(1 for c in cs if c["page"] is None)
        print(f"{name}: {len(cs)} chunks, pages {pages[:1]}..{pages[-1:]} ({len(pages)} found), "
              f"missing pages {gaps or 'none'}, chunks without page: {no_page}")
    if inspect_only:
        return
    print(f"Embedding {len(chunks)} brochure chunks...")
    vectors = embed_texts([c["embed_text"] for c in chunks], "RETRIEVAL_DOCUMENT", role="ingest")
    config.INDEX_DIR.mkdir(exist_ok=True)
    (config.INDEX_DIR / "brochure_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(config.INDEX_DIR / "brochure_vectors.npy", vectors)
    print("Brochure index saved.")


# ======================================================================
# Service reports: LLM extraction into a fixed schema
# ======================================================================
class ServiceRecord(BaseModel):
    equipment_model: Optional[str] = None     # e.g. "PF-DS"
    equipment_family: Optional[str] = None    # e.g. "Larox pressure filter"
    service_type: Optional[str] = None        # e.g. "filter cloth replacement"
    country: Optional[str] = None
    date: Optional[str] = None
    quantity: Optional[float] = None          # number of main items serviced (e.g. cloths)
    quantity_unit: Optional[str] = None
    labor_hours: Optional[float] = None
    technicians: Optional[int] = None
    parts_cost: Optional[float] = None
    labor_cost: Optional[float] = None
    travel_cost: Optional[float] = None
    total_cost: Optional[float] = None        # ONLY if the report states a total
    currency: Optional[str] = None            # ISO code or symbol exactly as the report uses it
    summary: str = ""                         # one neutral sentence about what was done


class ServiceRecordWithId(ServiceRecord):
    report_id: str = ""


class ExtractionBatch(BaseModel):
    records: list[ServiceRecordWithId]


EXTRACT_PROMPT = """Extract structured data from each field service report below.
Return exactly one record per report, with report_id copied exactly from its header.
Rules:
- Copy numbers exactly as written. Never estimate, round, convert currency or calculate totals.
- If a value is not stated in the report, use null.
- total_cost is only the total the report states; do not add up components yourself.
- summary: one neutral sentence describing the work done (no opinions).
- Reports are independent: never mix information between them.
"""


def extract_batch(reports: dict[str, str]) -> dict[str, dict]:
    """ONE LLM call for several reports -> {report_id: record}. Batching keeps us inside the daily request quota."""
    body = "\n".join(f"=== REPORT {rid} ===\n{text}\n" for rid, text in reports.items())
    resp = config.api_call("ingest", "gen", lambda: config.get_client("ingest").models.generate_content(
        model=config.EXTRACT_MODEL,
        contents=EXTRACT_PROMPT + "\n" + body,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionBatch,
        ),
    ))
    out = {}
    for rec in ExtractionBatch.model_validate_json(resp.text).records:
        if rec.report_id in reports:   # ignore ids the model made up
            d = rec.model_dump()
            d.pop("report_id", None)
            out[rec.report_id] = d
    return out


def build_service_index(force: bool = False):
    cache_dir = config.INDEX_DIR / "extractions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = {f.stem: f.read_text(encoding="utf-8", errors="ignore")
             for f in sorted(config.REPORT_DIR.glob("*.txt"))}
    todo = [rid for rid in texts if force or not (cache_dir / f"{rid}.json").exists()]

    size = max(1, config.EXTRACT_BATCH_SIZE)
    for s in range(0, len(todo), size):
        ids = todo[s:s + size]
        print(f"extracting reports {s + 1}-{s + len(ids)} of {len(todo)} (1 LLM call)")
        try:
            got = extract_batch({rid: texts[rid] for rid in ids})
        except config.DailyQuotaExceeded as e:
            print(f"  STOPPED: {e}\n  Building the index from the reports extracted so far.")
            break
        for rid, rec in got.items():
            (cache_dir / f"{rid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        missed = [rid for rid in ids if rid not in got]
        if missed:
            print(f"  WARNING: the model skipped {missed}; rerun ingest to retry only those")

    records = []
    for rid in texts:
        p = cache_dir / f"{rid}.json"
        if not p.exists():
            print(f"  (no extraction for {rid}, left out)")
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["report_id"] = rid
        records.append(rec)

    # Embed the service description so "filter cloth change" can match "cloth replacement".
    vec_texts = [f"{r.get('service_type') or ''} | {r.get('summary') or ''}" for r in records]
    vectors = embed_texts(vec_texts, "RETRIEVAL_DOCUMENT", role="ingest")
    (config.INDEX_DIR / "service_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(config.INDEX_DIR / "service_vectors.npy", vectors)
    missing = sum(1 for r in records if r.get("total_cost") is None)
    print(f"Service index saved: {len(records)} of {len(texts)} reports, {missing} without a stated total.")
    if len(records) < len(texts):
        print("  WARNING: some reports are not extracted yet. Rerun `python ingest.py --only services` to finish them.")


# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="chunk brochures only, no API calls")
    ap.add_argument("--only", choices=["brochures", "services"])
    ap.add_argument("--force", action="store_true", help="redo cached extractions")
    args = ap.parse_args()

    if args.inspect:
        build_brochure_index(inspect_only=True)
    else:
        if args.only in (None, "brochures"):
            build_brochure_index()
        if args.only in (None, "services"):
            build_service_index(force=args.force)