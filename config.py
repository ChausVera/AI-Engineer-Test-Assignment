"""Shared settings, API clients and the rate limiter.

Two API keys / roles:
  "ingest"  INGEST_API_KEY   a fresh free-tier account used only by ingest.py (falls back to GEMINI_API_KEY)
  "agent"   GEMINI_API_KEY   used by main.py: chat calls and one query-embedding per question

Free-tier limits seen in AI Studio: embeddings 100 requests/min, 30K tokens/min, 1K/day (every text in a
batch counts as a request); Gemini 3.6 Flash 5 requests/min, 250K tokens/min, 20 requests/day.
"""
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

# --- models (check AI Studio for current names; override via .env) ---
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # smaller than the 3072 default: plenty for a few hundred chunks
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", GEN_MODEL)
EXTRACT_BATCH_SIZE = int(os.getenv("EXTRACT_BATCH_SIZE", "8"))   # reports per LLM call: 50 reports -> 7 calls

# --- paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
BROCHURE_DIR = DATA_DIR / "brochures"
REPORT_DIR = DATA_DIR / "HistoricalServices"
INDEX_DIR = Path("index")   # built by ingest.py
LOG_DIR = Path("logs")      # tool-call traces
FLOTATION_CSV = Path(os.getenv("FLOTATION_CSV", DATA_DIR / "flotation_process_data_hourly.csv"))

# --- retrieval / agent ---
TOP_K = 5
RRF_K = 60                  # reciprocal-rank-fusion constant
MIN_DENSE_SCORE = 0.55      # below this best cosine score -> "no confident match". TUNE ON YOUR EVALS.
MAX_TOOL_STEPS = 5          # hard cap on tool-calling rounds per user message

# Display names (file stem -> name). Anything not listed falls back to the file name with _ and - as spaces.
PRODUCT_NAMES = {
    "ac-filter": "AC Filter",
    "cellstation": "CellStation",
    "columncell": "ColumnCell",
    "concorde-cell": "Concorde Cell",
    "filter-optimizer": "Filter Optimizer",
    "floatforce-upgrade": "FloatForce Upgrade",
    "gm-filter": "GM Filter",
    "larox-pf-brochure-a": "Larox PF (brochure A)",
    "larox-pf-brochure-b": "Larox PF (brochure B)",
    "larox-pf-ds": "Larox PF-DS",
    "md-series-pump": "MD Series Pump",
    "orion-series-pump": "Orion Series Pump",
}


# ---------------------------------------------------------------- clients
_clients: dict[str, genai.Client] = {}


def get_client(role: str = "agent") -> genai.Client:
    if role not in _clients:
        key = (os.getenv("INGEST_API_KEY") or os.environ["GEMINI_API_KEY"]) if role == "ingest" \
            else os.environ["GEMINI_API_KEY"]
        _clients[role] = genai.Client(api_key=key)
    return _clients[role]


# ---------------------------------------------------------------- rate limiting
def _budget(role: str, kind: str) -> tuple[int, int]:
    """(requests/min, tokens/min) we allow ourselves: a little under the real limits.
    Override with env vars, e.g. AGENT_GEN_RPM=60 or INGEST_EMBED_TPM=20000."""
    rpm, tpm = {"embed": (90, 25_000), "gen": (4, 200_000)}[kind]
    p = f"{role}_{kind}".upper()
    return int(os.getenv(f"{p}_RPM", rpm)), int(os.getenv(f"{p}_TPM", tpm))


class RateLimiter:
    """Sliding 60-second window over requests and tokens.

    wait() blocks until a call fits inside the budget, then records it. The window is saved to disk,
    so restarting a script right after a crash does not forget what was just used.
    """
    WINDOW = 60.0

    def __init__(self, name: str, rpm: int, tpm: int):
        self.rpm, self.tpm = rpm, tpm
        self.path = INDEX_DIR / f".rate_{name}.json"
        try:
            now = time.time()
            self.events = [e for e in json.loads(self.path.read_text(encoding="utf-8")) if now - e[0] < self.WINDOW]
        except Exception:
            self.events = []

    def wait(self, requests: int = 1, tokens: int = 0):
        while True:
            now = time.time()
            self.events = [e for e in self.events if now - e[0] < self.WINDOW]
            fits = (sum(e[1] for e in self.events) + requests <= self.rpm
                    and sum(e[2] for e in self.events) + tokens <= self.tpm)
            if fits or not self.events:   # a single oversized call is allowed through an empty window
                break
            time.sleep(self.events[0][0] + self.WINDOW - now + 0.2)   # until the oldest call leaves the window
        self.events.append([time.time(), requests, tokens])
        try:
            INDEX_DIR.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.events), encoding="utf-8")
        except OSError:
            pass


_limiters: dict[str, RateLimiter] = {}


def limiter(role: str, kind: str) -> RateLimiter:
    name = f"{role}_{kind}"
    if name not in _limiters:
        _limiters[name] = RateLimiter(name, *_budget(role, kind))
    return _limiters[name]


class DailyQuotaExceeded(RuntimeError):
    pass


def api_call(role: str, kind: str, fn, requests: int = 1, tokens: int = 0, tries: int = 6):
    """Run fn() under the rate limiter. Retries transient errors; stops at once on a daily quota error.

    role: "ingest" | "agent".  kind: "embed" | "gen".
    """
    for attempt in range(tries):
        limiter(role, kind).wait(requests, tokens)
        try:
            return fn()
        except errors.APIError as e:
            if e.code == 429 and "perday" in str(e).lower().replace(" ", "").replace("_", ""):
                raise DailyQuotaExceeded(
                    f"Daily quota used up for the '{role}' key ({kind}). It resets at midnight Pacific. "
                    "Progress is cached; rerun later.") from e
            if e.code not in (429, 500, 503) or attempt == tries - 1:
                raise
            time.sleep(min(2 ** attempt * 5, 60) + random.random())