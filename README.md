# Metso Sales Assistant Agent

A command-line chat agent for a salesperson on a call with a mining customer. It can:

1. **Answer product questions and recommend a product**, grounded in the product brochures.
2. **Estimate the cost of a field service visit**, computed from historical service reports.
3. **Assess a customer's flotation concentrate grades** (silica and iron) against benchmark plant data,
   computed from an hourly process dataset.

When the answer is not in its data, it says so. When required information is missing, it asks.

Part 3 is implemented as a benchmark comparison. A "deeper research" mode (dosing-timing analysis and a
list of extra data to request from the customer) is built but **switched off**; see
[Deeper research (disabled)](#deeper-research-disabled).

Built with the Gemini API (`google-genai`), plain Python, and no agent framework. This is a take-home
prototype, not a production system.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and fill in the keys (see [API keys](#api-keys)).
2. Put the data next to the code (it is not included in this repository):
   ```
   data/brochures/*.md                      # text extraction of the brochures
   data/HistoricalServices/*.txt            # field service reports
   data/flotation_process_data_hourly.csv   # hourly flotation plant data (part 3)
   ```
   The flotation file path can be overridden with `FLOTATION_CSV` in `.env`. If the file is missing, the
   chat still starts (parts 1 and 2 work); the agent then says it has no flotation data.
3. Build the indexes (once; parts 1 and 2 only, part 3 reads the CSV directly):
   ```bash
   python ingest.py --inspect       # optional, no API calls: check page detection and chunk counts
   python ingest.py                 # embeds brochures, extracts + embeds service reports
   ```
4. Chat:
   ```bash
   python main.py
   python main.py --verbose         # also prints every tool call and LLM call
   ```
   In the chat, `/reset` starts a new conversation and `/quit` exits.

Other ingest options: `--only brochures`, `--only services`, `--force` (redo cached extractions).

Flotation scripts, runnable without the chat and without any API calls:

```bash
python analysis.py data/flotation_process_data_hourly.csv          # data quality + lag/Granger checks
python assess_flotation.py 3.1 64.5 data/flotation_process_data_hourly.csv   # what the tool returns
```

### API keys

| Variable | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `main.py` (chat + one query embedding per question) | required |
| `INGEST_API_KEY` | `ingest.py` | optional; falls back to `GEMINI_API_KEY`. Lets ingestion use a separate account/quota. |

Model names are set in `config.py` and can be overridden in `.env` (`GEN_MODEL`, `EMBED_MODEL`, `EXTRACT_MODEL`).

## Architecture

```
OFFLINE (ingest.py, run once, cached)
  brochures/*.md  -> split by page, chunk -> embed -> index/brochure_*.{json,npy}
  reports/*.txt   -> LLM extraction into a fixed schema -> index/service_records.json
                  -> embed each report's service description -> index/service_vectors.npy

RUNTIME (main.py)
  user message -> router.py: Gemini + system prompt + tool declarations
                    |-- search_brochures(query, product?, diverse?)       -> products.py -> retrieval.py
                    |-- estimate_service_cost(equipment, service_type,
                    |                         country, quantity?)         -> services.py
                    |-- assess_flotation(silica_concentrate,
                    |                    iron_concentrate)                -> assess_flotation.py -> analysis.py
                    '-- (answer directly / ask a clarifying question: plain text, no tool)
                  -> grounding check on the draft answer -> reply
```

| File | Role |
|---|---|
| `main.py` | CLI loop and conversation handling |
| `router.py` | System prompt, tool declarations, the tool-calling loop, the grounding check |
| `retrieval.py` | Embeddings (with disk cache) and hybrid search |
| `products.py` | Brochure search tool: returns evidence or an explicit "no confident match" |
| `services.py` | Cost estimation from historical records, computed in plain Python |
| `assess_flotation.py` | Flotation tool: benchmark comparison of the customer's concentrate grades |
| `analysis.py` | Flotation data loader, data quality report, and verification of the lag relationships |
| `ingest.py` | Reads source files, builds the indexes |
| `config.py` | Settings, API clients, rate limiter |

## How it works

### Routing

There is no separate classifier. Gemini's function calling is the router: for each message the model
chooses to answer, ask a question, or call one or more tools. `router.py` runs the loop itself
(automatic function calling is disabled), which allows:

- a cap of 5 tool rounds per message (the last round forbids tool calls, so a reply is always produced),
- a JSONL trace of every tool call in `logs/trace.jsonl`,
- a check on the final answer (below).

The system prompt is the behavioural contract: product facts only from brochure search results, costs
only from the cost tool, flotation findings only from the flotation tool, quote figures exactly, cite
brochure and page, ask one short question when information is missing, and say so plainly when a tool
finds nothing.

### Product questions and recommendations

- Brochures are split on page markers into chunks of about 1,200 characters. Each chunk is embedded
  together with a `Product | Page` header, so queries mentioning a product land on the right brochure.
- Search is **hybrid**: Gemini embeddings (cosine similarity) plus BM25, merged with reciprocal rank
  fusion. Embeddings capture meaning; BM25 catches exact model names such as "PF-DS".
- If the best cosine score is below `MIN_DENSE_SCORE`, the tool returns `no_confident_match`. This is
  what drives the "this is not in the brochures" answer.
- For recommendations the model sets `diverse=true`, which returns at most one passage per brochure so
  several products can be compared.

### Service cost estimates

- **Ingest:** an LLM reads the reports once (8 per call) into a fixed schema (equipment, service type,
  country, quantity, parts/labor/travel/total cost, currency, summary). It is told to copy numbers
  exactly, use null for anything not stated, and never calculate.
- **Estimate (no LLM arithmetic):** the tool first validates that equipment, service type and country
  are present. If not, it returns `missing_fields` and the model asks the user. It then scores each past
  visit by equipment match (rule-based), service-type similarity (embeddings) and country, keeps the
  comparable ones, and computes min / median / max **per currency** (currencies are never mixed).
- The result includes `used_report_ids` (exactly the reports behind the statistics), the reports left
  out for lacking a total, a confidence level based on the best-supported single currency, and caveats
  generated in code: cases from other countries, cases from a different model in the same family,
  derived totals, quantity mismatch, few cases per currency. With no comparable case it returns
  `no_comparable_cases` instead of a number.

### Flotation assessment

Used when the salesperson gives the customer's concentrate grades (for example "3.1% silica and
64.5% iron, and they think that is normal"). The data is a public dataset of six months of hourly
readings from a Brazilian iron-ore flotation plant (March to September 2017, 4,097 hourly rows).
Everything numeric is computed by `assess_flotation.py` with pandas and NumPy; the model never does
the arithmetic and only writes the answer from the returned figures.

The tool (`assess_flotation(silica_concentrate, iron_concentrate)`) returns:

- **Benchmark position:** the share of benchmark hours with lower silica and with higher iron, the
  benchmark medians, the silica expected at the customer's iron grade (a linear fit of silica on iron in
  the concentrate) with the customer's deviation from it in standard deviations, and the median silica
  in hours with a similar iron grade (within 0.5 points).
- **Limitations** that the answer must state: the benchmark is a different plant, and there is no
  tailings data, so recovery cannot be assessed.

Both grades are required; if one is missing the model asks for it. If the CSV is not found the tool
returns `data_unavailable` and the agent says it has no flotation data. Data quality is deliberately not
part of the tool's output; it is checked in `analysis.py`.

### Verifying the data and the relationships (`analysis.py`)

`analysis.py` is run by hand and is how the flotation data was validated.

- **Data quality report:** row count and date range, duplicate timestamps, missing hours (on the
  benchmark file: 318 of 4,415 hours, all from one 13-day gap in March, handled by working on a full
  hourly index), missing values in the columns used (none), and columns dominated by a single value
  (none).
- **Lag correlations:** amina flow against concentrate silica at several lags, in both directions.
  On this file the values reproduce those reported in the published research on this dataset
  (silica leading amina: 0.194 at 1 h, 0.230 at 2 h, 0.255 at 3 h; amina leading silica: 0.145 at 1 h,
  0.052 at 6 h), 5 of 5 reference values within tolerance.
- **Granger-style check:** whether the past of one series improves the prediction of the other beyond its
  own history, in levels and in first differences (to remove slow shared drift). Silica's past predicts
  amina 6.8 to 14.1 times better (in R² gain) than the reverse, so the data is consistent with dosing that
  follows silica by a few hours. The effect is small (about 1 to 2 percent of the variation in amina
  flow), and it is a predictive lead, not proof of cause.

### Deeper research (disabled)

A second mode of the flotation tool is implemented but commented out. When enabled, the agent would
first answer from the benchmark and ask whether the salesperson wants deeper research. On a yes, it
would call the tool again with `deeper_research=true`, which adds the timing findings above (with their
limitations) and a prioritised list of extra data to request from the customer: operator and control
logs, lab sampling versus reporting times, the customer's own hourly data, tailings data, ore
characteristics, reagent and water details, and higher-frequency data.

It is disabled to keep the live answer short and tied to the data. To re-enable it, uncomment the marked
blocks in `assess_flotation.py` (the `deeper_research` parameter and the branch at the end of
`assess_flotation()`), add the parameter back to the wrapper and the declaration in `router.py`, and
restore the deeper-research rule in the system prompt.

### Grounding check

After each answer, `router.py` extracts the numbers from the text and checks that each appears in this
turn's tool results or in the user's own messages. If not, the model is asked once to rewrite the
answer; if it still contains unverified numbers, they are flagged at the end of the reply. This catches
invented figures. It does not catch misread ones: it checks presence, not meaning.

### Rate limits and caching

The code was written for free-tier limits (embeddings: 100 requests/min, 30K tokens/min, 1K/day;
Gemini 3.6 Flash: 5 requests/min, 20/day). `config.py` has a sliding-window rate limiter (state saved
to disk) that keeps calls under a budget slightly below those limits. A daily-quota error stops
ingestion cleanly and keeps the progress made. Every embedding and every report extraction is cached
on disk, so reruns cost no API calls. Budgets can be overridden via environment variables, for example
`AGENT_GEN_RPM=60`. A flotation question costs two LLM calls (choose the tool, write the answer), or
three if the grounding check asks for a rewrite.

## Example prompts

- "We have Larox pressure filters but performance drops whenever our feed grade changes"
- "A customer in Chile needs 12 filter cloths replaced on a Larox PF-DS tower press. What should I budget for the service visit?"
- "The customer is getting 3.1% silica and 64.5% iron in their concentrate. They think that's normal. What do you think?"

## Known limitations

- **Heuristics are not tuned.** `MIN_DENSE_SCORE`, `MIN_COMPARABLE`, the similarity weights and the
  cosine scaling in `services.py` are starting values. They should be tuned on real scores and test questions.
- **Extraction quality decides cost quality.** `index/service_records.json` should be spot-checked
  against the original reports (currencies and totals in particular).
- **Page detection** assumes the page-marker format in `PAGE_RE` (`ingest.py`); `python ingest.py --inspect`
  shows missing pages and chunks without a page.
- **Product display names** are set in `config.PRODUCT_NAMES` (derived from file names).
- **Flattened tables** in the text extraction can lose structure; specifications that live in tables may be
  incomplete or garbled.
- **No automated evaluation set** is included yet.
- The grounding check verifies that figures exist in the tool results, not that they are used correctly.
- **Flotation benchmark is one plant.** The data comes from a different plant, years earlier, so the
  comparison is a reference point and not a diagnosis of the customer's process. It is an hourly aggregate
  of the raw measurements.
- **No tailings data,** so recovery and mass yield cannot be computed. Any suggestion to raise grade may
  cost recovery.
- **Flotation timing results are weak and not causal** (correlations of about 0.25 at best), and they were
  not controlled for feed quality. A natural next step is a version of the Granger-style check that also
  conditions on feed silica.
- **Flotation tool has no automated tests** beyond running the scripts above on the real file.

## Scaling to much more data

- **Brochures:** replace the NumPy file with a vector database (for example pgvector or Qdrant), add a
  document-level index in front of chunk search, and a reranker after it.
- **Service reports:** extraction becomes a batch pipeline; the estimator moves to SQL or a dataframe
  with the same logic.
- **Flotation:** the CSV is read on each call. For larger or live plant data, move it to a database or
  a cached dataframe and compute the benchmark statistics on a schedule.
- **Quality:** a growing evaluation set, and re-embedding whenever the embedding model changes (the
  cache key includes the model name).

## Generated folders (do not commit)

`index/` (indexes, caches, rate-limit state), `logs/` (tool-call traces), `data/`, `.env`.
