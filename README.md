# Craival — AI-Powered Restaurant Recommendation System

> **Zomato Milestone 1 AI** · A zero-cost, two-stage **Retrieval-Augmented Generation (RAG)** restaurant recommender for Bangalore. It combines structured filtering (SQLite), semantic search (ChromaDB), and an LLM (via the OpenAI-compatible Groq/xAI API) to produce ranked, explained recommendations — with a premium Zomato-inspired web UI.

---

## What it does

Give it your preferences — in plain English ("cheap Italian near Koramangala, rated 4+") or via filters (Location, Budget, Cuisine, Rating) — and it returns the top picks with a short, personalized reason for each. Every recommended restaurant is cross-checked against the real dataset, so the AI never invents a place.

- **Natural-language or structured input**
- **Two-stage retrieval**: hard filters narrow thousands → dozens; semantic search ranks the best matches
- **LLM ranking + explanations** with strict anti-hallucination validation
- **Graceful fallback** to rule-based ranking if the LLM is unavailable
- **Conversational follow-ups** ("cheaper options", "show me something else")
- **Response caching** and **structured request logging**
- **Premium web UI**: dark/light theme, glassmorphism cards, restaurant detail pages, mobile-optimized

---

## Architecture (high level)

```
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────┐
│  User input  │──▶│  Input Parser    │──▶│  Two-Stage      │──▶│  LLM         │
│ (NL or form) │   │  (entities)      │   │  Retrieval      │   │  Recommender │
└──────────────┘   └──────────────────┘   │  SQLite+Chroma  │   │  (+fallback) │
                                           └─────────────────┘   └──────┬───────┘
                                                                        │
                          ┌─────────────────────────────────────────────┘
                          ▼
                 ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐
                 │  Orchestrator    │──▶│  Cache +     │──▶│  Web UI /    │
                 │  (pipeline glue) │   │  Session +   │   │  CLI output  │
                 └──────────────────┘   │  JSONL logs  │   └──────────────┘
                                        └──────────────┘
```

Full design: [`Docs/architecture.md`](Docs/architecture.md) · Context & dataset notes: [`Docs/context.md`](Docs/context.md)

---

## Quick start (< 5 minutes)

```bash
# 1. Clone and enter the project
cd "Zomato Milestone 1 AI"

# 2. Create a virtual environment (Python >= 3.10)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
copy .env.example .env       # Windows  (cp on macOS/Linux)
# then edit .env and set XAI_API_KEY to a free Groq key from https://console.groq.com

# 5. Run the web app  (data stores build automatically on first run)
python -m src.web
# open http://127.0.0.1:5000
```

> **First run builds the data stores automatically.** If `data/restaurants.db` or
> `data/chroma_store/` are missing, the app downloads/caches the dataset and builds
> both stores once (a few minutes, one-time). Subsequent runs start instantly.

### Command-line interface (optional)
```bash
python -m src.app            # interactive prompt
python -m src.app --smoke    # runs 5 sample queries end-to-end
```

---

## Demo queries to try

| Query | Showcases |
|-------|-----------|
| `cheap Italian in Koramangala` | Natural-language parsing + budget/cuisine/location |
| `upscale dining for a date night in Indiranagar` | Semantic search on ambiance ("date night" → romantic) |
| `family-friendly place in BTM, medium budget` | Preference synonyms + budget tier |
| `best rated biryani in Koramangala` | Rating sort + dish/cuisine matching |
| `mongolian under ₹50 rated 4.9 in Koramangala` | Progressive relaxation (over-constrained → relaxed with notice) |

In the UI you can also click any card to open its **detail page**, give 👍/👎 feedback, and use **Refine Search** for conversational follow-ups.

---

## Tech stack (100% free / open-source)

| Layer | Technology | Cost |
|-------|-----------|------|
| Language | Python ≥ 3.10 | Free |
| Structured store | SQLite (stdlib) + FTS5 | Free |
| Vector store | ChromaDB | Free (OSS) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Free (OSS) |
| LLM | Groq / xAI via the OpenAI-compatible SDK | Free tier |
| Data | Zomato dataset (HuggingFace) | Free |
| Web | Flask + Tailwind (CDN) + vanilla JS | Free |
| Fuzzy matching | `thefuzz` | Free |
| Tests | `pytest`, `pytest-cov` | Free |

---

## Project layout

```
src/
  data_loader.py       # download + clean dataset
  data_layer.py        # SQLite + ChromaDB stores and queries
  input_parser.py      # NL / structured → UserPreferences
  retrieval_engine.py  # two-stage retrieval + relaxation + context guard
  prompt_builder.py    # template-driven CoT prompts
  recommender.py       # LLM adapter, parsing, anti-hallucination, fallback
  orchestrator.py      # end-to-end pipeline
  cache.py             # TTL + LRU response cache
  session_manager.py   # conversational session memory
  logger.py            # structured JSONL request logs
  logging_config.py    # console logging setup
  app.py               # CLI entry point + first-run data bootstrap
  web.py               # Flask web app (UI + JSON API)
prompts/               # editable prompt templates (.txt)
templates/ static/     # web UI
tests/                 # pytest suite + evaluate.py
Docs/                  # architecture, context, edge cases, plan
```

---

## Testing & evaluation

```bash
pytest                       # run the full suite (74 tests)
pytest --cov=src             # with coverage (≈93% on core modules)
python -m tests.evaluate     # quality/perf metrics against the real LLM
```

**Latest evaluation results** (12 representative queries):

| Metric | Result | Target |
|--------|--------|--------|
| Hallucination rate | **0%** | 0% |
| Fallback rate | 0% | graceful |
| Cache hit latency | ~0–10 ms | < 100 ms |
| End-to-end latency (fresh) | ~6 s avg | < 5 s* |

\* Fresh-query latency is bound by the free LLM's response time (retrieval itself is ~0.5 s; cached queries return in ~0 ms).

---

## Configuration (`.env`)

| Variable | Purpose | Default |
|----------|---------|---------|
| `XAI_API_KEY` | LLM API key (Groq/xAI) | — (required) |
| `XAI_BASE_URL` | OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Model name | `llama-3.3-70b-versatile` |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | Generation params | `0.4` / `1024` |
| `DB_PATH` / `VECTOR_DB_PATH` / `DATA_PATH` | Data store paths | under `data/` |
| `CACHE_ENABLED` / `CACHE_TTL` | Response cache | `true` / `3600` |
| `SESSION_ENABLED` / `SESSION_TTL` | Conversational memory | `true` / `1800` |
| `LOG_LEVEL` / `LOG_PATH` | Console level / JSONL log path | `INFO` / `logs/queries.jsonl` |

---

## Notes & limitations

- Data scope is **Bangalore** restaurants (per the source dataset).
- The bundled web server is Flask's development server — fine for the demo. For production use a WSGI server (e.g. `waitress` / `gunicorn`).
- Recommendations are validated against the dataset; if the LLM proposes anything unmatched, it's dropped (zero hallucinations).
- `.env` (your API key) and the generated `data/`, `logs/` are git-ignored. Never commit your real key; rotate it if it was ever shared.
