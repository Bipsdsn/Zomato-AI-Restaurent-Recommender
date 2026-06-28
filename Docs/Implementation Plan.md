# Implementation Plan — AI-Powered Restaurant Recommendation System

> **Project:** Zomato Milestone 1 AI  
> **Architecture Pattern:** Two-Stage RAG with Semantic Search  
> **Total Estimated Effort:** ~40–50 hours (solo developer)  
> **Constraint:** Zero-cost stack (Grok API free credits, all OSS)  
> **Last Updated:** 2026-06-17

---

## Plan Overview

| Phase | Name | Est. Hours | Dependencies | Key Deliverable |
|-------|------|-----------|--------------|-----------------|
| 0 | Project Setup & Environment | 2–3 | None | Runnable project skeleton |
| 1 | Data Ingestion & Storage Layer | 6–8 | Phase 0 | SQLite + ChromaDB populated |
| 2 | Input Parser Module | 4–5 | Phase 1 | Structured entity extraction |
| 3 | Two-Stage Retrieval Engine | 5–6 | Phase 1 | Hard filter + semantic rank |
| 4 | Prompt Engineering Module | 3–4 | Phase 3 | Template-driven CoT prompts |
| 5 | LLM Recommender (Grok Adapter) | 4–5 | Phase 4 | Working LLM call + validation |
| 6 | Orchestrator & Integration | 4–5 | Phases 2–5 | End-to-end pipeline working |
| 7 | Caching & Session Management | 3–4 | Phase 6 | TTL cache + conversational follow-ups |
| 8 | Observability & Logging | 2–3 | Phase 6 | Structured query logs |
| 9 | Presentation Layer — Premium Web UI | 8–10 | Phase 6 | Stunning Flask + HTML/CSS/JS frontend |
| 10 | Testing & Evaluation | 4–5 | Phase 6 | Unit + integration tests |
| 11 | Polish, Docs & Submission | 2–3 | All | README, demo, submission-ready |

---

## Phase 0: Project Setup & Environment

**Goal:** Runnable project skeleton with all dependencies installable in one command.

### Tasks

1. **Create project directory structure**
   ```
   Zomato Milestone 1 AI/
   ├── Docs/
   ├── data/
   ├── prompts/
   ├── src/
   │   └── __init__.py
   ├── templates/
   ├── static/
   ├── logs/
   ├── tests/
   ├── .env.example
   ├── .gitignore
   ├── requirements.txt
   └── README.md
   ```

2. **Set up Python virtual environment**
   - Python ≥ 3.10
   - Create `venv` and `requirements.txt`

3. **Install core dependencies**
   ```
   pandas>=2.0
   datasets>=2.14
   openai>=1.0
   python-dotenv>=1.0
   chromadb>=0.4
   sentence-transformers>=2.2
   thefuzz>=0.20
   ```

4. **Configure environment variables**
   - Create `.env.example` with all config keys (see architecture §9c)
   - Register for Grok API key at [console.x.ai](https://console.x.ai)
   - Set `XAI_API_KEY`, `XAI_BASE_URL`, `LLM_MODEL` etc.

5. **Initialize Git repo**
   - Add `.gitignore` (exclude `.env`, `data/`, `__pycache__/`, `logs/`, `chroma_store/`)
   - Initial commit

6. **Create `src/models.py`** with shared data classes
   - `UserPreferences`
   - `RetrievalResult`
   - `Recommendation`
   - `RecommendationResponse`

### Acceptance Criteria
- [ ] `pip install -r requirements.txt` succeeds without errors
- [ ] `python -c "from src.models import UserPreferences"` imports cleanly
- [ ] `.env` loaded via `python-dotenv` and API key accessible
- [ ] Git repo initialized with clean first commit

### Output Files
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `src/__init__.py`
- `src/models.py`
- `README.md` (basic setup instructions)

---

## Phase 1: Data Ingestion & Storage Layer

**Goal:** Load, clean, and persist Zomato dataset into both SQLite (structured queries) and ChromaDB (vector search).

### Tasks

#### 1a. Dataset Download & Exploration (1–2 hrs)

1. **Download dataset** from HuggingFace
   ```python
   from datasets import load_dataset
   ds = load_dataset("ManikaSaini/zomato-restaurant-recommendation")
   ```
2. **Explore schema** — identify columns, nulls, data types, value distributions
3. **Save local CSV cache** at `data/zomato.csv` for offline use

#### 1b. Data Preprocessing (`data_loader.py`) (2–3 hrs)

1. **Parse `rate` column**
   - Extract numeric value from `"4.1/5"` format
   - Handle special values: `"NEW"` → flag as `is_new=True`, set rate to None
   - Handle `"-"` and null → set rate to None
   - Drop rows where both `rate` and `approx_cost` are null

2. **Parse `approx_cost` column**
   - Remove commas: `"1,200"` → `1200`
   - Convert to integer
   - Handle null → drop row

3. **Normalize `location`**
   - Lowercase, strip whitespace
   - Standardize common variants (e.g., `" BTM "` → `"btm"`)

4. **Split `cuisines`**
   - `"North Indian, Chinese"` → `["north indian", "chinese"]`
   - Normalize casing

5. **Convert boolean fields**
   - `online_order`: `"Yes"` → `True`, `"No"` → `False`
   - `book_table`: `"Yes"` → `True`, `"No"` → `False`

6. **Derive new columns**
   - `budget_tier`: Based on `approx_cost`
     - Low: ₹0–₹300
     - Medium: ₹301–₹800
     - High: ₹801+
   - `is_new`: `True` if original rate was `"NEW"`

#### 1c. SQLite Structured Store (`data_layer.py` — Part 1) (2–3 hrs)

1. **Create SQLite database** at `data/restaurants.db`
2. **Define schema** — `restaurants` table with all columns (see architecture §3.2)
3. **Create indexes** on:
   - `location` (exact match, primary filter)
   - `approx_cost` (range queries)
   - `rate` (range queries)
   - `budget_tier` (exact match)
   - `cuisines` (FTS5 full-text search index)
4. **Bulk insert** cleaned data
5. **Implement interface methods:**
   - `init_db(csv_path) → None`
   - `query_structured(filters: dict) → List[dict]`
   - `get_available_locations() → List[str]`
   - `get_available_cuisines() → List[str]`

#### 1d. ChromaDB Vector Store (`data_layer.py` — Part 2) (2–3 hrs)

1. **Initialize ChromaDB** persistent client at `data/chroma_store/`
2. **Create collection** `restaurant_embeddings`
3. **Build composite documents** per restaurant:
   ```
   "{name}. Cuisines: {cuisines}. Type: {rest_type}. 
    Popular dishes: {dish_liked}. Cost ₹{cost} for two. 
    Rating: {rate}/5 with {votes} votes. Location: {location}."
   ```
4. **Generate embeddings** using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
5. **Store with metadata** — id, name, location, approx_cost, rate, budget_tier
6. **Implement interface:**
   - `query_semantic(text: str, filter_ids: List[int], top_k: int) → List[dict]`

### Acceptance Criteria
- [ ] `data/zomato.csv` downloaded and cached locally
- [ ] All parsing handles edge cases (NEW, null, commas, bad format)
- [ ] `data/restaurants.db` created with indexed tables
- [ ] `data/chroma_store/` created with all restaurant embeddings
- [ ] `query_structured({"location": "btm", "budget_max": 800})` returns correct results
- [ ] `query_semantic("cozy Italian place", ids=[...])` returns semantically relevant restaurants
- [ ] Embedding generation completes in < 10 minutes for full dataset

### Output Files
- `src/data_loader.py`
- `src/data_layer.py`
- `data/zomato.csv`
- `data/restaurants.db` (auto-generated)
- `data/chroma_store/` (auto-generated)

---

## Phase 2: Input Parser Module

**Goal:** Parse natural language or structured input into a validated `UserPreferences` object.

### Tasks

#### 2a. Synonym Map & Configuration (1 hr)

1. **Define `SYNONYM_MAP`** for budget terms:
   - `low`: cheap, budget-friendly, pocket-friendly, affordable, inexpensive, economical
   - `medium`: moderate, mid-range, reasonable, not too expensive
   - `high`: premium, expensive, upscale, fine dining, luxury, splurge

2. **Define preference synonyms:**
   - `family-friendly`: kid-friendly, family, children
   - `quick service`: fast, quick bite, grab and go
   - `romantic`: date night, cozy, intimate
   - `rooftop`: terrace, outdoor, open-air

3. **Load known locations and cuisines** from data layer
   - `get_available_locations()` → used for fuzzy matching
   - `get_available_cuisines()` → used for extraction

#### 2b. Structured Input Handler (1 hr)

1. **Validate form/dict inputs** — check required fields (location, budget)
2. **Normalize values** — lowercase, strip, map budget to `budget_max`
3. **Return `UserPreferences`** object

#### 2c. Natural Language Parser (2–3 hrs)

1. **Option A (Recommended): Rule-based NLP**
   - Location extraction: fuzzy match against known locations (`thefuzz` library)
   - Budget extraction: match against synonym map
   - Cuisine extraction: match against known cuisine list
   - Rating extraction: regex for numbers (e.g., "rated 4+")
   - Additional prefs extraction: match against preference synonyms

2. **Option B (Stretch): LLM-based extraction**
   - Send to Grok: "Extract location, budget, cuisine, rating from: '{input}'. Return JSON."
   - Parse JSON response
   - Validate against known values

3. **Implement `resolve_location()`**
   - Fuzzy match using `thefuzz.process.extractOne`
   - Threshold: 80% match → accept; below → raise `InvalidLocationError`
   - Suggest closest matches if no exact match

4. **Implement `parse_user_input(raw_input: str | dict) → UserPreferences`**

### Acceptance Criteria
- [ ] `parse("something cheap near Koramangala")` → `{location: "koramangala", budget: "low"}`
- [ ] `parse("best Italian under 1000 in Indiranagar")` → `{location: "indiranagar", cuisine: "Italian", budget_max: 1000}`
- [ ] `parse("pocket-friendly Chinese, rated 4+")` → `{budget: "low", cuisine: "Chinese", min_rating: 4.0}`
- [ ] Invalid location triggers helpful error with suggestions
- [ ] Structured dict input validates and normalizes correctly

### Output Files
- `src/input_parser.py`

---

## Phase 3: Two-Stage Retrieval Engine

**Goal:** Implement the two-pass retrieval — hard structural filters first, then semantic vector ranking.

### Tasks

#### 3a. Stage 1 — Hard Filters (2 hrs)

1. **Build SQL query** from `UserPreferences`:
   ```sql
   SELECT * FROM restaurants
   WHERE location = :location
     AND approx_cost <= :budget_max
     AND rate >= :min_rating
   ```
2. **Add optional cuisine filter** (FTS search on cuisines column)
3. **Implement progressive relaxation** when results < 3:
   - Step 1: Remove cuisine constraint
   - Step 2: Expand budget by ±20%
   - Step 3: Lower min_rating by 0.5
   - Step 4: Return top-rated in location (last resort)
4. **Track which filters were relaxed** → include in response

#### 3b. Stage 2 — Semantic Ranking (2–3 hrs)

1. **Build semantic query** from user preferences:
   ```python
   query = f"{cuisine} restaurant, {additional_prefs}, {budget_tier} price range"
   ```
2. **Query ChromaDB** with Stage 1 result IDs as filter
   - `collection.query(query_texts=[query], where={"id": {"$in": stage1_ids}}, n_results=top_k)`
3. **Return top-K** (K = 10–15) ranked by cosine similarity
4. **Attach similarity scores** to each result

#### 3c. Context Window Guard (1 hr)

1. **Define `MAX_LLM_CANDIDATES = 15`**
2. **If Stage 2 returns > 15**, apply composite pre-ranking:
   ```
   composite_score = (similarity * 0.5) + (normalized_rating * 0.3) + (normalized_votes * 0.2)
   ```
3. **Take top 15** by composite score

4. **Implement `retrieve(preferences: UserPreferences) → RetrievalResult`**

### Acceptance Criteria
- [ ] Stage 1 returns correct structural matches for given location + budget
- [ ] Progressive relaxation fires when < 3 results and informs which filters relaxed
- [ ] Stage 2 returns semantically relevant subset (e.g., "cozy Italian" ranks Italian restaurants with ambiance keywords higher)
- [ ] Context window guard caps output at 15 candidates
- [ ] Full retrieve() pipeline runs in < 500ms

### Output Files
- `src/retrieval_engine.py`

---

## Phase 4: Prompt Engineering Module

**Goal:** Build a dedicated, template-driven prompt builder with chain-of-thought reasoning.

### Tasks

#### 4a. Create Prompt Templates (1.5 hrs)

1. **`prompts/system.txt`** — System instruction
   ```
   You are an expert restaurant recommendation assistant for Zomato.
   IMPORTANT: Only recommend restaurants from the provided data. Do NOT invent any.
   Be friendly, concise, and helpful.
   ```

2. **`prompts/user_context.txt`** — User preferences block
   ```
   A user is looking for restaurants with these preferences:
   - Location: {location}
   - Budget: {budget_tier} (up to ₹{budget_max} for two)
   - Cuisine: {cuisine}
   - Minimum Rating: {min_rating}
   - Additional Preferences: {additional_prefs}
   ```

3. **`prompts/restaurant_data.txt`** — Token-efficient serialization format
   ```
   [{idx}] {name} | {cuisines} | ★{rating} ({votes} votes) | ₹{cost} | {rest_type} | Online: {online} | Book: {book} | Popular: {dishes}
   ```

4. **`prompts/cot_reasoning.txt`** — Chain-of-thought instruction
   ```
   Before ranking, analyze each restaurant:
   - How well does it match cuisine preference?
   - Is it within budget?
   - Does rating meet expectations?
   - Does restaurant type fit additional preferences?
   Then rank the top 5 from best to worst fit.
   ```

5. **`prompts/output_format.txt`** — Output structure specification
   ```
   Return your response as a JSON array with this structure:
   [{"rank": 1, "name": "...", "explanation": "..."}, ...]
   ```

#### 4b. Implement Prompt Builder (2 hrs)

1. **Load templates** from `prompts/` directory
2. **Serialize restaurant data** in token-efficient format (key fields only)
3. **Estimate token count** — ensure total prompt stays under model limit
4. **Handle session context** — if conversational, inject previous turn context
5. **Implement `build_prompt(preferences, restaurants, session=None) → str`**

#### 4c. Token Budget Management (0.5 hrs)

1. **Estimate tokens** — ~4 chars per token heuristic
2. **If over budget** — reduce restaurant count or trim optional fields
3. **Log token estimate** for cost tracking

### Acceptance Criteria
- [ ] Templates load correctly from `prompts/` directory
- [ ] Prompt includes all 5 sections (system, user, data, CoT, format)
- [ ] Token estimate stays under 2048 input tokens for 15 restaurants
- [ ] Changing a template requires zero code changes
- [ ] Anti-hallucination instruction is always present

### Output Files
- `src/prompt_builder.py`
- `prompts/system.txt`
- `prompts/user_context.txt`
- `prompts/restaurant_data.txt`
- `prompts/cot_reasoning.txt`
- `prompts/output_format.txt`

---

## Phase 5: LLM Recommender (Grok Adapter)

**Goal:** Implement the LLM call via Grok API with response parsing, validation, and fallback.

### Tasks

#### 5a. LLM Adapter Pattern (1.5 hrs)

1. **Define abstract base class `LLMAdapter`**
   - `call(prompt: str) → str`
   - `get_provider_name() → str`
   - `get_model_name() → str`

2. **Implement `GrokAdapter(LLMAdapter)`** — Primary (FREE)
   ```python
   from openai import OpenAI
   client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
   response = client.chat.completions.create(
       model="grok-3-mini-fast",
       messages=[{"role": "user", "content": prompt}],
       temperature=0.4,
       max_tokens=1024
   )
   ```

3. **Configuration** — temperature, max_tokens, timeout from `.env`

#### 5b. Response Parsing (1.5 hrs)

1. **Primary: JSON parsing** — parse structured JSON output from LLM
2. **Fallback: Regex extraction** — if JSON fails, extract restaurant names + explanations via regex patterns
3. **Handle malformed responses** gracefully

#### 5c. Anti-Hallucination Validation (1 hr)

1. **Cross-validate** every recommended restaurant name against `candidates_df`
2. **Drop hallucinated entries** — names not found in dataset
3. **Log `hallucinations_dropped` count**
4. **If all recommendations are hallucinated** → invoke fallback

#### 5d. Fallback Ranking (1 hr)

1. **Implement `fallback_ranking(candidates_df) → List[Recommendation]`**
   - Sort by: `rating DESC → votes DESC → cost ASC`
   - Return top 5 with generic explanation
   - Mark `source = "fallback"`
2. **Trigger conditions:** LLM timeout, rate limit, all outputs invalid

### Acceptance Criteria
- [ ] Grok API call succeeds with test prompt
- [ ] JSON response parsed correctly into `List[Recommendation]`
- [ ] Hallucinated restaurant names are detected and dropped
- [ ] Fallback ranking works when LLM is unavailable
- [ ] Timeout handling works (30s timeout → 1 retry → fallback)
- [ ] Response parsing handles both JSON and non-JSON LLM outputs

### Output Files
- `src/recommender.py`

---

## Phase 6: Orchestrator & Integration

**Goal:** Wire all components together into a single `process_request()` method. First end-to-end working pipeline.

### Tasks

#### 6a. Orchestrator Class (2–3 hrs)

1. **Implement `RecommendationOrchestrator`** — inject all components via constructor
2. **Implement `process_request(raw_input, session_id=None) → RecommendationResponse`**
   - Step 1: Parse input → `UserPreferences`
   - Step 2: Check cache (if enabled) → return cached if hit
   - Step 3: Retrieve (two-stage) → `RetrievalResult`
   - Step 4: Build prompt → `str`
   - Step 5: Get recommendations → `List[Recommendation]`
   - Step 6: Cache response (if enabled)
   - Step 7: Update session (if session_id provided)
   - Step 8: Log everything
   - Step 9: Return `RecommendationResponse`

3. **Error routing at each step:**
   - Parse failure → ask user for structured input
   - Retrieval empty → progressive relaxation (already in retrieval engine)
   - LLM failure → fallback ranking
   - Cache failure → skip cache, proceed normally

#### 6b. Application Entry Point (1 hr)

1. **Create `src/app.py`** — initialize all components and orchestrator
2. **Wire dependencies:**
   ```python
   orchestrator = RecommendationOrchestrator(
       input_parser=InputParser(data_layer),
       data_layer=DataLayer(db_path, chroma_path),
       retrieval=RetrievalEngine(data_layer),
       prompt_builder=PromptBuilder(templates_dir),
       recommender=Recommender(adapter=GrokAdapter()),
       cache=Cache(strategy="memory"),
       session_mgr=SessionManager(),
       logger=Logger(log_path)
   )
   ```
3. **Create simple CLI loop** for testing:
   ```python
   while True:
       user_input = input("What are you looking for? > ")
       response = orchestrator.process_request(user_input)
       display(response)
   ```

#### 6c. End-to-End Smoke Test (1 hr)

1. **Test full pipeline** with 5 sample queries:
   - `"cheap Italian in Koramangala"`
   - `"best rated Chinese under 500"`
   - `"upscale dining for date night in Indiranagar"`
   - `"family-friendly place in BTM, medium budget"`
   - `"something quick near Whitefield"`
2. **Verify:** input parsed → restaurants retrieved → prompt built → LLM called → response validated → output displayed
3. **Fix integration issues** between modules

### Acceptance Criteria
- [ ] `orchestrator.process_request("cheap Italian in Koramangala")` returns valid recommendations
- [ ] End-to-end latency < 5 seconds (including LLM call)
- [ ] All 5 smoke test queries produce reasonable output
- [ ] Fallback triggers correctly when LLM is artificially disabled
- [ ] No hallucinated restaurant names in output

### Output Files
- `src/orchestrator.py`
- `src/app.py`

---

## Phase 7: Caching & Session Management

**Goal:** Add response caching for repeated queries and session memory for follow-up conversations.

### Tasks

#### 7a. Caching Layer (`cache.py`) (1.5–2 hrs)

1. **Implement `InMemoryCache`** (dict-based, dev/demo mode)
2. **Cache key generation:**
   ```python
   key = sha256(f"{location}|{budget}|{cuisine}|{rating}|{prefs}")
   ```
3. **TTL-based expiration** — default 3600 seconds (1 hour)
4. **Max entries cap** — 1000 entries, LRU eviction
5. **Cache stats:** hits, misses, hit_rate
6. **Interface:**
   - `get(key) → Optional[CachedResponse]`
   - `set(key, response, ttl) → None`
   - `invalidate(key) → None`
   - `get_stats() → CacheStats`

#### 7b. Session Manager (`session_manager.py`) (2–3 hrs)

1. **Implement `InMemoryStore`** session storage
2. **Session lifecycle:**
   - `create_session(prefs) → session_id`
   - `get_session(session_id) → Session | None`
   - `update_session(session_id, prefs, recs) → None`
   - `expire_session(session_id) → None`

3. **Implement `merge_with_session(session_id, new_input) → UserPreferences`**
   - `"cheaper options"` → keep location, lower budget
   - `"show me something else"` → exclude previous recommendations
   - `"try Chinese instead"` → swap cuisine, keep rest
   - Merge new partial preferences with base session context

4. **Configuration:**
   - `SESSION_TTL = 1800` (30 minutes)
   - `MAX_HISTORY = 5` turns
   - Auto-cleanup expired sessions

### Acceptance Criteria
- [ ] Second identical query returns from cache (< 50ms vs. ~3s)
- [ ] Cache hit rate logged correctly
- [ ] TTL expiration works (entry disappears after TTL)
- [ ] Session follow-up works: "Italian in BTM" → "something cheaper" modifies budget only
- [ ] "Show me something else" excludes previous recommendations
- [ ] Session expires after 30 minutes of inactivity

### Output Files
- `src/cache.py`
- `src/session_manager.py`

---

## Phase 8: Observability & Logging

**Goal:** Log every request end-to-end for debugging, evaluation, and future fine-tuning.

### Tasks

#### 8a. Structured Logger (`logger.py`) (2–3 hrs)

1. **Define log schema** (per request):
   - `request_id` (UUID)
   - `timestamp` (ISO 8601)
   - `session_id`
   - Input: `raw_input`, `parsed_prefs`
   - Retrieval: `stage1_count`, `stage2_count`, `filters_relaxed`, `retrieval_ms`
   - Prompt: `token_estimate`, `template_version`
   - LLM: `provider`, `model`, `llm_ms`, `source`, `hallucinations_dropped`
   - Output: `recommendations` (names + ranks), `total_ms`
   - Feedback: `user_accepted`, `thumbs_up` (optional, async)

2. **Storage format:** JSON Lines (`logs/queries.jsonl`)
   - One JSON object per line per request
   - Append-only for performance

3. **Interface:**
   - `log_request(RequestLog) → None`
   - `log_feedback(request_id, FeedbackData) → None`
   - `get_metrics(time_range) → DashboardMetrics`

4. **Metrics dashboard data** (optional):
   - Average latency, P95 latency
   - Cache hit rate
   - Fallback rate
   - Hallucination rate
   - Top queried locations, cuisines

### Acceptance Criteria
- [ ] Every request produces a JSONL log entry in `logs/queries.jsonl`
- [ ] Log contains all fields: input, retrieval stats, prompt info, LLM response, latency
- [ ] Logs are machine-parseable (valid JSON per line)
- [ ] Any request is reproducible from log data
- [ ] Log file doesn't grow unbounded (rotate at 10MB)

### Output Files
- `src/logger.py`
- `logs/` directory (auto-created)

---

## Phase 9: Presentation Layer — Premium Web UI

**Goal:** A visually stunning, modern web UI that creates a premium first impression — not a basic form, but a polished product-quality interface inspired by Zomato's design language.

**Est. Hours:** 8–10 (increased for quality)

### Design Philosophy

> **The user should be wowed at first glance.** The UI must feel like a real product, not a hackathon demo. Every interaction should feel smooth, intentional, and premium.

### Design System & Aesthetic Spec

| Element              | Specification                                                      |
|----------------------|--------------------------------------------------------------------|
| **Color Palette**    | Zomato-inspired: `#E23744` (primary red), `#1C1C1C` (dark bg), `#2D2D2D` (card bg), `#F4F4F4` (light text), `#FFD700` (star gold), `#4CAF50` (success green) |
| **Typography**       | Google Fonts — `Inter` (UI text), `Outfit` (headings) — no browser defaults |
| **Card Style**       | Glassmorphism — translucent frosted-glass cards with `backdrop-filter: blur(12px)`, subtle borders, soft shadows |
| **Animations**       | Micro-animations on every interaction: card entrance (staggered fade-up), hover lift + shadow, button pulse, AI text typewriter effect |
| **Layout**           | Responsive CSS Grid — 1-col mobile, 2-col tablet, 3-col desktop |
| **Dark Mode**        | Default dark theme with `prefers-color-scheme` auto-detect; manual toggle |
| **Border Radius**    | `12px–16px` on cards, `8px` on inputs, `24px` on pills/badges |
| **Spacing**          | 8px grid system, generous whitespace |

### Tasks

#### 9a. Flask Backend Routes (1 hr)

1. **`GET /`** → Render `index.html` with dynamic location + cuisine lists
2. **`POST /api/recommend`** → JSON API: accepts user preferences, returns recommendations
3. **`POST /api/feedback`** → Capture thumbs up/down per recommendation
4. **`GET /api/locations`** → Return available locations (for autocomplete)
5. **`GET /api/cuisines`** → Return available cuisines (for autocomplete)

#### 9b. Landing Page — Hero Section (`templates/index.html`) (2 hrs)

1. **Hero Banner:**
   - Full-width gradient background (`#E23744` → `#1C1C1C` diagonal)
   - Large heading: *"Discover Your Next Favorite Restaurant"* (Outfit font, 48px)
   - Subtitle: *"AI-powered recommendations tailored to your taste"* (Inter, muted)
   - Subtle floating food emoji particles (CSS keyframe animation)

2. **Smart Search Bar — The Star Component:**
   - Large, centered, glassmorphism input bar (width: 70%)
   - Placeholder that cycles through examples with typing animation:
     - *"Try: cheap Italian near Koramangala..."*
     - *"Try: rooftop dining for a date night..."*
     - *"Try: family-friendly under ₹500 in BTM..."*
   - Real-time search icon → loading spinner transition on submit
   - Keyboard shortcut: `Enter` to search, `Escape` to clear

3. **Quick Filter Pills (below search bar):**
   - Horizontal scrollable row of filter pills
   - Location pill (dropdown), Budget pill (Low/Med/High toggle), Cuisine pill (multi-select)
   - Min Rating pill (star-based selector)
   - Pills have `active` state with subtle color transition
   - Users can mix natural-language search + pills

#### 9c. Results Section (2.5 hrs)

1. **Results Header:**
   - *"🍽️ Top 5 Picks for You"* with result count
   - Relaxation badge if filters were loosened: *"We expanded your search — budget filter relaxed"*
   - Source indicator: `AI Ranked` / `Popularity Based` / `Cached` with colored dot

2. **Restaurant Recommendation Cards:**
   - **Glassmorphism card** — frosted glass effect, `rgba(255,255,255,0.05)` bg, blur backdrop
   - **Staggered entrance animation** — cards fade-up sequentially (100ms delay each)
   - **Card layout (each card):**
     ```
     ┌─────────────────────────────────────────────────────┐
     │  #1  ★★★★☆ (4.2)    ₹650 for two    🟢 Online Order │
     │  ──────────────────────────────────────────────────── │
     │  Restaurant Name (large, bold, Outfit)                │
     │  North Indian, Chinese  •  Casual Dining              │
     │  📍 Koramangala  •  👍 342 votes                      │
     │                                                       │
     │  ┌── AI Insight ─────────────────────────────────┐   │
     │  │  "This is a great match because..."           │   │
     │  │  (typewriter animation, subtle gradient bg)   │   │
     │  └───────────────────────────────────────────────┘   │
     │                                                       │
     │  ┌──────┐ ┌──────┐                                   │
     │  │  👍  │ │  👎  │   feedback buttons                │
     │  └──────┘ └──────┘                                   │
     └─────────────────────────────────────────────────────┘
     ```
   - **Hover effect:** Card lifts 4px, shadow deepens, subtle border glow
   - **Star rating:** Custom CSS stars (gold fill, half-star support)
   - **Feature badges:** Pill-style tags for Online Order, Book Table, Popular Dish
   - **AI Insight block:** Distinct styled block with gradient left-border (`#E23744` → `#FFD700`), typewriter text animation

3. **Empty / Error States:**
   - No results: Friendly illustration + *"No matches found. Try adjusting your preferences."* with suggested actions
   - API error: Graceful message + *"Our AI is taking a break. Here are popular picks instead."* → shows fallback results
   - Loading state: Animated skeleton cards (shimmer effect) while API processes

#### 9d. Styling — `static/style.css` (2 hrs)

1. **CSS Custom Properties (design tokens):**
   ```css
   :root {
     --primary: #E23744;
     --bg-dark: #1C1C1C;
     --card-bg: rgba(45, 45, 45, 0.7);
     --text-primary: #F4F4F4;
     --text-muted: #A0A0A0;
     --star-gold: #FFD700;
     --success: #4CAF50;
     --radius-card: 16px;
     --radius-input: 8px;
     --radius-pill: 24px;
     --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
   }
   ```

2. **Glassmorphism mixin:**
   - `backdrop-filter: blur(12px); background: var(--card-bg); border: 1px solid rgba(255,255,255,0.08);`

3. **Keyframe animations:**
   - `@keyframes fadeUp` — card entrance (0→1 opacity, 20px→0 translateY)
   - `@keyframes shimmer` — skeleton loading effect
   - `@keyframes typewriter` — AI insight text reveal
   - `@keyframes pulse` — search button pulse on hover
   - `@keyframes float` — hero emoji particles

4. **Responsive breakpoints:**
   - Mobile: `<768px` → single column, full-width cards, collapsible filters
   - Tablet: `768px–1024px` → 2-column grid
   - Desktop: `>1024px` → 3-column grid, side filter panel option

5. **Dark mode** — default, with `@media (prefers-color-scheme: light)` override
   - Light mode swaps: `--bg-dark: #FAFAFA`, `--card-bg: rgba(255,255,255,0.8)`, `--text-primary: #1C1C1C`

#### 9e. Client-Side Logic — `static/script.js` (1.5 hrs)

1. **Search handling:**
   - Debounced input (300ms) for autocomplete suggestions
   - `fetch('/api/recommend')` with loading state management
   - Response renders cards dynamically (DOM injection)

2. **Typing placeholder animation:**
   - Cycles through 4–5 example queries with typing/deleting effect (pure JS)

3. **Card animations:**
   - Intersection Observer for scroll-triggered entrance
   - Staggered `animation-delay` per card

4. **Feedback:**
   - Thumbs up/down with immediate visual feedback (color change + count increment)
   - Sends async `POST /api/feedback`

5. **Filter pill interactivity:**
   - Toggle active state on click
   - Sync pill values with hidden form fields
   - Update search URL params for shareable links

6. **Accessibility:**
   - All interactive elements focusable with keyboard
   - ARIA labels on custom widgets
   - Reduced-motion media query disables animations

### Acceptance Criteria
- [ ] Web UI loads at `http://localhost:5000` with a premium dark-themed landing page
- [ ] Hero section has animated cycling placeholder + gradient background
- [ ] Search bar accepts both natural language AND structured filter pills
- [ ] Results display as glassmorphism cards with staggered fade-up animation
- [ ] AI explanation renders with typewriter effect in a distinct styled block
- [ ] Star ratings display correctly (including half-stars)
- [ ] Responsive: works on mobile (single-col), tablet (2-col), desktop (3-col)
- [ ] Loading state shows animated skeleton cards
- [ ] Error states display friendly messages with fallback suggestions
- [ ] Thumbs up/down feedback works with visual confirmation
- [ ] Page scores 90+ on Lighthouse performance audit
- [ ] All animations respect `prefers-reduced-motion`

### Output Files
- `src/app.py` (Flask routes + API endpoints)
- `templates/index.html` (full premium template)
- `static/style.css` (design system + glassmorphism + animations)
- `static/script.js` (search, animations, feedback, autocomplete)

---

## Phase 10: Testing & Evaluation

**Goal:** Unit tests for each module + integration tests for the full pipeline.

### Tasks

#### 10a. Unit Tests (2–3 hrs)

1. **`tests/test_input_parser.py`**
   - Natural language parsing (10+ cases from examples table)
   - Synonym resolution (budget terms, preference terms)
   - Fuzzy location matching (typos, abbreviations)
   - Structured input validation
   - Edge cases: empty input, gibberish, multiple cuisines

2. **`tests/test_data_layer.py`**
   - SQLite query correctness (location, budget, rating filters)
   - ChromaDB semantic query returns relevant results
   - Index performance (query < 50ms)
   - Empty result handling

3. **`tests/test_retrieval_engine.py`**
   - Stage 1 hard filter correctness
   - Stage 2 semantic ranking quality
   - Progressive relaxation (< 3 results → relaxes correctly)
   - Context window guard (caps at 15)
   - Combined two-stage flow

4. **`tests/test_prompt_builder.py`**
   - Template loading
   - Variable substitution
   - Token budget not exceeded
   - Anti-hallucination instruction present in all prompts

5. **`tests/test_recommender.py`**
   - Mock LLM response → correct parsing
   - Hallucination detection (fake restaurant name → dropped)
   - Fallback ranking (sort by rating × votes)
   - Timeout handling

6. **`tests/test_cache.py`**
   - Cache hit/miss
   - TTL expiration
   - Key generation consistency

7. **`tests/test_session_manager.py`**
   - Create → get → update → expire lifecycle
   - Merge logic ("cheaper", "something else", "try Chinese")

#### 10b. Integration Tests (1–2 hrs)

1. **`tests/test_orchestrator.py`**
   - Full pipeline: input → output (mocked LLM)
   - Fallback triggers when LLM mocked to fail
   - Cache integration (second call returns cached)
   - Session follow-up (two-turn conversation)

2. **End-to-end with real LLM** (manual, not automated):
   - 10 diverse queries → verify output quality
   - Check no hallucinations
   - Measure latency

#### 10c. Evaluation Metrics (1 hr)

1. **Recommendation relevance** — do recommendations match stated preferences?
2. **Hallucination rate** — % of LLM outputs containing fake restaurants
3. **Latency** — end-to-end P50, P95
4. **Cache effectiveness** — hit rate over repeated queries
5. **Fallback rate** — % of requests going to rule-based ranking
6. **Create evaluation script** that runs 20 test queries and reports metrics

### Acceptance Criteria
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Integration test passes with mocked LLM
- [ ] Hallucination rate = 0% (all outputs validated)
- [ ] Average latency < 5s (including LLM call)
- [ ] Cache hit returns in < 100ms
- [ ] At least 80% test coverage on core modules

### Output Files
- `tests/test_input_parser.py`
- `tests/test_data_layer.py`
- `tests/test_retrieval_engine.py`
- `tests/test_prompt_builder.py`
- `tests/test_recommender.py`
- `tests/test_cache.py`
- `tests/test_session_manager.py`
- `tests/test_orchestrator.py`
- `tests/evaluate.py` (evaluation script)

---

## Phase 11: Polish, Documentation & Submission

**Goal:** Production-ready documentation, clean code, and a professional submission.

### Tasks

#### 11a. README.md (1 hr)

1. **Project overview** — what it does, architecture diagram (ASCII)
2. **Quick start guide:**
   ```
   git clone <repo>
   cd "Zomato Milestone 1 AI"
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # Add your XAI_API_KEY
   python src/data_loader.py  # One-time data setup
   python src/app.py  # Start the app
   ```
3. **Architecture overview** — link to `Docs/architecture.md`
4. **Screenshots/demo** — sample input/output
5. **Tech stack table** with cost breakdown (all free)
6. **Evaluation results** — latency, cache hit rate, sample outputs

#### 11b. Code Cleanup (1 hr)

1. **Docstrings** — every public function has a docstring
2. **Type hints** — all function signatures annotated
3. **Remove debug prints** — use logger instead
4. **Consistent formatting** — run `black` or `ruff format`
5. **No hardcoded values** — everything in `.env` or config

#### 11c. Demo Preparation (1 hr)

1. **Prepare 5 compelling demo queries** that showcase:
   - Natural language understanding
   - Semantic search (ambiance, "date night")
   - Progressive relaxation
   - Session follow-up ("something else")
   - Fallback graceful degradation
2. **Record a short walkthrough** or prepare screenshots
3. **Ensure first-run experience is smooth** (data auto-downloads if missing)

### Acceptance Criteria
- [ ] README is clear enough for a reviewer to set up and run in < 5 minutes
- [ ] All code has docstrings and type hints
- [ ] No hardcoded API keys or paths
- [ ] Demo queries produce impressive, consistent results
- [ ] Project runs on a fresh machine with only `pip install` + `.env` setup

### Output Files
- `README.md` (complete)
- All `src/*.py` files cleaned up
- `Docs/architecture.md` (final)
- `Docs/context.md` (final)

---

## Dependency Graph (Build Order)

```
Phase 0 ─────────────────────────────────────────────────────────────────
   │
   ▼
Phase 1 (Data Layer) ────────────────────────────────────────────────────
   │                    \
   ▼                     ▼
Phase 2 (Parser)    Phase 3 (Retrieval) ─────────────────────────────────
                         │
                         ▼
                    Phase 4 (Prompt) ─────────────────────────────────────
                         │
                         ▼
                    Phase 5 (Recommender) ────────────────────────────────
                         │
   ┌─────────────────────┼─────────────────────┐
   │                     │                     │
   ▼                     ▼                     ▼
Phase 7 (Cache)    Phase 6 (Orchestrator)  Phase 8 (Logging)
                         │
                         ▼
                    Phase 9 (UI) ─────────────────────────────────────────
                         │
                         ▼
                    Phase 10 (Testing) ───────────────────────────────────
                         │
                         ▼
                    Phase 11 (Polish) ────────────────────────────────────
```

**Critical Path:** Phase 0 → 1 → 3 → 4 → 5 → 6 → 9

**Parallelizable:**
- Phase 2 (Parser) can be built in parallel with Phase 3 (Retrieval)
- Phase 7 (Cache) and Phase 8 (Logging) can be built in parallel after Phase 6
- Phase 10 (Testing) can start incrementally from Phase 2 onwards

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Grok free credits run out | Medium | High | Aggressive caching; fallback to rule-based; monitor usage |
| ChromaDB embedding slow on CPU | Medium | Medium | Pre-compute once; persist to disk; don't re-embed |
| Dataset too large for memory | Low | Medium | SQLite handles disk-based queries; don't load full df |
| LLM hallucinates frequently | Medium | Medium | Strict validation; anti-hallucination prompt; fallback |
| HuggingFace dataset unavailable | Low | Low | Cache CSV locally on first download |
| Sentence-transformers download slow | Low | Low | One-time download; model cached in `~/.cache/` |

---

## Success Metrics (MVP Acceptance)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| End-to-end latency | < 5 seconds | Timer in orchestrator |
| Recommendation relevance | 4/5 queries feel appropriate | Manual evaluation |
| Hallucination rate | 0% | Validation check in recommender |
| Cache hit latency | < 100ms | Timer in cache layer |
| Fallback graceful | No crashes when LLM unavailable | Disable API → verify output |
| Code modularity | Any component swappable | Swap adapter → still works |
| Zero cost | $0 spent | Monitor xAI console |

---

> **Next Step:** Begin Phase 0 — set up project structure, install dependencies, and register for Grok API key.
