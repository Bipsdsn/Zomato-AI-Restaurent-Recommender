# Architecture Document — AI-Powered Restaurant Recommendation System

> **Project:** Zomato Milestone 1 AI  
> **Pattern:** Two-Stage Retrieval-Augmented Generation (RAG) with Semantic Search  
> **Last Updated:** 2026-06-17

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Deep-Dive](#3-component-deep-dive)
   - 3.1 [Input Parser](#31-input-parser-input_parserpy)
   - 3.2 [Data Layer & Storage](#32-data-layer--storage-data_layerpy)
   - 3.3 [Two-Stage Retrieval Engine](#33-two-stage-retrieval-engine-retrieval_enginepy)
   - 3.4 [Prompt Engineering Module](#34-prompt-engineering-module-prompt_builderpy)
   - 3.5 [Recommender (LLM Adapter)](#35-recommender-llm-adapter-recommenderpy)
   - 3.6 [Caching Layer](#36-caching-layer-cachepy)
   - 3.7 [Orchestrator / Controller](#37-orchestrator--controller-orchestratorpy)
   - 3.8 [Session Manager](#38-session-manager-session_managerpy)
   - 3.9 [Observability & Logging](#39-observability--logging-loggerpy)
4. [Data Flow & Sequence Diagrams](#4-data-flow--sequence-diagrams)
5. [Data Layer](#5-data-layer)
6. [Module Interface Contracts](#6-module-interface-contracts)
7. [Prompt Engineering Architecture](#7-prompt-engineering-architecture)
8. [Error Handling & Resilience](#8-error-handling--resilience)
9. [Tech Stack & Dependencies](#9-tech-stack--dependencies)
10. [Project Directory Structure](#10-project-directory-structure)
11. [Scalability & Future Extensions](#11-scalability--future-extensions)

---

## 1. System Overview

The system follows a **five-stage pipeline** architecture built on two-stage retrieval with semantic understanding:

| Stage               | Responsibility                                                          | Core Technology                  |
|---------------------|-------------------------------------------------------------------------|----------------------------------|
| **Parse**           | Extract structured entities from natural-language user input            | NLP / LLM pre-processing        |
| **Hard Filter**     | Apply non-negotiable constraints (location, budget ceiling)             | SQLite / pandas structured query |
| **Semantic Rank**   | Vector similarity search on filtered subset for soft preference match   | ChromaDB / FAISS embeddings      |
| **Augment + Generate** | Build chain-of-thought prompt → LLM reasons, ranks, and explains   | Prompt template engine + Grok API (xAI) |
| **Cache + Observe** | Cache responses for repeat queries; log everything for evaluation       | TTL cache + structured logging   |

### Design Principles

- **Separation of Concerns** — Each pipeline stage lives in its own module with a clean interface contract
- **Two-Stage Retrieval** — Hard filters narrow thousands → dozens; semantic ranking picks top candidates
- **Dedicated Prompt Module** — Prompt templates are decoupled from application logic for independent iteration
- **Fail-Safe Degradation** — If the LLM or vector DB is unavailable, fall back to rule-based ranking
- **Data Integrity** — LLM output is always cross-validated against source data (no hallucinated restaurants)
- **Stateless Core with Optional Sessions** — Each request is independently processable; session context is opt-in for conversational follow-ups
- **Zero-Cost Stack** — Every component is free: Grok API (free credits via xAI), ChromaDB (OSS), SQLite (stdlib), sentence-transformers (OSS), HuggingFace datasets (free)
- **Modular LLM Adapter** — Grok is the primary LLM; the adapter pattern allows future swaps without touching upstream code
- **Observability First** — Every query is logged end-to-end: input → retrieval → prompt → response → latency
- **Context Window Management** — Pre-ranking limits candidates to 10–15 before LLM injection to control token cost
- **Caching for Cost & Latency** — Frequent query patterns return cached LLM responses with a configurable TTL

---

## 2. Architecture Diagram

### 2a. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                                  │
│                     (Flask + Premium HTML/CSS/JS)                                 │
│   ┌──────────────────────────────────────────────────────────────────────┐        │
│   │  Web UI — Dark-themed, Glassmorphism, Responsive, Zomato-inspired  │        │
│   │  • Hero section with animated search bar (cycling placeholders)    │        │
│   │  • Quick filter pills (location, budget, cuisine, rating)          │        │
│   │  • Glassmorphism recommendation cards with staggered fade-up       │        │
│   │  • AI insight typewriter animation + star ratings + feedback       │        │
│   │  • Skeleton loading states, error handling, dark/light mode       │        │
│   └──────────────────────────────────┬───────────────────────────────────┘        │
│          └─────────────────────┼──────────────────────────┘                       │
│                                │                                                  │
└────────────────────────────────┼──────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼──────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                                         │
│                                │                                                  │
│                    ┌───────────▼───────────┐                                      │
│                    │   orchestrator.py     │                                      │
│                    │   (Central Controller)│                                      │
│                    │                       │                                      │
│                    │ • Coordinates flow    │                                      │
│                    │ • Error routing       │◀────── session_manager.py            │
│                    │ • Fallback decisions  │        (Optional session context)    │
│                    └───┬───┬───┬───┬───┬───┘                                      │
│                        │   │   │   │   │                                          │
└────────────────────────┼───┼───┼───┼───┼──────────────────────────────────────────┘
                         │   │   │   │   │
    ┌────────────────────┘   │   │   │   └────────────────────┐
    │              ┌─────────┘   │   └─────────┐              │
    │              │             │             │              │
┌───┼──────────────┼─────────────┼─────────────┼──────────────┼─────────────────────┐
│   │         APPLICATION LAYER  │             │              │                      │
│   │              │             │             │              │                      │
│ ┌─▼────────┐ ┌──▼──────────┐ ┌▼───────────┐ ┌▼────────────┐ ┌▼────────────┐      │
│ │ input_   │ │ retrieval_  │ │ prompt_    │ │ recommender │ │ cache.py   │      │
│ │ parser   │ │ engine.py   │ │ builder.py │ │    .py      │ │            │      │
│ │ .py      │ │             │ │            │ │             │ │ • TTL cache│      │
│ │          │ │ ┌─────────┐ │ │ • Template │ │ • LLM call  │ │ • Hash key │      │
│ │ • NLP    │ │ │ Stage 1 │ │ │   module   │ │ • Parse     │ │ • Hit/miss │      │
│ │ • Entity │ │ │ Hard    │ │ │ • Chain-of-│ │ • Validate  │ │   metrics  │      │
│ │   extract│ │ │ Filter  │ │ │   thought  │ │ • Fallback  │ │            │      │
│ │ • Synonym│ │ ├─────────┤ │ │ • Token    │ │ • Adapter   │ └────────────┘      │
│ │   resolve│ │ │ Stage 2 │ │ │   budget   │ │   pattern   │                      │
│ │          │ │ │ Semantic │ │ │ • Format   │ │             │                      │
│ │          │ │ │ Rank     │ │ │   spec     │ │             │                      │
│ └──────────┘ │ └─────────┘ │ └────────────┘ └──────┬──────┘                      │
│              └──────┬──────┘                        │                              │
│                     │                               │                              │
└─────────────────────┼───────────────────────────────┼──────────────────────────────┘
                      │                               │
┌─────────────────────┼───────────────────────────────┼──────────────────────────────┐
│                DATA LAYER                           │                              │
│     ┌───────────────┼──────────────┐                │                              │
│     │               │              │                │                              │
│ ┌───▼─────────┐ ┌───▼────────┐ ┌───▼────────┐  ┌───▼─────────┐  ┌──────────────┐ │
│ │ SQLite      │ │ ChromaDB / │ │ data_      │  │ Grok API    │  │ logger.py    │ │
│ │ (free,      │ │ FAISS      │ │ loader.py  │  │ (xAI)       │  │              │ │
│ │  stdlib)    │ │ (free, OSS)│ │            │  │             │  │ • Query log  │ │
│ │ Structured  │ │ Vector     │ │ • HF load  │  │ • Free $25  │  │ • Prompt log │ │
│ │ queries     │ │ embeddings │ │ • CSV cache│  │   signup    │  │ • Response   │ │
│ │ (location,  │ │ (cuisine,  │ │ • Clean    │  │   credits   │  │ • Latency    │ │
│ │  cost, etc.)│ │  reviews,  │ │ • Normalize│  │ • OpenAI-   │  │ • Feedback   │ │
│ │             │ │  ambiance) │ │ • Embed    │  │   compatible│  │              │ │
│ └──────┬──────┘ └──────┬─────┘ └──────┬─────┘  └──────┬──────┘  └──────────────┘ │
│        │               │              │               │                            │
│ ┌──────▼───────────────▼──────┐   ┌───▼────────┐  ┌───▼─────────┐                 │
│ │  Zomato Dataset             │   │  Embedding │  │  xAI API    │                 │
│ │  (HuggingFace / local CSV)  │   │  Model     │  │  Endpoint   │                 │
│ │  (free dataset)             │   │  (sentence │  │  api.x.ai   │                 │
│ │                             │   │  transformers)│ │  (free tier)│                 │
│ └─────────────────────────────┘   └────────────┘  └─────────────┘                 │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### 2b. Request Flow (Two-Stage Retrieval)

```
                                    ┌─────────────┐
                                    │  Cache Hit? │
                                    └──────┬──────┘
                                       YES │ NO
                              ┌────────────┘ └────────────────────────────────┐
                              │                                               │
                              ▼                                               ▼
                    ┌─────────────────┐                             ┌─────────────────┐
                    │ Return Cached   │                             │  Input Parser   │
                    │ Response        │                             │  (NLP extract)  │
                    └─────────────────┘                             └────────┬────────┘
                                                                            │
                                                                            ▼
                                                              ┌──────────────────────────┐
                                                              │  STAGE 1: Hard Filters   │
                                                              │  (SQLite / pandas)       │
                                                              │  Location, Budget, Rating│
                                                              └────────────┬─────────────┘
                                                                           │
                                                                   ~50–200 candidates
                                                                           │
                                                              ┌────────────▼─────────────┐
                                                              │  STAGE 2: Semantic Rank  │
                                                              │  (ChromaDB / FAISS)      │
                                                              │  Cuisine, reviews,       │
                                                              │  ambiance, soft prefs    │
                                                              └────────────┬─────────────┘
                                                                           │
                                                                   Top 10–15 candidates
                                                                           │
                                                              ┌────────────▼─────────────┐
                                                              │  Prompt Builder          │
                                                              │  (Chain-of-Thought)      │
                                                              └────────────┬─────────────┘
                                                                           │
                                                              ┌────────────▼─────────────┐
                                                              │  LLM Recommender         │
                                                              │  (Reason → Rank → Explain)│
                                                              └────────────┬─────────────┘
                                                                           │
                                                              ┌────────────▼─────────────┐
                                                              │  Cache + Log + Display   │
                                                              └──────────────────────────┘
```

---

## 3. Component Deep-Dive

### 3.1 Input Parser (`input_parser.py`)

> **Improvement #9** — Build input parsing as its own module. Don't pass raw user text directly to filters.

**Responsibility:** Extract structured entities from natural-language or form input. Handle synonyms, colloquial terms, and ambiguous references before anything reaches the filter pipeline.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       input_parser.py                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  parse_user_input(raw_input: str | dict) → UserPreferences           │
│  │                                                                   │
│  ├── IF structured input (form/dict):                                │
│  │   └── Validate fields, normalize, return UserPreferences          │
│  │                                                                   │
│  ├── IF natural language (free text):                                │
│  │   ├── Option A: LLM-based entity extraction                      │
│  │   │   Send to LLM: "Extract location, budget, cuisine,           │
│  │   │   rating from: '{raw_input}'. Return JSON."                   │
│  │   │                                                               │
│  │   └── Option B: Rule-based NLP (regex + synonym map)              │
│  │       ├── Location extraction (known location list + fuzzy match) │
│  │       ├── Budget extraction (synonym map below)                   │
│  │       ├── Cuisine extraction (known cuisine list + fuzzy match)   │
│  │       └── Rating extraction (number parsing)                      │
│  │                                                                   │
│  └── Return: UserPreferences (structured, validated)                 │
│                                                                      │
│  SYNONYM_MAP = {                                                     │
│      "budget": {                                                     │
│          "low":  ["cheap", "budget-friendly", "pocket-friendly",     │
│                   "affordable", "inexpensive", "economical"],        │
│          "medium": ["moderate", "mid-range", "reasonable",           │
│                     "not too expensive"],                            │
│          "high": ["premium", "expensive", "upscale", "fine dining", │
│                   "luxury", "splurge"]                               │
│      },                                                              │
│      "prefs": {                                                      │
│          "family-friendly": ["kid-friendly", "family", "children"],  │
│          "quick service":   ["fast", "quick bite", "grab and go"],   │
│          "romantic":        ["date night", "cozy", "intimate"],      │
│          "rooftop":         ["terrace", "outdoor", "open-air"]       │
│      }                                                               │
│  }                                                                   │
│                                                                      │
│  resolve_location(raw: str, known_locations: List[str]) → str        │
│  │  Fuzzy match against known locations in dataset                   │
│  │  Returns best match or raises InvalidLocationError                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Natural Language → Structured Examples:**

| User Says                                        | Parsed Output                                                       |
|--------------------------------------------------|---------------------------------------------------------------------|
| `"something cheap near Koramangala"`             | `location: "koramangala", budget: "low"`                            |
| `"best Italian under 1000 in Indiranagar"`       | `location: "indiranagar", cuisine: "Italian", budget_max: 1000`     |
| `"pocket-friendly Chinese, rated 4+"`            | `budget: "low", cuisine: "Chinese", min_rating: 4.0`               |
| `"upscale place for a date night in Whitefield"` | `location: "whitefield", budget: "high", prefs: "romantic"`        |

---

### 3.2 Data Layer & Storage (`data_layer.py`)

> **Improvement #7** — Separate the data layer from the application layer. Use a proper data store.

**Responsibility:** Manage all data persistence — structured restaurant data in a relational store, embeddings in a vector store. Replace in-memory CSV loading with indexed, query-optimized storage.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         data_layer.py                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              STRUCTURED DATA STORE                       │        │
│  │              (SQLite / PostgreSQL)                        │        │
│  │                                                          │        │
│  │  restaurants TABLE:                                      │        │
│  │  ┌────────────────────────────────────────────────┐     │        │
│  │  │ id          INTEGER PRIMARY KEY                │     │        │
│  │  │ name        TEXT NOT NULL                      │     │        │
│  │  │ location    TEXT NOT NULL  — INDEX             │     │        │
│  │  │ cuisines    TEXT NOT NULL  — INDEX             │     │        │
│  │  │ approx_cost INTEGER       — INDEX             │     │        │
│  │  │ rate        REAL                               │     │        │
│  │  │ votes       INTEGER                            │     │        │
│  │  │ online_order BOOLEAN                           │     │        │
│  │  │ book_table  BOOLEAN                            │     │        │
│  │  │ rest_type   TEXT                               │     │        │
│  │  │ dish_liked  TEXT                               │     │        │
│  │  │ listed_in_type TEXT                            │     │        │
│  │  │ budget_tier TEXT          — INDEX              │     │        │
│  │  │ is_new      BOOLEAN                            │     │        │
│  │  └────────────────────────────────────────────────┘     │        │
│  │                                                          │        │
│  │  Indexed Fields (fast lookups):                          │        │
│  │  • location   — Primary filter                           │        │
│  │  • cuisines   — FTS (Full-Text Search) index             │        │
│  │  • approx_cost — Range queries                           │        │
│  │  • budget_tier — Exact match                             │        │
│  │  • rate       — Range queries                            │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              VECTOR DATA STORE                           │        │
│  │              (ChromaDB / FAISS)                           │        │
│  │                                                          │        │
│  │  Collection: restaurant_embeddings                       │        │
│  │  ┌────────────────────────────────────────────────┐     │        │
│  │  │ id          — matches structured store ID      │     │        │
│  │  │ embedding   — sentence-transformers vector     │     │        │
│  │  │ metadata    — {name, location, cuisines, cost} │     │        │
│  │  │ document    — composite text for embedding:    │     │        │
│  │  │   "{name}. {cuisines}. {rest_type}.            │     │        │
│  │  │    {dish_liked}. Cost ₹{cost} for two.         │     │        │
│  │  │    {rating}/5 rating. {location}."             │     │        │
│  │  └────────────────────────────────────────────────┘     │        │
│  │                                                          │        │
│  │  Embedding Model: sentence-transformers/all-MiniLM-L6-v2│        │
│  │  Dimensions: 384                                         │        │
│  │  Distance Metric: Cosine Similarity                      │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                      │
│  INTERFACE:                                                          │
│  • init_db(csv_path) → None            # One-time setup             │
│  • query_structured(filters) → List[dict]   # SQL queries           │
│  • query_semantic(text, ids) → List[dict]   # Vector search         │
│  • get_available_locations() → List[str]                             │
│  • get_available_cuisines() → List[str]                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why Two Stores:**

| Aspect               | Structured (SQLite)                    | Vector (ChromaDB)                           |
|----------------------|----------------------------------------|---------------------------------------------|
| **Query Type**       | Exact match, range, equality           | Semantic similarity, fuzzy                  |
| **Best For**         | Location, budget ceiling, min rating   | Cuisine match, ambiance, review sentiment   |
| **Speed**            | Sub-millisecond with indexes           | ~10–50ms for top-K search                    |
| **Example**          | `WHERE location = 'btm' AND cost < 800`| `"cozy Italian place"` → closest embeddings |

---

### 3.3 Two-Stage Retrieval Engine (`retrieval_engine.py`)

> **Improvements #1 & #2** — Replace basic filtering with RAG pipeline. Add two-stage retrieval.

**Responsibility:** Two-pass retrieval — hard structural filters first, then semantic ranking for soft preferences.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     retrieval_engine.py                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  retrieve(preferences: UserPreferences) → RetrievalResult            │
│  │                                                                   │
│  ├── STAGE 1: Hard Filters (Non-Negotiable Constraints)              │
│  │   │                                                               │
│  │   ├── Query structured DB:                                        │
│  │   │   SELECT * FROM restaurants                                   │
│  │   │   WHERE location = :location                                  │
│  │   │     AND approx_cost <= :budget_max                            │
│  │   │     AND rate >= :min_rating                                   │
│  │   │                                                               │
│  │   ├── If results < MIN_CANDIDATES (3):                            │
│  │   │   └── Progressive relaxation (see below)                      │
│  │   │                                                               │
│  │   └── Output: ~50–200 structurally valid candidates               │
│  │                                                                   │
│  ├── STAGE 2: Semantic Ranking (Soft Preference Matching)            │
│  │   │                                                               │
│  │   ├── Build semantic query from user input:                       │
│  │   │   query = f"{cuisine} restaurant, {additional_prefs},         │
│  │   │           {budget_tier} price range"                          │
│  │   │   e.g., "budget-friendly Italian, family-friendly"            │
│  │   │                                                               │
│  │   ├── Search vector store (ChromaDB/FAISS):                       │
│  │   │   Filter by IDs from Stage 1                                  │
│  │   │   Return top-K by cosine similarity (K = 10–15)               │
│  │   │                                                               │
│  │   ├── Each result has a similarity_score (0.0–1.0)                │
│  │   │                                                               │
│  │   └── Output: Top 10–15 semantically ranked candidates            │
│  │                                                                   │
│  └── Return: RetrievalResult                                         │
│       ├── candidates: List[Restaurant]  (ranked, 10–15 items)        │
│       ├── total_matched: int            (Stage 1 count)              │
│       └── filters_relaxed: List[str]    (which filters were relaxed) │
│                                                                      │
│  RELAXATION LOGIC (Progressive):                                     │
│  │  Step 1: Remove cuisine constraint                                │
│  │  Step 2: Expand budget by ±20%                                    │
│  │  Step 3: Lower min_rating by 0.5                                  │
│  │  Step 4: If still < 3, return top-rated in location               │
│  │  → User is informed which constraints were relaxed                │
│                                                                      │
│  CONTEXT WINDOW GUARD:                                               │
│  │  MAX_LLM_CANDIDATES = 15                                         │
│  │  If Stage 2 returns > 15, pre-rank by:                            │
│  │    composite_score = (similarity * 0.5) +                         │
│  │                      (normalized_rating * 0.3) +                  │
│  │                      (normalized_votes * 0.2)                     │
│  │  Take top 15                                                      │
│                                                                      │
│  BUDGET_MAP = {                                                      │
│      "low":    (0, 300),                                             │
│      "medium": (0, 800),                                             │
│      "high":   (0, 99999)                                            │
│  }                                                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Two-Stage Flow Diagram:**

```
          Thousands of restaurants in DB
                      │
         ┌────────────▼────────────┐
         │   STAGE 1: Hard Filter  │
         │   (SQL / structured)    │
         │                         │
         │   ✓ Location match      │
         │   ✓ Budget ceiling      │
         │   ✓ Min rating          │
         └────────────┬────────────┘
                      │
              ~50–200 candidates
                      │
         ┌────────────▼────────────┐
         │   STAGE 2: Semantic     │
         │   (Vector similarity)   │
         │                         │
         │   ✓ Cuisine affinity    │
         │   ✓ Ambiance match      │
         │   ✓ Review sentiment    │
         │   ✓ "budget-friendly"   │
         │     → cost < ₹500       │
         └────────────┬────────────┘
                      │
              Top 10–15 candidates
                      │
         ┌────────────▼────────────┐
         │   Context Window Guard  │
         │   (pre-rank if > 15)    │
         └────────────┬────────────┘
                      │
            10–15 candidates → LLM
```

**Semantic Search Advantage Over Basic Filtering:**

| User Input                    | Basic Filter Result          | Semantic Search Result                   |
|-------------------------------|------------------------------|------------------------------------------|
| `"budget-friendly"`           | ❌ No `budget-friendly` field | ✅ Maps to cost < ₹500 via embedding     |
| `"cozy Italian place"`        | ❌ Can only match cuisine     | ✅ Matches cuisine + ambiance in reviews |
| `"good for a date"`           | ❌ No filter available        | ✅ Finds "romantic", "intimate" in data  |
| `"like Toscano but cheaper"`  | ❌ Cannot process             | ✅ Finds similar restaurants at lower cost|

---

### 3.4 Prompt Engineering Module (`prompt_builder.py`)

> **Improvement #3** — Introduce prompt engineering layer as a separate module with chain-of-thought prompting.

**Responsibility:** A dedicated, isolated module for constructing LLM prompts. Decoupled from all application logic so prompt templates can be iterated independently without code changes.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     prompt_builder.py                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TEMPLATES_DIR = "prompts/"                                          │
│  │  ├── system.txt           # System instruction template           │
│  │  ├── user_context.txt     # User preferences template             │
│  │  ├── restaurant_data.txt  # Restaurant serialization template     │
│  │  ├── cot_reasoning.txt    # Chain-of-thought instruction          │
│  │  └── output_format.txt    # Output structure specification        │
│  │                                                                   │
│  build_prompt(preferences, restaurants, session=None) → PromptParts  │
│  │                                                                   │
│  ├── 1. SYSTEM MESSAGE (from system.txt)                             │
│  │   Role: "Expert Zomato restaurant advisor"                        │
│  │   Constraints: Only recommend from provided data                  │
│  │   Anti-hallucination: "Do NOT invent restaurants"                 │
│  │   Tone: Friendly, concise, helpful                                │
│  │                                                                   │
│  ├── 2. USER CONTEXT (from user_context.txt)                         │
│  │   Preferences + session history (if conversational)               │
│  │                                                                   │
│  ├── 3. RESTAURANT DATA (from restaurant_data.txt)                   │
│  │   Token-efficient serialization (key fields only):                │
│  │   ┌──────────────────────────────────────────────┐               │
│  │   │ [1] Toscano | Italian, Mediterranean |       │               │
│  │   │     ★4.3 (1200 votes) | ₹1500 | Casual      │               │
│  │   │     Dining | 🟢 Online 🟢 Book |             │               │
│  │   │     Popular: Margherita, Tiramisu             │               │
│  │   └──────────────────────────────────────────────┘               │
│  │                                                                   │
│  ├── 4. CHAIN-OF-THOUGHT INSTRUCTIONS (from cot_reasoning.txt)       │
│  │   "Before ranking, first analyze each restaurant:                 │
│  │    - How well does it match the user's cuisine preference?        │
│  │    - Is it within budget?                                         │
│  │    - Does the rating meet expectations?                           │
│  │    - Does the restaurant type fit additional preferences?         │
│  │    Then produce your final ranked list."                          │
│  │                                                                   │
│  ├── 5. OUTPUT FORMAT (from output_format.txt)                       │
│  │   - Rank top 5                                                    │
│  │   - Per-restaurant: name, match score, explanation                │
│  │   - Standout dishes / features                                    │
│  │   - Honest about imperfect matches                                │
│  │   - JSON structured output (for reliable parsing)                 │
│  │                                                                   │
│  └── Returns: PromptParts (system_msg, user_msg, or combined)        │
│                                                                      │
│  serialize_restaurants(df, max_tokens_per_entry=80) → str            │
│  │  Token-efficient format: key fields only                          │
│  │  Omit reviews, full descriptions to save context window           │
│                                                                      │
│  estimate_token_count(text) → int                                    │
│  │  Approximate token count (~4 chars per token)                     │
│  │  Warn if exceeding context budget                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Chain-of-Thought Prompting (vs Direct Ranking):**

```
DIRECT (Old):                        CHAIN-OF-THOUGHT (New):
┌────────────────────────┐           ┌────────────────────────────────┐
│ "Rank these 10          │           │ "For each restaurant:          │
│  restaurants for the    │           │  1. Assess cuisine match       │
│  user. Return top 5."   │           │  2. Check budget fit           │
│                         │           │  3. Evaluate rating            │
│ → LLM may skip reasoning│          │  4. Consider add'l prefs       │
│ → Inconsistent rankings │           │  5. Assign a fit score (1-10) │
│                         │           │                                │
│                         │           │  Then rank by fit score and    │
│                         │           │  explain your reasoning."      │
│                         │           │                                │
│                         │           │ → LLM reasons explicitly       │
│                         │           │ → More consistent rankings     │
│                         │           │ → Better explanations          │
└────────────────────────┘           └────────────────────────────────┘
```

**Why Separate Prompts Directory:**

- Prompt changes don't require code changes → faster iteration
- Version control prompts independently
- A/B test different prompt strategies without touching logic
- Non-engineers (product managers) can review and edit prompts

---

### 3.5 Recommender — LLM Adapter (`recommender.py`)

> **Improvement #8 (partial)** — LLM adapter pattern makes providers swappable.

**Responsibility:** Abstracted LLM interface using **Grok (xAI)** as the primary (and free) provider. The xAI API is **OpenAI-compatible**, so it uses the standard `openai` Python SDK with a custom `base_url`. Adapter pattern allows future swaps without touching upstream code.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       recommender.py                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────┐                     │
│  │  LLMAdapter (Abstract Base Class)          │                     │
│  │  ├── call(prompt) → str                    │                     │
│  │  ├── get_provider_name() → str             │                     │
│  │  └── get_model_name() → str                │                     │
│  ├────────────────────────────────────────────┤                     │
│  │  GrokAdapter(LLMAdapter) ← PRIMARY (FREE) │                     │
│  │  └── call() → openai SDK + base_url=       │                     │
│  │              "https://api.x.ai/v1"         │                     │
│  │     model: "grok-3-mini-fast" (cheapest)   │                     │
│  ├────────────────────────────────────────────┤                     │
│  │  LocalAdapter(LLMAdapter) ← FALLBACK       │                     │
│  │  └── call() → local Ollama / vLLM endpoint │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                      │
│  HOW GROK API WORKS (OpenAI-Compatible):                             │
│  ┌────────────────────────────────────────────────────────┐         │
│  │  from openai import OpenAI                             │         │
│  │                                                        │         │
│  │  client = OpenAI(                                      │         │
│  │      api_key=os.getenv("XAI_API_KEY"),                │         │
│  │      base_url="https://api.x.ai/v1",                  │         │
│  │  )                                                     │         │
│  │                                                        │         │
│  │  response = client.chat.completions.create(            │         │
│  │      model="grok-3-mini-fast",                         │         │
│  │      messages=[{"role": "user", "content": prompt}],   │         │
│  │      temperature=0.4,                                  │         │
│  │      max_tokens=1024,                                  │         │
│  │  )                                                     │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                      │
│  get_recommendations(prompt, candidates_df, adapter) → List[Rec]    │
│  │                                                                   │
│  ├── 1. Call adapter.call(prompt)                                    │
│  │   ├── Temperature: configurable (0.3–0.5)                        │
│  │   ├── Max Tokens: configurable (~1000)                            │
│  │   ├── Timeout: 30 seconds                                        │
│  │   └── Retry: 1 attempt on failure                                 │
│  │                                                                   │
│  ├── 2. Parse Response                                               │
│  │   ├── Try: JSON structured output                                 │
│  │   └── Fallback: Regex extraction of names + explanations          │
│  │                                                                   │
│  ├── 3. Validate (Anti-Hallucination)                                │
│  │   ├── Every recommended name MUST exist in candidates_df          │
│  │   ├── Drop hallucinated entries                                   │
│  │   └── If all dropped → invoke fallback_ranking()                  │
│  │                                                                   │
│  └── 4. Return: List[Recommendation]                                 │
│                                                                      │
│  fallback_ranking(candidates_df) → List[Rec]                         │
│  │  Rule-based ranking when LLM is unavailable:                      │
│  │  Sort by: rating DESC → votes DESC → cost ASC                    │
│  │  Return top 5 with generic explanation                            │
│  │  source = "fallback"                                              │
│                                                                      │
│  LLM_CONFIG (from .env):                                             │
│  ┌──────────────┬──────────────────────────────────┐                │
│  │ Parameter    │ Grok (grok-3-mini-fast)          │                │
│  ├──────────────┼──────────────────────────────────┤                │
│  │ Temperature  │ 0.4                              │                │
│  │ Max Tokens   │ 1024                             │                │
│  │ Top-P        │ 0.95                             │                │
│  │ Timeout      │ 30s                              │                │
│  │ Retry        │ 1 (then fallback)                │                │
│  │ Base URL     │ https://api.x.ai/v1              │                │
│  └──────────────┴──────────────────────────────────┘                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 3.6 Caching Layer (`cache.py`)

> **Improvement #5** — Implement a caching layer for frequent query patterns.

**Responsibility:** Cache LLM responses keyed by hashed user preferences. Reduces API costs and latency for repeated or similar queries.

```
┌──────────────────────────────────────────────────────────────────────┐
│                           cache.py                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CacheStrategy:                                                      │
│  ├── InMemoryCache   (dict-based, single process, dev/demo)         │
│  ├── RedisCache      (distributed, production)                       │
│  └── FileCache       (JSON file, simple persistence)                 │
│                                                                      │
│  generate_cache_key(preferences: UserPreferences) → str              │
│  │  Hash of normalized preferences:                                  │
│  │  key = sha256(f"{location}|{budget}|{cuisine}|{rating}|{prefs}") │
│  │  Example: "best_italian_koramangala_medium" → "a3f8c2..."        │
│                                                                      │
│  get(key: str) → Optional[CachedResponse]                            │
│  │  Returns cached LLM response if:                                  │
│  │  1. Key exists in cache                                           │
│  │  2. Entry has not expired (TTL check)                             │
│  │  Returns None on miss                                             │
│                                                                      │
│  set(key: str, response: List[Recommendation], ttl: int) → None     │
│  │  Store response with timestamp                                    │
│  │  Default TTL: 3600 seconds (1 hour)                               │
│                                                                      │
│  invalidate(key: str) → None                                         │
│  invalidate_all() → None                                             │
│                                                                      │
│  get_stats() → CacheStats                                            │
│  │  hits: int, misses: int, hit_rate: float, entries: int            │
│                                                                      │
│  CONFIGURATION:                                                      │
│  ├── CACHE_ENABLED = True                                            │
│  ├── CACHE_TTL = 3600          # seconds                             │
│  ├── CACHE_MAX_SIZE = 1000     # max entries                         │
│  └── CACHE_STRATEGY = "memory" # "memory", "redis", "file"          │
│                                                                      │
│  FLOW:                                                               │
│  ┌──────────┐    ┌─────────┐    ┌──────────────────┐               │
│  │  Request  │───▶│  Cache  │───▶│  HIT: Return     │               │
│  │          │    │  Lookup │    │  cached response  │               │
│  └──────────┘    └────┬────┘    └──────────────────┘               │
│                       │ MISS                                         │
│                       ▼                                               │
│                  ┌─────────┐    ┌──────────────────┐               │
│                  │  Full   │───▶│  Store in cache   │               │
│                  │  Pipeline│   │  with TTL          │               │
│                  └─────────┘    └──────────────────┘               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 3.7 Orchestrator / Controller (`orchestrator.py`)

> **Improvement #8** — Introduce a central controller that coordinates the entire flow.

**Responsibility:** Central coordinator that wires all components together. Each component is independently testable and replaceable. This is the only module that knows the full pipeline.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       orchestrator.py                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  class RecommendationOrchestrator:                                   │
│  │                                                                   │
│  │  __init__(                                                        │
│  │      input_parser:    InputParser,                                │
│  │      data_layer:      DataLayer,                                  │
│  │      retrieval:       RetrievalEngine,                            │
│  │      prompt_builder:  PromptBuilder,                              │
│  │      recommender:     Recommender,                                │
│  │      cache:           Cache,                                      │
│  │      session_mgr:     SessionManager,                             │
│  │      logger:          Logger                                      │
│  │  )                                                                │
│  │                                                                   │
│  │  process_request(raw_input, session_id=None) → RecommendationResp│
│  │  │                                                                │
│  │  ├── 1. PARSE INPUT                                               │
│  │  │   prefs = input_parser.parse(raw_input)                        │
│  │  │   If session_id: merge with session context                    │
│  │  │                                                                │
│  │  ├── 2. CHECK CACHE                                               │
│  │  │   cache_key = cache.generate_key(prefs)                        │
│  │  │   cached = cache.get(cache_key)                                │
│  │  │   If cached: log("cache_hit") → return cached                  │
│  │  │                                                                │
│  │  ├── 3. RETRIEVE (Two-Stage)                                      │
│  │  │   result = retrieval.retrieve(prefs)                           │
│  │  │                                                                │
│  │  ├── 4. BUILD PROMPT                                              │
│  │  │   prompt = prompt_builder.build(prefs, result.candidates)      │
│  │  │                                                                │
│  │  ├── 5. GET RECOMMENDATIONS                                       │
│  │  │   try:                                                         │
│  │  │       recs = recommender.get_recommendations(                  │
│  │  │           prompt, result.candidates                            │
│  │  │       )                                                        │
│  │  │   except LLMError:                                             │
│  │  │       recs = recommender.fallback_ranking(result.candidates)   │
│  │  │                                                                │
│  │  ├── 6. CACHE RESPONSE                                            │
│  │  │   cache.set(cache_key, recs)                                   │
│  │  │                                                                │
│  │  ├── 7. UPDATE SESSION                                            │
│  │  │   If session_id: session_mgr.update(session_id, prefs, recs)  │
│  │  │                                                                │
│  │  ├── 8. LOG EVERYTHING                                            │
│  │  │   logger.log_request(prefs, result, prompt, recs, latency)    │
│  │  │                                                                │
│  │  └── 9. RETURN                                                    │
│  │       RecommendationResponse(recs, result.filters_relaxed, ...)   │
│  │                                                                   │
│  │  COMPONENT SWAP EXAMPLE:                                          │
│  │  ┌──────────────────────────────────────────────────────┐        │
│  │  │  # Default: Grok (free)                              │        │
│  │  │  orchestrator = RecommendationOrchestrator(          │        │
│  │  │      ...                                             │        │
│  │  │      recommender=Recommender(adapter=GrokAdapter()), │        │
│  │  │      # FALLBACK: recommender=Recommender(            │        │
│  │  │      #     adapter=LocalAdapter()),  # Ollama        │        │
│  │  │      ...                                             │        │
│  │  │  )                                                   │        │
│  │  └──────────────────────────────────────────────────────┘        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Why an Orchestrator:**

| Without Orchestrator                     | With Orchestrator                               |
|------------------------------------------|------------------------------------------------|
| `app.py` knows all component internals   | `app.py` only calls `orchestrator.process()`   |
| Changing flow requires editing `app.py`  | Change flow in orchestrator only               |
| Hard to test components in isolation     | Each component injected → mock in tests        |
| Swapping LLM requires code changes      | Only adapter class changes                     |
| No single place for cross-cutting logic  | Logging, caching, sessions all in one place    |

---

### 3.8 Session Manager (`session_manager.py`)

> **Improvement #11** — Stateless core with optional session memory.

**Responsibility:** Maintain short-lived session context so users can say "show me something else" or "cheaper options" without repeating their full preferences. Sessions are opt-in; the core pipeline remains stateless.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     session_manager.py                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Storage Options:                                                    │
│  ├── InMemoryStore     (dict, single process, dev)                  │
│  └── RedisStore        (distributed, production)                     │
│                                                                      │
│  @dataclass                                                          │
│  Session:                                                            │
│      session_id: str                                                 │
│      created_at: datetime                                            │
│      last_active: datetime                                           │
│      base_preferences: UserPreferences        # Original prefs      │
│      history: List[TurnRecord]                 # Past interactions   │
│      excluded_restaurants: Set[str]            # "Show me something  │
│                                                #  else" tracking     │
│                                                                      │
│  @dataclass                                                          │
│  TurnRecord:                                                         │
│      turn_id: int                                                    │
│      user_input: str                                                 │
│      parsed_prefs: UserPreferences                                   │
│      recommendations: List[str]    # restaurant names               │
│      timestamp: datetime                                             │
│                                                                      │
│  INTERFACE:                                                          │
│  • create_session(prefs) → session_id                                │
│  • get_session(session_id) → Session | None                          │
│  • update_session(session_id, prefs, recs) → None                   │
│  • merge_with_session(session_id, new_input) → UserPreferences       │
│  │   Merge new input with session context:                           │
│  │   "cheaper options" → keep location, lower budget                 │
│  │   "show me something else" → exclude previous recs               │
│  │   "try Chinese instead" → swap cuisine, keep rest                │
│  • expire_session(session_id) → None                                 │
│                                                                      │
│  SESSION_CONFIG:                                                     │
│  ├── SESSION_TTL = 1800          # 30 minutes                        │
│  ├── MAX_HISTORY = 5             # last 5 turns                      │
│  └── CLEANUP_INTERVAL = 300      # garbage collect every 5 min       │
│                                                                      │
│  CONVERSATIONAL FLOW EXAMPLE:                                        │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Turn 1: "Italian in Koramangala under ₹1000"           │       │
│  │  → Session created with base prefs                       │       │
│  │  → Returns: Toscano, La Pinoz, etc.                      │       │
│  │                                                          │       │
│  │  Turn 2: "something cheaper"                             │       │
│  │  → Merges: keep location + cuisine, lower budget to low  │       │
│  │  → Returns: different set                                │       │
│  │                                                          │       │
│  │  Turn 3: "show me something else"                        │       │
│  │  → Excludes previous recommendations                     │       │
│  │  → Returns: new restaurants not shown before              │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 3.9 Observability & Logging (`logger.py`)

> **Improvement #10** — Add observability and logging for every query.

**Responsibility:** Structured logging of every request through the pipeline. Tracks what happened, how long it took, and (optionally) what the user thought of the results. This data becomes the evaluation dataset and future fine-tuning source.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          logger.py                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WHAT IS LOGGED (per request):                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  request_id:        UUID                                   │     │
│  │  timestamp:         ISO 8601                               │     │
│  │  session_id:        str | null                             │     │
│  │                                                            │     │
│  │  INPUT:                                                    │     │
│  │  ├── raw_input:     str (original user text)               │     │
│  │  └── parsed_prefs:  UserPreferences (structured)           │     │
│  │                                                            │     │
│  │  RETRIEVAL:                                                │     │
│  │  ├── stage1_count:  int (hard filter results)              │     │
│  │  ├── stage2_count:  int (semantic ranking results)         │     │
│  │  ├── filters_relaxed: List[str]                            │     │
│  │  └── retrieval_ms:  int (latency)                          │     │
│  │                                                            │     │
│  │  PROMPT:                                                   │     │
│  │  ├── prompt_text:   str (full prompt sent to LLM)          │     │
│  │  ├── token_estimate: int                                   │     │
│  │  └── template_version: str                                 │     │
│  │                                                            │     │
│  │  LLM:                                                      │     │
│  │  ├── provider:      str ("grok" / "local")                 │     │
│  │  ├── model:         str ("grok-3-mini-fast")               │     │
│  │  ├── raw_response:  str (full LLM output)                  │     │
│  │  ├── llm_ms:        int (latency)                          │     │
│  │  ├── source:        str ("llm" / "fallback" / "cache")     │     │
│  │  └── hallucinations_dropped: int                           │     │
│  │                                                            │     │
│  │  OUTPUT:                                                   │     │
│  │  ├── recommendations: List[{name, rank, score}]            │     │
│  │  └── total_ms:      int (end-to-end latency)               │     │
│  │                                                            │     │
│  │  FEEDBACK (optional, async):                               │     │
│  │  ├── user_accepted: str | null (which rec was clicked)     │     │
│  │  ├── user_rejected: bool                                   │     │
│  │  └── thumbs_up:     bool | null                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  INTERFACE:                                                          │
│  • log_request(RequestLog) → None                                   │
│  • log_feedback(request_id, FeedbackData) → None                    │
│  • get_metrics(time_range) → DashboardMetrics                       │
│  │  ├── avg_latency, p95_latency                                    │
│  │  ├── cache_hit_rate                                               │
│  │  ├── fallback_rate                                                │
│  │  ├── hallucination_rate                                           │
│  │  └── top_queried_locations, cuisines                              │
│                                                                      │
│  STORAGE:                                                            │
│  ├── Dev:   JSON Lines file (logs/queries.jsonl)                    │
│  ├── Prod:  Structured logging → BigQuery / PostgreSQL              │
│  └── Dashboard: Optional Streamlit/Grafana for metrics              │
│                                                                      │
│  WHY THIS MATTERS:                                                   │
│  • Evaluation dataset for prompt quality assessment                  │
│  • Fine-tuning data source (input → preferred output pairs)         │
│  • Debugging: reproduce any request from logs                       │
│  • Business metrics: popular locations, cuisines, peak times        │
│  • Cost tracking: LLM API calls, cache savings                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow & Sequence Diagrams

### 4a. End-to-End Sequence Diagram (Improved Pipeline)

```
 User      Orchestrator  Input    Session    Cache     Retrieval  Prompt    Recommender  LLM     Logger
  │            │         Parser    Mgr        │        Engine    Builder      │         API       │
  │            │           │        │          │          │         │          │          │        │
  │  Request   │           │        │          │          │         │          │          │        │
  │───────────▶│           │        │          │          │         │          │          │        │
  │            │  parse()  │        │          │          │         │          │          │        │
  │            │──────────▶│        │          │          │         │          │          │        │
  │            │  prefs    │        │          │          │         │          │          │        │
  │            │◀──────────│        │          │          │         │          │          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  merge_session()   │          │          │         │          │          │        │
  │            │───────────────────▶│          │          │         │          │          │        │
  │            │  merged_prefs      │          │          │         │          │          │        │
  │            │◀───────────────────│          │          │         │          │          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  cache.get(key)    │          │          │         │          │          │        │
  │            │──────────────────────────────▶│          │         │          │          │        │
  │            │  MISS                         │          │         │          │          │        │
  │            │◀──────────────────────────────│          │         │          │          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  retrieve(prefs)   │          │          │         │          │          │        │
  │            │─────────────────────────────────────────▶│         │          │          │        │
  │            │           │        │          │   ┌──────┴──────┐  │          │          │        │
  │            │           │        │          │   │ Stage 1:    │  │          │          │        │
  │            │           │        │          │   │ SQL Filter  │  │          │          │        │
  │            │           │        │          │   ├─────────────┤  │          │          │        │
  │            │           │        │          │   │ Stage 2:    │  │          │          │        │
  │            │           │        │          │   │ Vector Rank │  │          │          │        │
  │            │           │        │          │   └──────┬──────┘  │          │          │        │
  │            │  candidates (10–15)│          │          │         │          │          │        │
  │            │◀─────────────────────────────────────────│         │          │          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  build_prompt(prefs, candidates)         │         │          │          │        │
  │            │─────────────────────────────────────────────────▶│          │          │        │
  │            │  prompt (CoT)                                     │          │          │        │
  │            │◀──────────────────────────────────────────────────│          │          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  get_recommendations(prompt, candidates) │         │          │          │        │
  │            │────────────────────────────────────────────────────────────▶│          │        │
  │            │           │        │          │          │         │          │ call()   │        │
  │            │           │        │          │          │         │          │─────────▶│        │
  │            │           │        │          │          │         │          │ response │        │
  │            │           │        │          │          │         │          │◀─────────│        │
  │            │           │        │          │          │         │  validate + parse    │        │
  │            │  List[Recommendation]                    │         │          │          │        │
  │            │◀────────────────────────────────────────────────────────────│          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  cache.set(key, recs)         │          │         │          │          │        │
  │            │──────────────────────────────▶│          │         │          │          │        │
  │            │           │        │          │          │         │          │          │        │
  │            │  log_request(full_context)    │          │         │          │          │        │
  │            │──────────────────────────────────────────────────────────────────────────────────▶│
  │            │           │        │          │          │         │          │          │        │
  │  Results   │           │        │          │          │         │          │          │        │
  │◀───────────│           │        │          │          │         │          │          │        │
```

### 4b. Data Transformation Pipeline

```
     RAW DATASET                CLEANED + INDEXED              TWO-STAGE RETRIEVAL
  ┌─────────────────┐       ┌─────────────────────┐       ┌─────────────────────────┐
  │ rate: "4.1/5"   │       │ SQLite:              │       │ Stage 1 (SQL):          │
  │ cost: "1,200"   │  ETL  │ ┌─────────────────┐ │ Hard  │ WHERE location='btm'    │
  │ cuisines: "N.I.,│ ────▶ │ │ rate: 4.1       │ │ ────▶ │   AND cost <= 800       │
  │   Chinese"      │       │ │ cost: 1200      │ │Filter │   AND rate >= 3.5       │
  │ online: "Yes"   │       │ │ location: "btm" │ │       │                         │
  │ city: " BTM "   │       │ │ INDEX on: loc,  │ │       │ → ~50–200 candidates    │
  └─────────────────┘       │ │ cost, cuisine   │ │       └────────────┬────────────┘
                            │ └─────────────────┘ │                    │
                            │                     │       ┌────────────▼────────────┐
                            │ ChromaDB:            │       │ Stage 2 (Vector):       │
                            │ ┌─────────────────┐ │ Sem.  │ query: "cozy Italian    │
                            │ │ embedding: [..] │ │ ────▶ │   family-friendly"      │
                            │ │ doc: "Toscano.  │ │ Rank  │                         │
                            │ │  Italian, Med..." │       │ → Top 10–15 by cosine   │
                            │ └─────────────────┘ │       └────────────┬────────────┘
                            └─────────────────────┘                    │
                                                                       ▼
                                                          ┌──────────────────────────┐
                                                          │  Serialize (token-       │
                                                          │  efficient) + CoT Prompt │
                                                          └────────────┬─────────────┘
                                                                       │
                                                          ┌────────────▼─────────────┐
                                                          │  LLM: Reason → Rank →   │
                                                          │  Explain (top 5)         │
                                                          └────────────┬─────────────┘
                                                                       │
                                                          ┌────────────▼─────────────┐
                                                          │  Validate + Cache + Log  │
                                                          └──────────────────────────┘
```

---

## 5. Data Layer

### 5a. Dual-Store Architecture

> **Improvement #7** — Separate structured store from vector store.

| Aspect               | Structured Store (SQLite)              | Vector Store (ChromaDB / FAISS)               |
|----------------------|----------------------------------------|-----------------------------------------------|
| **Purpose**          | Exact match, range queries             | Semantic similarity search                    |
| **Query Type**       | `WHERE location = 'btm' AND cost < 800`| `"cozy Italian"` → nearest embeddings        |
| **Indexed Fields**   | location, approx_cost, rate, cuisines  | Composite document embeddings                 |
| **Speed**            | Sub-ms with indexes                    | ~10–50ms for top-K                             |
| **When Used**        | Stage 1: Hard filters                  | Stage 2: Semantic ranking                     |
| **Data Size**        | Full dataset (~50K rows)               | Full dataset (~50K embeddings)                |

### 5b. Dataset Schema (Post-Preprocessing)

| Column             | Type           | Nullable | Description                            | Indexed? | Derived? |
|--------------------|----------------|----------|----------------------------------------|----------|----------|
| `id`               | `int`          | No       | Auto-increment primary key             | PK       | **Yes**  |
| `name`             | `str`          | No       | Restaurant name                        | No       | No       |
| `online_order`     | `bool`         | No       | Online ordering available              | No       | No       |
| `book_table`       | `bool`         | No       | Table booking available                | No       | No       |
| `rate`             | `float`        | Yes*     | Numeric rating (0.0–5.0)              | **Yes**  | No       |
| `votes`            | `int`          | No       | Total number of votes                  | No       | No       |
| `approx_cost`      | `int`          | Yes*     | Cost for two (₹)                       | **Yes**  | No       |
| `location`         | `str`          | No       | Normalized locality name               | **Yes**  | No       |
| `cuisines`         | `List[str]`    | No       | List of cuisines served                | **FTS**  | No       |
| `rest_type`        | `str`          | Yes      | Restaurant type (Casual Dining, etc.)  | No       | No       |
| `dish_liked`       | `str`          | Yes      | Popular dishes                         | No       | No       |
| `listed_in_type`   | `str`          | Yes      | Dining type (Buffet, Delivery, etc.)   | No       | No       |
| `budget_tier`      | `str`          | No       | `"low"` / `"medium"` / `"high"`       | **Yes**  | **Yes**  |
| `is_new`           | `bool`         | No       | `True` if rating was `"NEW"`           | No       | **Yes**  |

> *Rows with null `rate` or `approx_cost` are dropped during preprocessing.  
> **FTS** = Full-Text Search index for substring/multi-value matching.

### 5c. Embedding Document Schema (ChromaDB)

```python
# What gets embedded per restaurant:
document = (
    f"{name}. "
    f"Cuisines: {', '.join(cuisines)}. "
    f"Type: {rest_type}. "
    f"Popular dishes: {dish_liked}. "
    f"Cost ₹{approx_cost} for two. "
    f"Rating: {rate}/5 with {votes} votes. "
    f"Location: {location}."
)

# Metadata stored alongside (for filtering):
metadata = {
    "id": restaurant_id,
    "name": name,
    "location": location,
    "approx_cost": approx_cost,
    "rate": rate,
    "budget_tier": budget_tier
}
```

---

## 6. Module Interface Contracts

### 6a. Core Data Objects

```python
@dataclass
class UserPreferences:
    location: str                       # Required — normalized locality
    budget: str                         # Required — "low", "medium", "high"
    budget_max: int                     # Derived — max cost from BUDGET_MAP
    cuisine: Optional[str] = None       # Optional — e.g., "Italian"
    min_rating: float = 0.0             # Optional — minimum rating threshold
    additional_prefs: str = ""          # Optional — free text for LLM + semantic
    raw_input: str = ""                 # Original user text (for logging)

@dataclass
class RetrievalResult:
    candidates: List[dict]              # Restaurants passing both stages
    total_stage1: int                   # Count after hard filters
    total_stage2: int                   # Count after semantic ranking
    filters_relaxed: List[str]          # Which filters were relaxed (if any)
    similarity_scores: List[float]      # Per-candidate semantic scores

@dataclass
class Recommendation:
    rank: int                           # 1-based position in ranked list
    name: str                           # Restaurant name (from dataset)
    cuisines: List[str]                 # Cuisines served
    rating: float                       # Numeric rating
    votes: int                          # Vote count
    cost_for_two: int                   # Cost in ₹
    online_order: bool                  # Online ordering available
    book_table: bool                    # Table booking available
    explanation: str                    # AI-generated reasoning
    source: str                         # "llm", "fallback", or "cache"
    similarity_score: float = 0.0       # Semantic similarity (0–1)

@dataclass
class RecommendationResponse:
    recommendations: List[Recommendation]
    filters_relaxed: List[str]
    source: str                         # "llm", "fallback", "cache"
    latency_ms: int                     # End-to-end latency
    request_id: str                     # UUID for tracking
```

### 6b. Inter-Module Call Graph

```
orchestrator.py (Central Controller)
 │
 ├── input_parser.parse(raw_input)                → UserPreferences
 │
 ├── session_manager.merge(session_id, prefs)      → UserPreferences (enriched)
 │
 ├── cache.get(cache_key)                          → Optional[List[Recommendation]]
 │       └── (if HIT: skip steps 3–5, return cached)
 │
 ├── retrieval_engine.retrieve(prefs)              → RetrievalResult
 │       ├── (Stage 1) data_layer.query_structured(filters) → List[dict]
 │       └── (Stage 2) data_layer.query_semantic(text, ids) → List[dict]
 │
 ├── prompt_builder.build(prefs, candidates)       → str (prompt)
 │       └── loads templates from prompts/ directory
 │
 ├── recommender.get_recommendations(prompt, df)   → List[Recommendation]
 │       ├── (calls LLM via adapter)
 │       ├── (validates against candidates)
 │       └── (fallback_ranking on failure)
 │
 ├── cache.set(cache_key, recs)                    → None
 │
 ├── session_manager.update(session_id, prefs, recs) → None
 │
 └── logger.log_request(full_context)              → None
```

---

## 7. Prompt Engineering Architecture

> **Improvement #3** — Dedicated prompt module with chain-of-thought prompting.

### 7a. Prompt Template Layers

| Layer                    | Content                                              | Token Budget | File                  |
|--------------------------|------------------------------------------------------|--------------|-----------------------|
| **System Instruction**   | Role, constraints, tone, anti-hallucination rules    | ~120 tokens  | `prompts/system.txt`  |
| **User Preferences**     | Location, budget, cuisine, rating, additional prefs  | ~60 tokens   | `prompts/user_context.txt` |
| **Restaurant Data**      | 10–15 token-efficient serialized entries             | ~500 tokens  | `prompts/restaurant_data.txt` |
| **Chain-of-Thought**     | Step-by-step reasoning instructions                  | ~150 tokens  | `prompts/cot_reasoning.txt` |
| **Output Format Spec**   | JSON structure, ranking rules, honesty clause        | ~100 tokens  | `prompts/output_format.txt` |
| **Total Input**          |                                                      | **~930 tokens** |                    |
| **Reserved for Output**  |                                                      | **~1000 tokens** |                   |

### 7b. Context Window Management

> **Improvement #4** — Define limits and compress data.

| Control                         | Implementation                                          |
|---------------------------------|---------------------------------------------------------|
| Max restaurants in prompt       | 15 (hard cap)                                           |
| Pre-ranking when > 15           | Composite score (similarity × 0.5 + rating × 0.3 + votes × 0.2) |
| Token-efficient serialization   | Key fields only, no full descriptions or reviews        |
| Token estimation                | ~4 chars per token, warn if > 2000 input tokens         |
| Dynamic compression             | If > budget, drop `dish_liked` and `rest_type` fields   |

### 7c. Anti-Hallucination Guardrails

1. **Prompt-level:** Explicit instruction — *"Only recommend restaurants from the numbered list below. Do NOT invent, assume, or reference any restaurant not in this list."*
2. **Chain-of-Thought:** LLM must reference restaurant numbers — *"Refer to each restaurant by its [number] when reasoning."*
3. **Post-processing:** Cross-validate every restaurant name in LLM output against `candidates_df['name']`
4. **Fallback:** If validation drops all entries, invoke `fallback_ranking()` instead

### 7d. LLM Configuration — Grok (xAI)

| Parameter        | Grok (`grok-3-mini-fast`) | Local Fallback (Ollama) | Notes                            |
|------------------|---------------------------|-------------------------|----------------------------------|
| Temperature      | `0.4`                     | `0.4`                   | Balanced creativity/consistency  |
| Max Output Tokens| `1024`                    | `1024`                  | ~5 recommendations + CoT        |
| Top-P            | `0.95`                    | `0.95`                  | Default                          |
| Timeout          | `30s`                     | `60s`                   | Local models may be slower       |
| Retry on Failure | 1 retry                   | 2 retries               | Then rule-based fallback         |
| Base URL         | `https://api.x.ai/v1`     | `http://localhost:11434` | OpenAI-compatible endpoints     |
| SDK              | `openai` (Python)         | `openai` (Python)       | Same SDK, different base_url     |
| **Cost**         | **Free (signup credits)**  | **Free (local GPU)**    | **$0 for this project**          |

---

## 8. Error Handling & Resilience

> **Improvement #6** — Add fallback and graceful degradation at every layer.

### 8a. Error Categories & Recovery

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           ERROR HANDLING MATRIX                                │
├──────────────────────┬────────────────────┬───────────────────────────────────┤
│ Error Category       │ Detection          │ Recovery Strategy                 │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Dataset Load Fail    │ FileNotFoundError  │ Download from HuggingFace →      │
│                      │ ConnectionError    │ cache locally → retry             │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ DB Init Failure      │ sqlite3.Error      │ Fallback to pandas in-memory     │
│                      │                    │ (degrade gracefully)              │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Vector Store Down    │ ChromaDB error     │ Skip Stage 2; send Stage 1       │
│                      │                    │ results (sorted by rating) to LLM│
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Invalid Location     │ 0 results Stage 1  │ Fuzzy match → suggest locations; │
│                      │                    │ show available list to user       │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Too Few Results      │ len(stage1) < 3    │ Progressive filter relaxation    │
│                      │                    │ (cuisine → budget → rating)      │
│                      │                    │ → inform user what was relaxed   │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Too Many Results     │ len(stage1) > 200  │ Stage 2 semantic ranking handles │
│                      │                    │ this naturally; cap at top 15    │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ NLP Parse Failure    │ InputParser can't  │ Fallback: ask user to fill form  │
│                      │ extract entities   │ (structured input)               │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ LLM API Timeout      │ TimeoutError       │ 1 retry → fallback_ranking()    │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ LLM Rate Limited     │ HTTP 429           │ Exponential backoff (1 attempt)  │
│                      │                    │ → fallback_ranking()             │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ LLM Hallucination    │ Name not in df     │ Drop invalid entries; if all     │
│                      │                    │ invalid → fallback_ranking()     │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Malformed LLM Output │ JSONDecodeError    │ Regex fallback parsing →         │
│                      │                    │ fallback_ranking() if fails      │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Missing API Key      │ AuthError on start │ Clear error message with .env    │
│                      │                    │ setup instructions               │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Cache Failure        │ Redis/file error   │ Skip cache; process normally     │
│                      │                    │ (cache is non-critical)          │
├──────────────────────┼────────────────────┼───────────────────────────────────┤
│ Session Expired      │ TTL exceeded       │ Treat as new session; log        │
│                      │                    │ warning; don't error             │
└──────────────────────┴────────────────────┴───────────────────────────────────┘
```

### 8b. Degradation Cascade

```
                 FULL PIPELINE                    DEGRADED MODES
            ┌─────────────────┐
            │  Normal Flow    │
            │  (all systems   │
            │   operational)  │
            └────────┬────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   Vector DB      LLM API      Cache
    down?          down?        down?
        │            │            │
        ▼            ▼            ▼
  ┌───────────┐ ┌──────────┐ ┌──────────┐
  │ Skip      │ │ fallback │ │ Skip     │
  │ Stage 2   │ │ ranking  │ │ cache    │
  │           │ │          │ │          │
  │ Send      │ │ Sort by: │ │ Process  │
  │ Stage 1   │ │ rating × │ │ normally │
  │ results   │ │ votes    │ │ every    │
  │ to LLM    │ │          │ │ request  │
  │ (sorted)  │ │ Generic  │ │          │
  └───────────┘ │ explain  │ └──────────┘
                └──────────┘

  Impact:       Impact:       Impact:
  Lower         No AI-gen     Higher
  relevance     explanations  latency +
  accuracy      but still     API cost
                functional
```

---

## 9. Tech Stack & Dependencies

### 9a. Runtime Dependencies

| Package                        | Version   | Purpose                                  | Cost     |
|--------------------------------|-----------|------------------------------------------|----------|
| `python`                       | ≥ 3.10    | Runtime                                  | **Free** |
| `pandas`                       | ≥ 2.0     | Data loading, cleaning, initial ETL      | **Free** |
| `datasets`                     | ≥ 2.14    | HuggingFace dataset loading              | **Free** |
| `openai`                       | ≥ 1.0     | Grok API via OpenAI-compatible SDK       | **Free** |
| `python-dotenv`                | ≥ 1.0     | Load API keys from `.env`                | **Free** |
| `chromadb`                     | ≥ 0.4     | Vector store for semantic search (OSS)   | **Free** |
| `sentence-transformers`        | ≥ 2.2     | Embedding model (`all-MiniLM-L6-v2`)     | **Free** |
| `sqlite3`                      | (stdlib)  | Structured data store                    | **Free** |

> **No `google-generativeai` or paid SDK needed.** The `openai` package connects to xAI's Grok API
> by simply setting `base_url="https://api.x.ai/v1"`. This is the same SDK — zero extra dependencies.

### 9b. Dev / Optional Dependencies

| Package                        | Purpose                                          |
|--------------------------------|--------------------------------------------------|
| `flask`                        | Web UI backend (chosen frontend framework)       |
| `pytest`                       | Unit tests                                       |
| `jupyter`                      | Notebook-based interaction (optional backup)     |
| `thefuzz` (fuzzywuzzy)         | Fuzzy string matching for location resolution    |

### 9c. Environment Configuration

```
.env
├── XAI_API_KEY=<your-xai-api-key>               # Grok API key (get free at console.x.ai)
├── XAI_BASE_URL=https://api.x.ai/v1             # xAI endpoint (OpenAI-compatible)
├── LLM_MODEL=grok-3-mini-fast                   # Model name (cheapest Grok model)
├── LLM_TEMPERATURE=0.4                          # Response creativity
├── LLM_MAX_TOKENS=1024                          # Max output tokens
├── DATA_PATH=data/zomato.csv                    # Local dataset cache path
├── DB_PATH=data/restaurants.db                  # SQLite database path
├── VECTOR_DB_PATH=data/chroma_store             # ChromaDB persistence path
├── CACHE_ENABLED=true                           # Enable/disable response cache
├── CACHE_TTL=3600                               # Cache time-to-live (seconds)
├── CACHE_STRATEGY=memory                        # "memory", "redis", "file"
├── SESSION_ENABLED=true                         # Enable/disable session memory
├── SESSION_TTL=1800                             # Session timeout (seconds)
├── LOG_LEVEL=INFO                               # Logging level
└── LOG_PATH=logs/queries.jsonl                  # Query log output path
```

### 9d. Free-Tier Cost Strategy

> **Goal: $0 total project cost.** Every component in this stack is free.

| Component              | How It's Free                                                          |
|------------------------|------------------------------------------------------------------------|
| **Grok API (xAI)**     | $25 free signup credits at [console.x.ai](https://console.x.ai). Use `grok-3-mini-fast` (cheapest model) + aggressive caching to stay within budget. Optionally opt into data-sharing for $150/mo additional credits. |
| **HuggingFace Dataset**| Public dataset, free to download                                       |
| **ChromaDB**           | Open-source, runs locally, no cloud costs                              |
| **sentence-transformers** | Open-source model, runs locally on CPU                             |
| **SQLite**             | Python stdlib, no server needed                                        |
| **Python + all libs**  | All pip packages are open-source                                       |

**Token Cost Optimization:**
- `grok-3-mini-fast` is the cheapest xAI model
- Aggressive caching (TTL = 1 hour) avoids repeat API calls
- Context window guard limits to 15 restaurants per prompt (~930 input tokens)
- Chain-of-thought adds ~150 tokens but improves output quality (fewer retries)
- **Estimated usage:** ~50 unique queries before credits run out (with caching, effectively unlimited repeats)

---

## 10. Project Directory Structure

```
Zomato Milestone 1 AI/
│
├── Docs/
│   ├── Problemstatement.txt          # Original problem statement
│   ├── context.md                    # Full project context & requirements
│   └── architecture.md               # This file — system architecture
│
├── data/
│   ├── zomato.csv                    # Downloaded / cached Zomato dataset
│   ├── restaurants.db                # SQLite structured store (auto-generated)
│   └── chroma_store/                 # ChromaDB vector store (auto-generated)
│
├── prompts/                          # Prompt templates (editable without code changes)
│   ├── system.txt                    # System instruction template
│   ├── user_context.txt              # User preferences template
│   ├── restaurant_data.txt           # Restaurant serialization template
│   ├── cot_reasoning.txt             # Chain-of-thought instruction
│   └── output_format.txt             # Output structure specification
│
├── src/
│   ├── __init__.py                   # Package init
│   ├── orchestrator.py               # Central controller — coordinates full pipeline
│   ├── input_parser.py               # NLP entity extraction + synonym resolution
│   ├── data_layer.py                 # SQLite + ChromaDB dual-store management
│   ├── data_loader.py                # Dataset ingestion, preprocessing, ETL into stores
│   ├── retrieval_engine.py           # Two-stage retrieval (hard filter + semantic rank)
│   ├── prompt_builder.py             # Prompt construction from templates + CoT
│   ├── recommender.py                # LLM adapter + response parsing + validation
│   ├── cache.py                      # TTL-based response caching
│   ├── session_manager.py            # Optional session memory for follow-up queries
│   ├── logger.py                     # Structured observability logging
│   ├── models.py                     # Shared data classes (UserPreferences, Recommendation, etc.)
│   └── app.py                        # Application entry point (web/CLI/notebook)
│
├── templates/                        # Flask HTML templates (premium UI)
│   └── index.html                    # Full-page premium template (hero + search + cards)
│
├── static/                           # Static assets (premium design system)
│   ├── style.css                     # Design tokens, glassmorphism, animations, responsive grid
│   └── script.js                     # Search logic, typing animation, card rendering, feedback
│
├── logs/                             # Query logs (auto-generated)
│   └── queries.jsonl                 # Structured request/response logs
│
├── tests/
│   ├── test_input_parser.py          # NLP parsing + synonym resolution tests
│   ├── test_data_layer.py            # SQLite + ChromaDB integration tests
│   ├── test_retrieval_engine.py      # Two-stage retrieval tests
│   ├── test_prompt_builder.py        # Prompt construction + token budget tests
│   ├── test_recommender.py           # LLM adapter + fallback tests
│   ├── test_cache.py                 # Cache hit/miss/TTL tests
│   ├── test_session_manager.py       # Session lifecycle tests
│   └── test_orchestrator.py          # End-to-end integration tests
│
├── .env                              # API keys (NOT committed to git)
├── .env.example                      # Template for .env setup
├── .gitignore                        # Excludes .env, data/, __pycache__/, logs/
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview & setup instructions
```

---

## 11. Scalability & Future Extensions

### 11a. Current Scope (Milestone 1)

- Single-city dataset (primarily Bangalore)
- Two-stage retrieval (SQL + vector search)
- Chain-of-thought LLM prompting
- Optional session memory for follow-ups
- In-memory caching with TTL
- Structured query logging

### 11b. What's Built for Extension

| Capability Built Now              | Future Extension It Enables                         |
|-----------------------------------|-----------------------------------------------------|
| **Dual data store**               | Scale to multi-city, millions of restaurants        |
| **Vector embeddings**             | Add review sentiment, image embeddings, taste profiles |
| **Adapter pattern (LLM)**         | Swap to fine-tuned model, A/B test models           |
| **Separate prompt templates**     | Non-engineer prompt iteration, A/B test prompts     |
| **Structured logging**            | Fine-tuning data, analytics dashboard, cost tracking|
| **Session manager**               | Multi-turn conversation, user preference learning   |
| **Caching layer**                 | Redis cluster for production, CDN-level caching     |
| **Orchestrator pattern**          | Plugin new stages (re-ranking, personalization)     |

### 11c. Performance Considerations

| Concern                 | Current Mitigation                       | At Scale                                  |
|-------------------------|------------------------------------------|-------------------------------------------|
| Dataset load time       | One-time ETL into SQLite + ChromaDB      | PostgreSQL + pgvector / managed vector DB |
| Filter latency          | SQLite indexed queries (~sub-ms)         | Connection pooling, read replicas         |
| Semantic search latency | ChromaDB top-K (~10–50ms)                | FAISS GPU index / managed Vertex AI search|
| LLM latency             | Single API call (~1–3s)                  | Async calls, streaming, response caching  |
| Prompt token cost       | Cap at 15 restaurants, key fields only   | Summarize via smaller model before main LLM |
| Cache effectiveness     | In-memory TTL cache                      | Redis cluster, distributed cache          |
| Log volume              | JSONL file                               | BigQuery/Elasticsearch for analytics      |

---

## Improvement Traceability

| # | Improvement                               | Where It's Implemented                                   |
|---|-------------------------------------------|----------------------------------------------------------|
| 1 | RAG pipeline with vector DB               | §3.2 Data Layer, §3.3 Retrieval Engine (Stage 2)        |
| 2 | Two-stage retrieval architecture          | §3.3 Retrieval Engine (Stage 1 + Stage 2)                |
| 3 | Separate prompt engineering module        | §3.4 Prompt Builder, `prompts/` directory                |
| 4 | Context window management                 | §3.3 Context Window Guard, §7b Token Budget              |
| 5 | Caching layer                             | §3.6 Cache, §9c Configuration                            |
| 6 | Fallback and graceful degradation         | §3.5 fallback_ranking(), §8 Error Matrix, §8b Cascade   |
| 7 | Separate data layer (SQLite + vector DB)  | §3.2 Data Layer, §5 Dual-Store Architecture              |
| 8 | Orchestration / controller layer          | §3.7 Orchestrator, §6b Call Graph                        |
| 9 | Input parsing module                      | §3.1 Input Parser, synonym map, fuzzy matching           |
| 10| Observability and logging                 | §3.9 Logger, structured JSONL logging                    |
| 11| Stateless + optional session memory       | §3.8 Session Manager, conversational follow-ups          |

---

> **Source Document:** [context.md](file:///c:/Users/KAbIRAJ/Desktop/NextLeap%20PM%20Fellowship/Zomato%20Milestone%201%20AI/Docs/context.md)  
> **Problem Statement:** [Problemstatement.txt](file:///c:/Users/KAbIRAJ/Desktop/NextLeap%20PM%20Fellowship/Zomato%20Milestone%201%20AI/Docs/Problemstatement.txt)
