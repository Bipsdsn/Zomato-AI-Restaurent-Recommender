# Project Context: AI-Powered Restaurant Recommendation System (Zomato Use Case)

## Problem Statement

Build an **AI-powered restaurant recommendation service** inspired by Zomato. The system should intelligently suggest restaurants based on user preferences by combining structured data with a Large Language Model (LLM).

The core idea is a **Retrieval-Augmented Generation (RAG)-style pipeline**: structured restaurant data is filtered first (retrieval), and the filtered subset is then passed to an LLM for intelligent ranking and natural-language explanation (generation).

> **💰 Cost Constraint:** The entire project must be built and run **for free** — no paid APIs, no cloud costs. All tools (Grok API, ChromaDB, SQLite, HuggingFace) are free/open-source.

---

## Objective

Design and implement an application that:

1. **Takes user preferences** — location, budget, cuisine type, minimum rating, and optional qualitative preferences (e.g., "family-friendly", "quick service", "rooftop dining")
2. **Uses a real-world dataset** — the Zomato restaurant dataset hosted on Hugging Face
3. **Leverages an LLM** — to generate personalized, human-like recommendations with reasoning
4. **Displays clear and useful results** — in a user-friendly format (web UI, CLI, or notebook)

---

## High-Level Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  User Input  │────▶│  Data Filtering  │────▶│   LLM Prompt    │────▶│   Output /   │
│  (Prefs UI)  │     │  (pandas/SQL)    │     │  Construction   │     │   Display    │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                           │                         │
                    ┌──────┴──────┐           ┌──────┴──────┐
                    │   Zomato    │           │  Grok API   │
                    │  Dataset    │           │  (xAI)      │
                    │ (HuggingFace)│          │  FREE tier  │
                    └─────────────┘           └─────────────┘
```

---

## System Workflow — Detailed Breakdown

### 1. Data Ingestion

**Source:**  
[ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) (Hugging Face Datasets)

**Key Fields to Extract & Normalize:**

| Field               | Expected Type | Description                                           | Preprocessing Notes                                      |
|----------------------|---------------|-------------------------------------------------------|----------------------------------------------------------|
| `name`              | `string`      | Restaurant name                                       | Strip whitespace, normalize casing                       |
| `online_order`      | `string`      | Whether online ordering is available (`Yes` / `No`)   | Convert to boolean                                       |
| `book_table`        | `string`      | Whether table booking is available (`Yes` / `No`)     | Convert to boolean                                       |
| `rate`              | `string`      | Rating (e.g., `"4.1/5"`, `"NEW"`, `"-"`)             | Parse to float; handle `"NEW"`, `"-"`, `null` gracefully |
| `votes`             | `int`         | Total number of votes/reviews                         | Use as a popularity signal                               |
| `approx_cost`       | `string`      | Approximate cost for two people (e.g., `"800"`)       | Parse to int; remove commas if present                   |
| `listed_in(type)`   | `string`      | Dining type (Buffet, Delivery, Dine-out, etc.)        | Use for filtering                                        |
| `listed_in(city)`   | `string`      | Locality/area within the city                         | Primary location filter                                  |
| `cuisines`          | `string`      | Comma-separated cuisines (e.g., `"North Indian, Chinese"`) | Split into list; normalize casing                   |
| `rest_type`         | `string`      | Restaurant type (Casual Dining, Quick Bites, etc.)    | Useful for "additional preferences" matching             |
| `dish_liked`        | `string`      | Popular dishes (may be empty)                         | Optional; can enrich LLM context                         |
| `reviews_list`      | `string`      | Sample reviews (stringified list)                     | Optional; can parse for sentiment or LLM context         |

**Preprocessing Steps:**

1. Load dataset via `datasets` library (`load_dataset`) or download CSV
2. Drop rows with completely missing critical fields (`name`, `rate`, `approx_cost`)
3. Parse `rate` column — extract numeric value, flag `"NEW"` restaurants separately
4. Convert `approx_cost` to integer
5. Split `cuisines` into a list of individual cuisine types
6. Normalize location strings (lowercase, strip)
7. Create derived columns if needed (e.g., `budget_tier` from `approx_cost`)

---

### 2. User Input

Collect the following preferences from the user:

| Preference               | Input Type     | Examples                                  | Required? | How It's Used                                             |
|--------------------------|----------------|-------------------------------------------|-----------|-----------------------------------------------------------|
| **Location / Area**      | Text / Dropdown| `"Koramangala"`, `"Connaught Place"`      | ✅ Yes    | Filters `listed_in(city)` column                          |
| **Budget**               | Select         | `Low`, `Medium`, `High`                   | ✅ Yes    | Maps to `approx_cost` ranges (see below)                  |
| **Cuisine**              | Text / Multi   | `"Italian"`, `"Chinese"`, `"North Indian"`| ❌ No     | Filters `cuisines` column (substring match)               |
| **Minimum Rating**       | Slider / Number| `3.5`, `4.0`                              | ❌ No     | Filters `rate >= min_rating`                               |
| **Additional Preferences**| Free Text     | `"family-friendly"`, `"quick service"`    | ❌ No     | Passed directly to LLM prompt for qualitative reasoning   |

**Budget Tier Mapping (suggested):**

| Tier       | `approx_cost` Range (₹) |
|------------|--------------------------|
| **Low**    | ₹0 – ₹300               |
| **Medium** | ₹301 – ₹800             |
| **High**   | ₹801+                   |

> These thresholds should be calibrated based on the actual dataset distribution.

---

### 3. Integration Layer (Filtering + Prompt Construction)

**Step 3a — Data Filtering:**

Apply hard filters to the dataset based on user inputs:

```
filtered_df = df[
    (df['location'] == user_location) &
    (df['approx_cost'] <= budget_max) &
    (df['rate'] >= min_rating) &
    (df['cuisines'].str.contains(cuisine, case=False))  # if cuisine provided
]
```

- If filters return **too few results** (< 3), progressively relax constraints:
  1. Remove cuisine filter
  2. Expand budget range by ±20%
  3. Lower minimum rating by 0.5
- If filters return **too many results** (> 20), narrow by:
  1. Sorting by `votes` (popularity) and taking top 15–20
  2. Adding dining-type filter if available

**Step 3b — Prompt Construction:**

Build a structured prompt for the LLM containing:

1. **System instruction** — Define the LLM's role (expert food recommender)
2. **User preferences** — Location, budget, cuisine, rating, additional prefs
3. **Restaurant data** — Serialized as a numbered list or JSON array (top 10–15 candidates)
4. **Output format instructions** — Ask for ranked list with explanations

**Example Prompt Template (for Grok):**

```
You are an expert restaurant recommendation assistant for Zomato.

A user is looking for restaurants with these preferences:
- Location: {location}
- Budget: {budget_tier} (up to ₹{budget_max} for two)
- Cuisine: {cuisine}
- Minimum Rating: {min_rating}
- Additional Preferences: {additional_prefs}

Here are the matching restaurants:
{serialized_restaurant_data}

IMPORTANT: Only recommend restaurants from the list above. Do NOT invent any.

Please:
1. First, briefly analyze each restaurant's fit for this user.
2. Then rank the top 5 restaurants from best to worst fit.
3. For each, explain WHY it's a good match for this user.
4. Mention any standout dishes or features.
5. If none are a perfect match, say so honestly.
```

---

### 4. Recommendation Engine (LLM)

**LLM Provider: Grok by xAI (FREE)**

| Setting              | Value                              | Notes                                     |
|----------------------|------------------------------------|-------------------------------------------|
| **Provider**         | xAI (Grok)                         | Free $25 signup credits at [console.x.ai](https://console.x.ai) |
| **Model**            | `grok-3-mini-fast`                 | Cheapest Grok model — maximizes free credit usage |
| **SDK**              | `openai` Python package            | xAI API is OpenAI-compatible — same SDK, different `base_url` |
| **Base URL**         | `https://api.x.ai/v1`             | Set via `base_url` param in OpenAI client |
| **Temperature**      | `0.4`                              | Balanced creativity/consistency           |
| **Max Tokens**       | `~1024`                            | Enough for 5 recommendations + CoT reasoning |
| **Response Format**  | JSON or numbered list              | For reliable parsing                      |

**How the Grok API Works (code snippet):**

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

response = client.chat.completions.create(
    model="grok-3-mini-fast",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.4,
    max_tokens=1024,
)
```

> **Why Grok?** Free signup credits, OpenAI-compatible API (zero learning curve), and strong reasoning for structured ranking tasks. No `google-generativeai` or paid SDK required.

**What the LLM Should Do:**

- ✅ **Rank** the filtered restaurants based on how well they match user preferences
- ✅ **Explain** each recommendation — why it fits the user's needs
- ✅ **Highlight** standout features (popular dishes, high votes, table booking, online ordering)
- ✅ **Summarize** the overall recommendation set (optional closing paragraph)
- ❌ **NOT hallucinate** restaurants that aren't in the provided data
- ❌ **NOT ignore** user constraints (budget, rating, etc.)

---

### 5. Output Display

Present the top recommendations in a structured, user-friendly format:

| Field                        | Source                | Description                                             |
|------------------------------|----------------------|---------------------------------------------------------|
| **Rank**                     | LLM output           | Position in the ranked list (1–5)                       |
| **Restaurant Name**          | Dataset + LLM        | Name of the restaurant                                  |
| **Cuisine(s)**               | Dataset              | Types of cuisine served                                 |
| **Rating**                   | Dataset              | Aggregate rating (e.g., `4.2/5`)                        |
| **Votes**                    | Dataset              | Number of user votes (social proof)                     |
| **Estimated Cost for Two**   | Dataset              | Approximate cost in ₹                                   |
| **Online Order / Table Book**| Dataset              | Availability indicators                                 |
| **AI Explanation**           | LLM output           | Personalized reason why this restaurant is recommended   |

**Display Format:**

- **Web App (Primary)** — Premium Flask UI with dark-themed glassmorphism cards, animated search bar with cycling placeholders, AI insight typewriter effect, star ratings, filter pills, responsive grid layout, and skeleton loading states

---

## Edge Cases & Error Handling

| Scenario                                  | Handling Strategy                                          |
|-------------------------------------------|------------------------------------------------------------|
| No restaurants match all filters          | Relax filters progressively; inform user                   |
| Dataset has missing/malformed ratings     | Default to `0.0` or exclude; flag as "Unrated"             |
| User enters an invalid location           | Show available locations; suggest closest match             |
| LLM hallucinates a restaurant             | Cross-validate LLM output against the filtered dataset     |
| LLM API is unavailable / rate-limited     | Fallback to rule-based ranking (sort by rating × votes)    |
| Cost field is missing                     | Exclude from budget filtering; note "Cost not available"   |

---

## Data Source

- **Dataset:** [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation) (Hugging Face)
- **Format:** CSV / Parquet via Hugging Face `datasets` library
- **Scope:** Restaurants primarily in Bangalore (based on typical Zomato datasets)

---

## Tech Stack (100% Free)

| Layer                | Technology                                    | Purpose                              | Cost     |
|----------------------|-----------------------------------------------|--------------------------------------|----------|
| **Language**         | Python 3.10+                                  | Primary development language         | **Free** |
| **Data Processing**  | `pandas`, `datasets` (Hugging Face)           | Loading, cleaning, filtering data    | **Free** |
| **LLM Integration**  | `openai` SDK → Grok API (`api.x.ai/v1`)      | Sending prompts, receiving responses | **Free** |
| **Frontend (Web)**   | Flask + vanilla HTML/CSS/JS (premium design)  | Zomato-inspired dark UI with glassmorphism  | **Free** |
| **Typography**       | Google Fonts (`Inter`, `Outfit`)              | Modern typography — no browser defaults    | **Free** |
| **Environment**      | `.env` for API keys, `requirements.txt`       | Configuration & dependency management| **Free** |

> **No paid dependencies.** The `openai` Python package connects to xAI's Grok API by setting `base_url="https://api.x.ai/v1"`. Get your free API key at [console.x.ai](https://console.x.ai).

---

## Evaluation Criteria

| Criterion                     | Weight | What's Being Assessed                                      |
|-------------------------------|--------|------------------------------------------------------------|
| **Functional Correctness**    | High   | Does it take input, filter data, call LLM, show results?  |
| **Data Handling**             | Medium | Proper preprocessing, missing value handling, normalization|
| **Prompt Engineering**        | High   | Quality of LLM prompt; structured, clear, constrained     |
| **Output Quality**            | High   | Are recommendations relevant, explained, well-formatted?  |
| **Code Quality**              | Medium | Clean, modular, documented code                           |
| **User Experience**           | Medium | Intuitive input, clear output, error handling              |

---

## Project Structure (Suggested)

```
Zomato Milestone 1 AI/
├── Docs/
│   ├── Problemstatement.txt      # Original problem statement
│   └── context.md                # This file — full project context
├── data/
│   └── zomato.csv                # Downloaded/cached dataset
├── src/
│   ├── data_loader.py            # Data ingestion & preprocessing
│   ├── filters.py                # User preference filtering logic
│   ├── prompt_builder.py         # LLM prompt construction
│   ├── recommender.py            # LLM API call & response parsing
│   └── app.py                    # Main application entry point
├── .env                          # API keys (not committed to git)
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview & setup instructions
```
