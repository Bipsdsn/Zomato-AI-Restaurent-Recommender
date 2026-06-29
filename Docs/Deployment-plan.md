# Deployment Plan — Craival (Zomato AI Restaurant Recommender)

> **Goal:** Deploy the **backend (Flask API + RAG pipeline)** on **Railway** and the
> **frontend (static UI)** on **Vercel**.
>
> **Last Updated:** 2026-06-29

---

## 0. Architecture after the split

Today the Flask app (`src/web.py`) serves **both** the JSON API and the HTML/CSS/JS
UI (via Jinja templates + `static/`). To host the UI on Vercel and the API on
Railway, we separate them:

```
┌─────────────────────────────┐         HTTPS (CORS)        ┌──────────────────────────────┐
│  Vercel (static frontend)   │  ───────────────────────▶   │  Railway (Flask API backend)  │
│  index.html + style.css     │   /api/recommend            │  gunicorn → src.web:app        │
│  + script.js (API_BASE URL) │   /api/locations ...        │  SQLite + ChromaDB + LLM       │
└─────────────────────────────┘                             └──────────────────────────────┘
```

Key consequences (handled in the steps below):
1. The frontend must call the backend by **absolute URL** (not relative `/api/...`).
2. The backend must enable **CORS** for the Vercel domain.
3. The backend must bind `0.0.0.0:$PORT` and run under a **production WSGI server** (gunicorn).
4. The Jinja template must become a **static** `index.html` (no `{{ }}` / `url_for`).
5. The **data stores** (SQLite + ChromaDB) must be built/persisted on Railway.

> **Simpler alternative (recommended if you don't strictly need a split):** deploy the
> whole Flask app (UI + API) on Railway alone — no CORS, no static-export work. The
> Vercel split is documented below because it was requested.

---

## PART A — Backend on Railway

### A1. Add production dependencies
Append to `requirements.txt`:
```
gunicorn>=21.2.0
flask-cors>=4.0.0
```

### A2. Enable CORS and a stats endpoint (code changes in `src/web.py`)
```python
from flask_cors import CORS

# after `app = Flask(...)`
CORS(app, resources={r"/api/*": {"origins": [
    "https://<your-vercel-app>.vercel.app",
    "http://localhost:3000",            # local frontend dev
]}})

# Add a stats endpoint so the static frontend can show counts
@app.route("/api/stats")
def api_stats():
    return jsonify({
        "restaurant_count": _orchestrator.get_restaurant_count(),
        "location_count": len(_orchestrator.get_locations()),
    })
```

### A3. Bind host/port for Railway
Railway injects `$PORT`. Do **not** use Flask's dev server in production — use gunicorn.
Create a **`Procfile`** in the project root:
```
web: gunicorn src.web:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 --preload
```
Notes:
- `--workers 1 --preload`: the orchestrator loads SQLite + ChromaDB + the embedding
  model once at startup. A single preloaded worker avoids loading the model N times
  and keeps memory within Railway's free/Hobby limits. Use `--threads` for concurrency.
- `--timeout 180`: first request may trigger the embedding-model download.

### A4. Pin the Python version
Create **`runtime.txt`** (or set in Railway settings):
```
python-3.12.x
```
(Match a version Railway supports; 3.11/3.12 recommended over 3.13 for wheel availability.)

### A5. Data stores on Railway (important)
The `data/` directory (SQLite DB + ChromaDB) is **git-ignored**, so it won't be in the
repo. Two options:

**Option 1 — Build on first boot (simplest).** `create_orchestrator()` already calls
`ensure_data_ready()`, which downloads the dataset and builds both stores if missing.
On Railway this runs on first startup (slow: a few minutes, downloads dataset +
embedding model). Works, but rebuilds on every fresh deploy unless persisted.

**Option 2 — Persist with a Railway Volume (recommended).**
- In Railway, add a **Volume** mounted at `/app/data`.
- Set env vars so the app uses it:
  - `DB_PATH=/app/data/restaurants.db`
  - `VECTOR_DB_PATH=/app/data/chroma_store`
  - `DATA_PATH=/app/data/zomato.csv`
- First boot builds into the volume; later deploys reuse it (fast starts).

### A6. Environment variables (Railway → Variables)
```
XAI_API_KEY=<your_groq_key>          # set in Railway dashboard, never in git
XAI_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=1024
CACHE_ENABLED=true
SESSION_ENABLED=true
LOG_LEVEL=INFO
# If using a volume (Option 2):
DB_PATH=/app/data/restaurants.db
VECTOR_DB_PATH=/app/data/chroma_store
DATA_PATH=/app/data/zomato.csv
```

### A7. Deploy steps
1. Push the repo to GitHub (see project README for the safe push procedure).
2. Railway → **New Project → Deploy from GitHub repo** → select this repo.
3. Railway auto-detects Python and installs `requirements.txt`.
4. Add the env vars (A6) and (optionally) the Volume (A5).
5. Confirm the start command matches the `Procfile`.
6. Deploy. Watch logs for: data build (first time) → `Running on ...`.
7. Note the public URL, e.g. `https://craival-api.up.railway.app`.

### A8. Verify the backend
```bash
curl https://<your-railway-app>.up.railway.app/api/locations
curl -X POST https://<your-railway-app>.up.railway.app/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"cheap Italian in Koramangala"}'
```

---

## PART B — Frontend on Vercel

The UI must become a **static** site (Vercel serves static files / it is not a Python host).

### B1. Create a static frontend folder
Create `frontend/` with:
```
frontend/
  index.html      # static version of templates/index.html
  style.css       # copy of static/style.css
  script.js       # copy of static/script.js (with API_BASE)
  config.js       # defines the backend URL
```

### B2. Make `index.html` static
Replace the Jinja bits in `templates/index.html`:
- Remove `{{ url_for('static', filename='style.css') }}?v=N` → `style.css`
- Remove `{{ url_for('static', filename='script.js') }}?v=N` → `script.js` (add `<script src="config.js"></script>` before it)
- Replace `{{ restaurant_count }}` / `{{ location_count }}` with placeholders
  (e.g. `<span id="stat-restaurants">…</span>`) that JS fills from `/api/stats`.

### B3. Configure the API base URL
`frontend/config.js`:
```js
// Point the static UI at the Railway backend.
window.API_BASE = "https://<your-railway-app>.up.railway.app";
```
In `frontend/script.js`, prefix every fetch:
```js
const API = window.API_BASE || "";
// e.g.
await fetch(`${API}/api/recommend`, { ... });
await fetch(`${API}/api/locations`);
await fetch(`${API}/api/cuisines`);
await fetch(`${API}/api/feedback`, { ... });
// and fetch counts on load:
fetch(`${API}/api/stats`).then(r => r.json()).then(s => {
  document.getElementById("stat-restaurants").textContent = s.restaurant_count;
  document.getElementById("stat-locations").textContent  = s.location_count;
});
```

### B4. Vercel config
Create `frontend/vercel.json` (static, no build step):
```json
{
  "version": 2,
  "cleanUrls": true,
  "headers": [
    { "source": "/(.*)", "headers": [
      { "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }
    ]}
  ]
}
```

### B5. Deploy steps
1. Vercel → **Add New → Project** → import the GitHub repo.
2. Set **Root Directory = `frontend`**.
3. Framework preset: **Other** (no build command; output dir = `frontend`).
4. Deploy → note the URL, e.g. `https://craival.vercel.app`.
5. Go back to **Railway** and set the CORS origin (A2) to this exact Vercel URL; redeploy backend.

### B6. Verify the frontend
- Open the Vercel URL, run a search, open a restaurant detail, submit feedback.
- In browser DevTools → Network, confirm `/api/*` calls go to the Railway domain and return `200` (no CORS errors).

---

## PART C — Post-deploy checklist

- [ ] Backend `/api/locations` returns data over HTTPS
- [ ] Frontend loads on Vercel and calls the Railway API (no CORS errors)
- [ ] `XAI_API_KEY` is set **only** in Railway env vars (never committed)
- [ ] `.env`, `data/`, `logs/` excluded by `.gitignore`
- [ ] CORS `origins` lists the exact Vercel domain
- [ ] Railway uses gunicorn (not the Flask dev server)
- [ ] (If used) Railway Volume persists `data/` across deploys
- [ ] Rotate the Groq key if it was ever committed to git history

---

## PART D — Cost & limits

| Service | Plan | Notes |
|---------|------|-------|
| Railway | Free/Hobby | Watch RAM — ChromaDB + embedding model are memory-heavy; 1 preloaded worker |
| Vercel | Hobby (free) | Static hosting is free and fast |
| Groq/xAI | Free tier | LLM calls; caching reduces usage |

**Memory tip:** the `all-MiniLM-L6-v2` embedding model + ChromaDB can exceed small
instances. If Railway OOMs, upgrade the instance or precompute embeddings into a
persisted volume so the model isn't reloaded.

---

## PART E — Files to add (summary)

| File | Where | Purpose |
|------|-------|---------|
| `Procfile` | project root | gunicorn start command for Railway |
| `runtime.txt` | project root | Pin Python version |
| `requirements.txt` | (edit) | add `gunicorn`, `flask-cors` |
| `src/web.py` | (edit) | CORS, `/api/stats`, ensure prod-safe |
| `frontend/index.html` | new | static UI (de-Jinja'd) |
| `frontend/style.css` | new | copy of `static/style.css` |
| `frontend/script.js` | new | copy with `API_BASE` prefix |
| `frontend/config.js` | new | backend URL |
| `frontend/vercel.json` | new | Vercel static config |

> Want me to generate these files (Procfile, runtime.txt, CORS + `/api/stats` code,
> and the `frontend/` static export) so it's deploy-ready? Just say the word.
