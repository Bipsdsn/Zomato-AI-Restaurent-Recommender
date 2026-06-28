"""
web.py -- Phase 9: Presentation Layer (Flask backend)

Serves the Craival web UI and exposes a small JSON API on top of the
existing RecommendationOrchestrator.

Routes:
  GET  /                -> render index.html (search + results SPA shell)
  POST /api/recommend   -> run the pipeline, return enriched recommendations
  POST /api/feedback    -> log a thumbs up/down for a recommendation
  GET  /api/locations   -> available Bangalore locations (autocomplete)
  GET  /api/cuisines    -> available cuisines (multi-select)

Run:
  python -m src.web
"""

import os
from dataclasses import asdict

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, make_response

load_dotenv()

from src.app import create_orchestrator
from src.logger import Logger, FeedbackData


# ---------------------------------------------------------------------------
#  App + singletons
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Build the orchestrator once at startup (loads SQLite + ChromaDB).
_orchestrator = create_orchestrator()
_logger = Logger()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _enrich(recommendation):
    """
    Combine an LLM Recommendation (rank/name/explanation/source) with the
    restaurant's display metadata from the data layer. Falls back to bare
    fields if the restaurant can't be matched.
    """
    rec = asdict(recommendation)
    details = _orchestrator.get_restaurant_details(recommendation.name)

    if details:
        rec.update({
            "cuisines": details.get("cuisines", ""),
            "rate": details.get("rate"),
            "votes": details.get("votes", 0),
            "approx_cost": details.get("approx_cost"),
            "rest_type": details.get("rest_type", ""),
            "online_order": bool(details.get("online_order")),
            "book_table": bool(details.get("book_table")),
            "dish_liked": details.get("dish_liked", ""),
            "location": details.get("location", ""),
            "budget_tier": details.get("budget_tier", ""),
            "is_new": bool(details.get("is_new")),
        })
    else:
        rec.update({
            "cuisines": "", "rate": None, "votes": 0, "approx_cost": None,
            "rest_type": "", "online_order": False, "book_table": False,
            "dish_liked": "", "location": "", "budget_tier": "", "is_new": False,
        })
    return rec


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    resp = make_response(render_template(
        "index.html",
        restaurant_count=_orchestrator.get_restaurant_count(),
        location_count=len(_orchestrator.get_locations()),
    ))
    # Always serve fresh HTML so updated ?v= asset links are picked up.
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/api/locations")
def api_locations():
    return jsonify({"locations": _orchestrator.get_locations()})


@app.route("/api/cuisines")
def api_cuisines():
    return jsonify({"cuisines": _orchestrator.get_cuisines()})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")

    # Two input modes: free-text query OR structured form fields.
    query = (data.get("query") or "").strip()
    if query:
        raw_input = query
    else:
        raw_input = {
            "location": data.get("location"),
            "cuisine": data.get("cuisine"),
            "budget_tier": data.get("budget_tier"),
            "budget_max": data.get("budget_max"),
            "min_rating": data.get("min_rating", 0.0),
            "additional_prefs": data.get("additional_prefs"),
        }

    try:
        response = _orchestrator.process_request(raw_input, session_id)
    except Exception as e:  # pragma: no cover - defensive
        app.logger.exception("recommend failed")
        return jsonify({"error": "processing_failed", "message": str(e)}), 500

    recommendations = [_enrich(r) for r in response.recommendations]
    source = recommendations[0]["source"] if recommendations else "none"

    return jsonify({
        "recommendations": recommendations,
        "filters_relaxed": response.filters_relaxed,
        "session_id": response.session_id,
        "processing_time_ms": response.processing_time_ms,
        "source": source,
        "count": len(recommendations),
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}
    restaurant = data.get("restaurant", "unknown")
    thumbs_up = data.get("thumbs_up")
    session_id = data.get("session_id", "anonymous")

    _logger.log_feedback(
        request_id=session_id,
        feedback=FeedbackData(
            user_accepted=bool(thumbs_up) if thumbs_up is not None else None,
            thumbs_up=thumbs_up,
        ),
    )
    return jsonify({"ok": True, "restaurant": restaurant})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    # debug=False so the heavy orchestrator isn't built twice by the reloader.
    app.run(host="127.0.0.1", port=port, debug=False)
