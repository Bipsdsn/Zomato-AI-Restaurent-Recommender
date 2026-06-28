"""Tests for the two-stage retrieval engine (Phase 3). Requires data stores."""

from src.models import UserPreferences
from src.retrieval_engine import MAX_LLM_CANDIDATES
from tests._helpers import requires_data

pytestmark = requires_data


def test_stage1_hard_filters(parser, retrieval_engine):
    prefs = parser.parse_user_input("cheap Italian in Koramangala")
    result = retrieval_engine.retrieve(prefs)
    assert result.candidates                       # found something
    assert result.stage1_count >= len(result.candidates)


def test_context_window_guard_caps_candidates(retrieval_engine):
    prefs = UserPreferences(location="btm")        # broad query
    result = retrieval_engine.retrieve(prefs)
    assert len(result.candidates) <= MAX_LLM_CANDIDATES


def test_progressive_relaxation_fires(retrieval_engine):
    # Over-constrained: tiny budget + impossible rating + niche cuisine
    prefs = UserPreferences(
        location="koramangala", budget_max=50, budget_tier="low",
        cuisine="mongolian", min_rating=4.9,
    )
    result = retrieval_engine.retrieve(prefs)
    assert result.filters_relaxed                  # some filter was relaxed
    assert isinstance(result.filters_relaxed, list)


def test_semantic_scores_attached(parser, retrieval_engine):
    prefs = parser.parse_user_input("cozy italian in indiranagar")
    result = retrieval_engine.retrieve(prefs)
    if result.candidates:
        assert any("similarity_score" in c for c in result.candidates)


def test_retrieve_returns_result_shape(parser, retrieval_engine):
    prefs = parser.parse_user_input("family-friendly place in BTM, medium budget")
    result = retrieval_engine.retrieve(prefs)
    assert hasattr(result, "candidates")
    assert hasattr(result, "filters_relaxed")
    assert hasattr(result, "stage1_count")
    assert hasattr(result, "stage2_count")
