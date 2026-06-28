"""Integration tests for the orchestrator (Phase 6) with a mocked LLM.

Wires real parser/retrieval/prompt components against the persisted data
stores, but injects a deterministic LLM adapter so no network calls occur.
"""

import pytest

from src.orchestrator import RecommendationOrchestrator
from src.recommender import Recommender
from src.cache import InMemoryCache
from src.session_manager import SessionManager
from tests._helpers import requires_data

pytestmark = requires_data


def _build(data_layer, parser, retrieval_engine, prompt_builder, adapter):
    recommender = Recommender(llm_adapter=adapter, prompt_builder=prompt_builder)
    return RecommendationOrchestrator(
        data_layer=data_layer,
        input_parser=parser,
        retrieval_engine=retrieval_engine,
        prompt_builder=prompt_builder,
        recommender=recommender,
        cache=InMemoryCache(),
        session_manager=SessionManager(),
    )


def test_full_pipeline_returns_validated_recs(
    data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter
):
    orch = _build(data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter())
    resp = orch.process_request("cheap Italian in Koramangala")
    assert resp.recommendations
    assert resp.recommendations[0].source == "llm"
    # All recommended names exist in the dataset (no hallucinations)
    for r in resp.recommendations:
        assert data_layer.get_restaurant_by_name(r.name) is not None


def test_fallback_when_llm_fails(
    data_layer, parser, retrieval_engine, prompt_builder, FailingLLMAdapter
):
    orch = _build(data_layer, parser, retrieval_engine, prompt_builder, FailingLLMAdapter())
    resp = orch.process_request("best rated Chinese under 800 in BTM")
    assert resp.recommendations
    assert all(r.source == "fallback" for r in resp.recommendations)


def test_cache_hit_on_repeat_query(
    data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter
):
    orch = _build(data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter())
    q = "family-friendly place in BTM, medium budget"
    first = orch.process_request(q)
    second = orch.process_request(q)
    # Same recommendations and the cache registered at least one hit
    assert [r.name for r in first.recommendations] == [r.name for r in second.recommendations]
    assert orch._cache.get_stats().hits >= 1


def test_session_followup_two_turns(
    data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter
):
    orch = _build(data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter())
    first = orch.process_request("Italian in Indiranagar")
    sid = first.session_id
    assert sid
    second = orch.process_request("show me something else", session_id=sid)
    assert second.session_id == sid
    assert isinstance(second.recommendations, list)


def test_invalid_location_returns_empty(
    data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter
):
    orch = _build(data_layer, parser, retrieval_engine, prompt_builder, MockLLMAdapter())
    # Structured input with an unmatchable location raises InvalidLocationError
    # inside the parser, which the orchestrator routes to an empty response.
    resp = orch.process_request({"location": "zzzqqqxyznowhere", "budget_tier": "low"})
    assert resp.recommendations == []
