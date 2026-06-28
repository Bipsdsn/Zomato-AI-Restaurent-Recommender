"""Unit tests for the LLM recommender: parsing, validation, fallback (Phase 5)."""

from src.recommender import (
    ResponseParser, HallucinationValidator, fallback_ranking, Recommender,
)
from src.models import Recommendation, UserPreferences


# ---- ResponseParser ----
def test_parse_clean_json():
    raw = '[{"rank":1,"name":"Toscano","explanation":"Great"},' \
          '{"rank":2,"name":"Spice Garden","explanation":"Nice"}]'
    recs = ResponseParser.parse(raw)
    assert len(recs) == 2
    assert recs[0].name == "Toscano"
    assert recs[0].source == "llm"


def test_parse_json_with_code_fences():
    raw = "```json\n[{\"rank\":1,\"name\":\"Toscano\",\"explanation\":\"x\"}]\n```"
    recs = ResponseParser.parse(raw)
    assert len(recs) == 1
    assert recs[0].name == "Toscano"


def test_parse_regex_fallback():
    raw = (
        "Here are my picks:\n"
        "1. **Spice Garden** - Amazing North Indian\n"
        "2. **Pasta Palace** - Best Italian around\n"
        "3. **Wok & Roll** - Solid Chinese\n"
    )
    recs = ResponseParser.parse(raw)
    assert len(recs) >= 2
    assert any("Spice Garden" in r.name for r in recs)


def test_parse_garbage_returns_empty():
    assert ResponseParser.parse("this is not parseable at all") == []


def test_parse_skips_none_and_null_names():
    raw = ('[{"rank":1,"name":"None","explanation":"x"},'
           '{"rank":2,"name":null,"explanation":"y"},'
           '{"rank":3,"name":"Toscano","explanation":"z"}]')
    recs = ResponseParser.parse(raw)
    assert [r.name for r in recs] == ["Toscano"]


# ---- HallucinationValidator ----
def test_validator_drops_hallucinations():
    recs = [
        Recommendation(rank=1, name="Toscano", explanation="x", source="llm"),
        Recommendation(rank=2, name="TOTALLY FAKE PLACE", explanation="y", source="llm"),
        Recommendation(rank=3, name="Spice Garden", explanation="z", source="llm"),
    ]
    candidates = [{"name": "Toscano"}, {"name": "Spice Garden"}, {"name": "Wok Express"}]
    valid, dropped = HallucinationValidator.validate(recs, candidates)
    assert dropped == 1
    assert len(valid) == 2
    assert [r.rank for r in valid] == [1, 2]   # re-ranked after drop


def test_validator_normalizes_to_dataset_name():
    recs = [Recommendation(rank=1, name="toscano", explanation="x", source="llm")]
    candidates = [{"name": "Toscano"}]
    valid, dropped = HallucinationValidator.validate(recs, candidates)
    assert dropped == 0
    assert valid[0].name == "Toscano"   # normalized to canonical casing


def test_validator_drops_ambiguous_partial_match():
    # "cafe" is contained in multiple known names → ambiguous → dropped
    recs = [Recommendation(rank=1, name="Cafe", explanation="x", source="llm")]
    candidates = [{"name": "Cafe Coffee Day"}, {"name": "Truffles Cafe"}]
    valid, dropped = HallucinationValidator.validate(recs, candidates)
    assert dropped == 1
    assert valid == []


def test_validator_accepts_unambiguous_partial_match():
    recs = [Recommendation(rank=1, name="Toscano Restaurant", explanation="x", source="llm")]
    candidates = [{"name": "Toscano"}, {"name": "Spice Garden"}]
    valid, dropped = HallucinationValidator.validate(recs, candidates)
    assert dropped == 0
    assert valid[0].name == "Toscano"   # single unambiguous match


def test_validator_drops_duplicate_recommendations():
    recs = [
        Recommendation(rank=1, name="Toscano", explanation="x", source="llm"),
        Recommendation(rank=2, name="Toscano", explanation="y", source="llm"),
    ]
    candidates = [{"name": "Toscano"}]
    valid, dropped = HallucinationValidator.validate(recs, candidates)
    assert dropped == 1
    assert len(valid) == 1


def test_validator_drops_too_short_name():
    # A 3-char fragment is below the containment threshold → dropped
    recs = [Recommendation(rank=1, name="Tos", explanation="x", source="llm")]
    candidates = [{"name": "Toscano"}]
    valid, dropped = HallucinationValidator.validate(recs, candidates)
    assert dropped == 1
    assert valid == []


# ---- fallback_ranking ----
def test_fallback_orders_by_rating_then_votes(sample_candidates):
    recs = fallback_ranking(sample_candidates)
    assert recs[0].name == "Toscano"        # rate 4.6 highest
    assert all(r.source == "fallback" for r in recs)
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))


def test_fallback_empty_candidates():
    assert fallback_ranking([]) == []


def test_fallback_dedupes_duplicate_names():
    cands = [
        {"name": "Natural Ice Cream", "rate": 4.6, "votes": 500, "approx_cost": 200},
        {"name": "Natural Ice Cream", "rate": 4.6, "votes": 500, "approx_cost": 200},
        {"name": "Gods Own Cafe", "rate": 4.4, "votes": 300, "approx_cost": 300},
    ]
    recs = fallback_ranking(cands)
    names = [r.name for r in recs]
    assert names == ["Natural Ice Cream", "Gods Own Cafe"]   # no duplicate
    assert [r.rank for r in recs] == [1, 2]


# ---- Recommender end-to-end (mocked LLM) ----
def test_recommender_llm_happy_path(MockLLMAdapter, prompt_builder, sample_candidates):
    rec = Recommender(llm_adapter=MockLLMAdapter(top_n=3), prompt_builder=prompt_builder)
    prefs = UserPreferences(location="koramangala", cuisine="italian", min_rating=4.0)
    out = rec.recommend(prefs, sample_candidates)
    assert len(out) >= 1
    assert all(r.source == "llm" for r in out)
    names = {c["name"] for c in sample_candidates}
    assert all(r.name in names for r in out)   # no hallucinations


def test_recommender_falls_back_on_llm_failure(FailingLLMAdapter, prompt_builder, sample_candidates):
    rec = Recommender(llm_adapter=FailingLLMAdapter(), prompt_builder=prompt_builder)
    prefs = UserPreferences(location="koramangala", cuisine="italian")
    out = rec.recommend(prefs, sample_candidates)
    assert len(out) >= 1
    assert all(r.source == "fallback" for r in out)


def test_recommender_empty_candidates_returns_empty(MockLLMAdapter, prompt_builder):
    rec = Recommender(llm_adapter=MockLLMAdapter(), prompt_builder=prompt_builder)
    assert rec.recommend(UserPreferences(location="x"), []) == []


# ---- GrokAdapter metadata (constructs client; no network call) ----
def test_grok_adapter_metadata():
    from src.recommender import GrokAdapter
    adapter = GrokAdapter()
    assert adapter.get_provider_name() == "xAI"
    assert isinstance(adapter.get_model_name(), str) and adapter.get_model_name()
