"""Unit tests for the template-driven prompt builder (Phase 4)."""

from src.prompt_builder import PromptBuilder, MAX_INPUT_TOKENS, MAX_RESTAURANTS
from src.models import UserPreferences


def _prefs(**kw):
    base = dict(location="koramangala", budget_tier="low", budget_max=300,
                cuisine="italian", min_rating=4.0, additional_prefs="romantic")
    base.update(kw)
    return UserPreferences(**base)


def _restaurants(n=15):
    out = []
    for i in range(n):
        out.append({
            "name": f"Restaurant {i}", "cuisines": "italian, continental",
            "rate": 4.0 + (i % 10) / 10, "votes": 100 + i,
            "approx_cost": 300 + i * 10, "rest_type": "Casual Dining",
            "online_order": bool(i % 2), "book_table": bool((i + 1) % 2),
            "dish_liked": "Pasta, Pizza, Risotto",
        })
    return out


def test_build_prompt_has_all_sections():
    builder = PromptBuilder()
    prompt = builder.build_prompt(_prefs(), _restaurants(5))
    assert set(prompt.keys()) == {"system", "user", "token_estimate"}
    assert prompt["system"]
    # user message contains context + data + format
    assert "Available Restaurants" in prompt["user"]
    assert "preferences" in prompt["user"].lower()
    assert "JSON" in prompt["user"]


def test_user_context_substitution():
    builder = PromptBuilder()
    prompt = builder.build_prompt(_prefs(location="indiranagar"), _restaurants(3))
    assert "indiranagar" in prompt["user"].lower()
    # placeholders should be replaced, not left raw
    assert "{location}" not in prompt["user"]
    assert "{budget_max}" not in prompt["user"]


def test_anti_hallucination_instruction_present():
    builder = PromptBuilder()
    sys = builder.build_prompt(_prefs(), _restaurants(3))["system"].lower()
    assert "do not invent" in sys or "hallucinate" in sys


def test_token_budget_respected_for_max_restaurants():
    builder = PromptBuilder()
    prompt = builder.build_prompt(_prefs(), _restaurants(MAX_RESTAURANTS))
    assert prompt["token_estimate"] <= MAX_INPUT_TOKENS


def test_restaurants_capped_at_max():
    builder = PromptBuilder()
    prompt = builder.build_prompt(_prefs(), _restaurants(40))
    # Only MAX_RESTAURANTS get serialized into the data block
    assert prompt["user"].count("[") >= 1
    # The 16th restaurant (index 15) must not appear
    assert "Restaurant 15" not in prompt["user"]


def test_session_context_injected():
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        _prefs(), _restaurants(3),
        session_context="User previously asked for Chinese food.",
    )
    assert "Previous Conversation" in prompt["user"]
    assert "Chinese" in prompt["user"]


def test_trim_to_budget_when_oversized():
    builder = PromptBuilder()
    # Oversized restaurants force the token-budget trimming path.
    big = []
    for i in range(MAX_RESTAURANTS):
        big.append({
            "name": f"Restaurant {i} " + ("X" * 400),
            "cuisines": "italian " * 60,
            "rate": 4.2, "votes": 100, "approx_cost": 500,
            "rest_type": "Casual Dining " * 20,
            "online_order": True, "book_table": True,
            "dish_liked": "Pasta",
        })
    prompt = builder.build_prompt(_prefs(), big)
    # After trimming the estimate must still fit the budget.
    assert prompt["token_estimate"] <= MAX_INPUT_TOKENS
