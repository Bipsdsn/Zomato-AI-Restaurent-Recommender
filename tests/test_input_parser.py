"""Unit tests for the input parser (Phase 2). Requires the SQLite store for
known-location / known-cuisine lists."""

import pytest

from src.input_parser import InputParser, InvalidLocationError
from tests._helpers import requires_data

pytestmark = requires_data


# ---- Natural language extraction ----
def test_parse_cheap_near_koramangala(parser):
    p = parser.parse_user_input("something cheap near Koramangala")
    assert p.location and "koramangala" in p.location   # may resolve to a sub-locality
    assert p.budget_tier == "low"


def test_parse_explicit_budget_and_cuisine(parser):
    p = parser.parse_user_input("best Italian under 1000 in Indiranagar")
    assert p.location == "indiranagar"
    assert p.cuisine == "italian"
    assert p.budget_max == 1000


def test_parse_rating_extraction(parser):
    p = parser.parse_user_input("pocket-friendly Chinese, rated 4+")
    assert p.budget_tier == "low"
    assert p.cuisine == "chinese"
    assert p.min_rating == 4.0


def test_parse_additional_prefs(parser):
    p = parser.parse_user_input("upscale place for a date night in Whitefield")
    assert p.location == "whitefield"
    assert p.budget_tier == "high"
    assert p.additional_prefs and "romantic" in p.additional_prefs


# ---- Fuzzy location resolution ----
def test_resolve_location_typo(parser):
    # "koramangla" is a near-miss; resolves to a koramangala locality
    assert "koramangala" in parser.resolve_location("koramangla")


def test_resolve_location_invalid_raises(parser):
    with pytest.raises(InvalidLocationError):
        parser.resolve_location("xyznonexistentplace")


# ---- Structured input ----
def test_structured_input_normalizes(parser):
    p = parser.parse_user_input({
        "location": "Koramangala", "budget_tier": "low", "cuisine": "Italian",
    })
    assert p.location and "koramangala" in p.location
    assert p.budget_tier == "low"
    assert p.budget_max == 300          # derived from tier default
    assert p.cuisine == "italian"


def test_structured_budget_max_derives_tier(parser):
    p = parser.parse_user_input({"location": "Bellandur", "budget_max": 250})
    assert p.budget_tier == "low"       # 250 → low tier


def test_empty_string_input_does_not_crash(parser):
    p = parser.parse_user_input("")
    assert p.location is None
