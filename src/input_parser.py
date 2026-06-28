"""
input_parser.py — Phase 2: Input Parser Module

Parses natural language or structured (form/dict) input into a validated
UserPreferences object. Handles synonyms, fuzzy location matching, cuisine
extraction, budget detection, and rating parsing.
"""

import re
from typing import List, Optional, Dict, Any, Tuple

from thefuzz import process as fuzz_process

from src.models import UserPreferences
from src.data_layer import DataLayer


# ═══════════════════════════════════════════════════════════════════════════
#  SYNONYM MAPS (from architecture §3.1)
# ═══════════════════════════════════════════════════════════════════════════

BUDGET_SYNONYMS = {
    "low": [
        "cheap", "budget", "budget-friendly", "pocket-friendly",
        "affordable", "inexpensive", "economical", "low cost",
        "low budget", "value for money",
    ],
    "medium": [
        "moderate", "mid-range", "mid range", "reasonable",
        "not too expensive", "average", "decent",
    ],
    "high": [
        "premium", "expensive", "upscale", "fine dining",
        "luxury", "luxurious", "splurge", "high-end", "posh",
        "fancy", "classy",
    ],
}

PREF_SYNONYMS = {
    "family-friendly": [
        "kid-friendly", "family", "children", "kids",
        "family friendly", "child-friendly",
    ],
    "quick service": [
        "fast", "quick", "quick bite", "grab and go",
        "fast food", "express", "takeaway",
    ],
    "romantic": [
        "date night", "date", "cozy", "intimate", "couple",
        "candlelight", "candle light", "romantic dinner",
    ],
    "rooftop": [
        "terrace", "outdoor", "open-air", "open air",
        "rooftop dining", "al fresco",
    ],
}

# Budget tier → default max cost mapping
BUDGET_TIER_DEFAULTS = {
    "low":    300,
    "medium": 800,
    "high":   2000,
}


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════

class InvalidLocationError(Exception):
    """Raised when no matching location is found."""
    def __init__(self, raw_input: str, suggestions: List[str]):
        self.raw_input = raw_input
        self.suggestions = suggestions
        msg = f"Location '{raw_input}' not found."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        super().__init__(msg)


# ═══════════════════════════════════════════════════════════════════════════
#  INPUT PARSER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class InputParser:
    """
    Parses raw user input (natural language or structured dict) into
    a validated UserPreferences object.
    
    Uses:
      - Fuzzy matching (thefuzz) for locations
      - Synonym maps for budget and preference terms
      - Known cuisine list from the data layer
      - Regex for rating and explicit budget amounts
    """

    def __init__(self, data_layer: DataLayer):
        self._data_layer = data_layer
        self._known_locations = data_layer.get_available_locations()
        self._known_cuisines = data_layer.get_available_cuisines()

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------
    def parse_user_input(self, raw_input) -> UserPreferences:
        """
        Main entry point: parse either a dict (structured) or str (NL).
        
        Args:
            raw_input: str (natural language) or dict (form fields)
        
        Returns:
            UserPreferences object with extracted fields.
        """
        if isinstance(raw_input, dict):
            return self._parse_structured(raw_input)
        elif isinstance(raw_input, str):
            return self._parse_natural_language(raw_input)
        else:
            raise ValueError(f"Unsupported input type: {type(raw_input)}")

    # ------------------------------------------------------------------
    #  STRUCTURED INPUT (form/dict)
    # ------------------------------------------------------------------
    def _parse_structured(self, data: Dict[str, Any]) -> UserPreferences:
        """Validate and normalize structured form input."""
        location = data.get("location", "").strip().lower() if data.get("location") else None
        cuisine = data.get("cuisine", "").strip().lower() if data.get("cuisine") else None
        budget_tier = data.get("budget_tier", "").strip().lower() if data.get("budget_tier") else None
        budget_max = data.get("budget_max")
        min_rating = float(data.get("min_rating", 0.0))
        additional_prefs = data.get("additional_prefs", "").strip() if data.get("additional_prefs") else None

        # Resolve location via fuzzy match if provided
        if location:
            location = self.resolve_location(location)

        # Derive budget_max from tier if not explicitly given
        if budget_max is not None:
            budget_max = int(budget_max)
        elif budget_tier and budget_tier in BUDGET_TIER_DEFAULTS:
            budget_max = BUDGET_TIER_DEFAULTS[budget_tier]

        # Derive budget_tier from budget_max if not explicitly given
        if not budget_tier and budget_max is not None:
            budget_tier = self._cost_to_tier(budget_max)

        return UserPreferences(
            location=location,
            budget_max=budget_max,
            budget_tier=budget_tier,
            cuisine=cuisine,
            min_rating=min_rating,
            additional_prefs=additional_prefs,
        )

    # ------------------------------------------------------------------
    #  NATURAL LANGUAGE PARSING (Rule-based NLP — Option A)
    # ------------------------------------------------------------------
    def _parse_natural_language(self, text: str) -> UserPreferences:
        """
        Extract structured preferences from free-form text using
        rule-based NLP: regex, synonym maps, and fuzzy matching.
        """
        text_lower = text.lower().strip()

        location = self._extract_location(text_lower)
        budget_max, budget_tier = self._extract_budget(text_lower)
        cuisine = self._extract_cuisine(text_lower)
        min_rating = self._extract_rating(text_lower)
        additional_prefs = self._extract_prefs(text_lower)

        # If budget_tier found but no explicit max, use default
        if budget_tier and budget_max is None:
            budget_max = BUDGET_TIER_DEFAULTS.get(budget_tier)

        # If explicit max found but no tier, derive it
        if budget_max is not None and not budget_tier:
            budget_tier = self._cost_to_tier(budget_max)

        return UserPreferences(
            location=location,
            budget_max=budget_max,
            budget_tier=budget_tier,
            cuisine=cuisine,
            min_rating=min_rating,
            additional_prefs=additional_prefs,
        )

    # ------------------------------------------------------------------
    #  LOCATION EXTRACTION + FUZZY MATCHING
    # ------------------------------------------------------------------
    def resolve_location(self, raw: str) -> str:
        """
        Fuzzy match a raw location string against known locations.
        
        - Threshold >= 80: accept the match
        - Below 80: raise InvalidLocationError with suggestions
        """
        raw = raw.strip().lower()

        # Exact match first
        if raw in self._known_locations:
            return raw

        # Fuzzy match
        result = fuzz_process.extractOne(raw, self._known_locations)
        if result and result[1] >= 80:
            return result[0]

        # No good match — suggest top 3
        suggestions = fuzz_process.extract(raw, self._known_locations, limit=3)
        top_names = [s[0] for s in suggestions if s[1] >= 50]
        raise InvalidLocationError(raw, top_names)

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Try to find a known location in the text via:
        1. Preposition patterns: "in X", "near X", "at X", "around X"
        2. Direct substring match against known locations
        3. Fuzzy match on remaining tokens
        """
        # Pattern 1: preposition-based extraction
        prep_patterns = [
            r"\b(?:in|near|at|around|from)\s+([a-z\s]+?)(?:\s*[,.]|\s+(?:for|with|under|below|above|rated|budget)|\s*$)",
        ]
        for pattern in prep_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                try:
                    return self.resolve_location(candidate)
                except InvalidLocationError:
                    pass  # Not a valid location, continue

        # Pattern 2: direct match — check if any known location appears in text
        # Sort by length descending to match longer names first (e.g., "mg road" before "mg")
        sorted_locations = sorted(self._known_locations, key=len, reverse=True)
        for loc in sorted_locations:
            if loc in text:
                return loc

        # Pattern 3: try each word as a candidate
        words = text.split()
        for word in words:
            word_clean = re.sub(r"[^a-z]", "", word)
            if len(word_clean) >= 3:
                try:
                    return self.resolve_location(word_clean)
                except InvalidLocationError:
                    continue

        return None

    # ------------------------------------------------------------------
    #  BUDGET EXTRACTION
    # ------------------------------------------------------------------
    def _extract_budget(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract budget info from text.
        Returns (budget_max, budget_tier).
        
        Checks:
        1. Explicit amounts: "under 500", "below 1000", "less than 800"
        2. Synonym matching: "cheap" → low, "premium" → high
        """
        budget_max = None
        budget_tier = None

        # Pattern 1: explicit amounts
        amount_patterns = [
            r"(?:under|below|less than|up to|max|within|budget of|upto)\s*₹?\s*(\d+)",
            r"₹\s*(\d+)\s*(?:or less|max|budget)",
            r"(\d{3,})\s*(?:rupees|rs|inr|budget)",
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text)
            if match:
                budget_max = int(match.group(1))
                break

        # Pattern 2: synonym matching
        for tier, synonyms in BUDGET_SYNONYMS.items():
            for synonym in synonyms:
                # Use word boundary matching for multi-word synonyms
                if len(synonym.split()) > 1:
                    if synonym in text:
                        budget_tier = tier
                        break
                else:
                    if re.search(r"\b" + re.escape(synonym) + r"\b", text):
                        budget_tier = tier
                        break
            if budget_tier:
                break

        return budget_max, budget_tier

    # ------------------------------------------------------------------
    #  CUISINE EXTRACTION
    # ------------------------------------------------------------------
    def _extract_cuisine(self, text: str) -> Optional[str]:
        """
        Extract cuisine by matching against known cuisine list.
        Matches longer cuisines first (e.g., "north indian" before "indian").
        """
        sorted_cuisines = sorted(self._known_cuisines, key=len, reverse=True)
        for cuisine in sorted_cuisines:
            if re.search(r"\b" + re.escape(cuisine) + r"\b", text):
                return cuisine
        return None

    # ------------------------------------------------------------------
    #  RATING EXTRACTION
    # ------------------------------------------------------------------
    def _extract_rating(self, text: str) -> float:
        """
        Extract minimum rating from text.
        Patterns: "rated 4+", "4+ stars", "rating above 3.5", "minimum 4"
        """
        rating_patterns = [
            r"(?:rated?|rating)\s*(?:above|over|>=?|at least|minimum)?\s*(\d+\.?\d*)\s*\+?",
            r"(\d+\.?\d*)\s*\+\s*(?:stars?|rated?|rating)",
            r"(?:minimum|min|at least)\s*(?:rating of)?\s*(\d+\.?\d*)",
            r"(?:above|over)\s*(\d+\.?\d*)\s*(?:stars?|rating|rated?)",
            r"(\d+\.?\d*)\s*(?:stars?\s+(?:or more|and above|minimum|\+))",
        ]
        for pattern in rating_patterns:
            match = re.search(pattern, text)
            if match:
                rating = float(match.group(1))
                if 0 <= rating <= 5:
                    return rating
        return 0.0

    # ------------------------------------------------------------------
    #  PREFERENCE / AMBIANCE EXTRACTION
    # ------------------------------------------------------------------
    def _extract_prefs(self, text: str) -> Optional[str]:
        """
        Extract additional preferences (ambiance, dining style) by
        matching against the preference synonym map.
        """
        matched_prefs = []
        for pref_key, synonyms in PREF_SYNONYMS.items():
            # Check if the canonical term itself is in the text
            if pref_key in text:
                matched_prefs.append(pref_key)
                continue
            # Check synonyms
            for synonym in synonyms:
                if len(synonym.split()) > 1:
                    if synonym in text:
                        matched_prefs.append(pref_key)
                        break
                else:
                    if re.search(r"\b" + re.escape(synonym) + r"\b", text):
                        matched_prefs.append(pref_key)
                        break

        return ", ".join(matched_prefs) if matched_prefs else None

    # ------------------------------------------------------------------
    #  HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _cost_to_tier(cost: int) -> str:
        """Convert an explicit cost value to a budget tier."""
        if cost <= 300:
            return "low"
        elif cost <= 800:
            return "medium"
        else:
            return "high"


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry-point for standalone testing
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    dl = DataLayer()

    parser = InputParser(dl)

    test_cases = [
        # Natural language tests (from acceptance criteria)
        "something cheap near Koramangala",
        "best Italian under 1000 in Indiranagar",
        "pocket-friendly Chinese, rated 4+",
        "upscale place for a date night in Whitefield",
        "family-friendly under 500 in BTM",
        "quick bite near MG Road",
        # Structured input test
        {"location": "Koramangala", "budget_tier": "low", "cuisine": "Italian"},
    ]

    print("=" * 70)
    print("Phase 2: Input Parser Tests")
    print("=" * 70)

    for i, tc in enumerate(test_cases, 1):
        try:
            result = parser.parse_user_input(tc)
            print(f"\n[{i}] Input: {tc}")
            print(f"    -> location={result.location}, budget_tier={result.budget_tier}, "
                  f"budget_max={result.budget_max}, cuisine={result.cuisine}, "
                  f"min_rating={result.min_rating}, prefs={result.additional_prefs}")
        except InvalidLocationError as e:
            print(f"\n[{i}] Input: {tc}")
            print(f"    X {e}")
        except Exception as e:
            print(f"\n[{i}] Input: {tc}")
            print(f"    X ERROR: {e}")

    # Test invalid location
    print(f"\n[8] Testing invalid location...")
    try:
        parser.resolve_location("xyznonexistent")
        print("    X Should have raised InvalidLocationError")
    except InvalidLocationError as e:
        print(f"    OK InvalidLocationError raised: {e}")

    dl.close()
    print(f"\n{'='*70}")
    print("Phase 2 COMPLETE")
    print("=" * 70)
