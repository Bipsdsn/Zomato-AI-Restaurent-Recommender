"""
recommender.py -- Phase 5: LLM Recommender (Grok Adapter)

Implements:
  - Abstract LLMAdapter base class
  - GrokAdapter: Grok API calls via openai SDK (free tier)
  - Response parsing: JSON primary, regex fallback
  - Anti-hallucination validation against candidate data
  - Fallback ranking when LLM is unavailable or returns garbage
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from src.models import Recommendation, UserPreferences
from src.prompt_builder import PromptBuilder
from src.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


# Names that indicate an empty/placeholder LLM result — never treated as real.
_JUNK_NAMES = {"", "none", "null", "n/a", "na", "unknown", "-"}


# ═══════════════════════════════════════════════════════════════════════════
#  ABSTRACT LLM ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

class LLMAdapter(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the raw text response."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  GROK ADAPTER (Primary — FREE via xAI)
# ═══════════════════════════════════════════════════════════════════════════

class GrokAdapter(LLMAdapter):
    """
    Grok LLM adapter using the openai SDK with xAI's API endpoint.
    
    Config loaded from environment variables:
      - XAI_API_KEY: API key from console.x.ai
      - XAI_BASE_URL: https://api.x.ai/v1
      - LLM_MODEL: grok-3-mini-fast (default)
      - LLM_TEMPERATURE: 0.4 (default)
      - LLM_MAX_TOKENS: 1024 (default)
    """

    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("XAI_API_KEY", "")
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        self._model = os.getenv("LLM_MODEL", "grok-3-mini-fast")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self._timeout = 30  # seconds

        if not api_key:
            logger.warning("XAI_API_KEY not set. LLM calls will fail; "
                           "fallback ranking will be used.")

        self._client = OpenAI(
            api_key=api_key or "dummy-key",
            base_url=base_url,
        )

    def call(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        """Call Grok API and return the raw text response."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Grok API error: %s", e)
            raise

    def get_provider_name(self) -> str:
        return "xAI"

    def get_model_name(self) -> str:
        return self._model


# ═══════════════════════════════════════════════════════════════════════════
#  RESPONSE PARSER
# ═══════════════════════════════════════════════════════════════════════════

class ResponseParser:
    """
    Parses LLM text output into a list of Recommendation objects.
    
    Strategy:
      1. Try JSON parsing (primary)
      2. Fall back to regex extraction
    """

    @staticmethod
    def parse(raw_response: str) -> List[Recommendation]:
        """Parse LLM response text into Recommendation objects."""
        # Try JSON first
        recs = ResponseParser._parse_json(raw_response)
        if recs:
            return recs

        # Fallback: regex extraction
        recs = ResponseParser._parse_regex(raw_response)
        if recs:
            return recs

        logger.warning("Could not parse LLM response")
        return []

    @staticmethod
    def _parse_json(raw: str) -> List[Recommendation]:
        """Try to extract a JSON array from the response."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Try to find a JSON array in the response
        # Look for the outermost [ ... ]
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        recommendations = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            explanation = str(item.get("explanation") or "").strip()
            rank = item.get("rank", len(recommendations) + 1)

            if name and name.lower() not in _JUNK_NAMES:
                recommendations.append(Recommendation(
                    rank=int(rank),
                    name=name,
                    explanation=explanation or "Recommended based on your preferences.",
                    source="llm",
                ))

        return recommendations

    @staticmethod
    def _parse_regex(raw: str) -> List[Recommendation]:
        """
        Fallback: extract restaurant names and explanations using regex.
        
        Handles patterns like:
          1. **Restaurant Name** - explanation text
          1. Restaurant Name: explanation text
          #1 Restaurant Name | explanation text
        """
        patterns = [
            # Pattern: "1. **Name** - explanation" or "1. **Name**: explanation"
            r"(\d+)\.\s*\*\*(.+?)\*\*\s*[-:]\s*(.+?)(?=\n\d+\.|\Z)",
            # Pattern: "1. Name - explanation"
            r"(\d+)\.\s*([^-:*\n]+?)\s*[-:]\s*(.+?)(?=\n\d+\.|\Z)",
            # Pattern: "#1 Name | explanation"
            r"#(\d+)\s+(.+?)\s*\|\s*(.+?)(?=\n#\d+|\Z)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, raw, re.DOTALL)
            if matches and len(matches) >= 2:
                recs = []
                for rank_str, name, explanation in matches:
                    name = name.strip().strip("*").strip()
                    explanation = explanation.strip()
                    if name and len(name) > 2 and name.lower() not in _JUNK_NAMES:
                        recs.append(Recommendation(
                            rank=int(rank_str),
                            name=name,
                            explanation=explanation or "Recommended.",
                            source="llm",
                        ))
                if recs:
                    return recs

        return []


# ═══════════════════════════════════════════════════════════════════════════
#  ANTI-HALLUCINATION VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class HallucinationValidator:
    """
    Cross-validates LLM recommendations against the actual candidate data.
    Drops any restaurant names not found in the dataset.
    """

    @staticmethod
    def validate(
        recommendations: List[Recommendation],
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[Recommendation], int]:
        """
        Validate recommendations against candidate restaurant names.
        
        Returns:
            (valid_recommendations, hallucinations_dropped_count)
        """
        # Build a set of known restaurant names (lowercase for matching)
        known_names = set()
        name_map = {}  # lowercase → original name
        for c in candidates:
            name = str(c.get("name", "")).strip()
            if name:
                known_names.add(name.lower())
                name_map[name.lower()] = name

        valid = []
        dropped = 0
        seen = set()  # avoid accepting the same dataset name twice

        for rec in recommendations:
            rec_name_lower = rec.name.strip().lower()

            resolved = None  # the canonical dataset name this rec maps to

            # 1) Exact match — always trusted.
            if rec_name_lower in known_names:
                resolved = name_map[rec_name_lower]
            else:
                # 2) Containment match — accepted ONLY when it is unambiguous,
                #    i.e. it points to exactly one known restaurant. If the name
                #    is too short, matches nothing, or matches several places,
                #    we drop it rather than guess (prevents hallucinations).
                if len(rec_name_lower) >= 4:
                    matches = {
                        known for known in known_names
                        if rec_name_lower in known or known in rec_name_lower
                    }
                    if len(matches) == 1:
                        resolved = name_map[next(iter(matches))]

            # Reject unmatched, ambiguous, or duplicate recommendations.
            if resolved is None or resolved.lower() in seen:
                dropped += 1
                logger.debug("Hallucination dropped: \"%s\"", rec.name)
                continue

            rec.name = resolved
            seen.add(resolved.lower())
            valid.append(rec)

        if dropped > 0:
            logger.info("Total hallucinations dropped: %d", dropped)

        # Re-rank after dropping
        for i, rec in enumerate(valid, 1):
            rec.rank = i

        return valid, dropped


# ═══════════════════════════════════════════════════════════════════════════
#  FALLBACK RANKING
# ═══════════════════════════════════════════════════════════════════════════

def fallback_ranking(
    candidates: List[Dict[str, Any]],
    preferences: Optional[UserPreferences] = None,
    top_n: int = 5,
) -> List[Recommendation]:
    """
    Deterministic fallback ranking when LLM is unavailable.
    
    Sort by: rating DESC -> votes DESC -> cost ASC
    Returns top_n with generic explanations.
    """
    if not candidates:
        return []

    # Sort: highest rating first, then most votes, then cheapest
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            -(c.get("rate") or 0),
            -(c.get("votes") or 0),
            (c.get("approx_cost") or 9999),
        ),
    )

    recommendations = []
    seen_names = set()  # the dataset has duplicate rows; show each place once
    for c in sorted_cands:
        name = str(c.get("name", "Unknown"))
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        rate = c.get("rate", "N/A")
        cost = c.get("approx_cost", "N/A")
        cuisines = c.get("cuisines", "")

        explanation = (
            f"Highly rated at {rate}/5 with {c.get('votes', 0)} votes. "
            f"Serves {cuisines} at Rs{cost} for two."
        )

        recommendations.append(Recommendation(
            rank=len(recommendations) + 1,
            name=name,
            explanation=explanation,
            source="fallback",
        ))
        if len(recommendations) >= top_n:
            break

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════
#  RECOMMENDER (Main class)
# ═══════════════════════════════════════════════════════════════════════════

class Recommender:
    """
    Full LLM recommendation pipeline:
      1. Build prompt from templates
      2. Call LLM (Grok)
      3. Parse response
      4. Validate against candidates (anti-hallucination)
      5. Fallback if needed
    """

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self._llm = llm_adapter or GrokAdapter()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._parser = ResponseParser()
        self._validator = HallucinationValidator()

    def recommend(
        self,
        preferences: UserPreferences,
        candidates: List[Dict[str, Any]],
        session_context: Optional[str] = None,
    ) -> List[Recommendation]:
        """
        Get restaurant recommendations via LLM with validation and fallback.
        
        Returns a list of validated Recommendation objects.
        """
        if not candidates:
            return []

        # Step 1: Build prompt
        prompt = self._prompt_builder.build_prompt(
            preferences, candidates, session_context
        )
        logger.info("Prompt built: ~%s tokens", prompt['token_estimate'])

        # Step 2: Call LLM (with retry)
        # Step 2: Call LLM (with retry)
        # --- Initialize metrics ---
        self.last_metrics = {
            "token_estimate": len(prompt["user"]) // 4,
            "template_version": "v1_cot",
            "provider": self._llm.provider if hasattr(self._llm, "provider") else "groq",
            "model": self._llm.model if hasattr(self._llm, "model") else "llama-3.3-70b-versatile",
            "llm_ms": 0,
            "source": "fallback",
            "hallucinations_dropped": 0
        }
        
        start_llm = time.time()
        raw_response = self._call_with_retry(
            prompt["system"], prompt["user"], retries=1
        )
        self.last_metrics["llm_ms"] = int((time.time() - start_llm) * 1000)

        if raw_response:
            # Step 3: Parse response
            recommendations = self._parser.parse(raw_response)

            if recommendations:
                # Step 4: Anti-hallucination validation
                valid_recs, dropped = self._validator.validate(
                    recommendations, candidates
                )
                self.last_metrics["hallucinations_dropped"] = dropped

                if valid_recs:
                    logger.info("LLM returned %d valid recommendations "
                                "(%d hallucinations dropped)", len(valid_recs), dropped)
                    self.last_metrics["source"] = "llm"
                    return valid_recs
                else:
                    logger.info("All LLM recommendations were hallucinated. Falling back.")
            else:
                logger.info("Could not parse LLM response. Falling back.")
        else:
            logger.info("LLM call failed. Using fallback ranking.")

        # Step 5: Fallback
        return fallback_ranking(candidates, preferences)

    def _call_with_retry(
        self, system: str, user: str, retries: int = 1
    ) -> Optional[str]:
        """Call LLM with retry on failure."""
        for attempt in range(retries + 1):
            try:
                start = time.time()
                response = self._llm.call(system, user)
                elapsed = time.time() - start
                logger.info("LLM response received in %.1fs (%d chars)", elapsed, len(response))
                return response
            except Exception as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.warning("Retry %d/%d after %ds...", attempt + 1, retries, wait)
                    time.sleep(wait)
                else:
                    logger.error("All %d attempts failed: %s", retries + 1, e)
                    return None


# ---------------------------------------------------------------------------
#  CLI entry-point for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_layer import DataLayer
    from src.input_parser import InputParser
    from src.retrieval_engine import RetrievalEngine

    dl = DataLayer()
    parser = InputParser(dl)
    engine = RetrievalEngine(dl)
    recommender = Recommender()

    print("=" * 70)
    print("Phase 5: LLM Recommender Tests")
    print("=" * 70)

    # --- Test 1: Response Parser (JSON) ---
    print("\n--- [1] JSON Parser Test ---")
    sample_json = '''[
        {"rank": 1, "name": "Test Restaurant", "explanation": "Great food"},
        {"rank": 2, "name": "Another Place", "explanation": "Nice ambiance"}
    ]'''
    parsed = ResponseParser.parse(sample_json)
    print(f"  Parsed {len(parsed)} recommendations from JSON")
    assert len(parsed) == 2, "JSON parsing failed"
    assert parsed[0].name == "Test Restaurant"
    print("  [OK] JSON parsing works")

    # --- Test 2: Response Parser (with code fences) ---
    print("\n--- [2] JSON with code fences ---")
    fenced = '```json\n' + sample_json + '\n```'
    parsed = ResponseParser.parse(fenced)
    assert len(parsed) == 2, "Fenced JSON parsing failed"
    print("  [OK] Code-fenced JSON parsing works")

    # --- Test 3: Response Parser (regex fallback) ---
    print("\n--- [3] Regex Fallback Parser Test ---")
    sample_text = """Here are my recommendations:
1. **Spice Garden** - Amazing North Indian food with a cozy ambiance
2. **Pasta Palace** - Best Italian in the area with great reviews
3. **Wok & Roll** - Excellent Chinese cuisine at affordable prices"""
    parsed = ResponseParser.parse(sample_text)
    print(f"  Parsed {len(parsed)} recommendations from regex")
    assert len(parsed) >= 2, "Regex parsing failed"
    print("  [OK] Regex fallback works")

    # --- Test 4: Anti-Hallucination ---
    print("\n--- [4] Anti-Hallucination Validation ---")
    fake_recs = [
        Recommendation(rank=1, name="The Hungers Zone", explanation="Great", source="llm"),
        Recommendation(rank=2, name="TOTALLY FAKE PLACE", explanation="Fake", source="llm"),
        Recommendation(rank=3, name="Chefie", explanation="Good", source="llm"),
    ]
    fake_candidates = [
        {"name": "The Hungers Zone", "rate": 3.8},
        {"name": "Chefie", "rate": 3.8},
        {"name": "Mid Night Hunting", "rate": 3.7},
    ]
    valid, dropped = HallucinationValidator.validate(fake_recs, fake_candidates)
    print(f"  Valid: {len(valid)}, Dropped: {dropped}")
    assert dropped == 1, f"Expected 1 hallucination, got {dropped}"
    assert len(valid) == 2
    print("  [OK] Hallucination detection works")

    # --- Test 5: Fallback Ranking ---
    print("\n--- [5] Fallback Ranking ---")
    prefs = parser.parse_user_input("cheap Italian in Koramangala")
    result = engine.retrieve(prefs)
    fb = fallback_ranking(result.candidates, prefs)
    print(f"  Fallback returned {len(fb)} recommendations")
    for r in fb[:3]:
        name = r.name.encode('ascii', 'replace').decode('ascii')
        print(f"    #{r.rank} {name} | source={r.source}")
    assert all(r.source == "fallback" for r in fb)
    print("  [OK] Fallback ranking works")

    # --- Test 6: Full pipeline (may fail if no API key) ---
    print("\n--- [6] Full Recommend Pipeline ---")
    recs = recommender.recommend(prefs, result.candidates)
    print(f"  Got {len(recs)} recommendations (source={recs[0].source if recs else 'none'})")
    for r in recs[:3]:
        name = r.name.encode('ascii', 'replace').decode('ascii')
        print(f"    #{r.rank} {name}: {r.explanation[:60]}...")

    dl.close()
    print(f"\n{'='*70}")
    print("Phase 5 COMPLETE")
    print("=" * 70)
