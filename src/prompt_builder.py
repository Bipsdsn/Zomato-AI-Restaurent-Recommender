"""
prompt_builder.py -- Phase 4: Prompt Engineering Module

Builds structured, template-driven prompts for the LLM recommender.
All templates are loaded from the prompts/ directory so changes
require zero code modifications.

Sections assembled in order:
  1. System instruction (anti-hallucination, persona)
  2. User preferences context
  3. Restaurant data (token-efficient serialization)
  4. Chain-of-thought reasoning instruction
  5. Output format specification
"""

import os
from typing import List, Dict, Any, Optional

from src.models import UserPreferences


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
CHARS_PER_TOKEN = 4          # heuristic: ~4 chars per token
MAX_INPUT_TOKENS = 2048      # budget for the full prompt
MAX_RESTAURANTS = 15         # absolute cap (matches context window guard)


# ---------------------------------------------------------------------------
#  Prompt Builder
# ---------------------------------------------------------------------------
class PromptBuilder:
    """
    Template-driven prompt builder for the restaurant recommendation LLM.

    Loads .txt templates from the prompts/ directory and assembles them
    into a single prompt with token budget management.
    """

    def __init__(self, prompts_dir: str = PROMPTS_DIR):
        self._prompts_dir = prompts_dir
        self._templates: Dict[str, str] = {}
        self._load_templates()

    # ------------------------------------------------------------------
    #  Template loading
    # ------------------------------------------------------------------
    def _load_templates(self) -> None:
        """Load all .txt templates from the prompts directory."""
        template_files = [
            "system",
            "user_context",
            "restaurant_data",
            "cot_reasoning",
            "output_format",
        ]
        for name in template_files:
            path = os.path.join(self._prompts_dir, f"{name}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._templates[name] = f.read().strip()
            else:
                print(f"[prompt_builder] WARNING: Template not found: {path}")
                self._templates[name] = ""

    def reload_templates(self) -> None:
        """Hot-reload templates from disk (zero-code-change updates)."""
        self._load_templates()

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        preferences: UserPreferences,
        restaurants: List[Dict[str, Any]],
        session_context: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Build the full prompt from templates + data.

        Returns a dict with:
          - "system": system instruction string
          - "user": concatenated user message (context + data + CoT + format)
          - "token_estimate": estimated token count
        """
        # 1. System instruction
        system_msg = self._templates.get("system", "")

        # 2. User preferences context
        user_context = self._render_user_context(preferences)

        # 3. Restaurant data block
        restaurant_block = self._serialize_restaurants(restaurants)

        # 4. Chain-of-thought reasoning
        cot_block = self._templates.get("cot_reasoning", "")

        # 5. Output format
        format_block = self._templates.get("output_format", "")

        # 6. Optional session / conversation context
        session_block = ""
        if session_context:
            session_block = (
                f"\n--- Previous Conversation Context ---\n"
                f"{session_context}\n"
                f"--- End Context ---\n"
            )

        # Assemble the user message
        user_msg_parts = [
            user_context,
            session_block,
            "\n--- Available Restaurants ---",
            restaurant_block,
            "--- End Restaurant Data ---\n",
            cot_block,
            "",
            format_block,
        ]
        user_msg = "\n".join(user_msg_parts)

        # Token budget check and trimming
        total_text = system_msg + user_msg
        token_estimate = self._estimate_tokens(total_text)

        if token_estimate > MAX_INPUT_TOKENS:
            # Trim restaurants to fit within budget
            user_msg, token_estimate = self._trim_to_budget(
                preferences, restaurants, cot_block, format_block,
                session_block, system_msg
            )

        return {
            "system": system_msg,
            "user": user_msg,
            "token_estimate": token_estimate,
        }

    # ------------------------------------------------------------------
    #  User context rendering
    # ------------------------------------------------------------------
    def _render_user_context(self, prefs: UserPreferences) -> str:
        """Fill the user_context template with preference values."""
        template = self._templates.get("user_context", "")

        replacements = {
            "{location}": prefs.location or "any",
            "{budget_tier}": prefs.budget_tier or "any",
            "{budget_max}": str(prefs.budget_max) if prefs.budget_max else "no limit",
            "{cuisine}": prefs.cuisine or "any",
            "{min_rating}": str(prefs.min_rating) if prefs.min_rating else "none",
            "{additional_prefs}": prefs.additional_prefs or "none",
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    # ------------------------------------------------------------------
    #  Restaurant serialization
    # ------------------------------------------------------------------
    def _serialize_restaurants(
        self, restaurants: List[Dict[str, Any]]
    ) -> str:
        """
        Serialize restaurants into the token-efficient format
        defined by the restaurant_data.txt template.
        """
        template = self._templates.get("restaurant_data", "")
        lines = []

        for idx, r in enumerate(restaurants[:MAX_RESTAURANTS], 1):
            line = template
            replacements = {
                "{idx}": str(idx),
                "{name}": str(r.get("name", "Unknown")),
                "{cuisines}": str(r.get("cuisines", "")),
                "{rating}": str(r.get("rate", "N/A")),
                "{votes}": str(r.get("votes", 0)),
                "{cost}": str(r.get("approx_cost", "N/A")),
                "{rest_type}": str(r.get("rest_type", "")),
                "{online}": "Yes" if r.get("online_order") else "No",
                "{book}": "Yes" if r.get("book_table") else "No",
                "{dishes}": str(r.get("dish_liked", ""))[:80],  # trim long dish lists
            }

            for placeholder, value in replacements.items():
                line = line.replace(placeholder, value)

            lines.append(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Token budget management
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using ~4 chars per token heuristic."""
        return len(text) // CHARS_PER_TOKEN

    def _trim_to_budget(
        self,
        preferences: UserPreferences,
        restaurants: List[Dict[str, Any]],
        cot_block: str,
        format_block: str,
        session_block: str,
        system_msg: str,
    ) -> tuple:
        """
        Progressively reduce restaurant count until the prompt
        fits within MAX_INPUT_TOKENS.
        """
        user_context = self._render_user_context(preferences)

        # Start from MAX_RESTAURANTS and reduce
        for limit in range(MAX_RESTAURANTS, 2, -1):
            trimmed_data = self._serialize_restaurants(restaurants[:limit])

            user_msg_parts = [
                user_context,
                session_block,
                "\n--- Available Restaurants ---",
                trimmed_data,
                "--- End Restaurant Data ---\n",
                cot_block,
                "",
                format_block,
            ]
            user_msg = "\n".join(user_msg_parts)
            total = system_msg + user_msg
            est = self._estimate_tokens(total)

            if est <= MAX_INPUT_TOKENS:
                print(f"[prompt_builder] Trimmed to {limit} restaurants "
                      f"({est} est. tokens)")
                return user_msg, est

        # Absolute minimum: 3 restaurants, no CoT
        minimal_data = self._serialize_restaurants(restaurants[:3])
        user_msg = "\n".join([
            user_context,
            "\n--- Available Restaurants ---",
            minimal_data,
            "--- End Restaurant Data ---\n",
            format_block,
        ])
        est = self._estimate_tokens(system_msg + user_msg)
        print(f"[prompt_builder] Minimal mode: 3 restaurants, no CoT "
              f"({est} est. tokens)")
        return user_msg, est


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

    test_query = "cheap Italian in Koramangala"
    print("=" * 70)
    print("Phase 4: Prompt Builder Tests")
    print("=" * 70)

    # Parse + Retrieve
    prefs = parser.parse_user_input(test_query)
    result = engine.retrieve(prefs)

    print(f"\nQuery: \"{test_query}\"")
    print(f"Parsed: loc={prefs.location}, budget={prefs.budget_tier}, cuisine={prefs.cuisine}")
    print(f"Retrieved: {len(result.candidates)} candidates")

    # Build prompt
    builder = PromptBuilder()
    prompt = builder.build_prompt(prefs, result.candidates)

    print(f"\n--- SYSTEM MESSAGE ({len(prompt['system'])} chars) ---")
    print(prompt["system"][:200])

    print(f"\n--- USER MESSAGE ({len(prompt['user'])} chars) ---")
    print(prompt["user"][:500])
    print("...")

    print(f"\nToken estimate: {prompt['token_estimate']}")
    assert prompt["token_estimate"] <= MAX_INPUT_TOKENS, \
        f"Over budget: {prompt['token_estimate']} > {MAX_INPUT_TOKENS}"
    print(f"OK: Within {MAX_INPUT_TOKENS} token budget")

    # Verify all 5 sections present
    checks = {
        "System prompt": "ONLY recommend" in prompt["system"],
        "User context": "preferences" in prompt["user"].lower(),
        "Restaurant data": "Available Restaurants" in prompt["user"],
        "CoT reasoning": "step-by-step" in prompt["user"] or "analyze" in prompt["user"],
        "Output format": "JSON" in prompt["user"],
        "Anti-hallucination": "NOT invent" in prompt["system"] or "NOT fabricate" in prompt["system"],
    }

    print(f"\nSection checks:")
    for name, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {name}")

    # Test with session context
    prompt_with_session = builder.build_prompt(
        prefs, result.candidates,
        session_context="User previously asked for Chinese food but wants to try something new."
    )
    assert "Previous Conversation" in prompt_with_session["user"]
    print(f"\n  [OK] Session context injection")

    dl.close()
    print(f"\n{'='*70}")
    print("Phase 4 COMPLETE")
    print("=" * 70)
