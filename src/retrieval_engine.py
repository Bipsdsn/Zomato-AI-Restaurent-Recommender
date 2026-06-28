"""
retrieval_engine.py — Phase 3: Two-Stage Retrieval Engine

Implements the two-pass retrieval pipeline:
  Stage 1: Hard structural filters via SQLite (location, budget, rating, cuisine)
           with progressive relaxation when results < 3.
  Stage 2: Semantic ranking via ChromaDB cosine similarity on soft preferences.
  Guard:   Context window cap at MAX_LLM_CANDIDATES = 15.
"""

import time
from typing import List, Dict, Any, Optional

from src.models import UserPreferences, RetrievalResult
from src.data_layer import DataLayer
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
MAX_LLM_CANDIDATES = 15
MIN_RESULTS_THRESHOLD = 3


# ---------------------------------------------------------------------------
#  Retrieval Engine
# ---------------------------------------------------------------------------
class RetrievalEngine:
    """
    Two-stage retrieval: hard SQL filters → semantic vector ranking.
    
    Stage 1 uses SQLite with progressive relaxation.
    Stage 2 uses ChromaDB cosine similarity with a composite re-ranking guard.
    """

    def __init__(self, data_layer: DataLayer):
        self._data_layer = data_layer

    # ==================================================================
    #  PUBLIC API
    # ==================================================================
    def retrieve(self, preferences: UserPreferences) -> RetrievalResult:
        """
        Full two-stage retrieval pipeline.
        
        Args:
            preferences: Validated UserPreferences from the input parser.
        
        Returns:
            RetrievalResult with up to MAX_LLM_CANDIDATES candidates,
            plus metadata on which filters were relaxed.
        """
        start = time.time()

        # ── Stage 1: Hard filters with progressive relaxation ──────────
        stage1_results, filters_relaxed = self._stage1_hard_filters(preferences)
        stage1_count = len(stage1_results)

        # ── Stage 2: Semantic ranking (only if we have results) ────────
        if stage1_results:
            stage2_results = self._stage2_semantic_rank(
                preferences, stage1_results
            )
        else:
            stage2_results = []
        stage2_count = len(stage2_results)

        # ── Context window guard ──────────────────────────────────────
        final_candidates = self._context_window_guard(stage2_results)

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info("Stage1=%d -> Stage2=%d -> Final=%d (%dms)",
                    stage1_count, stage2_count, len(final_candidates), elapsed_ms)

        return RetrievalResult(
            candidates=final_candidates,
            filters_relaxed=filters_relaxed,
            stage1_count=stage1_count,
            stage2_count=stage2_count,
        )

    # ==================================================================
    #  STAGE 1: Hard Filters + Progressive Relaxation
    # ==================================================================
    def _stage1_hard_filters(
        self, prefs: UserPreferences
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Query SQLite with the user's hard constraints.
        If results < MIN_RESULTS_THRESHOLD, progressively relax filters.
        
        Returns:
            (results, list_of_relaxed_filter_names)
        """
        filters_relaxed: List[str] = []

        # Build initial filter dict
        filters = self._build_filter_dict(prefs)

        # Attempt 1: Full filters
        results = self._data_layer.query_structured(filters)
        if len(results) >= MIN_RESULTS_THRESHOLD:
            return results, filters_relaxed

        # ── Progressive relaxation ────────────────────────────────────

        # Step 1: Remove cuisine constraint
        if "cuisine" in filters and filters["cuisine"]:
            relaxed_filters = {k: v for k, v in filters.items() if k != "cuisine"}
            results = self._data_layer.query_structured(relaxed_filters)
            filters_relaxed.append("cuisine")
            if len(results) >= MIN_RESULTS_THRESHOLD:
                return results, filters_relaxed
            filters = relaxed_filters

        # Step 2: Expand budget by +20%
        if "budget_max" in filters and filters["budget_max"]:
            expanded = dict(filters)
            expanded["budget_max"] = int(filters["budget_max"] * 1.2)
            results = self._data_layer.query_structured(expanded)
            filters_relaxed.append("budget (+20%)")
            if len(results) >= MIN_RESULTS_THRESHOLD:
                return results, filters_relaxed
            filters = expanded

        # Step 3: Lower min_rating by 0.5
        if "min_rating" in filters and filters["min_rating"] and filters["min_rating"] > 0:
            relaxed_rating = dict(filters)
            relaxed_rating["min_rating"] = max(0, filters["min_rating"] - 0.5)
            results = self._data_layer.query_structured(relaxed_rating)
            filters_relaxed.append("min_rating (-0.5)")
            if len(results) >= MIN_RESULTS_THRESHOLD:
                return results, filters_relaxed
            filters = relaxed_rating

        # Step 4: Last resort — just location, no other filters
        if "location" in filters and filters["location"]:
            last_resort = {"location": filters["location"]}
            results = self._data_layer.query_structured(last_resort)
            filters_relaxed.append("all filters (location only)")
            return results, filters_relaxed

        # Step 5: Absolute fallback — no filters at all (top rated globally)
        results = self._data_layer.query_structured({})
        filters_relaxed.append("all filters removed")
        return results[:50], filters_relaxed

    def _build_filter_dict(self, prefs: UserPreferences) -> Dict[str, Any]:
        """Convert UserPreferences into a filter dict for query_structured()."""
        filters: Dict[str, Any] = {}

        if prefs.location:
            filters["location"] = prefs.location

        if prefs.budget_max is not None:
            filters["budget_max"] = prefs.budget_max

        if prefs.min_rating and prefs.min_rating > 0:
            filters["min_rating"] = prefs.min_rating

        if prefs.cuisine:
            filters["cuisine"] = prefs.cuisine

        if prefs.budget_tier:
            filters["budget_tier"] = prefs.budget_tier

        return filters

    # ==================================================================
    #  STAGE 2: Semantic Ranking
    # ==================================================================
    def _stage2_semantic_rank(
        self,
        prefs: UserPreferences,
        stage1_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Re-rank Stage 1 results using ChromaDB semantic similarity.
        
        Builds a natural language query from the user's soft preferences
        and queries the vector store filtered to Stage 1 IDs.
        """
        # Build the semantic query string
        query = self._build_semantic_query(prefs)

        # Get Stage 1 IDs for filtering
        stage1_ids = [r["id"] for r in stage1_results if "id" in r]

        if not stage1_ids or not query:
            # If no IDs or no meaningful query, return Stage 1 as-is
            return stage1_results

        # Query ChromaDB with Stage 1 IDs as filter
        semantic_results = self._data_layer.query_semantic(
            text=query,
            filter_ids=stage1_ids,
            top_k=min(len(stage1_ids), MAX_LLM_CANDIDATES * 2),
        )

        if not semantic_results:
            return stage1_results

        # Merge semantic scores back onto the full restaurant data
        # Create a lookup from Stage 1 results (keyed by id)
        stage1_lookup = {r["id"]: r for r in stage1_results}

        merged = []
        seen_ids = set()
        for sem in semantic_results:
            rid = sem.get("id")
            if rid in stage1_lookup and rid not in seen_ids:
                entry = dict(stage1_lookup[rid])
                entry["similarity_score"] = sem.get("similarity_score", 0.0)
                merged.append(entry)
                seen_ids.add(rid)

        # Add any Stage 1 results not in semantic results (with score 0)
        for r in stage1_results:
            if r["id"] not in seen_ids:
                entry = dict(r)
                entry["similarity_score"] = 0.0
                merged.append(entry)
                seen_ids.add(r["id"])

        return merged

    def _build_semantic_query(self, prefs: UserPreferences) -> str:
        """
        Build a natural language query from user preferences for
        ChromaDB semantic search.
        """
        parts = []

        if prefs.cuisine:
            parts.append(f"{prefs.cuisine} restaurant")

        if prefs.additional_prefs:
            parts.append(prefs.additional_prefs)

        if prefs.budget_tier:
            parts.append(f"{prefs.budget_tier} price range")

        if prefs.location:
            parts.append(f"in {prefs.location}")

        # If nothing specific, use a generic query
        if not parts:
            return "good popular restaurant"

        return ", ".join(parts)

    # ==================================================================
    #  CONTEXT WINDOW GUARD
    # ==================================================================
    def _context_window_guard(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        If candidates exceed MAX_LLM_CANDIDATES, apply composite scoring
        and return only the top MAX_LLM_CANDIDATES.
        
        composite_score = (similarity * 0.5) + (norm_rating * 0.3) + (norm_votes * 0.2)
        """
        if len(candidates) <= MAX_LLM_CANDIDATES:
            return candidates

        # Compute normalized values for scoring
        max_rating = max((c.get("rate") or 0) for c in candidates) or 1
        max_votes = max((c.get("votes") or 0) for c in candidates) or 1

        scored = []
        for c in candidates:
            similarity = c.get("similarity_score", 0.0)
            norm_rating = (c.get("rate") or 0) / max_rating
            norm_votes = (c.get("votes") or 0) / max_votes

            composite = (similarity * 0.5) + (norm_rating * 0.3) + (norm_votes * 0.2)
            entry = dict(c)
            entry["composite_score"] = composite
            scored.append(entry)

        # Sort descending by composite score, take top N
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored[:MAX_LLM_CANDIDATES]


# ---------------------------------------------------------------------------
#  CLI entry-point for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.input_parser import InputParser

    dl = DataLayer()
    parser = InputParser(dl)
    engine = RetrievalEngine(dl)

    test_queries = [
        "cheap Italian in Koramangala",
        "best rated Chinese under 500",
        "upscale dining for date night in Indiranagar",
        "family-friendly place in BTM, medium budget",
        "something quick near Whitefield",
    ]

    print("=" * 70)
    print("Phase 3: Two-Stage Retrieval Engine Tests")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- [{i}] \"{query}\" ---")
        prefs = parser.parse_user_input(query)
        print(f"  Parsed: loc={prefs.location}, budget={prefs.budget_tier}/{prefs.budget_max}, "
              f"cuisine={prefs.cuisine}, rating={prefs.min_rating}, prefs={prefs.additional_prefs}")

        result = engine.retrieve(prefs)
        print(f"  Stage1={result.stage1_count}, Stage2={result.stage2_count}, "
              f"Final={len(result.candidates)}")

        if result.filters_relaxed:
            print(f"  Relaxed: {result.filters_relaxed}")

        for j, c in enumerate(result.candidates[:3], 1):
            score = c.get("similarity_score", "N/A")
            comp = c.get("composite_score", "")
            score_str = f"sim={score:.3f}" if isinstance(score, float) else ""
            comp_str = f" comp={comp:.3f}" if isinstance(comp, float) else ""
            name = c['name'].encode('ascii', 'replace').decode('ascii')
            print(f"  #{j} {name} | Rs{c['approx_cost']} | "
                  f"Rate:{c.get('rate', 'N/A')} | {score_str}{comp_str}")

    # Test progressive relaxation with a very restrictive query
    print(f"\n--- [6] Testing progressive relaxation ---")
    tight_prefs = UserPreferences(
        location="koramangala",
        budget_max=50,
        budget_tier="low",
        cuisine="mongolian",
        min_rating=4.9,
    )
    result = engine.retrieve(tight_prefs)
    print(f"  Relaxed: {result.filters_relaxed}")
    print(f"  Results: {len(result.candidates)}")

    # Test context window guard
    print(f"\n--- [7] Testing context window guard ---")
    broad_prefs = UserPreferences(location="btm")
    result = engine.retrieve(broad_prefs)
    print(f"  Capped at MAX={MAX_LLM_CANDIDATES}: {len(result.candidates)} candidates")
    assert len(result.candidates) <= MAX_LLM_CANDIDATES, "Guard failed!"
    print(f"  OK - Guard working correctly")

    dl.close()
    print(f"\n{'='*70}")
    print("Phase 3 COMPLETE")
    print("=" * 70)
