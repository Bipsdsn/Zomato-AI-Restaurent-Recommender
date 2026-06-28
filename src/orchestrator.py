"""
orchestrator.py -- Phase 6: Orchestrator & Integration

Wires all components into a single process_request() pipeline:
  Input Parser -> Retrieval Engine -> Prompt Builder -> LLM Recommender

Handles error routing at each step with graceful degradation.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional
import os

from src.models import UserPreferences, RecommendationResponse
from src.data_layer import DataLayer
from src.input_parser import InputParser, InvalidLocationError
from src.retrieval_engine import RetrievalEngine
from src.prompt_builder import PromptBuilder
from src.recommender import Recommender, fallback_ranking
from src.cache import InMemoryCache
from src.logging_config import get_logger

logger = get_logger(__name__)
from src.session_manager import SessionManager


class RecommendationOrchestrator:
    """
    Central orchestrator that drives the full recommendation pipeline.
    
    All components are injected via constructor for testability.
    """

    def __init__(
        self,
        data_layer: DataLayer,
        input_parser: InputParser,
        retrieval_engine: RetrievalEngine,
        prompt_builder: PromptBuilder,
        recommender: Recommender,
        cache: Optional[InMemoryCache] = None,
        session_manager: Optional[SessionManager] = None
    ):
        self._data_layer = data_layer
        self._parser = input_parser
        self._retrieval = retrieval_engine
        self._prompt_builder = prompt_builder
        self._recommender = recommender
        
        self._cache = cache if cache else InMemoryCache()
        self._session_manager = session_manager if session_manager else SessionManager()
        
        # Load Logger
        from src.logger import Logger
        self._logger = Logger()
        
        # Load configs
        self._cache_enabled = str(os.getenv("CACHE_ENABLED", "true")).lower() == "true"
        self._session_enabled = str(os.getenv("SESSION_ENABLED", "true")).lower() == "true"

    # ==================================================================
    #  PUBLIC API
    # ==================================================================
    def process_request(
        self,
        raw_input,
        session_id: Optional[str] = None,
    ) -> RecommendationResponse:
        """
        Full end-to-end pipeline:
          1. Parse input -> UserPreferences
          2. Check cache (if enabled)
          3. Retrieve candidates (two-stage)
          4. Get LLM recommendations (with fallback)
          5. Update session and cache
          6. Return RecommendationResponse
        """
        start = time.time()

        # ── Step 1: Parse Input ────────────────────────────────────────
        try:
            if isinstance(raw_input, str):
                preferences = self._parser.parse_user_input(raw_input)
            elif isinstance(raw_input, dict):
                preferences = self._parser.parse_user_input(raw_input)
            else:
                preferences = raw_input  # Already a UserPreferences
        except InvalidLocationError as e:
            elapsed = int((time.time() - start) * 1000)
            return RecommendationResponse(
                recommendations=[],
                filters_relaxed=[],
                session_id=session_id,
                processing_time_ms=elapsed,
            )
        except Exception as e:
            logger.warning("Parse error: %s", e)
            elapsed = int((time.time() - start) * 1000)
            return RecommendationResponse(
                recommendations=[],
                filters_relaxed=[],
                session_id=session_id,
                processing_time_ms=elapsed,
            )
            
        # ── Step 1b: Handle Session Context ────────────────────────────
        if self._session_enabled and isinstance(raw_input, str):
            # If session exists, merge the context
            if session_id:
                preferences = self._session_manager.merge_with_session(session_id, raw_input, preferences)
            
            # Create or ensure session_id is active
            session_id = self._session_manager.get_or_create_session(session_id, preferences)

        logger.info("Parsed: loc=%s, budget=%s/%s, cuisine=%s, rating=%s, prefs=%s",
                    preferences.location, preferences.budget_tier, preferences.budget_max,
                    preferences.cuisine, preferences.min_rating, preferences.additional_prefs)

        # ── Step 2: Check Cache ────────────────────────────────────────
        cache_key = None
        if self._cache_enabled:
            cache_key = self._cache.generate_key(preferences)
            cached_resp = self._cache.get(cache_key)
            if cached_resp:
                logger.info("Cache HIT for key %s...", cache_key[:8])
                elapsed = int((time.time() - start) * 1000)
                cached_resp.processing_time_ms = elapsed
                cached_resp.session_id = session_id
                
                # Update session history
                if self._session_enabled and session_id:
                    self._session_manager.update_session(session_id, preferences, cached_resp.recommendations)
                return cached_resp
            else:
                logger.info("Cache MISS for key %s...", cache_key[:8])

        # ── Step 3: Retrieve Candidates ────────────────────────────────
        try:
            retrieval_result = self._retrieval.retrieve(preferences)
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            elapsed = int((time.time() - start) * 1000)
            return RecommendationResponse(
                recommendations=[],
                filters_relaxed=[],
                session_id=session_id,
                processing_time_ms=elapsed,
            )

        candidates = retrieval_result.candidates
        filters_relaxed = retrieval_result.filters_relaxed

        if not candidates:
            logger.info("No candidates found even after relaxation.")
            elapsed = int((time.time() - start) * 1000)
            return RecommendationResponse(
                recommendations=[],
                filters_relaxed=filters_relaxed,
                session_id=session_id,
                processing_time_ms=elapsed,
            )

        # ── Step 4: Get Recommendations ────────────────────────────────
        try:
            # Pass session history as context if enabled
            session_context = None
            if self._session_enabled and session_id:
                session = self._session_manager.get_session(session_id)
                if session and len(session.history) > 0:
                    session_context = "Previous recommendations given to user:\n"
                    for i, past_recs in enumerate(session.history):
                        names = [r.name for r in past_recs]
                        session_context += f"Turn -{len(session.history)-i}: {', '.join(names)}\n"
                        
            recommendations = self._recommender.recommend(
                preferences, candidates, session_context
            )
            # Recommender saves metrics in last_metrics
            llm_metrics = getattr(self._recommender, "last_metrics", {})
        except Exception as e:
            logger.warning("Recommender error: %s. Using fallback.", e)
            recommendations = fallback_ranking(candidates, preferences)
            llm_metrics = {
                "token_estimate": 0, "template_version": "fallback",
                "provider": "none", "model": "none", "llm_ms": 0,
                "source": "fallback", "hallucinations_dropped": 0
            }

        # ── Step 5: Build Response & Update State ──────────────────────
        elapsed = int((time.time() - start) * 1000)

        response = RecommendationResponse(
            recommendations=recommendations,
            filters_relaxed=filters_relaxed,
            session_id=session_id,
            processing_time_ms=elapsed,
        )
        
        # ── Step 6: Log Request ────────────────────────────────────────
        request_id = str(uuid.uuid4())
        
        from dataclasses import asdict
        from src.logger import RequestLog
        
        log_entry = RequestLog(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            raw_input=str(raw_input),
            parsed_prefs=asdict(preferences),
            stage1_count=retrieval_result.stage1_count if 'retrieval_result' in locals() else 0,
            stage2_count=retrieval_result.stage2_count if 'retrieval_result' in locals() else 0,
            filters_relaxed=filters_relaxed,
            retrieval_ms=0,  # We didn't time retrieval precisely, but could
            token_estimate=llm_metrics.get("token_estimate", 0),
            template_version=llm_metrics.get("template_version", "unknown"),
            provider=llm_metrics.get("provider", "unknown"),
            model=llm_metrics.get("model", "unknown"),
            llm_ms=llm_metrics.get("llm_ms", 0),
            source=llm_metrics.get("source", "fallback"),
            hallucinations_dropped=llm_metrics.get("hallucinations_dropped", 0),
            recommendations=[asdict(r) for r in recommendations],
            total_ms=elapsed
        )
        
        self._logger.log_request(log_entry)
        
        # Cache the result
        if self._cache_enabled and cache_key:
            # We don't cache session_id in the shared cache, so make a copy or strip it? 
            # The cache holds RecommendationResponse. We can just set session_id=None before caching and restore it.
            resp_to_cache = RecommendationResponse(
                recommendations=recommendations,
                filters_relaxed=filters_relaxed,
                session_id=None,
                processing_time_ms=elapsed,
            )
            self._cache.set(cache_key, resp_to_cache)
            
        # Update session
        if self._session_enabled and session_id:
            self._session_manager.update_session(session_id, preferences, recommendations)

        logger.info("Done: %d recs in %dms", len(recommendations), elapsed)

        return response

    # ==================================================================
    #  METADATA HELPERS (for the UI)
    # ==================================================================
    def get_locations(self):
        """Return list of available locations for dropdowns/filters."""
        return self._data_layer.get_available_locations()

    def get_cuisines(self):
        """Return list of available cuisines for dropdowns/filters."""
        return self._data_layer.get_available_cuisines()

    def get_restaurant_count(self):
        """Return total restaurant count for display."""
        return self._data_layer.get_restaurant_count()

    def get_restaurant_details(self, name: str, location=None):
        """Look up full display metadata for a recommended restaurant by name."""
        return self._data_layer.get_restaurant_by_name(name, location)
