"""
cache.py -- Phase 7a: Caching Layer

Implements an in-memory cache for recommendation responses to avoid redundant LLM calls.
Features LRU eviction and TTL-based expiration.
"""

import hashlib
import time
from typing import Optional, Tuple
from collections import OrderedDict

from src.models import RecommendationResponse, UserPreferences

class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InMemoryCache:
    """
    In-memory cache using OrderedDict for LRU eviction.
    """
    def __init__(self, max_entries: int = 1000, default_ttl: int = 3600):
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        # OrderedDict: end is most recently used, start is least recently used
        self._store: OrderedDict[str, Tuple[RecommendationResponse, float]] = OrderedDict()
        self._stats = CacheStats()

    @staticmethod
    def generate_key(prefs: UserPreferences) -> str:
        """Generate a deterministic SHA256 cache key from UserPreferences."""
        key_str = f"{prefs.location}|{prefs.budget_tier}|{prefs.budget_max}|{prefs.cuisine}|{prefs.min_rating}|{prefs.additional_prefs}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get(self, key: str) -> Optional[RecommendationResponse]:
        """Retrieve a cached response if it exists and is not expired."""
        if key not in self._store:
            self._stats.misses += 1
            return None
        
        response, expiry = self._store[key]
        if time.time() > expiry:
            # Expired
            self.invalidate(key)
            self._stats.misses += 1
            return None
        
        # Move to end (most recently used)
        self._store.move_to_end(key)
        self._stats.hits += 1
        return response

    def set(self, key: str, response: RecommendationResponse, ttl: Optional[int] = None) -> None:
        """Cache a response with an optional TTL."""
        if ttl is None:
            ttl = self._default_ttl
        
        expiry = time.time() + ttl
        
        if key in self._store:
            self._store.move_to_end(key)
        
        self._store[key] = (response, expiry)
        
        # LRU Eviction
        if len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache."""
        if key in self._store:
            del self._store[key]

    def get_stats(self) -> CacheStats:
        """Get cache performance stats."""
        return self._stats
