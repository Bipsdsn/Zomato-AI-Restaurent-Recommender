"""Unit tests for the in-memory caching layer (Phase 7a)."""

from src.cache import InMemoryCache, CacheStats
from src.models import UserPreferences, RecommendationResponse, Recommendation


def _resp(name="Toscano"):
    return RecommendationResponse(
        recommendations=[Recommendation(rank=1, name=name, explanation="x")],
        filters_relaxed=[],
    )


def _prefs(**kw):
    base = dict(location="koramangala", budget_tier="low", budget_max=300,
                cuisine="italian", min_rating=4.0, additional_prefs=None)
    base.update(kw)
    return UserPreferences(**base)


def test_key_is_deterministic_and_sensitive():
    k1 = InMemoryCache.generate_key(_prefs())
    k2 = InMemoryCache.generate_key(_prefs())
    k3 = InMemoryCache.generate_key(_prefs(cuisine="chinese"))
    assert k1 == k2          # same prefs → same key
    assert k1 != k3          # different prefs → different key
    assert len(k1) == 64     # sha256 hexdigest


def test_set_then_get_hit():
    cache = InMemoryCache()
    key = "abc"
    cache.set(key, _resp("A"))
    got = cache.get(key)
    assert got is not None
    assert got.recommendations[0].name == "A"


def test_get_miss_returns_none():
    cache = InMemoryCache()
    assert cache.get("nope") is None


def test_ttl_expiration():
    cache = InMemoryCache()
    cache.set("k", _resp(), ttl=-10)   # already expired
    assert cache.get("k") is None


def test_lru_eviction():
    cache = InMemoryCache(max_entries=2)
    cache.set("a", _resp("A"))
    cache.set("b", _resp("B"))
    cache.set("c", _resp("C"))         # evicts least-recently-used "a"
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_lru_touch_on_get():
    cache = InMemoryCache(max_entries=2)
    cache.set("a", _resp("A"))
    cache.set("b", _resp("B"))
    cache.get("a")                     # "a" now most-recently-used
    cache.set("c", _resp("C"))         # evicts "b" instead of "a"
    assert cache.get("a") is not None
    assert cache.get("b") is None


def test_stats_hit_rate():
    cache = InMemoryCache()
    cache.set("k", _resp())
    cache.get("k")        # hit
    cache.get("missing")  # miss
    stats = cache.get_stats()
    assert isinstance(stats, CacheStats)
    assert stats.hits == 1
    assert stats.misses == 1
    assert abs(stats.hit_rate - 0.5) < 1e-9


def test_invalidate():
    cache = InMemoryCache()
    cache.set("k", _resp())
    cache.invalidate("k")
    assert cache.get("k") is None
