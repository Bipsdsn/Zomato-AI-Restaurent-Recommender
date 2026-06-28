"""Unit tests for session management & conversational merge logic (Phase 7b)."""

from src.session_manager import SessionManager, InMemoryStore, Session
from src.models import UserPreferences, Recommendation


def _prefs(**kw):
    base = dict(location="btm", budget_tier="medium", budget_max=800,
                cuisine="italian", min_rating=4.0, additional_prefs=None)
    base.update(kw)
    return UserPreferences(**base)


def _recs(*names):
    return [Recommendation(rank=i + 1, name=n, explanation="x")
            for i, n in enumerate(names)]


# ---- Lifecycle ----
def test_create_get_update_lifecycle():
    mgr = SessionManager()
    sid = mgr.get_or_create_session(None, _prefs())
    assert sid

    session = mgr.get_session(sid)
    assert isinstance(session, Session)
    assert session.preferences.location == "btm"

    mgr.update_session(sid, _prefs(), _recs("Toscano", "Spice Garden"))
    session = mgr.get_session(sid)
    assert len(session.history) == 1
    assert session.history[0][0].name == "Toscano"


def test_get_or_create_reuses_existing():
    mgr = SessionManager()
    sid = mgr.get_or_create_session(None, _prefs())
    same = mgr.get_or_create_session(sid, _prefs())
    assert sid == same


def test_invalid_session_id_creates_new():
    mgr = SessionManager()
    sid = mgr.get_or_create_session("does-not-exist", _prefs())
    assert sid != "does-not-exist"
    assert mgr.get_session(sid) is not None


def test_expired_session_returns_none():
    store = InMemoryStore(ttl=-1)          # everything immediately stale
    sid = store.create_session(_prefs())
    assert store.get_session(sid) is None


def test_history_capped_at_max():
    store = InMemoryStore(max_history=3)
    sid = store.create_session(_prefs())
    for i in range(5):
        store.update_session(sid, _prefs(), _recs(f"R{i}"))
    session = store._sessions[sid]
    assert len(session.history) == 3       # only last 3 kept
    assert session.history[-1][0].name == "R4"


# ---- Merge logic ----
def test_merge_new_field_overrides_base():
    mgr = SessionManager()
    sid = mgr.get_or_create_session(None, _prefs(cuisine="italian"))
    merged = mgr.merge_with_session(sid, "try chinese instead", _prefs(cuisine="chinese"))
    assert merged.cuisine == "chinese"     # new wins
    assert merged.location == "btm"        # base retained


def test_merge_keeps_base_when_new_missing():
    mgr = SessionManager()
    sid = mgr.get_or_create_session(None, _prefs(location="btm", cuisine="italian"))
    # New prefs with no cuisine/location → base values retained
    partial = UserPreferences(min_rating=4.5)
    merged = mgr.merge_with_session(sid, "rated higher", partial)
    assert merged.location == "btm"
    assert merged.cuisine == "italian"
    assert merged.min_rating == 4.5


def test_merge_something_else_adds_exclusions():
    mgr = SessionManager()
    sid = mgr.get_or_create_session(None, _prefs())
    mgr.update_session(sid, _prefs(), _recs("Toscano", "Spice Garden"))
    merged = mgr.merge_with_session(sid, "show me something else", _prefs())
    assert merged.additional_prefs is not None
    assert "Do NOT recommend" in merged.additional_prefs
    assert "Toscano" in merged.additional_prefs
    assert "Spice Garden" in merged.additional_prefs


def test_merge_unknown_session_returns_new_prefs():
    mgr = SessionManager()
    new = _prefs(cuisine="thai")
    merged = mgr.merge_with_session("ghost", "anything", new)
    assert merged.cuisine == "thai"
