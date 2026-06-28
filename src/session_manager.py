"""
session_manager.py -- Phase 7b: Session Management

Implements an in-memory session store to manage user conversation state across multiple turns.
Supports context merging for follow-up queries (e.g., "cheaper options", "something else").
"""

import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.models import UserPreferences, Recommendation

@dataclass
class Session:
    session_id: str
    preferences: UserPreferences
    history: List[List[Recommendation]] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)


class InMemoryStore:
    """In-memory storage for active sessions with TTL."""
    def __init__(self, ttl: int = 1800, max_history: int = 5):
        self._ttl = ttl
        self._max_history = max_history
        self._sessions: Dict[str, Session] = {}

    def _cleanup(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [sid for sid, sess in self._sessions.items() if now - sess.last_accessed > self._ttl]
        for sid in expired:
            self.expire_session(sid)

    def create_session(self, prefs: UserPreferences) -> str:
        """Create a new session and return the ID."""
        self._cleanup()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(session_id=session_id, preferences=prefs)
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID and update its last accessed time."""
        self._cleanup()
        session = self._sessions.get(session_id)
        if session:
            session.last_accessed = time.time()
        return session

    def update_session(self, session_id: str, prefs: UserPreferences, recs: List[Recommendation]) -> None:
        """Update session state with new preferences and history."""
        session = self.get_session(session_id)
        if session:
            session.preferences = prefs
            session.history.append(recs)
            if len(session.history) > self._max_history:
                session.history = session.history[-self._max_history:]
            session.last_accessed = time.time()

    def expire_session(self, session_id: str) -> None:
        """Manually expire/remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]


class SessionManager:
    """Manages conversational session context and preference merging."""
    def __init__(self, ttl: int = 1800, max_history: int = 5):
        self._store = InMemoryStore(ttl=ttl, max_history=max_history)

    def get_or_create_session(self, session_id: Optional[str], prefs: UserPreferences) -> str:
        """Get an existing session ID or create a new one."""
        if session_id:
            session = self._store.get_session(session_id)
            if session:
                return session_id
        return self._store.create_session(prefs)
        
    def merge_with_session(self, session_id: str, raw_input: str, parsed_new_prefs: UserPreferences) -> UserPreferences:
        """
        Merge new partial preferences with base session context.
        Handles conversational follow-ups like:
        - "cheaper options" (keeps location, updates budget)
        - "something else" (adds exclusions based on history)
        """
        session = self._store.get_session(session_id)
        if not session:
            return parsed_new_prefs
            
        base_prefs = session.preferences
        
        # Merge fields: take new if provided, otherwise keep base
        merged = UserPreferences(
            location=parsed_new_prefs.location or base_prefs.location,
            budget_tier=parsed_new_prefs.budget_tier or base_prefs.budget_tier,
            budget_max=parsed_new_prefs.budget_max or base_prefs.budget_max,
            cuisine=parsed_new_prefs.cuisine or base_prefs.cuisine,
            min_rating=parsed_new_prefs.min_rating or base_prefs.min_rating,
            additional_prefs=parsed_new_prefs.additional_prefs or base_prefs.additional_prefs
        )
        
        # Handle "something else" / "different" exclusions
        lower_input = str(raw_input).lower()
        if any(word in lower_input for word in ["else", "different", "other"]):
            excluded_names = []
            for recs in session.history:
                for r in recs:
                    if r.name not in excluded_names:
                        excluded_names.append(r.name)
                        
            if excluded_names:
                exclusion_str = f"Do NOT recommend: {', '.join(excluded_names)}."
                if merged.additional_prefs:
                    merged.additional_prefs += " | " + exclusion_str
                else:
                    merged.additional_prefs = exclusion_str
                    
        return merged

    def update_session(self, session_id: str, prefs: UserPreferences, recs: List[Recommendation]) -> None:
        """Persist updated context back to the store."""
        self._store.update_session(session_id, prefs, recs)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session from store."""
        return self._store.get_session(session_id)
