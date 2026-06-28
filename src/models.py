from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class UserPreferences:
    """Structured user preferences extracted from input."""
    location: Optional[str] = None
    budget_max: Optional[int] = None
    budget_tier: Optional[str] = None        # "low", "medium", "high"
    cuisine: Optional[str] = None
    min_rating: float = 0.0
    additional_prefs: Optional[str] = None


@dataclass
class RetrievalResult:
    """Output of the two-stage retrieval engine."""
    candidates: List[Dict[str, Any]]
    filters_relaxed: List[str] = field(default_factory=list)
    stage1_count: int = 0
    stage2_count: int = 0


@dataclass
class Recommendation:
    """A single restaurant recommendation from the LLM or fallback."""
    rank: int
    name: str
    explanation: str
    score: Optional[float] = None
    source: str = "llm"  # 'llm', 'fallback', 'cache'


@dataclass
class RecommendationResponse:
    """Full response returned to the user."""
    recommendations: List[Recommendation]
    filters_relaxed: List[str]
    session_id: Optional[str] = None
    processing_time_ms: int = 0
