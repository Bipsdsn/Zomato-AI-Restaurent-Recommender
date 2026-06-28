"""
logger.py -- Phase 8a: Structured Logger

Implements structured JSON logging for observability and evaluation.
Logs every request end-to-end to `logs/queries.jsonl` with rotation at 10MB.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional

from src.models import UserPreferences, Recommendation

@dataclass
class RequestLog:
    """Schema for a single recommendation request log."""
    request_id: str
    timestamp: str
    session_id: Optional[str]
    
    # Input
    raw_input: str
    parsed_prefs: Dict[str, Any]
    
    # Retrieval
    stage1_count: int
    stage2_count: int
    filters_relaxed: List[str]
    retrieval_ms: int
    
    # Prompt
    token_estimate: int
    template_version: str
    
    # LLM
    provider: str
    model: str
    llm_ms: int
    source: str
    hallucinations_dropped: int
    
    # Output
    recommendations: List[Dict[str, Any]]
    total_ms: int
    
    # Feedback (async)
    user_accepted: Optional[bool] = None
    thumbs_up: Optional[bool] = None

@dataclass
class FeedbackData:
    user_accepted: Optional[bool] = None
    thumbs_up: Optional[bool] = None

class Logger:
    """
    Structured logger for the recommendation pipeline.
    Uses RotatingFileHandler to cap file size at 10MB.
    """
    def __init__(self, log_path: str = "logs/queries.jsonl", max_bytes: int = 10 * 1024 * 1024):
        self._log_path = log_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        
        # Set up standard library logger
        self._logger = logging.getLogger("ZomatoRecommender")
        self._logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers if instantiated multiple times
        if not self._logger.handlers:
            # Rotating file handler: max 10MB, keep 5 backups
            handler = RotatingFileHandler(
                self._log_path, maxBytes=max_bytes, backupCount=5, encoding="utf-8"
            )
            # Just log the raw message (which will be JSON)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def log_request(self, log_data: RequestLog) -> None:
        """Log a complete request as a JSON line."""
        try:
            log_dict = asdict(log_data)
            self._logger.info(json.dumps(log_dict))
        except Exception as e:
            print(f"[logger] Error logging request: {e}")

    def log_feedback(self, request_id: str, feedback: Any) -> None:
        """Log user feedback for a specific request. Appended as a new line."""
        try:
            feedback_data = asdict(feedback) if hasattr(feedback, '__dataclass_fields__') else feedback
            feedback_log = {
                "event_type": "feedback",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "feedback": feedback_data
            }
            self._logger.info(json.dumps(feedback_log))
        except Exception as e:
            print(f"[logger] Error logging feedback: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate basic metrics from the log file (for optional dashboard)."""
        metrics = {
            "total_requests": 0,
            "average_latency_ms": 0,
            "fallback_rate": 0.0,
            "total_hallucinations_dropped": 0,
            "llm_requests": 0
        }
        
        if not os.path.exists(self._log_path):
            return metrics
            
        total_time = 0
        fallback_count = 0
        
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "event_type" in data and data["event_type"] == "feedback":
                            continue
                            
                        metrics["total_requests"] += 1
                        total_time += data.get("total_ms", 0)
                        metrics["total_hallucinations_dropped"] += data.get("hallucinations_dropped", 0)
                        
                        source = data.get("source", "")
                        if source == "fallback":
                            fallback_count += 1
                        elif source == "llm":
                            metrics["llm_requests"] += 1
                            
                    except json.JSONDecodeError:
                        continue
                        
            if metrics["total_requests"] > 0:
                metrics["average_latency_ms"] = total_time / metrics["total_requests"]
                metrics["fallback_rate"] = fallback_count / metrics["total_requests"]
                
        except Exception as e:
            print(f"[logger] Error reading logs for metrics: {e}")
            
        return metrics
