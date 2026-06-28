"""Unit tests for the structured logger (Phase 8)."""

import json
import logging
import os

from src.logger import Logger, RequestLog, FeedbackData


def _fresh_logger(path):
    # The logger uses a shared stdlib logger ("ZomatoRecommender"); clear its
    # handlers so each test writes to its own temp file.
    logging.getLogger("ZomatoRecommender").handlers.clear()
    return Logger(log_path=path)


def _request(request_id, source, **over):
    base = dict(
        request_id=request_id, timestamp="2026-06-28T00:00:00+00:00", session_id="s1",
        raw_input="cheap italian in btm", parsed_prefs={"location": "btm"},
        stage1_count=5, stage2_count=3, filters_relaxed=[], retrieval_ms=10,
        token_estimate=100, template_version="v1_cot", provider="groq", model="llama",
        llm_ms=200, source=source, hallucinations_dropped=1,
        recommendations=[{"rank": 1, "name": "X"}], total_ms=100,
    )
    base.update(over)
    return RequestLog(**base)


def test_log_request_writes_valid_jsonl(tmp_path):
    path = str(tmp_path / "q.jsonl")
    logger = _fresh_logger(path)
    logger.log_request(_request("r1", "llm"))

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        line = f.readline()
    data = json.loads(line)                      # must be valid JSON
    assert data["request_id"] == "r1"
    assert data["source"] == "llm"
    assert data["parsed_prefs"]["location"] == "btm"


def test_log_feedback_appends_event(tmp_path):
    path = str(tmp_path / "q.jsonl")
    logger = _fresh_logger(path)
    logger.log_request(_request("r1", "llm"))
    logger.log_feedback("r1", FeedbackData(user_accepted=True, thumbs_up=True))

    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 2
    fb = json.loads(lines[1])
    assert fb["event_type"] == "feedback"
    assert fb["request_id"] == "r1"
    assert fb["feedback"]["thumbs_up"] is True


def test_get_metrics_aggregates(tmp_path):
    path = str(tmp_path / "q.jsonl")
    logger = _fresh_logger(path)
    logger.log_request(_request("a", "llm"))
    logger.log_request(_request("b", "fallback"))
    logger.log_feedback("a", FeedbackData(thumbs_up=False))   # ignored by metrics

    m = logger.get_metrics()
    assert m["total_requests"] == 2
    assert m["llm_requests"] == 1
    assert abs(m["fallback_rate"] - 0.5) < 1e-9
    assert m["total_hallucinations_dropped"] == 2
    assert m["average_latency_ms"] == 100


def test_get_metrics_empty_when_no_log(tmp_path):
    logger = _fresh_logger(str(tmp_path / "missing.jsonl"))
    m = logger.get_metrics()
    assert m["total_requests"] == 0
    assert m["fallback_rate"] == 0.0
