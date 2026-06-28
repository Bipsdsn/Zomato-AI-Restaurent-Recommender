"""
Shared pytest fixtures and test doubles for the Zomato Milestone 1 AI suite.

Heavy resources (SQLite + ChromaDB via DataLayer) are built once per session.
Tests that need real data are skipped automatically if the data stores are
not present on disk.
"""

import os
import re
import json

import pytest

from tests._helpers import DB_PATH, CHROMA_PATH, DATA_AVAILABLE as _DATA_AVAILABLE


# ---------------------------------------------------------------------------
#  Heavy shared resources (session-scoped)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def data_layer():
    """Real DataLayer backed by the persisted SQLite + ChromaDB stores."""
    if not _DATA_AVAILABLE:
        pytest.skip("data stores not available")
    from src.data_layer import DataLayer
    dl = DataLayer(db_path=DB_PATH, chroma_path=CHROMA_PATH)
    yield dl
    dl.close()


@pytest.fixture(scope="session")
def parser(data_layer):
    from src.input_parser import InputParser
    return InputParser(data_layer)


@pytest.fixture(scope="session")
def retrieval_engine(data_layer):
    from src.retrieval_engine import RetrievalEngine
    return RetrievalEngine(data_layer)


@pytest.fixture(scope="session")
def prompt_builder():
    from src.prompt_builder import PromptBuilder
    return PromptBuilder()


# ---------------------------------------------------------------------------
#  Test doubles for the LLM adapter
# ---------------------------------------------------------------------------
@pytest.fixture
def MockLLMAdapter():
    """
    Factory for a deterministic LLM adapter that echoes back the first few
    restaurant names found in the prompt as a valid JSON array. This drives
    the LLM "happy path" without any network calls.
    """
    from src.recommender import LLMAdapter

    class _MockAdapter(LLMAdapter):
        def __init__(self, top_n=3):
            self._top_n = top_n

        def call(self, system_prompt: str, user_prompt: str) -> str:
            names = re.findall(r"\[\d+\]\s*(.+?)\s*\|", user_prompt)
            names = names[: self._top_n] or ["Unknown"]
            payload = [
                {"rank": i + 1, "name": n.strip(),
                 "explanation": f"Great match #{i + 1} for your preferences."}
                for i, n in enumerate(names)
            ]
            return json.dumps(payload)

        def get_provider_name(self) -> str:
            return "mock"

        def get_model_name(self) -> str:
            return "mock-model"

    return _MockAdapter


@pytest.fixture
def FailingLLMAdapter():
    """Factory for an adapter that always raises — used to test fallback."""
    from src.recommender import LLMAdapter

    class _FailingAdapter(LLMAdapter):
        def call(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("simulated LLM outage")

        def get_provider_name(self) -> str:
            return "mock-failing"

        def get_model_name(self) -> str:
            return "none"

    return _FailingAdapter


# ---------------------------------------------------------------------------
#  Lightweight sample data (no DB needed)
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_candidates():
    return [
        {"name": "Toscano", "cuisines": "italian, continental", "rate": 4.6,
         "votes": 1200, "approx_cost": 1500, "rest_type": "Casual Dining",
         "online_order": True, "book_table": True, "dish_liked": "Pasta, Pizza"},
        {"name": "Spice Garden", "cuisines": "north indian, chinese", "rate": 4.2,
         "votes": 800, "approx_cost": 600, "rest_type": "Casual Dining",
         "online_order": True, "book_table": False, "dish_liked": "Biryani"},
        {"name": "Wok Express", "cuisines": "chinese, thai", "rate": 3.9,
         "votes": 300, "approx_cost": 400, "rest_type": "Quick Bites",
         "online_order": False, "book_table": False, "dish_liked": "Noodles"},
    ]
