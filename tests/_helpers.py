"""Shared test helpers — data-availability gate for data-backed tests."""

import os
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "restaurants.db")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_store")

DATA_AVAILABLE = os.path.exists(DB_PATH) and os.path.isdir(CHROMA_PATH)

requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="data/restaurants.db or data/chroma_store missing — run data_loader + data_layer first",
)
