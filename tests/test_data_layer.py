"""Tests for the data layer: SQLite structured queries + ChromaDB semantic
search + metadata helpers (Phase 1). Requires persisted data stores."""

from tests._helpers import requires_data

pytestmark = requires_data


def test_structured_query_respects_location_and_budget(data_layer):
    rows = data_layer.query_structured({"location": "btm", "budget_max": 800})
    assert isinstance(rows, list)
    assert len(rows) > 0
    for r in rows:
        assert r["location"] == "btm"
        assert r["approx_cost"] <= 800


def test_structured_query_min_rating(data_layer):
    rows = data_layer.query_structured({"location": "bellandur", "min_rating": 4.5})
    for r in rows:
        assert r["rate"] is None or r["rate"] >= 4.5


def test_structured_query_empty_filters_returns_many(data_layer):
    rows = data_layer.query_structured({})
    assert len(rows) > 10


def test_metadata_locations_and_cuisines(data_layer):
    locations = data_layer.get_available_locations()
    cuisines = data_layer.get_available_cuisines()
    assert "bellandur" in locations
    assert len(cuisines) > 0
    assert data_layer.get_restaurant_count() > 0


def test_get_restaurant_by_name(data_layer):
    rows = data_layer.query_structured({"location": "bellandur"})
    assert rows
    target = rows[0]["name"]
    found = data_layer.get_restaurant_by_name(target)
    assert found is not None
    assert found["name"].lower() == target.lower()


def test_get_restaurant_by_name_missing(data_layer):
    assert data_layer.get_restaurant_by_name("Definitely Not A Real Place 99999") is None


def test_semantic_query_returns_relevant(data_layer):
    results = data_layer.query_semantic("cozy italian place", top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0
    # similarity score attached
    assert "similarity_score" in results[0]


def test_semantic_query_filtered_by_ids(data_layer):
    base = data_layer.query_structured({"location": "bellandur"})
    ids = [r["id"] for r in base[:20]]
    assert ids
    results = data_layer.query_semantic("good food", filter_ids=ids, top_k=10)
    returned_ids = {r.get("id") for r in results}
    assert returned_ids.issubset(set(ids))


# ---------------------------------------------------------------------------
#  Ingestion build path (exercises init_db / _init_sqlite / _init_chroma)
#  Builds a tiny fresh store in a temp dir — no dependency on prebuilt data.
# ---------------------------------------------------------------------------
def test_ingestion_build_and_query(tmp_path):
    import pandas as pd
    from src.data_layer import DataLayer

    df = pd.DataFrame([
        {"name": "Alpha Cafe", "location": "btm", "cuisines": "italian, cafe",
         "approx_cost": 400, "rate": 4.3, "votes": 120, "online_order": True,
         "book_table": False, "rest_type": "Cafe", "dish_liked": "Pasta",
         "listed_in_type": "Delivery", "budget_tier": "medium", "is_new": False},
        {"name": "Beta Biryani", "location": "btm", "cuisines": "biryani, north indian",
         "approx_cost": 300, "rate": 4.1, "votes": 300, "online_order": True,
         "book_table": True, "rest_type": "Casual Dining", "dish_liked": "Biryani",
         "listed_in_type": "Dine-out", "budget_tier": "low", "is_new": False},
        {"name": "Gamma Grill", "location": "koramangala", "cuisines": "bbq, grill",
         "approx_cost": 900, "rate": 4.6, "votes": 500, "online_order": False,
         "book_table": True, "rest_type": "Fine Dining", "dish_liked": "Kebab",
         "listed_in_type": "Dine-out", "budget_tier": "high", "is_new": False},
    ])

    dl = DataLayer(db_path=str(tmp_path / "r.db"), chroma_path=str(tmp_path / "chroma"))
    dl.init_db(df)
    try:
        assert dl.get_restaurant_count() == 3
        locs = dl.get_available_locations()
        assert "btm" in locs and "koramangala" in locs
        assert dl.get_available_cuisines()                       # non-empty

        rows = dl.query_structured({"location": "btm", "budget_max": 400})
        assert len(rows) >= 1
        assert all(r["location"] == "btm" for r in rows)

        sem = dl.query_semantic("italian cafe", top_k=3)
        assert isinstance(sem, list) and len(sem) >= 1

        found = dl.get_restaurant_by_name("Alpha Cafe")
        assert found and found["name"] == "Alpha Cafe"
    finally:
        dl.close()
