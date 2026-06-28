"""
data_layer.py — Phase 1c/1d: SQLite Structured Store + ChromaDB Vector Store

Provides a unified DataLayer class that wraps:
  - SQLite for structured queries (location, budget, rating filters)
  - ChromaDB for semantic similarity search (cuisine, ambiance, reviews)
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional

import pandas as pd
import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------------------------
# DataLayer — unified interface over both stores
# ---------------------------------------------------------------------------
class DataLayer:
    """
    Dual-store data layer for restaurant data.
    
    - SQLite: fast structured queries with indexes on location, cost, rating
    - ChromaDB: semantic vector search on composite restaurant descriptions
    """

    def __init__(self, db_path: str = "data/restaurants.db",
                 chroma_path: str = "data/chroma_store"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._conn: Optional[sqlite3.Connection] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    # ======================================================================
    #  INITIALIZATION
    # ======================================================================
    def init_db(self, df: pd.DataFrame) -> None:
        """
        One-time setup: populate both SQLite and ChromaDB from a
        preprocessed DataFrame.
        """
        print("[data_layer] Initializing SQLite...")
        self._init_sqlite(df)
        print("[data_layer] Initializing ChromaDB...")
        self._init_chroma(df)
        print("[data_layer] Both stores ready.")

    # ------------------------------------------------------------------
    #  SQLite setup
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        """Get (or create) an SQLite connection with row_factory."""
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            # check_same_thread=False: the connection is reused across Flask
            # worker threads. Access is read-only at request time, so this is
            # safe; WAL mode further allows concurrent reads.
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
        return self._conn

    def _init_sqlite(self, df: pd.DataFrame) -> None:
        """Create the restaurants table, indexes, and FTS5 table."""
        conn = self._get_conn()
        cur = conn.cursor()

        # Drop existing tables for a clean rebuild
        cur.execute("DROP TABLE IF EXISTS restaurants")
        cur.execute("DROP TABLE IF EXISTS restaurants_fts")

        # Main table
        cur.execute("""
            CREATE TABLE restaurants (
                id              INTEGER PRIMARY KEY,
                name            TEXT NOT NULL,
                location        TEXT NOT NULL,
                cuisines        TEXT NOT NULL DEFAULT '',
                approx_cost     INTEGER,
                rate            REAL,
                votes           INTEGER DEFAULT 0,
                online_order    BOOLEAN DEFAULT 0,
                book_table      BOOLEAN DEFAULT 0,
                rest_type       TEXT DEFAULT '',
                dish_liked      TEXT DEFAULT '',
                listed_in_type  TEXT DEFAULT '',
                budget_tier     TEXT DEFAULT 'medium',
                is_new          BOOLEAN DEFAULT 0
            )
        """)

        # Indexes for hard-filter queries
        cur.execute("CREATE INDEX idx_location    ON restaurants(location)")
        cur.execute("CREATE INDEX idx_approx_cost ON restaurants(approx_cost)")
        cur.execute("CREATE INDEX idx_rate        ON restaurants(rate)")
        cur.execute("CREATE INDEX idx_budget_tier ON restaurants(budget_tier)")

        # FTS5 virtual table for cuisine full-text search
        cur.execute("""
            CREATE VIRTUAL TABLE restaurants_fts USING fts5(
                cuisines,
                content='restaurants',
                content_rowid='id'
            )
        """)

        # Bulk insert
        cols = [
            "name", "location", "cuisines", "approx_cost", "rate", "votes",
            "online_order", "book_table", "rest_type", "dish_liked",
            "listed_in_type", "budget_tier", "is_new",
        ]
        available_cols = [c for c in cols if c in df.columns]
        placeholders = ", ".join(["?"] * len(available_cols))
        col_str = ", ".join(available_cols)

        rows = []
        for _, row in df.iterrows():
            values = []
            for c in available_cols:
                val = row[c]
                if pd.isna(val):
                    values.append(None)
                elif isinstance(val, bool):
                    values.append(int(val))
                else:
                    values.append(val)
            rows.append(tuple(values))

        cur.executemany(
            f"INSERT INTO restaurants ({col_str}) VALUES ({placeholders})",
            rows,
        )

        # Populate FTS index
        cur.execute("""
            INSERT INTO restaurants_fts(rowid, cuisines)
            SELECT id, cuisines FROM restaurants
        """)

        conn.commit()
        print(f"[data_layer] SQLite: {len(rows)} restaurants inserted with indexes.")

    # ------------------------------------------------------------------
    #  ChromaDB setup
    # ------------------------------------------------------------------
    def _get_chroma_collection(self):
        """Get (or create) the ChromaDB collection."""
        if self._collection is None:
            os.makedirs(self.chroma_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="restaurant_embeddings",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _build_document(self, row: pd.Series) -> str:
        """
        Build a composite text document for embedding.
        Format from architecture §3.2:
          "{name}. Cuisines: {cuisines}. Type: {rest_type}.
           Popular dishes: {dish_liked}. Cost ₹{cost} for two.
           Rating: {rate}/5 with {votes} votes. Location: {location}."
        """
        name = row.get("name", "Unknown")
        cuisines = row.get("cuisines", "")
        rest_type = row.get("rest_type", "")
        dish_liked = row.get("dish_liked", "")
        cost = row.get("approx_cost", "N/A")
        rate = row.get("rate", "N/A")
        votes = row.get("votes", 0)
        location = row.get("location", "")

        parts = [f"{name}."]
        if cuisines:
            parts.append(f"Cuisines: {cuisines}.")
        if rest_type:
            parts.append(f"Type: {rest_type}.")
        if dish_liked:
            parts.append(f"Popular dishes: {dish_liked}.")
        parts.append(f"Cost ₹{cost} for two.")
        if rate and rate != "N/A":
            parts.append(f"Rating: {rate}/5 with {votes} votes.")
        parts.append(f"Location: {location}.")

        return " ".join(parts)

    def _init_chroma(self, df: pd.DataFrame) -> None:
        """Embed all restaurants into ChromaDB with metadata."""
        collection = self._get_chroma_collection()

        # Delete existing data for clean rebuild
        existing = collection.count()
        if existing > 0:
            print(f"[data_layer] Clearing {existing} existing ChromaDB entries...")
            collection.delete(where={"id": {"$gte": 0}})
            # Fallback: get all IDs and delete
            try:
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
            except Exception:
                pass

        # Build documents and metadata
        batch_size = 500
        total = len(df)
        print(f"[data_layer] Embedding {total} restaurants (batch_size={batch_size})...")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_df = df.iloc[start:end]

            ids = [str(start + i) for i in range(len(batch_df))]
            documents = [self._build_document(row) for _, row in batch_df.iterrows()]
            metadatas = []

            for idx, (_, row) in enumerate(batch_df.iterrows()):
                meta = {
                    "id": start + idx,
                    "name": str(row.get("name", "")),
                    "location": str(row.get("location", "")),
                    "approx_cost": int(row["approx_cost"]) if pd.notna(row.get("approx_cost")) else 0,
                    "rate": float(row["rate"]) if pd.notna(row.get("rate")) else 0.0,
                    "budget_tier": str(row.get("budget_tier", "medium")),
                }
                metadatas.append(meta)

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            pct = int((end / total) * 100)
            print(f"  [{pct:3d}%] Embedded {end}/{total} restaurants")

        print(f"[data_layer] ChromaDB: {collection.count()} embeddings stored.")

    # ======================================================================
    #  QUERY INTERFACES
    # ======================================================================

    # ------------------------------------------------------------------
    #  SQLite structured queries
    # ------------------------------------------------------------------
    def query_structured(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query SQLite with structured filters.
        
        Supported filter keys:
          - location (str): exact match (lowercase)
          - budget_max (int): approx_cost <= value
          - min_rating (float): rate >= value
          - budget_tier (str): exact match
          - cuisine (str): FTS match on cuisines column
        
        Returns list of restaurant dicts.
        """
        conn = self._get_conn()
        conditions = []
        params = []

        if "location" in filters and filters["location"]:
            conditions.append("r.location = ?")
            params.append(filters["location"].lower())

        if "budget_max" in filters and filters["budget_max"] is not None:
            conditions.append("r.approx_cost <= ?")
            params.append(int(filters["budget_max"]))

        if "min_rating" in filters and filters["min_rating"]:
            conditions.append("r.rate >= ?")
            params.append(float(filters["min_rating"]))

        if "budget_tier" in filters and filters["budget_tier"]:
            conditions.append("r.budget_tier = ?")
            params.append(filters["budget_tier"].lower())

        # Cuisine: use FTS5 for fuzzy matching
        if "cuisine" in filters and filters["cuisine"]:
            cuisine_term = filters["cuisine"].lower().strip()
            conditions.append(
                "r.id IN (SELECT rowid FROM restaurants_fts WHERE cuisines MATCH ?)"
            )
            params.append(f'"{cuisine_term}"')

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT r.id, r.name, r.location, r.cuisines, r.approx_cost,
                   r.rate, r.votes, r.online_order, r.book_table,
                   r.rest_type, r.dish_liked, r.listed_in_type,
                   r.budget_tier, r.is_new
            FROM restaurants r
            WHERE {where_clause}
            ORDER BY r.rate DESC NULLS LAST, r.votes DESC
        """

        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    #  ChromaDB semantic queries
    # ------------------------------------------------------------------
    def query_semantic(self, text: str,
                       filter_ids: Optional[List[int]] = None,
                       top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Semantic similarity search over restaurant embeddings.
        
        Args:
            text: natural language query (e.g., "cozy Italian place")
            filter_ids: if provided, only search within these restaurant IDs
            top_k: max number of results to return
        
        Returns list of dicts with restaurant metadata + distance score.
        """
        collection = self._get_chroma_collection()
        total_count = collection.count()
        if total_count == 0:
            return []

        # Query broadly — ChromaDB handles the semantic ranking
        n = min(top_k, total_count)
        if n <= 0:
            return []

        # If filtering by IDs, we need to fetch more results and filter in Python
        # because ChromaDB's `ids` parameter is unreliable for large sets.
        fetch_n = n
        if filter_ids is not None and len(filter_ids) > 0:
            # Fetch extra so we have enough after filtering
            fetch_n = min(total_count, max(n * 5, 100))

        results = collection.query(
            query_texts=[text],
            n_results=fetch_n,
        )

        # Unpack ChromaDB results into a flat list of dicts
        output = []
        if results and results["metadatas"] and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                entry = dict(meta)
                if results["distances"] and results["distances"][0]:
                    entry["similarity_score"] = 1 - results["distances"][0][i]  # cosine → similarity
                if results["documents"] and results["documents"][0]:
                    entry["document"] = results["documents"][0][i]
                output.append(entry)

        # Filter to specific IDs if provided (post-query filtering)
        if filter_ids is not None and len(filter_ids) > 0:
            filter_set = set(filter_ids)
            output = [o for o in output if o.get("id") in filter_set]

        return output[:top_k]

    # ------------------------------------------------------------------
    #  Metadata queries
    # ------------------------------------------------------------------
    def get_available_locations(self) -> List[str]:
        """Return sorted list of all unique locations."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT DISTINCT location FROM restaurants ORDER BY location"
        )
        return [row[0] for row in cur.fetchall()]

    def get_available_cuisines(self) -> List[str]:
        """Return sorted list of all unique individual cuisines."""
        conn = self._get_conn()
        cur = conn.execute("SELECT cuisines FROM restaurants")
        cuisine_set = set()
        for row in cur.fetchall():
            if row[0]:
                for c in row[0].split(", "):
                    c = c.strip()
                    if c:
                        cuisine_set.add(c)
        return sorted(cuisine_set)

    def get_restaurant_count(self) -> int:
        """Return total number of restaurants in SQLite."""
        conn = self._get_conn()
        cur = conn.execute("SELECT COUNT(*) FROM restaurants")
        return cur.fetchone()[0]

    def get_restaurant_by_name(
        self, name: str, location: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single restaurant's full details by name (case-insensitive),
        optionally scoped to a location. Used to enrich LLM recommendations
        (which only carry name + explanation) with display metadata.

        Returns the best match (highest votes) or None.
        """
        if not name:
            return None

        conn = self._get_conn()
        conditions = ["LOWER(r.name) = ?"]
        params: List[Any] = [name.strip().lower()]

        if location:
            conditions.append("r.location = ?")
            params.append(location.strip().lower())

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT r.id, r.name, r.location, r.cuisines, r.approx_cost,
                   r.rate, r.votes, r.online_order, r.book_table,
                   r.rest_type, r.dish_liked, r.listed_in_type,
                   r.budget_tier, r.is_new
            FROM restaurants r
            WHERE {where_clause}
            ORDER BY r.votes DESC
            LIMIT 1
        """
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Close SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# CLI entry-point: build both stores from scratch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_and_preprocess

    print("=" * 60)
    print("Phase 1: Building Data Layer")
    print("=" * 60)

    # Step 1: Load and preprocess
    df = load_and_preprocess()

    # Step 2: Initialize both stores
    layer = DataLayer()
    layer.init_db(df)

    # Step 3: Verify
    print(f"\n{'='*60}")
    print("Verification Tests")
    print("=" * 60)

    # Test structured query
    results = layer.query_structured({"location": "btm", "budget_max": 800})
    print(f"\n✓ query_structured(location='btm', budget_max=800) → {len(results)} results")
    if results:
        print(f"  First: {results[0]['name']} | ₹{results[0]['approx_cost']} | ★{results[0]['rate']}")

    # Test semantic query
    sem_results = layer.query_semantic("cozy Italian place", top_k=5)
    print(f"\n✓ query_semantic('cozy Italian place') → {len(sem_results)} results")
    if sem_results:
        for r in sem_results[:3]:
            print(f"  {r['name']} | score={r.get('similarity_score', 'N/A'):.3f}")

    # Test metadata
    locations = layer.get_available_locations()
    cuisines = layer.get_available_cuisines()
    print(f"\n✓ Locations: {len(locations)} unique")
    print(f"✓ Cuisines:  {len(cuisines)} unique")
    print(f"✓ Total restaurants: {layer.get_restaurant_count()}")

    layer.close()
    print(f"\n{'='*60}")
    print("Phase 1 COMPLETE")
    print("=" * 60)
