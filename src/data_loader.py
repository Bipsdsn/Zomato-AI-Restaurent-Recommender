"""
data_loader.py — Phase 1a/1b: Dataset Download & Preprocessing

Downloads the Zomato dataset from HuggingFace, cleans it, and caches
a local CSV at data/zomato.csv for offline use.
"""

import os
import re
import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Budget tier thresholds (from architecture §3.2 + context.md)
# ---------------------------------------------------------------------------
BUDGET_TIERS = {
    "low":    (0, 300),
    "medium": (301, 800),
    "high":   (801, float("inf")),
}


def _classify_budget(cost: int) -> str:
    """Map an approx_cost value to a budget tier label."""
    for tier, (lo, hi) in BUDGET_TIERS.items():
        if lo <= cost <= hi:
            return tier
    return "high"


# ---------------------------------------------------------------------------
# Column parsers
# ---------------------------------------------------------------------------
def _parse_rate(val) -> tuple:
    """
    Parse the rate column.
    Returns (numeric_rate_or_None, is_new_bool).
    
    Examples:
      "4.1/5"  → (4.1, False)
      "NEW"    → (None, True)
      "-"      → (None, False)
      None/NaN → (None, False)
    """
    if pd.isna(val):
        return None, False

    val_str = str(val).strip()

    if val_str.upper() == "NEW":
        return None, True

    if val_str in ("-", ""):
        return None, False

    # Try to extract a numeric value like "4.1/5" or just "4.1"
    match = re.search(r"(\d+\.?\d*)", val_str)
    if match:
        return float(match.group(1)), False

    return None, False


def _parse_cost(val) -> int | None:
    """
    Parse approx_cost: remove commas, convert to int.
    Returns None if unparseable.
    """
    if pd.isna(val):
        return None
    val_str = str(val).strip().replace(",", "")
    try:
        return int(float(val_str))
    except (ValueError, TypeError):
        return None


def _parse_bool(val) -> bool:
    """Convert 'Yes'/'No' strings to bool."""
    if pd.isna(val):
        return False
    return str(val).strip().lower() == "yes"


def _normalize_location(val) -> str | None:
    """Lowercase and strip whitespace from location."""
    if pd.isna(val):
        return None
    return str(val).strip().lower()


def _split_cuisines(val) -> str:
    """
    Normalize cuisines: lowercase, strip each item.
    Keep as a comma-separated string for SQLite FTS compatibility.
    """
    if pd.isna(val):
        return ""
    parts = [c.strip().lower() for c in str(val).split(",") if c.strip()]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------
def download_dataset(cache_path: str = "data/zomato.csv") -> pd.DataFrame:
    """
    Download the Zomato dataset from HuggingFace and cache locally.
    If the cache file already exists, load from disk instead.
    """
    if os.path.exists(cache_path):
        print(f"[data_loader] Loading cached dataset from {cache_path}")
        return pd.read_csv(cache_path)

    print("[data_loader] Downloading dataset from HuggingFace...")
    ds = load_dataset("ManikaSaini/zomato-restaurant-recommendation")

    # The dataset may have a 'train' split — flatten it
    if "train" in ds:
        df = ds["train"].to_pandas()
    else:
        df = pd.DataFrame(ds)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[data_loader] Dataset cached at {cache_path}  ({len(df)} rows)")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the raw Zomato dataframe.
    
    Steps:
      1. Parse rate → numeric + is_new flag
      2. Parse approx_cost → int
      3. Normalize location
      4. Split/normalize cuisines
      5. Convert boolean fields
      6. Derive budget_tier
      7. Drop rows missing critical fields
    """
    df = df.copy()

    # ── 1. Rate parsing ────────────────────────────────────────────────
    rate_col = None
    for col_name in ("rate", "rating"):
        if col_name in df.columns:
            rate_col = col_name
            break

    if rate_col:
        parsed = df[rate_col].apply(_parse_rate)
        df["rate"] = parsed.apply(lambda x: x[0])
        df["is_new"] = parsed.apply(lambda x: x[1])
    else:
        df["rate"] = None
        df["is_new"] = False

    # ── 2. Cost parsing ────────────────────────────────────────────────
    cost_col = None
    for col_name in ("approx_cost(for two people)", "approx_cost", "cost"):
        if col_name in df.columns:
            cost_col = col_name
            break

    if cost_col:
        df["approx_cost"] = df[cost_col].apply(_parse_cost)
    else:
        df["approx_cost"] = None

    # ── 3. Location normalization ──────────────────────────────────────
    loc_col = None
    for col_name in ("listed_in(city)", "location", "city"):
        if col_name in df.columns:
            loc_col = col_name
            break

    if loc_col:
        df["location"] = df[loc_col].apply(_normalize_location)
    else:
        df["location"] = None

    # ── 4. Cuisines ────────────────────────────────────────────────────
    if "cuisines" in df.columns:
        df["cuisines"] = df["cuisines"].apply(_split_cuisines)
    else:
        df["cuisines"] = ""

    # ── 5. Boolean fields ──────────────────────────────────────────────
    if "online_order" in df.columns:
        df["online_order"] = df["online_order"].apply(_parse_bool)
    else:
        df["online_order"] = False

    if "book_table" in df.columns:
        df["book_table"] = df["book_table"].apply(_parse_bool)
    else:
        df["book_table"] = False

    # ── 6. Derive budget_tier ──────────────────────────────────────────
    df["budget_tier"] = df["approx_cost"].apply(
        lambda x: _classify_budget(x) if pd.notna(x) else None
    )

    # ── 7. Normalize rest_type, dish_liked, listed_in_type ─────────────
    for col in ("rest_type", "dish_liked"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    if "listed_in(type)" in df.columns:
        df["listed_in_type"] = df["listed_in(type)"].fillna("")
    elif "listed_in_type" not in df.columns:
        df["listed_in_type"] = ""

    # ── 8. Ensure 'name' and 'votes' exist ─────────────────────────────
    if "name" not in df.columns:
        df["name"] = "Unknown"
    df["name"] = df["name"].fillna("Unknown")

    if "votes" not in df.columns:
        df["votes"] = 0
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

    # ── 9. Drop rows missing critical fields ───────────────────────────
    before = len(df)
    df = df.dropna(subset=["approx_cost"])
    df = df[df["location"].notna() & (df["location"] != "")]
    df = df[df["name"] != "Unknown"]
    after = len(df)
    print(f"[data_loader] Dropped {before - after} rows with missing critical fields. {after} rows remain.")

    # ── 10. Reset index → will become the restaurant ID ────────────────
    df = df.reset_index(drop=True)
    df.index.name = "id"

    # ── 11. Select final columns ───────────────────────────────────────
    final_cols = [
        "name", "location", "cuisines", "approx_cost", "rate", "votes",
        "online_order", "book_table", "rest_type", "dish_liked",
        "listed_in_type", "budget_tier", "is_new",
    ]
    return df[[c for c in final_cols if c in df.columns]]


def load_and_preprocess(cache_path: str = "data/zomato.csv") -> pd.DataFrame:
    """Convenience: download (or load cached) + preprocess in one call."""
    raw = download_dataset(cache_path)
    return preprocess(raw)


# ---------------------------------------------------------------------------
# CLI entry-point for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_preprocess()
    print(f"\n{'='*60}")
    print(f"Dataset ready: {len(df)} restaurants")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample rows:")
    print(df.head(3).to_string())
    print(f"\nLocation counts (top 10):")
    print(df["location"].value_counts().head(10))
    print(f"\nBudget tier distribution:")
    print(df["budget_tier"].value_counts())
    print(f"\nRate stats:")
    print(df["rate"].describe())
