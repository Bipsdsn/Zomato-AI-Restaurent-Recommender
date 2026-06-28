"""
app.py -- Phase 6b: Application Entry Point

Initializes all components, wires the orchestrator, and provides
a simple CLI loop for interactive testing.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.data_layer import DataLayer
from src.input_parser import InputParser
from src.retrieval_engine import RetrievalEngine
from src.prompt_builder import PromptBuilder
from src.recommender import Recommender, GrokAdapter
from src.orchestrator import RecommendationOrchestrator
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def ensure_data_ready(db_path: str, chroma_path: str) -> None:
    """
    First-run bootstrap: if the SQLite DB or ChromaDB store is missing, build
    them from the dataset (downloading/caching the CSV if needed). This lets the
    project run on a fresh machine with no manual data step.
    """
    needs_build = not os.path.exists(db_path) or not os.path.isdir(chroma_path)
    if not needs_build:
        return

    logger.info("Data stores missing — building them (one-time setup)...")
    from src.data_loader import load_and_preprocess
    csv_path = os.getenv("DATA_PATH", "data/zomato.csv")
    df = load_and_preprocess(csv_path)
    DataLayer(db_path=db_path, chroma_path=chroma_path).init_db(df)
    logger.info("Data stores ready.")


def create_orchestrator() -> RecommendationOrchestrator:
    """
    Initialize all components and wire the orchestrator.
    Uses paths from .env or defaults. Builds data stores on first run.
    """
    configure_logging()
    db_path = os.getenv("DB_PATH", "data/restaurants.db")
    chroma_path = os.getenv("VECTOR_DB_PATH", "data/chroma_store")

    ensure_data_ready(db_path, chroma_path)

    # Initialize components
    data_layer = DataLayer(db_path=db_path, chroma_path=chroma_path)
    input_parser = InputParser(data_layer)
    retrieval_engine = RetrievalEngine(data_layer)
    prompt_builder = PromptBuilder()
    llm_adapter = GrokAdapter()
    recommender = Recommender(llm_adapter=llm_adapter, prompt_builder=prompt_builder)

    # Wire orchestrator
    orchestrator = RecommendationOrchestrator(
        data_layer=data_layer,
        input_parser=input_parser,
        retrieval_engine=retrieval_engine,
        prompt_builder=prompt_builder,
        recommender=recommender,
    )

    return orchestrator


def display_response(response):
    """Pretty-print a RecommendationResponse to the console."""
    if not response.recommendations:
        print("\n  No recommendations found. Try a different query.\n")
        return

    if response.filters_relaxed:
        relaxed = ", ".join(response.filters_relaxed)
        print(f"\n  Note: Some filters were relaxed to find results: {relaxed}")

    print(f"\n  Found {len(response.recommendations)} recommendations "
          f"({response.processing_time_ms}ms):\n")

    for rec in response.recommendations:
        name = rec.name.encode('ascii', 'replace').decode('ascii')
        source_tag = f" [{rec.source}]" if rec.source != "llm" else ""
        print(f"  #{rec.rank} {name}{source_tag}")
        print(f"     {rec.explanation}")
        print()


def run_cli():
    """Interactive CLI loop for testing the recommendation system."""
    print("=" * 60)
    print("  Zomato Restaurant Recommendation System")
    print("  Type your preferences or 'quit' to exit")
    print("=" * 60)

    orchestrator = create_orchestrator()

    count = orchestrator.get_restaurant_count()
    locs = len(orchestrator.get_locations())
    print(f"\n  Loaded {count} restaurants across {locs} locations.\n")

    session_id = None
    while True:
        try:
            user_input = input("  What are you looking for? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
            
        if user_input.lower() == "clear":
            session_id = None
            print("  [Session cleared]")
            continue

        response = orchestrator.process_request(user_input, session_id)
        session_id = response.session_id
        display_response(response)


def run_smoke_test():
    """Run the 5 smoke test queries from the acceptance criteria."""
    print("=" * 60)
    print("  Phase 6c: End-to-End Smoke Test")
    print("=" * 60)

    orchestrator = create_orchestrator()

    test_queries = [
        "cheap Italian in Koramangala",
        "best rated Chinese under 500",
        "upscale dining for date night in Indiranagar",
        "family-friendly place in BTM, medium budget",
        "something quick near Whitefield",
    ]

    all_passed = True
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*60}")
        print(f"  [{i}/5] \"{query}\"")
        print(f"{'-'*60}")

        response = orchestrator.process_request(query)

        if response.recommendations:
            print(f"  OK: {len(response.recommendations)} recs, "
                  f"{response.processing_time_ms}ms, "
                  f"source={response.recommendations[0].source}")
            for r in response.recommendations[:2]:
                name = r.name.encode('ascii', 'replace').decode('ascii')
                print(f"    #{r.rank} {name}: {r.explanation[:80]}...")
        else:
            print(f"  FAIL: No recommendations returned")
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL 5 SMOKE TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        run_smoke_test()
    else:
        run_cli()
