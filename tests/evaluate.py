"""
evaluate.py -- Phase 10c: Evaluation harness (manual, uses the REAL LLM).

Runs a batch of representative queries through the full orchestrator and
reports quality/performance metrics:
  - end-to-end latency P50 / P95 / avg
  - hallucination rate (recommended names must exist in the dataset)
  - fallback rate (share of requests served by rule-based ranking)
  - cache effectiveness (hit rate over a repeated batch)

Run from the project root:
    python -m tests.evaluate
"""

import statistics
import time

from src.app import create_orchestrator


QUERIES = [
    "cheap Italian in Koramangala",
    "best rated Chinese under 500",
    "upscale dining for date night in Indiranagar",
    "family-friendly place in BTM, medium budget",
    "something quick near Whitefield",
    "rooftop dining in Indiranagar",
    "pocket-friendly North Indian in BTM rated 4+",
    "fine dining for an anniversary in Koramangala",
    "vegetarian friendly cafe in Jayanagar",
    "late night food near MG Road",
    "biryani under 400 in BTM",
    "premium sushi in Indiranagar",
]


def _percentile(values, pct):
    if not values:
        return 0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


def main():
    print("=" * 70)
    print("  Phase 10 — Evaluation Harness")
    print("=" * 70)

    orch = create_orchestrator()

    latencies = []
    fallback_count = 0
    hallucinations = 0
    total_recs = 0

    for i, q in enumerate(QUERIES, 1):
        start = time.time()
        resp = orch.process_request(q)
        elapsed_ms = int((time.time() - start) * 1000)
        latencies.append(elapsed_ms)

        src = resp.recommendations[0].source if resp.recommendations else "none"
        if src == "fallback":
            fallback_count += 1

        # Validate every recommended name against the dataset.
        for r in resp.recommendations:
            total_recs += 1
            if orch.get_restaurant_details(r.name) is None:
                hallucinations += 1

        names = ", ".join(r.name for r in resp.recommendations[:3])
        print(f"[{i:2d}/{len(QUERIES)}] {elapsed_ms:5d}ms  src={src:9s}  "
              f"{len(resp.recommendations)} recs  | {names[:60]}")

    # --- Cache effectiveness: replay the batch, expect hits ---
    replay_start = time.time()
    for q in QUERIES:
        orch.process_request(q)
    replay_ms = int((time.time() - replay_start) * 1000)

    n = len(QUERIES)
    print("\n" + "-" * 70)
    print("  METRICS")
    print("-" * 70)
    print(f"  Queries run            : {n}")
    print(f"  Avg latency            : {int(statistics.mean(latencies))} ms")
    print(f"  P50 latency            : {_percentile(latencies, 50)} ms")
    print(f"  P95 latency            : {_percentile(latencies, 95)} ms")
    print(f"  Fallback rate          : {fallback_count / n * 100:.1f}%")
    hall_rate = (hallucinations / total_recs * 100) if total_recs else 0.0
    print(f"  Hallucination rate     : {hall_rate:.1f}%  ({hallucinations}/{total_recs})")
    print(f"  Cached replay (total)  : {replay_ms} ms for {n} queries "
          f"(~{replay_ms // max(n,1)} ms/query)")
    print("=" * 70)

    # Light assertions for a quick pass/fail signal.
    if hall_rate == 0.0:
        print("  PASS: zero hallucinations")
    else:
        print("  WARN: hallucinations detected")


if __name__ == "__main__":
    main()
