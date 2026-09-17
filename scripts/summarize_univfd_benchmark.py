"""
UnivFD Benchmark Summarizer
Generates an operational diagnostic summary from benchmark results.
DISCLAIMER: This script does NOT compute accuracy, precision, recall, or F1 scores
because the benchmark is a diagnostic operational harness and not a certified statistical dataset.
"""

import os
import sys
import json
from collections import Counter
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "benchmark", "results", "univfd_results.json")


def summarize_benchmark():
    print("=" * 75)
    print("UNIVFD BENCHMARK OPERATIONAL SUMMARY")
    print("Harness: Diagnostic Operational Analysis")
    print("=" * 75 + "\n")

    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        print("Please run scripts/run_univfd_benchmark.py first.")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    total_inputs = len(records)
    available_inputs = [r for r in records if r.get("available", False)]
    missing_inputs = [r for r in records if not r.get("available", False)]

    successful_inferences = [
        r for r in available_inputs if r.get("prediction") in {"REAL", "FAKE", "UNCERTAIN"}
    ]
    rejected_or_corrupted = [
        r for r in available_inputs if r.get("prediction") == "ERROR"
    ]

    predictions_counter = Counter(r.get("prediction") for r in records)
    categories_counter = Counter(r.get("category") for r in records)
    unknown_gt_entries = [r for r in records if r.get("ground_truth") == "UNKNOWN"]

    # Calculate average inference time over successful inferences
    times = [
        r["inference_time_ms"] for r in successful_inferences
        if r.get("inference_time_ms") is not None
    ]
    avg_inference_time = sum(times) / len(times) if times else 0.0

    print("--- 1. Input Inventory ---")
    print(f"Total entries registered:    {total_inputs}")
    print(f"Physically available inputs: {len(available_inputs)}")
    print(f"Missing inputs on disk:      {len(missing_inputs)}")
    print(f"UNKNOWN ground-truth labels: {len(unknown_gt_entries)}")
    print()

    print("--- 2. Inference & Error Breakdown ---")
    print(f"Successful model passes:     {len(successful_inferences)}")
    print(f"Rejected / corrupted inputs: {len(rejected_or_corrupted)}")
    print(f"Average inference latency:   {avg_inference_time:.2f} ms")
    print()

    print("--- 3. Prediction Counts ---")
    for pred, count in predictions_counter.items():
        print(f"  {pred:<25}: {count}")
    print()

    print("--- 4. Counts by Category ---")
    for cat, count in categories_counter.items():
        print(f"  {cat:<25}: {count}")
    print()

    print("--- 5. Missing / Skipped Inputs ---")
    if missing_inputs:
        for r in missing_inputs:
            print(f"  - [{r['id']}] {r['filename']} ({r['error']})")
    else:
        print("  None. All registered inputs were present.")
    print()

    print("--- 6. Captured Errors & Warnings ---")
    warnings_found = set()
    errors_found = []
    for r in records:
        if r.get("warning"):
            warnings_found.add(r["warning"])
        if r.get("error"):
            errors_found.append((r["id"], r["error"]))

    print(f"Unique Warnings ({len(warnings_found)}):")
    for w in warnings_found:
        print(f"  - \"{w}\"")
    print(f"Recorded Errors ({len(errors_found)}):")
    for eid, err in errors_found:
        print(f"  - [{eid}]: {err}")
    print()

    print("=" * 75)
    print("IMPORTANT DISCLAIMER ON EVALUATION METRICS:")
    print("In accordance with scientific benchmarking standards, no accuracy, precision,")
    print("recall, or F1 scores are reported. The benchmark inputs serve as operational")
    print("diagnostics, not a statistically certified ground-truth evaluation dataset.")
    print("=" * 75)


if __name__ == "__main__":
    summarize_benchmark()
