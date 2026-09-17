"""
UnivFD Benchmark Runner
Executes the isolated UniversalFakeDetect adapter over the benchmark manifest.
Records comprehensive results, handles missing and corrupted inputs gracefully,
and exports structured JSON and CSV reports.
"""

import os
import sys
import json
import csv
import time
from typing import Dict, Any, List

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.univfd_adapter import UnivFDAdapter
from src.detector_schema import PREDICTION_ERROR

BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")
IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


def run_benchmark():
    print("=" * 75)
    print("RUNNING UNIVFD BENCHMARK")
    print("Harness: Isolated UnivFDAdapter | Target Space: Multimodel_setup")
    print("=" * 75 + "\n")

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata manifest not found at: {METADATA_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    print(f"Loaded {len(entries)} benchmark entries from {METADATA_FILE}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    adapter = UnivFDAdapter(lazy_load=True)

    results_data: List[Dict[str, Any]] = []

    for entry in entries:
        entry_id = entry.get("id", "unknown")
        rel_filename = entry.get("filename", "")
        category = entry.get("category", "unknown")
        ground_truth = entry.get("ground_truth", "UNKNOWN")
        source_claim = entry.get("source_claim", "")

        full_path = os.path.join(IMAGES_DIR, rel_filename)
        file_available = os.path.exists(full_path)

        print(f"[{entry_id}] Category: {category} | File: {rel_filename}")
        print(f"  Available on disk: {file_available}")
        print(f"  Ground Truth:      {ground_truth}")

        if not file_available:
            print("  Status:            SKIPPED (File not present in benchmark/images/)")
            record = {
                "id": entry_id,
                "filename": rel_filename,
                "category": category,
                "ground_truth": ground_truth,
                "source_claim": source_claim,
                "available": False,
                "detector_name": adapter.DETECTOR_NAME,
                "model_name": adapter.MODEL_NAME,
                "prediction": "SKIPPED_MISSING_FILE",
                "fake_probability": None,
                "real_probability": None,
                "raw_logit": None,
                "confidence": None,
                "threshold": adapter.DECISION_THRESHOLD,
                "device": None,
                "preprocessing": adapter.PREPROCESSING_NAME,
                "inference_time_ms": None,
                "error": "File not physically present in repository benchmark/images/ directory.",
                "warning": adapter.LIMITATION_WARNING,
            }
            results_data.append(record)
            print()
            continue

        # Execute UnivFD inference
        try:
            result = adapter.predict(full_path)
            res_dict = result.to_dict()

            record = {
                "id": entry_id,
                "filename": rel_filename,
                "category": category,
                "ground_truth": ground_truth,
                "source_claim": source_claim,
                "available": True,
                **res_dict
            }
            print(f"  Prediction:        {result.prediction}")
            print(f"  Raw Logit:         {result.raw_logit}")
            print(f"  Fake Probability:  {result.fake_probability}")
            print(f"  Inference Latency: {result.inference_time_ms} ms")
            if result.error:
                print(f"  Error Caught:      {result.error}")

        except Exception as exec_err:
            print(f"  Exception during execution: {exec_err}")
            record = {
                "id": entry_id,
                "filename": rel_filename,
                "category": category,
                "ground_truth": ground_truth,
                "source_claim": source_claim,
                "available": True,
                "detector_name": adapter.DETECTOR_NAME,
                "model_name": adapter.MODEL_NAME,
                "prediction": PREDICTION_ERROR,
                "fake_probability": None,
                "real_probability": None,
                "raw_logit": None,
                "confidence": None,
                "threshold": adapter.DECISION_THRESHOLD,
                "device": None,
                "preprocessing": adapter.PREPROCESSING_NAME,
                "inference_time_ms": None,
                "error": f"Unhandled harness exception: {str(exec_err)}",
                "warning": adapter.LIMITATION_WARNING,
            }

        results_data.append(record)
        print()

    # Save JSON report
    out_json = os.path.join(RESULTS_DIR, "univfd_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Wrote benchmark results JSON: {out_json}")

    # Save CSV report
    out_csv = os.path.join(RESULTS_DIR, "univfd_results.csv")
    csv_columns = [
        "id", "filename", "category", "ground_truth", "available",
        "detector_name", "model_name", "prediction", "fake_probability",
        "real_probability", "raw_logit", "confidence", "threshold",
        "device", "preprocessing", "inference_time_ms", "error", "warning"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in results_data:
            writer.writerow(r)
    print(f"Wrote benchmark results CSV:  {out_csv}")

    return results_data


if __name__ == "__main__":
    run_benchmark()
