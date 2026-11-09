"""
Multi-Model Independent Detector Benchmark Runner (Memory-Safe Sequential Execution)
Dynamically discovers and executes all registered detectors from src.detector_registry
over the benchmark manifest sequentially.

Memory Management Strategy:
- Sequential execution: Evaluates one detector completely across all benchmark samples
  before initializing the next, preventing simultaneous memory residency of multiple heavy models.
- Explicit lifecycle release: Invokes unload_model() on adapters, releases references,
  clears registry cache, and runs garbage collection + CUDA cache clearing after each detector.
- Process isolation option: Supports optional subprocess mode (--mode subprocess) for complete
  OS-level virtual memory reclamation between models.
- Fault isolation: An unexpected failure in one detector produces structured error results
  and does not block subsequent detectors.
- Records RSS memory metrics across all lifecycle stages.
"""

import os
import sys
import json
import csv
import time
import gc
import argparse
import subprocess
from typing import Dict, Any, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detector_registry import list_detectors, get_detector, get_detector_info, reset_instances
from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)

BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")
IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size in megabytes."""
    if PSUTIL_AVAILABLE:
        try:
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0
    return 0.0


def run_detector_inprocess(
    det_name: str,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates a single detector across all benchmark entries in the current process,
    explicitly managing memory before, during, and after inference.
    """
    mem_before = get_current_rss_mb()
    load_start = time.perf_counter()
    load_err = None
    adapter = None

    try:
        # Instantiate without singleton caching to facilitate immediate garbage collection
        adapter = get_detector(det_name, lazy_load=False, singleton=False)
        load_time = round(time.perf_counter() - load_start, 2)
    except Exception as exc:
        load_err = str(exc)
        load_time = round(time.perf_counter() - load_start, 2)
        print(f"  [ERROR] Failed to load detector '{det_name}': {load_err}")

    mem_loaded = get_current_rss_mb()
    results: List[Dict[str, Any]] = []

    info = get_detector_info(det_name)
    det_meta = info.get("metadata", {})
    fallback_det_name = det_meta.get("detector_name", det_name)
    fallback_model_name = det_meta.get("model_name", "Unknown")

    for entry in entries:
        entry_id = entry["id"]
        rel_path = entry["filename"]
        category = entry["category"]
        explicit_label = entry.get("label", entry.get("ground_truth", "UNKNOWN"))
        source_claim = entry.get("source_claim", "")
        full_path = os.path.join(IMAGES_DIR, rel_path)

        if load_err is not None or adapter is None:
            err_result = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"Model loading failed: {load_err}",
            )
            res_dict = err_result.to_dict()
        elif not os.path.exists(full_path):
            err_result = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"Benchmark image not found: {full_path}",
            )
            res_dict = err_result.to_dict()
        else:
            try:
                res = adapter.predict(full_path)
                res_dict = res.to_dict()
            except Exception as inf_exc:
                err_result = DetectorResult.create_error(
                    detector_name=fallback_det_name,
                    model_name=fallback_model_name,
                    error_message=f"Inference exception: {str(inf_exc)}",
                )
                res_dict = err_result.to_dict()

        det_record = {
            "id": entry_id,
            "filename": rel_path,
            "category": category,
            "label": explicit_label,
            "source_claim": source_claim,
            **res_dict
        }
        results.append(det_record)

    mem_inferred = get_current_rss_mb()

    # Explicit memory cleanup
    if adapter is not None:
        if hasattr(adapter, "unload_model"):
            try:
                adapter.unload_model()
            except Exception as unl_err:
                print(f"  [WARN] Exception in unload_model for {det_name}: {unl_err}")
        del adapter
        adapter = None

    reset_instances()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_after = get_current_rss_mb()

    return {
        "results": results,
        "metrics": {
            "load_time_sec": load_time,
            "mem_before_mb": mem_before,
            "mem_loaded_mb": mem_loaded,
            "mem_inferred_mb": mem_inferred,
            "mem_after_mb": mem_after,
            "mem_freed_mb": round(mem_inferred - mem_after, 2),
        }
    }


def run_detector_subprocess(
    det_name: str,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates a single detector in an isolated child Python process.
    Guarantees 100% operating-system level reclamation of all memory allocations upon exit.
    """
    mem_before = get_current_rss_mb()
    start_t = time.perf_counter()

    sub_script = f"""
import sys, os, json
sys.path.insert(0, r"{BASE_DIR}")
from src.detector_registry import get_detector, get_detector_info
from src.detector_schema import DetectorResult

det_name = "{det_name}"
metadata_file = r"{METADATA_FILE}"
images_dir = r"{IMAGES_DIR}"

with open(metadata_file, "r", encoding="utf-8") as f:
    entries = json.load(f)["entries"]

info = get_detector_info(det_name)
det_meta = info.get("metadata", {{}})
fallback_det_name = det_meta.get("detector_name", det_name)
fallback_model_name = det_meta.get("model_name", "Unknown")

load_err = None
adapter = None
try:
    adapter = get_detector(det_name, lazy_load=False, singleton=False)
except Exception as e:
    load_err = str(e)

results = []
for entry in entries:
    entry_id = entry["id"]
    rel_path = entry["filename"]
    category = entry["category"]
    explicit_label = entry.get("label", entry.get("ground_truth", "UNKNOWN"))
    source_claim = entry.get("source_claim", "")
    full_path = os.path.join(images_dir, rel_path)

    if load_err is not None or adapter is None:
        err_res = DetectorResult.create_error(
            detector_name=fallback_det_name,
            model_name=fallback_model_name,
            error_message=f"Model loading failed: {{load_err}}"
        )
        res_dict = err_res.to_dict()
    elif not os.path.exists(full_path):
        err_res = DetectorResult.create_error(
            detector_name=fallback_det_name,
            model_name=fallback_model_name,
            error_message=f"Benchmark image not found: {{full_path}}"
        )
        res_dict = err_res.to_dict()
    else:
        try:
            res = adapter.predict(full_path)
            res_dict = res.to_dict()
        except Exception as inf_e:
            err_res = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"Inference exception: {{inf_e}}"
            )
            res_dict = err_res.to_dict()

    record = {{
        "id": entry_id,
        "filename": rel_path,
        "category": category,
        "label": explicit_label,
        "source_claim": source_claim,
        **res_dict
    }}
    results.append(record)

print("---BEGIN_DETECTOR_RESULTS---")
print(json.dumps(results))
"""
    proc = subprocess.run(
        [sys.executable, "-c", sub_script],
        capture_output=True,
        text=True,
    )
    elapsed = round(time.perf_counter() - start_t, 2)
    mem_after = get_current_rss_mb()

    if proc.returncode != 0:
        print(f"  [ERROR] Subprocess for {det_name} exited with code {proc.returncode}")
        print(f"  STDERR: {proc.stderr.strip()[:200]}")
        # Build error records
        info = get_detector_info(det_name)
        det_meta = info.get("metadata", {})
        results = [
            {
                "id": entry["id"],
                "filename": entry["filename"],
                "category": entry["category"],
                "label": entry.get("label", entry.get("ground_truth", "UNKNOWN")),
                "source_claim": entry.get("source_claim", ""),
                **DetectorResult.create_error(
                    detector_name=det_meta.get("detector_name", det_name),
                    model_name=det_meta.get("model_name", "Unknown"),
                    error_message=f"Subprocess crashed with code {proc.returncode}: {proc.stderr.strip()[:100]}",
                ).to_dict()
            }
            for entry in entries
        ]
    else:
        out = proc.stdout
        marker = "---BEGIN_DETECTOR_RESULTS---"
        if marker in out:
            json_text = out.split(marker)[1].strip()
            results = json.loads(json_text)
        else:
            raise RuntimeError(f"Could not parse subprocess output marker for {det_name}")

    return {
        "results": results,
        "metrics": {
            "load_time_sec": elapsed,
            "mem_before_mb": mem_before,
            "mem_loaded_mb": mem_before,
            "mem_inferred_mb": mem_before,
            "mem_after_mb": mem_after,
            "mem_freed_mb": 0.0,
            "mode": "subprocess",
        }
    }


def run_benchmark(mode: str = "inprocess"):
    print("=" * 75)
    print(f"INDEPENDENT MULTI-MODEL DETECTOR BENCHMARK (Mode: {mode.upper()})")
    print("Policy: Sequential evaluation. Memory-safe lifecycle. Strict independence.")
    print("=" * 75 + "\n")

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    print(f"Loaded {len(entries)} benchmark entries from {METADATA_FILE}")

    registered_names = list_detectors()
    print(f"Discovered {len(registered_names)} registered detectors: {registered_names}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_detector_records: Dict[str, List[Dict[str, Any]]] = {}
    memory_metrics: Dict[str, Dict[str, Any]] = {}

    # Sequential execution loop across detectors
    for idx, det_name in enumerate(registered_names, start=1):
        print("-" * 75)
        print(f"[{idx}/{len(registered_names)}] Evaluating Detector: '{det_name}'")

        if mode == "subprocess":
            out_data = run_detector_subprocess(det_name, entries)
        else:
            out_data = run_detector_inprocess(det_name, entries)

        records = out_data["results"]
        metrics = out_data["metrics"]

        per_detector_records[det_name] = records
        memory_metrics[det_name] = metrics

        # Print per-detector summary
        print(f"  Timing: {metrics.get('load_time_sec', 0.0)}s | "
              f"Pre-RSS: {metrics.get('mem_before_mb', 0)} MB | "
              f"Peak-RSS: {metrics.get('mem_inferred_mb', 0)} MB | "
              f"Post-Cleanup RSS: {metrics.get('mem_after_mb', 0)} MB")

        for r in records:
            pred = r.get("prediction", "UNKNOWN")
            fake_p = r.get("fake_probability")
            p_str = f"{fake_p:.4f}" if fake_p is not None else "None"
            err_str = f" | err={r.get('error')[:30]}..." if r.get("error") else ""
            print(f"    [{r['id']:<10}] pred={pred:<6} | fake_prob={p_str:<6}{err_str}")

        print()

    # Assemble long-format and side-by-side comparison tables
    long_format_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []

    for entry in entries:
        entry_id = entry["id"]
        rel_path = entry["filename"]
        category = entry["category"]
        explicit_label = entry.get("label", entry.get("ground_truth", "UNKNOWN"))

        comp_entry: Dict[str, Any] = {
            "id": entry_id,
            "filename": rel_path,
            "category": category,
            "explicit_label": explicit_label,
        }

        predictions_for_agreement = []

        for det_name in registered_names:
            det_recs = per_detector_records.get(det_name, [])
            match = next((r for r in det_recs if r["id"] == entry_id), None)
            if match is None:
                continue

            # Long-format entry
            long_record = {
                "sample_id": entry_id,
                "filename": rel_path,
                "category": category,
                "ground_truth": explicit_label,
                "detector": det_name,
                "detector_name": match.get("detector_name"),
                "model_name": match.get("model_name"),
                "prediction": match.get("prediction"),
                "fake_probability": match.get("fake_probability"),
                "real_probability": match.get("real_probability"),
                "raw_logit": match.get("raw_logit"),
                "confidence": match.get("confidence"),
                "threshold": match.get("threshold"),
                "device": match.get("device"),
                "latency_ms": match.get("inference_time_ms"),
                "error": match.get("error"),
                "warning": match.get("warning"),
            }
            long_format_records.append(long_record)

            # Side-by-side comparison fields
            comp_entry[f"{det_name}_prediction"] = match.get("prediction")
            comp_entry[f"{det_name}_fake_prob"] = match.get("fake_probability")
            comp_entry[f"{det_name}_confidence"] = match.get("confidence")
            comp_entry[f"{det_name}_logit"] = match.get("raw_logit")
            comp_entry[f"{det_name}_error"] = match.get("error")

            predictions_for_agreement.append(match.get("prediction"))

        # Observational agreement tracking only - NEVER combined into prediction
        all_same = len(set(predictions_for_agreement)) == 1
        comp_entry["agreement_status"] = "AGREE" if all_same else "DISAGREE"
        comparison_records.append(comp_entry)

    # 1. Export per-detector results (e.g. univfd_results.json, vit_results.json)
    for det_name, records in per_detector_records.items():
        out_path = os.path.join(RESULTS_DIR, f"{det_name}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote {det_name} results: {out_path}")

    # 2. Export long-format independent predictions table (JSON & CSV)
    long_json_path = os.path.join(RESULTS_DIR, "all_detector_predictions.json")
    with open(long_json_path, "w", encoding="utf-8") as f:
        json.dump(long_format_records, f, indent=2)

    long_csv_path = os.path.join(RESULTS_DIR, "all_detector_predictions.csv")
    long_keys = [
        "sample_id", "filename", "category", "ground_truth", "detector",
        "detector_name", "model_name", "prediction", "fake_probability",
        "real_probability", "raw_logit", "confidence", "threshold",
        "device", "latency_ms", "error", "warning"
    ]
    with open(long_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=long_keys, extrasaction="ignore")
        writer.writeheader()
        for r in long_format_records:
            writer.writerow(r)

    # 3. Export side-by-side comparison table (JSON & CSV)
    comp_json_path = os.path.join(RESULTS_DIR, "detector_comparison.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    comp_csv_path = os.path.join(RESULTS_DIR, "detector_comparison.csv")
    comp_keys = ["id", "filename", "category", "explicit_label"]
    for det_name in registered_names:
        comp_keys.extend([
            f"{det_name}_prediction",
            f"{det_name}_fake_prob",
            f"{det_name}_confidence",
            f"{det_name}_logit",
            f"{det_name}_error",
        ])
    comp_keys.append("agreement_status")

    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comp_keys, extrasaction="ignore")
        writer.writeheader()
        for r in comparison_records:
            writer.writerow(r)

    print("\n" + "=" * 75)
    print("BENCHMARK EXECUTION COMPLETED (MEMORY-SAFE)")
    print(f"  Long-format JSON: {long_json_path}")
    print(f"  Long-format CSV:  {long_csv_path}")
    print(f"  Comparison JSON:  {comp_json_path}")
    print(f"  Comparison CSV:   {comp_csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Memory-Safe Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["inprocess", "subprocess"],
        default="inprocess",
        help="Execution mode: 'inprocess' (sequential with explicit lifecycle release) or 'subprocess' (separate child process per detector)."
    )
    args = parser.parse_args()
    run_benchmark(mode=args.mode)
