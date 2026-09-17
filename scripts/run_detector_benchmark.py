"""
Multi-Model Independent Detector Benchmark Runner
Runs UniversalFakeDetect (UnivFDAdapter) and Vision Transformer (ViTAdapter)
independently over the benchmark manifest.
Records per-detector outputs and explicitly compares predictions side-by-side
without combining predictions or applying an ensemble/agreement layer.
"""

import os
import sys
import json
import csv
import time
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.univfd_adapter import UnivFDAdapter
from src.vit_adapter import ViTAdapter
from src.detector_schema import PREDICTION_REAL, PREDICTION_FAKE, PREDICTION_ERROR

BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")
IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


def run_benchmark():
    print("=" * 75)
    print("INDEPENDENT DETECTOR BENCHMARK EXECUTION")
    print("Detectors: UnivFDAdapter (CLIP ViT-L/14) & ViTAdapter (ViT-Base/16)")
    print("Policy: Strict independence. No combining. No agreement layer.")
    print("=" * 75 + "\n")

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    print(f"Loaded {len(entries)} benchmark entries.\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Initializing adapters...")
    univfd_adapter = UnivFDAdapter(lazy_load=True)
    vit_adapter = ViTAdapter(lazy_load=True)

    univfd_records: List[Dict[str, Any]] = []
    vit_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []

    for entry in entries:
        entry_id = entry["id"]
        rel_path = entry["filename"]
        category = entry["category"]
        explicit_label = entry.get("label", entry.get("ground_truth", "UNKNOWN"))
        source_claim = entry.get("source_claim", "")
        full_path = os.path.join(IMAGES_DIR, rel_path)
        exists = os.path.exists(full_path)

        print("-" * 70)
        print(f"[{entry_id}] File: {rel_path} | Category: {category} | Label: {explicit_label}")
        print(f"  Physical Path: {full_path} (Exists: {exists})")

        if not exists:
            print("  Status: File missing, skipping.")
            continue

        # 1. Run UnivFD independently
        res_univfd = univfd_adapter.predict(full_path)
        u_dict = res_univfd.to_dict()
        u_record = {
            "id": entry_id,
            "filename": rel_path,
            "category": category,
            "label": explicit_label,
            "source_claim": source_claim,
            **u_dict
        }
        univfd_records.append(u_record)

        # 2. Run ViT independently
        res_vit = vit_adapter.predict(full_path)
        v_dict = res_vit.to_dict()
        v_record = {
            "id": entry_id,
            "filename": rel_path,
            "category": category,
            "label": explicit_label,
            "source_claim": source_claim,
            **v_dict
        }
        vit_records.append(v_record)

        # 3. Compare without combining
        agreement = (res_univfd.prediction == res_vit.prediction)
        agreement_str = "AGREE" if agreement else "DISAGREE"

        comp_record = {
            "id": entry_id,
            "filename": rel_path,
            "category": category,
            "explicit_label": explicit_label,
            # UnivFD outputs
            "univfd_prediction": res_univfd.prediction,
            "univfd_fake_prob": res_univfd.fake_probability,
            "univfd_confidence": res_univfd.confidence,
            "univfd_logit": res_univfd.raw_logit,
            "univfd_error": res_univfd.error,
            # ViT outputs
            "vit_prediction": res_vit.prediction,
            "vit_fake_prob": res_vit.fake_probability,
            "vit_confidence": res_vit.confidence,
            "vit_logit": res_vit.raw_logit,
            "vit_error": res_vit.error,
            # Agreement tracking
            "agreement_status": agreement_str,
        }
        comparison_records.append(comp_record)

        print(f"  UnivFD: pred={res_univfd.prediction:<6} | fake_prob={str(res_univfd.fake_probability):<8} | logit={str(res_univfd.raw_logit):<8}")
        print(f"  ViT:    pred={res_vit.prediction:<6} | fake_prob={str(res_vit.fake_probability):<8} | logit={str(res_vit.raw_logit):<8}")
        print(f"  Status: {agreement_str}")

    # Export UnivFD results
    with open(os.path.join(RESULTS_DIR, "univfd_results.json"), "w", encoding="utf-8") as f:
        json.dump(univfd_records, f, indent=2)

    # Export ViT results
    with open(os.path.join(RESULTS_DIR, "vit_results.json"), "w", encoding="utf-8") as f:
        json.dump(vit_records, f, indent=2)

    # Export Comparison results
    comp_json_path = os.path.join(RESULTS_DIR, "detector_comparison.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    comp_csv_path = os.path.join(RESULTS_DIR, "detector_comparison.csv")
    csv_keys = [
        "id", "filename", "category", "explicit_label",
        "univfd_prediction", "univfd_fake_prob", "univfd_confidence", "univfd_logit", "univfd_error",
        "vit_prediction", "vit_fake_prob", "vit_confidence", "vit_logit", "vit_error",
        "agreement_status"
    ]
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for r in comparison_records:
            writer.writerow(r)

    print("\n" + "=" * 75)
    print("BENCHMARK EXECUTION COMPLETED")
    print(f"  UnivFD JSON:     {os.path.join(RESULTS_DIR, 'univfd_results.json')}")
    print(f"  ViT JSON:        {os.path.join(RESULTS_DIR, 'vit_results.json')}")
    print(f"  Comparison JSON: {comp_json_path}")
    print(f"  Comparison CSV:  {comp_csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
