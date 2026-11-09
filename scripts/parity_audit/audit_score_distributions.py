"""
Audit Phase 6 — Score Distribution Analysis
Analyzes the distribution of fake probabilities across REAL, Midjourney, and ADM samples
for all three detectors and fusion strategies.
Calculates min, max, mean, median, std, p25, p75, p90, p95.
"""

import os
import sys
import csv
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results")

VIT_CSV = os.path.join(RESULTS_DIR, "phase7_vit_predictions.csv")
UNIVFD_CSV = os.path.join(RESULTS_DIR, "phase7_univfd_predictions.csv")
CONVNEXT_CSV = os.path.join(RESULTS_DIR, "phase7_convnext_predictions.csv")
COMBINED_CSV = os.path.join(RESULTS_DIR, "phase7_combined_predictions.csv")
OUT_JSON = os.path.join(RESULTS_DIR, "parity_audit_score_distributions.json")

def compute_distribution_stats(scores: list) -> dict:
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "count": len(arr),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
    }

def main():
    print("=" * 80)
    print("AUDIT PHASE 6: SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # 1. Load Manifest to group sample IDs by generator
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    real_ids = {r["sample_id"] for r in manifest_rows if r["label"] == "REAL"}
    mj_ids = {r["sample_id"] for r in manifest_rows if r["generator_family"] == "midjourney"}
    adm_ids = {r["sample_id"] for r in manifest_rows if r["generator_family"] == "adm"}

    print(f"Manifest groups: REAL={len(real_ids)}, Midjourney={len(mj_ids)}, ADM={len(adm_ids)}")

    models = [
        ("ViT (dima806)", VIT_CSV, "fake_probability"),
        ("UnivFD (CLIP ViT-L/14)", UNIVFD_CSV, "fake_probability"),
        ("ConvNeXt (DRCT)", CONVNEXT_CSV, "fake_probability"),
    ]

    all_distributions = {}

    for model_name, csv_path, score_col in models:
        print(f"\n--- Model: {model_name} ---")
        scores_by_id = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores_by_id[row["sample_id"]] = float(row[score_col])

        real_scores = [scores_by_id[sid] for sid in real_ids if sid in scores_by_id]
        mj_scores = [scores_by_id[sid] for sid in mj_ids if sid in scores_by_id]
        adm_scores = [scores_by_id[sid] for sid in adm_ids if sid in scores_by_id]
        fake_scores = mj_scores + adm_scores

        dist_real = compute_distribution_stats(real_scores)
        dist_mj = compute_distribution_stats(mj_scores)
        dist_adm = compute_distribution_stats(adm_scores)
        dist_fake = compute_distribution_stats(fake_scores)

        all_distributions[model_name] = {
            "real": dist_real,
            "midjourney": dist_mj,
            "adm": dist_adm,
            "fake_all": dist_fake,
        }

        print(f"{'Partition':<16} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'P25':<8} {'P75':<8} {'P90':<8} {'P95':<8}")
        print("-" * 88)
        for part_name, dist in [("REAL (N=100)", dist_real), ("MJ (N=50)", dist_mj), ("ADM (N=50)", dist_adm), ("ALL FAKE (N=100)", dist_fake)]:
            print(f"{part_name:<16} {dist['min']:<8.4f} {dist['max']:<8.4f} {dist['mean']:<8.4f} "
                  f"{dist['median']:<8.4f} {dist['std']:<8.4f} {dist['p25']:<8.4f} "
                  f"{dist['p75']:<8.4f} {dist['p90']:<8.4f} {dist['p95']:<8.4f}")

    # Also check ensemble strategies from combined CSV
    print("\n--- Decision-Fusion Strategies ---")
    fusion_strategies = [
        ("Majority Vote Fake Count / Probability", COMBINED_CSV, "majority_vote_fake_probability"),
        ("Mean Fake Probability", COMBINED_CSV, "mean_fake_probability"),
    ]
    for strat_name, csv_path, col in fusion_strategies:
        scores_by_id = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if col in row and row[col]:
                    scores_by_id[row["sample_id"]] = float(row[col])

        if scores_by_id:
            real_scores = [scores_by_id[sid] for sid in real_ids if sid in scores_by_id]
            mj_scores = [scores_by_id[sid] for sid in mj_ids if sid in scores_by_id]
            adm_scores = [scores_by_id[sid] for sid in adm_ids if sid in scores_by_id]
            dist_real = compute_distribution_stats(real_scores)
            dist_mj = compute_distribution_stats(mj_scores)
            dist_adm = compute_distribution_stats(adm_scores)
            all_distributions[strat_name] = {
                "real": dist_real,
                "midjourney": dist_mj,
                "adm": dist_adm,
            }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_distributions, f, indent=2)
    print(f"\nSaved distribution statistics to: {OUT_JSON}")

if __name__ == "__main__":
    main()
