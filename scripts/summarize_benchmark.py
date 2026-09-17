"""
Benchmark Summary Generator
Parses detector_comparison.json and prints/saves an objective, comparative summary.
Reports agreements and disagreements without combining predictions.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "benchmark", "results", "detector_comparison.json")


def summarize():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    agreements = [r for r in records if r["agreement_status"] == "AGREE"]
    disagreements = [r for r in records if r["agreement_status"] == "DISAGREE"]

    print("=" * 90)
    print("MULTI-MODEL DETECTOR BENCHMARK SUMMARY")
    print("Detectors evaluated independently: UniversalFakeDetect vs. ViT (dima806)")
    print("=" * 90)
    print(f"{'ID':<10} | {'Category':<14} | {'Label':<12} | {'UnivFD (Conf)':<18} | {'ViT (Conf)':<18} | {'Status'}")
    print("-" * 90)

    for r in records:
        entry_id = r["id"]
        cat = r["category"]
        lbl = r["explicit_label"]

        u_pred = r["univfd_prediction"]
        u_conf = f"{r['univfd_confidence']*100:.1f}%" if r["univfd_confidence"] is not None else "N/A"
        u_str = f"{u_pred} ({u_conf})"

        v_pred = r["vit_prediction"]
        v_conf = f"{r['vit_confidence']*100:.1f}%" if r["vit_confidence"] is not None else "N/A"
        v_str = f"{v_pred} ({v_conf})"

        status = r["agreement_status"]
        print(f"{entry_id:<10} | {cat:<14} | {lbl:<12} | {u_str:<18} | {v_str:<18} | {status}")

    print("=" * 90)
    print(f"TOTAL SAMPLES:          {total}")
    print(f"IDENTICAL PREDICTIONS:  {len(agreements)} / {total} ({len(agreements)/total*100:.1f}%)")
    print(f"DISAGREEMENTS:          {len(disagreements)} / {total} ({len(disagreements)/total*100:.1f}%)")
    print("=" * 90)

    if disagreements:
        print("\nDETAILED DISAGREEMENTS (Predictions NOT combined):")
        for d in disagreements:
            print(f"  Sample ID: {d['id']} ({d['filename']})")
            print(f"    Explicit Label:    {d['explicit_label']}")
            print(f"    UnivFD Output:     {d['univfd_prediction']} (fake_prob={d['univfd_fake_prob']}, logit={d['univfd_logit']})")
            print(f"    ViT Output:        {d['vit_prediction']} (fake_prob={d['vit_fake_prob']}, logit={d['vit_logit']})")
            print(f"    Note: Each detector operated in total isolation; predictions are not combined.\n")
    else:
        print("\nNo disagreements observed across tested samples.")
        print("Note: Both detectors concurred on all samples tested (including error rejections and predictions).")


if __name__ == "__main__":
    summarize()
