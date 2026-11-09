"""
Benchmark Summary Generator
Dynamically parses detector_comparison.json and prints an objective, comparative summary
for all evaluated detectors without combining predictions.
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

    if not records:
        print("No benchmark records found.")
        return

    total = len(records)
    first_record = records[0]

    # Discover detectors dynamically from record keys (ending with '_prediction')
    detector_keys = [
        k[:-11] for k in first_record.keys()
        if k.endswith("_prediction")
    ]

    agreements = [r for r in records if r.get("agreement_status") == "AGREE"]
    disagreements = [r for r in records if r.get("agreement_status") == "DISAGREE"]

    col_width = 18
    header_detectors = " | ".join(f"{d.upper() + ' (Conf)':<{col_width}}" for d in detector_keys)
    table_header = f"{'ID':<10} | {'Category':<14} | {'Label':<12} | {header_detectors} | {'Status'}"
    separator = "-" * len(table_header)

    print("=" * len(table_header))
    print(f"MULTI-MODEL DETECTOR BENCHMARK SUMMARY ({len(detector_keys)} Detectors Registered)")
    print(f"Detectors evaluated independently: {', '.join(detector_keys)}")
    print("Policy: Independent evaluation. Predictions NOT combined into ensemble.")
    print("=" * len(table_header))
    print(table_header)
    print(separator)

    for r in records:
        entry_id = r["id"]
        cat = r["category"]
        lbl = r["explicit_label"]

        det_cols = []
        for d in detector_keys:
            pred = r.get(f"{d}_prediction", "N/A")
            conf_val = r.get(f"{d}_confidence")
            conf_str = f"{conf_val * 100:.1f}%" if conf_val is not None else "N/A"
            det_cols.append(f"{pred} ({conf_str}):<{col_width}")
            # Format nicely
            det_cols[-1] = f"{pred} ({conf_str})"[:col_width].ljust(col_width)

        det_str = " | ".join(det_cols)
        status = r.get("agreement_status", "UNKNOWN")
        print(f"{entry_id:<10} | {cat:<14} | {lbl:<12} | {det_str} | {status}")

    print("=" * len(table_header))
    print(f"TOTAL SAMPLES:          {total}")
    print(f"IDENTICAL PREDICTIONS:  {len(agreements)} / {total} ({len(agreements)/total*100:.1f}%)")
    print(f"DISAGREEMENTS:          {len(disagreements)} / {total} ({len(disagreements)/total*100:.1f}%)")
    print("=" * len(table_header))

    if disagreements:
        print("\nDETAILED DISAGREEMENTS (Predictions NOT combined):")
        for d in disagreements:
            print(f"  Sample ID: {d['id']} ({d['filename']})")
            print(f"    Explicit Label:    {d['explicit_label']}")
            for det in detector_keys:
                print(f"    {det.upper()} Output:     {d.get(f'{det}_prediction')} (fake_prob={d.get(f'{det}_fake_prob')}, logit={d.get(f'{det}_logit')})")
            print("    Note: Each detector operated in total isolation; predictions are not combined.\n")
    else:
        print("\nObservational Concurrence: Both detectors produced identical discrete predictions across all 5 samples.")
        print("Note: Discrete agreement at 0.50 threshold does NOT indicate identical feature representations or accuracy.")


if __name__ == "__main__":
    summarize()
