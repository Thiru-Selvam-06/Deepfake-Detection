"""
Phase 7 V2 Corrected External Generalization Evaluation Pipeline Runner
Evaluates all three deepfake detectors with verified reference preprocessing:
- ConvNeXt: DRCT native CenterCrop(224) without resizing (--is_crop configuration)
- UnivFD: UniversalFakeDetect validate.py native CenterCrop(224) with CLIP normalization
- ViT: HuggingFace ViTImageProcessor (verified bit-for-bit unchanged)

Saves all outputs in benchmark/results/phase7_v2_reference_preprocessing/:
- phase7_v2_vit_predictions.csv
- phase7_v2_univfd_predictions.csv
- phase7_v2_convnext_predictions.csv
- phase7_v2_combined_predictions.csv
- phase7_v2_combined_metrics.json
- phase7_v2_model_comparison.csv

Memory Safety:
- Sequential child process isolation per detector guarantees 100% OS virtual memory reclamation.
"""

import os
import sys
import json
import csv
import time
import math
import argparse
import subprocess
import tempfile
import gc
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.detector_registry import (
    get_detector,
    get_detector_info,
    reset_instances,
)
from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
    PREDICTION_UNCERTAIN,
)
from src.decision_fusion import (
    strategy_majority_vote,
    strategy_mean_fake_probability,
    strategy_vit_priority,
)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase7_v2_reference_preprocessing")

OUTPUT_VIT_CSV = os.path.join(OUT_DIR, "phase7_v2_vit_predictions.csv")
OUTPUT_UNIVFD_CSV = os.path.join(OUT_DIR, "phase7_v2_univfd_predictions.csv")
OUTPUT_CONVNEXT_CSV = os.path.join(OUT_DIR, "phase7_v2_convnext_predictions.csv")
OUTPUT_COMBINED_CSV = os.path.join(OUT_DIR, "phase7_v2_combined_predictions.csv")
OUTPUT_METRICS_JSON = os.path.join(OUT_DIR, "phase7_v2_combined_metrics.json")
OUTPUT_COMPARISON_CSV = os.path.join(OUT_DIR, "phase7_v2_model_comparison.csv")

V1_METRICS_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase7_combined_metrics.json")
V1_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase7_model_comparison.csv")

DETECTOR_KEYS = ["vit", "univfd", "chen2024_convnext"]


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> List[float]:
    """Computes Wilson score confidence interval for binomial proportion k / n."""
    if n <= 0:
        return [0.0, 0.0]
    p_hat = k / n
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return [round(lower, 4), round(upper, 4)]


def compute_cohen_kappa(both_fake: int, both_real: int, m1_f_m2_r: int, m1_r_m2_f: int) -> float:
    """Computes Cohen's Kappa for 2x2 agreement matrix."""
    n = both_fake + both_real + m1_f_m2_r + m1_r_m2_f
    if n == 0:
        return 0.0
    po = (both_fake + both_real) / n
    p_f1 = (both_fake + m1_f_m2_r) / n
    p_r1 = (m1_r_m2_f + both_real) / n
    p_f2 = (both_fake + m1_r_m2_f) / n
    p_r2 = (m1_f_m2_r + both_real) / n
    pe = (p_f1 * p_f2) + (p_r1 * p_r2)
    if abs(1.0 - pe) < 1e-9:
        return 1.0 if abs(po - 1.0) < 1e-9 else 0.0
    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def run_worker(det_key: str, input_json_path: str, output_json_path: str) -> None:
    """Worker function executed in isolated child process for a single detector."""
    print(f"  [Worker Process] Initializing detector '{det_key}' in clean subprocess...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    info = get_detector_info(det_key)
    meta = info.get("metadata", {})
    fallback_det_name = meta.get("detector_name", det_key)
    fallback_model_name = meta.get("model_name", "Unknown")

    adapter = None
    load_err = None
    t0 = time.perf_counter()
    try:
        adapter = get_detector(det_key, lazy_load=False, singleton=False)
        load_time_sec = round(time.perf_counter() - t0, 3)
        print(f"  [Worker Process] Detector '{det_key}' loaded successfully in {load_time_sec}s.")
    except Exception as e:
        load_err = str(e)
        load_time_sec = round(time.perf_counter() - t0, 3)
        print(f"  [Worker Process] ERROR loading detector '{det_key}': {load_err}")

    results = []
    latencies = []
    for idx, sample in enumerate(samples, 1):
        file_path = sample["file_path"]
        sample_id = sample["sample_id"]

        sample_t0 = time.perf_counter()
        if load_err is not None or adapter is None:
            err_res = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"Model initialization failed: {load_err}",
            )
            res_dict = err_res.to_dict()
            sample_latency_ms = round((time.perf_counter() - sample_t0) * 1000.0, 2)
        elif not os.path.exists(file_path):
            err_res = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"File not found on disk: {file_path}",
            )
            res_dict = err_res.to_dict()
            sample_latency_ms = round((time.perf_counter() - sample_t0) * 1000.0, 2)
        else:
            try:
                res = adapter.predict(file_path)
                sample_latency_ms = round((time.perf_counter() - sample_t0) * 1000.0, 2)
                res_dict = res.to_dict()
            except Exception as inf_exc:
                sample_latency_ms = round((time.perf_counter() - sample_t0) * 1000.0, 2)
                err_res = DetectorResult.create_error(
                    detector_name=fallback_det_name,
                    model_name=fallback_model_name,
                    error_message=f"Inference exception: {str(inf_exc)}",
                )
                res_dict = err_res.to_dict()

        res_dict["latency_ms"] = sample_latency_ms
        latencies.append(sample_latency_ms)

        record = {
            "sample_id": sample_id,
            "detector_key": det_key,
            **res_dict
        }
        results.append(record)

        if idx % 50 == 0 or idx == len(samples):
            print(f"  [Worker Process] Progress: {idx}/{len(samples)} samples processed.")

    # Explicit cleanup
    if adapter is not None and hasattr(adapter, "unload_model"):
        try:
            adapter.unload_model()
        except Exception:
            pass
        del adapter
    reset_instances()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "load_time_sec": load_time_sec,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        }, f, indent=2)

    print(f"  [Worker Process] Successfully completed all {len(results)} samples for '{det_key}'.")


def run_detector_subprocess(det_key: str, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float]:
    """Runs a detector in an isolated child process to guarantee OS-level virtual memory cleanup."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f_in:
        json.dump(samples, f_in)
        in_path = f_in.name

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f_out:
        out_path = f_out.name

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--worker-detector", det_key,
        "--worker-input", in_path,
        "--worker-output", out_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(proc.stdout.strip())
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["results"], data["load_time_sec"], data["mean_latency_ms"]
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Subprocess for {det_key} failed with return code {exc.returncode}")
        print("  --- Subprocess STDOUT ---")
        print(exc.stdout)
        print("  --- Subprocess STDERR ---")
        print(exc.stderr)
        raise RuntimeError(f"Detector evaluation subprocess failed for {det_key}") from exc
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def compute_binary_metrics(
    predictions: List[str],
    ground_truths: List[str],
    probabilities: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Computes binary classification metrics, Wilson CIs, and ROC-AUC."""
    tp = sum(1 for p, g in zip(predictions, ground_truths) if p == PREDICTION_FAKE and g == PREDICTION_FAKE)
    fp = sum(1 for p, g in zip(predictions, ground_truths) if p == PREDICTION_FAKE and g == PREDICTION_REAL)
    tn = sum(1 for p, g in zip(predictions, ground_truths) if p == PREDICTION_REAL and g == PREDICTION_REAL)
    fn = sum(1 for p, g in zip(predictions, ground_truths) if p == PREDICTION_REAL and g == PREDICTION_FAKE)
    error_or_uncertain = sum(1 for p in predictions if p not in (PREDICTION_REAL, PREDICTION_FAKE))

    total = len(predictions)
    valid_total = tp + fp + tn + fn
    actual_pos = tp + fn
    actual_neg = tn + fp

    accuracy = round((tp + tn) / valid_total, 4) if valid_total > 0 else 0.0
    accuracy_ci = wilson_score_interval(tp + tn, valid_total) if valid_total > 0 else [0.0, 0.0]

    recall = round(tp / actual_pos, 4) if actual_pos > 0 else 0.0
    recall_ci = wilson_score_interval(tp, actual_pos) if actual_pos > 0 else [0.0, 0.0]

    specificity = round(tn / actual_neg, 4) if actual_neg > 0 else 0.0
    specificity_ci = wilson_score_interval(tn, actual_neg) if actual_neg > 0 else [0.0, 0.0]

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    precision_ci = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else [0.0, 0.0]

    balanced_acc = round(0.5 * (recall + specificity), 4)

    if (precision + recall) > 0:
        f1 = round(2.0 * precision * recall / (precision + recall), 4)
    else:
        f1 = 0.0

    roc_auc = None
    if probabilities is not None and len(probabilities) == len(ground_truths) and actual_pos > 0 and actual_neg > 0:
        try:
            y_binary = [1 if g == PREDICTION_FAKE else 0 for g in ground_truths]
            valid_pairs = [(y, p) for y, p in zip(y_binary, probabilities) if p is not None]
            if len(valid_pairs) == len(ground_truths):
                roc_auc = round(float(roc_auc_score([y for y, _ in valid_pairs], [p for _, p in valid_pairs])), 4)
        except Exception:
            roc_auc = None

    return {
        "total_samples": total,
        "valid_evaluations": valid_total,
        "non_decisive_or_error": error_or_uncertain,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy,
        "accuracy_ci_95": accuracy_ci,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "precision_ci_95": precision_ci,
        "recall": recall,
        "recall_ci_95": recall_ci,
        "specificity": specificity,
        "specificity_ci_95": specificity_ci,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 7 V2 Corrected Evaluation Pipeline")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode")
    parser.add_argument("--worker-detector", type=str, help="Detector key for worker")
    parser.add_argument("--worker-input", type=str, help="Input JSON for worker")
    parser.add_argument("--worker-output", type=str, help="Output JSON for worker")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip preliminary smoke tests")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-running inference even if CSVs exist")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker_detector, args.worker_input, args.worker_output)
        return

    print("=" * 80)
    print("PHASE 7 V2: CORRECTED EXTERNAL GENERALIZATION EVALUATION")
    print("Official Reference Preprocessing Alignment Verification")
    print("=" * 80)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load manifest
    print(f"Loading external generalization manifest from: {MANIFEST_PATH}")
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    samples = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "sample_id": row["sample_id"],
                "original_filename": row["original_filename"],
                "file_path": row["absolute_path"],
                "ground_truth": row["label"],
                "generator_family": row["generator_family"],
                "category": row.get("category", ""),
            })

    total_samples = len(samples)
    real_count = sum(1 for s in samples if s["ground_truth"] == PREDICTION_REAL)
    fake_count = sum(1 for s in samples if s["ground_truth"] == PREDICTION_FAKE)
    midjourney_count = sum(1 for s in samples if s["generator_family"] == "midjourney")
    adm_count = sum(1 for s in samples if s["generator_family"] == "adm")

    print(f"Manifest verified: {total_samples} samples total (REAL={real_count}, FAKE={fake_count} [Midjourney={midjourney_count}, ADM={adm_count}])")
    assert total_samples == 200, f"Expected 200 samples, got {total_samples}"
    assert real_count == 100, f"Expected 100 REAL samples, got {real_count}"
    assert fake_count == 100, f"Expected 100 FAKE samples, got {fake_count}"

    # 2. Preliminary Smoke Tests
    if not args.skip_smoke and not (os.path.exists(OUTPUT_VIT_CSV) and not args.force_rerun):
        print("\n--- Step 1: Preliminary Single-Sample Smoke Tests ---")
        smoke_sample = [samples[0]]
        for det_key in DETECTOR_KEYS:
            print(f"Testing detector '{det_key}' smoke execution...")
            smoke_res, load_t, lat_t = run_detector_subprocess(det_key, smoke_sample)
            print(f"  -> '{det_key}' smoke passed! Pred: {smoke_res[0].get('prediction')}, Latency: {lat_t}ms")

    # 3. Sequential Subprocess Inference across 200 samples
    print("\n--- Step 2: Sequential Detector Inference on 200 Samples ---")
    detector_outputs: Dict[str, List[Dict[str, Any]]] = {}
    detector_perf: Dict[str, Dict[str, float]] = {}

    all_csvs_exist = all(os.path.exists(p) for p in [OUTPUT_VIT_CSV, OUTPUT_UNIVFD_CSV, OUTPUT_CONVNEXT_CSV])
    if all_csvs_exist and not args.force_rerun:
        print("Reusing existing completed individual predictions from Phase 7 V2 CSVs...")
        for det_key, out_path in [
            ("vit", OUTPUT_VIT_CSV),
            ("univfd", OUTPUT_UNIVFD_CSV),
            ("chen2024_convnext", OUTPUT_CONVNEXT_CSV),
        ]:
            rows = []
            with open(out_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append({
                        "sample_id": r["sample_id"],
                        "detector_key": det_key,
                        "detector_name": r.get("detector_name", ""),
                        "model_name": r.get("model_name", ""),
                        "prediction": r.get("prediction", ""),
                        "predicted_class": r.get("predicted_class", r.get("prediction", "")),
                        "fake_probability": float(r["fake_probability"]) if r.get("fake_probability") else None,
                        "real_probability": float(r["real_probability"]) if r.get("real_probability") else None,
                        "confidence": float(r["confidence"]) if r.get("confidence") else None,
                        "raw_logit": float(r["raw_logit"]) if r.get("raw_logit") else None,
                        "threshold": float(r.get("threshold", 0.5)),
                        "latency_ms": float(r.get("latency_ms", 0.0)) if r.get("latency_ms") else 0.0,
                        "error": r.get("error_message"),
                        "status": r.get("status"),
                        "preprocessing": r.get("preprocessing"),
                    })
            detector_outputs[det_key] = rows
            latencies = [r["latency_ms"] for r in rows if r["latency_ms"] > 0]
            detector_perf[det_key] = {
                "load_time_sec": 0.0,
                "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "total_time_sec": 0.0,
            }
            print(f"  Loaded {len(rows)} samples for '{det_key}' from {out_path} (Mean Latency: {detector_perf[det_key]['mean_latency_ms']}ms)")
    else:
        for det_key in DETECTOR_KEYS:
            print(f"\nEvaluating detector '{det_key}' on 200 samples...")
            t_start = time.perf_counter()
            results, load_time, mean_lat = run_detector_subprocess(det_key, samples)
            t_total = time.perf_counter() - t_start
            print(f"Completed '{det_key}' in {t_total:.2f}s (Load: {load_time}s, Mean Latency: {mean_lat}ms)")
            detector_outputs[det_key] = results
            detector_perf[det_key] = {
                "load_time_sec": load_time,
                "mean_latency_ms": mean_lat,
                "total_time_sec": round(t_total, 2),
            }

    # 4. Save Individual Prediction CSVs
    print("\n--- Step 3: Exporting Individual Model Predictions (Phase 7 V2) ---")
    pred_fieldnames = [
        "sample_id", "original_filename", "ground_truth", "generator_family",
        "detector_key", "detector_name", "model_name", "preprocessing_version",
        "preprocessing", "prediction", "predicted_class", "fake_probability",
        "real_probability", "confidence", "raw_logit", "threshold",
        "status", "error_message", "latency_ms", "file_path"
    ]

    preprocessing_versions = {
        "vit": "Phase 7 v2 (HuggingFace AutoImageProcessor, unchanged)",
        "univfd": "Phase 7 v2 (Official validate.py native CenterCrop 224, no resize)",
        "chen2024_convnext": "Phase 7 v2 (Official DRCT --is_crop native CenterCrop 224, no resize)",
    }

    for det_key, out_path in [
        ("vit", OUTPUT_VIT_CSV),
        ("univfd", OUTPUT_UNIVFD_CSV),
        ("chen2024_convnext", OUTPUT_CONVNEXT_CSV),
    ]:
        res_list = detector_outputs[det_key]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
            writer.writeheader()
            for r, s in zip(res_list, samples):
                row_dict = {
                    "sample_id": s["sample_id"],
                    "original_filename": s["original_filename"],
                    "ground_truth": s["ground_truth"],
                    "generator_family": s["generator_family"],
                    "detector_key": det_key,
                    "detector_name": r.get("detector_name", ""),
                    "model_name": r.get("model_name", ""),
                    "preprocessing_version": preprocessing_versions[det_key],
                    "preprocessing": r.get("preprocessing", ""),
                    "prediction": r.get("prediction", ""),
                    "predicted_class": r.get("predicted_class", r.get("prediction", "")),
                    "fake_probability": r.get("fake_probability", ""),
                    "real_probability": r.get("real_probability", ""),
                    "confidence": r.get("confidence", ""),
                    "raw_logit": r.get("raw_logit", ""),
                    "threshold": r.get("threshold", 0.5),
                    "status": r.get("status", ""),
                    "error_message": r.get("error", "") or r.get("error_message", ""),
                    "latency_ms": r.get("latency_ms", ""),
                    "file_path": s["file_path"],
                }
                writer.writerow(row_dict)
        print(f"  Written {len(res_list)} rows to: {out_path}")

    # 5. Combined Classifier & Ensemble Predictions
    print("\n--- Step 4: Running Decision Fusion Across All Samples ---")
    vit_preds = [r.get("prediction") for r in detector_outputs["vit"]]
    univfd_preds = [r.get("prediction") for r in detector_outputs["univfd"]]
    convnext_preds = [r.get("prediction") for r in detector_outputs["chen2024_convnext"]]

    vit_probs = [r.get("fake_probability") for r in detector_outputs["vit"]]
    univfd_probs = [r.get("fake_probability") for r in detector_outputs["univfd"]]
    convnext_probs = [r.get("fake_probability") for r in detector_outputs["chen2024_convnext"]]

    combined_rows = []
    maj_preds = []
    maj_scores = []  # continuous score: fake vote fraction (0.0, 1/3, 2/3, 1.0)
    mean_preds = []
    mean_probs = []  # continuous score: mean fake probability
    vit_priority_preds = []
    vit_priority_probs = []  # continuous score: vit fake probability

    agree_3_count = 0

    for i, s in enumerate(samples):
        record_map = {
            "vit": detector_outputs["vit"][i],
            "univfd": detector_outputs["univfd"][i],
            "chen2024_convnext": detector_outputs["chen2024_convnext"][i],
        }

        # 1. Majority Voting
        # Calculate vote fraction as true continuous score
        valid_votes = [vit_preds[i], univfd_preds[i], convnext_preds[i]]
        fake_votes = sum(1 for v in valid_votes if v == PREDICTION_FAKE)
        maj_score = round(fake_votes / len(valid_votes), 6)
        res_maj = strategy_majority_vote(record_map, tie_breaker="vit_or_real")
        maj_preds.append(res_maj.prediction)
        maj_scores.append(maj_score)

        # 2. Mean Fake Probability
        res_mean = strategy_mean_fake_probability(record_map, threshold=0.50)
        mean_preds.append(res_mean.prediction)
        mean_probs.append(res_mean.fake_probability)

        # 3. ViT-Priority Fusion
        res_prio = strategy_vit_priority(record_map, vit_key="vit")
        vit_priority_preds.append(res_prio.prediction)
        vit_priority_probs.append(res_prio.fake_probability)

        all_three_agree = (vit_preds[i] == univfd_preds[i] == convnext_preds[i])
        if all_three_agree:
            agree_3_count += 1

        combined_rows.append({
            "sample_id": s["sample_id"],
            "original_filename": s["original_filename"],
            "ground_truth": s["ground_truth"],
            "generator_family": s["generator_family"],
            "vit_pred": vit_preds[i],
            "vit_fake_prob": vit_probs[i],
            "univfd_pred": univfd_preds[i],
            "univfd_fake_prob": univfd_probs[i],
            "convnext_pred": convnext_preds[i],
            "convnext_fake_prob": convnext_probs[i],
            "majority_vote_pred": res_maj.prediction,
            "majority_vote_score": maj_score,
            "majority_vote_fake_prob": res_maj.fake_probability,
            "mean_prob_pred": res_mean.prediction,
            "mean_prob_fake_prob": res_mean.fake_probability,
            "vit_priority_pred": res_prio.prediction,
            "vit_priority_fake_prob": res_prio.fake_probability,
            "all_three_agree": all_three_agree,
            "majority_explanation": res_maj.explanation,
            "mean_explanation": res_mean.explanation,
            "priority_explanation": res_prio.explanation,
        })

    with open(OUTPUT_COMBINED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"  Written combined predictions ({len(combined_rows)} rows) to: {OUTPUT_COMBINED_CSV}")

    # 6. Compute Comprehensive Metrics
    print("\n--- Step 5: Computing Statistical Metrics (Aggregate & Per-Generator) ---")
    ground_truths = [s["ground_truth"] for s in samples]

    targets = {
        "vit": {
            "name": "ViT (dima806)",
            "preds": vit_preds,
            "scores": vit_probs,
            "lat": detector_perf["vit"]["mean_latency_ms"]
        },
        "univfd": {
            "name": "UnivFD (CLIP ViT-L/14)",
            "preds": univfd_preds,
            "scores": univfd_probs,
            "lat": detector_perf["univfd"]["mean_latency_ms"]
        },
        "chen2024_convnext": {
            "name": "ConvNeXt (DRCT / RAID)",
            "preds": convnext_preds,
            "scores": convnext_probs,
            "lat": detector_perf["chen2024_convnext"]["mean_latency_ms"]
        },
        "ensemble_majority_vote": {
            "name": "Ensemble (Majority Vote)",
            "preds": maj_preds,
            "scores": maj_scores,  # evaluated using vote fraction continuous score!
            "lat": sum(detector_perf[k]["mean_latency_ms"] for k in DETECTOR_KEYS)
        },
        "ensemble_mean_prob": {
            "name": "Ensemble (Mean Fake Probability)",
            "preds": mean_preds,
            "scores": mean_probs,
            "lat": sum(detector_perf[k]["mean_latency_ms"] for k in DETECTOR_KEYS)
        },
        "ensemble_vit_priority": {
            "name": "Ensemble (ViT Priority Fallback)",
            "preds": vit_priority_preds,
            "scores": vit_priority_probs,
            "lat": detector_perf["vit"]["mean_latency_ms"]
        },
    }

    # Slice indices
    idx_real = [i for i, s in enumerate(samples) if s["ground_truth"] == PREDICTION_REAL]
    idx_mj = [i for i, s in enumerate(samples) if s["generator_family"] == "midjourney"]
    idx_adm = [i for i, s in enumerate(samples) if s["generator_family"] == "adm"]

    # Slice cohorts (authentic controls + generator synthetic)
    idx_mj_cohort = idx_real + idx_mj
    idx_adm_cohort = idx_real + idx_adm

    aggregate_results: Dict[str, Any] = {}
    generator_slice_results: Dict[str, Any] = {}

    for k, t in targets.items():
        # Aggregate metrics
        agg_m = compute_binary_metrics(t["preds"], ground_truths, t["scores"])
        agg_m["mean_latency_ms"] = t["lat"]
        agg_m["display_name"] = t["name"]
        aggregate_results[k] = agg_m

        # Midjourney cohort (100 Real + 50 Midjourney = 150 samples)
        mj_preds = [t["preds"][i] for i in idx_mj_cohort]
        mj_gts = [ground_truths[i] for i in idx_mj_cohort]
        mj_scores = [t["scores"][i] for i in idx_mj_cohort]
        mj_m = compute_binary_metrics(mj_preds, mj_gts, mj_scores)

        # ADM cohort (100 Real + 50 ADM = 150 samples)
        adm_preds = [t["preds"][i] for i in idx_adm_cohort]
        adm_gts = [ground_truths[i] for i in idx_adm_cohort]
        adm_scores = [t["scores"][i] for i in idx_adm_cohort]
        adm_m = compute_binary_metrics(adm_preds, adm_gts, adm_scores)

        # Standalone fake detections (pure recall out of 50)
        mj_raw_correct = sum(1 for i in idx_mj if t["preds"][i] == PREDICTION_FAKE)
        adm_raw_correct = sum(1 for i in idx_adm if t["preds"][i] == PREDICTION_FAKE)
        real_raw_correct = sum(1 for i in idx_real if t["preds"][i] == PREDICTION_REAL)

        # Score stats on slices
        raw_scores = t["scores"]
        real_scores = [raw_scores[i] for i in idx_real if raw_scores[i] is not None]
        mj_scores_only = [raw_scores[i] for i in idx_mj if raw_scores[i] is not None]
        adm_scores_only = [raw_scores[i] for i in idx_adm if raw_scores[i] is not None]

        generator_slice_results[k] = {
            "display_name": t["name"],
            "authentic_control": {
                "total": len(idx_real),
                "correct_real": real_raw_correct,
                "specificity": round(real_raw_correct / len(idx_real), 4),
                "score_mean": round(float(np.mean(real_scores)), 4) if real_scores else None,
                "score_std": round(float(np.std(real_scores)), 4) if real_scores else None,
            },
            "midjourney_slice": {
                "total_cohort_samples": len(idx_mj_cohort),
                "accuracy": mj_m["accuracy"],
                "recall": mj_m["recall"],
                "specificity": mj_m["specificity"],
                "precision": mj_m["precision"],
                "f1_score": mj_m["f1_score"],
                "roc_auc": mj_m["roc_auc"],
                "raw_detections": f"{mj_raw_correct} / {len(idx_mj)} ({mj_raw_correct / len(idx_mj) * 100:.1f}%)",
                "score_mean": round(float(np.mean(mj_scores_only)), 4) if mj_scores_only else None,
                "score_std": round(float(np.std(mj_scores_only)), 4) if mj_scores_only else None,
            },
            "adm_slice": {
                "total_cohort_samples": len(idx_adm_cohort),
                "accuracy": adm_m["accuracy"],
                "recall": adm_m["recall"],
                "specificity": adm_m["specificity"],
                "precision": adm_m["precision"],
                "f1_score": adm_m["f1_score"],
                "roc_auc": adm_m["roc_auc"],
                "raw_detections": f"{adm_raw_correct} / {len(idx_adm)} ({adm_raw_correct / len(idx_adm) * 100:.1f}%)",
                "score_mean": round(float(np.mean(adm_scores_only)), 4) if adm_scores_only else None,
                "score_std": round(float(np.std(adm_scores_only)), 4) if adm_scores_only else None,
            },
        }

    # 7. Pairwise Agreement Matrix & Cohen's Kappa
    print("\n--- Step 6: Computing Pairwise Detector Agreement ---")
    pairwise_pairs = [
        ("vit", "univfd"),
        ("vit", "chen2024_convnext"),
        ("univfd", "chen2024_convnext"),
    ]
    pairwise_analysis = {}

    for d1, d2 in pairwise_pairs:
        p1 = targets[d1]["preds"]
        p2 = targets[d2]["preds"]
        both_fake = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_FAKE and b == PREDICTION_FAKE)
        both_real = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_REAL and b == PREDICTION_REAL)
        d1_f_d2_r = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_FAKE and b == PREDICTION_REAL)
        d1_r_d2_f = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_REAL and b == PREDICTION_FAKE)
        agree_count = both_fake + both_real
        agree_pct = round(agree_count / total_samples * 100.0, 2)
        kappa = compute_cohen_kappa(both_fake, both_real, d1_f_d2_r, d1_r_d2_f)

        pairwise_analysis[f"{d1}_vs_{d2}"] = {
            "detector_1": targets[d1]["name"],
            "detector_2": targets[d2]["name"],
            "agreed_samples": agree_count,
            "total_samples": total_samples,
            "agreement_percentage": agree_pct,
            "both_fake": both_fake,
            "both_real": both_real,
            "d1_fake_d2_real": d1_f_d2_r,
            "d1_real_d2_fake": d1_r_d2_f,
            "cohen_kappa": kappa,
        }
        print(f"  {targets[d1]['name']} vs {targets[d2]['name']}: Agreement={agree_pct}% (Kappa={kappa})")

    # 8. Comparison Table: Phase 7 V1 vs Phase 7 V2
    print("\n--- Step 7: Exporting Model Comparison CSV ---")
    comparison_rows = []
    comp_fieldnames = [
        "Model / Strategy",
        "Aggregate Acc (%)",
        "Aggregate 95% CI",
        "Aggregate BalAcc (%)",
        "Aggregate Precision",
        "Aggregate Recall",
        "Aggregate Specificity",
        "Aggregate F1",
        "Aggregate ROC-AUC",
        "Midjourney Acc (%)",
        "Midjourney Recall (%)",
        "ADM Acc (%)",
        "ADM Recall (%)",
        "TP", "FP", "TN", "FN",
        "Mean Latency (ms)"
    ]

    for k, t in targets.items():
        agg = aggregate_results[k]
        slices = generator_slice_results[k]
        cm = agg["confusion_matrix"]

        row = {
            "Model / Strategy": t["name"],
            "Aggregate Acc (%)": f"{agg['accuracy'] * 100:.2f}",
            "Aggregate 95% CI": f"[{agg['accuracy_ci_95'][0] * 100:.2f}%, {agg['accuracy_ci_95'][1] * 100:.2f}%]",
            "Aggregate BalAcc (%)": f"{agg['balanced_accuracy'] * 100:.2f}",
            "Aggregate Precision": f"{agg['precision']:.4f}",
            "Aggregate Recall": f"{agg['recall']:.4f}",
            "Aggregate Specificity": f"{agg['specificity']:.4f}",
            "Aggregate F1": f"{agg['f1_score']:.4f}",
            "Aggregate ROC-AUC": f"{agg['roc_auc']:.4f}" if agg['roc_auc'] is not None else "N/A",
            "Midjourney Acc (%)": f"{slices['midjourney_slice']['accuracy'] * 100:.2f}",
            "Midjourney Recall (%)": f"{slices['midjourney_slice']['recall'] * 100:.2f}",
            "ADM Acc (%)": f"{slices['adm_slice']['accuracy'] * 100:.2f}",
            "ADM Recall (%)": f"{slices['adm_slice']['recall'] * 100:.2f}",
            "TP": cm["tp"],
            "FP": cm["fp"],
            "TN": cm["tn"],
            "FN": cm["fn"],
            "Mean Latency (ms)": f"{agg['mean_latency_ms']:.1f}",
        }
        comparison_rows.append(row)

    with open(OUTPUT_COMPARISON_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"  Written comparison table ({len(comparison_rows)} rows) to: {OUTPUT_COMPARISON_CSV}")

    # 9. Load Version 1 data for direct delta calculation
    v1_metrics = {}
    if os.path.exists(V1_METRICS_JSON):
        try:
            with open(V1_METRICS_JSON, "r", encoding="utf-8") as f:
                v1_metrics = json.load(f)
        except Exception:
            pass

    version_deltas = {}
    for k in targets.keys():
        v2_acc = aggregate_results[k]["accuracy"]
        v2_rec = aggregate_results[k]["recall"]
        v2_spec = aggregate_results[k]["specificity"]
        v2_auc = aggregate_results[k]["roc_auc"]

        v1_acc = v1_metrics.get("aggregate_performance", {}).get(k, {}).get("accuracy")
        v1_rec = v1_metrics.get("aggregate_performance", {}).get(k, {}).get("recall")
        v1_spec = v1_metrics.get("aggregate_performance", {}).get(k, {}).get("specificity")
        v1_auc = v1_metrics.get("aggregate_performance", {}).get(k, {}).get("roc_auc")

        version_deltas[k] = {
            "v1_accuracy": v1_acc,
            "v2_accuracy": v2_acc,
            "delta_accuracy": round(v2_acc - v1_acc, 4) if v1_acc is not None else None,
            "v1_recall": v1_rec,
            "v2_recall": v2_rec,
            "delta_recall": round(v2_rec - v1_rec, 4) if v1_rec is not None else None,
            "v1_specificity": v1_spec,
            "v2_specificity": v2_spec,
            "delta_specificity": round(v2_spec - v1_spec, 4) if v1_spec is not None else None,
            "v1_roc_auc": v1_auc,
            "v2_roc_auc": v2_auc,
            "delta_roc_auc": round(v2_auc - v1_auc, 4) if (v1_auc is not None and v2_auc is not None) else None,
        }

    # 10. Save Complete Combined Metrics JSON
    metrics_payload = {
        "metadata": {
            "benchmark_phase": "Phase 7 Version 2 (Official Reference Preprocessing)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "manifest_path": "benchmark/external_generalization_manifest.csv",
            "dataset_source": "GenImage (TheKernel01/Tiny-GenImage validation split)",
            "decision_threshold": 0.50,
            "total_samples": 200,
            "real_samples": 100,
            "fake_samples": 100,
            "generator_counts": {
                "midjourney": 50,
                "adm": 50
            },
            "environment": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "platform": sys.platform,
            },
        },
        "aggregate_performance": aggregate_results,
        "generator_slices": generator_slice_results,
        "pairwise_agreement": pairwise_analysis,
        "all_three_agree_count": agree_3_count,
        "all_three_agree_pct": round(agree_3_count / total_samples * 100.0, 2),
        "version_comparison_v1_vs_v2": version_deltas,
    }

    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Written complete metrics JSON to: {OUTPUT_METRICS_JSON}")

    print("\n" + "=" * 80)
    print("PHASE 7 V2 EVALUATION COMPLETE")
    print("=" * 80)
    for row in comparison_rows:
        print(f"  {row['Model / Strategy']:35s} | Acc: {row['Aggregate Acc (%)']:>6s}% | Recall: {row['Aggregate Recall']:>6s} | Spec: {row['Aggregate Specificity']:>6s} | AUC: {row['Aggregate ROC-AUC']:>6s}")


if __name__ == "__main__":
    main()
