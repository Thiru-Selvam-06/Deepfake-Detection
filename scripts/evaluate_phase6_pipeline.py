"""
Phase 6 Evaluation Pipeline Runner
Evaluates all three integrated deepfake detectors (ViT, UnivFD, ConvNeXt/DRCT)
independently against benchmark/phase6_test_manifest.csv (200 pristine test images: 100 REAL, 100 FAKE).
Then evaluates ensemble combination strategies (Majority Vote, Mean Fake Probability, ViT-Priority Fallback).

Memory Safety:
- Sequential subprocess isolation per detector guarantees 100% OS-level reclamation of virtual memory.
- Complete garbage collection and CUDA cache clearing before and after each detector run.
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

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

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
    DecisionFusionPipeline,
    strategy_vit_only,
    strategy_majority_vote,
    strategy_mean_fake_probability,
    strategy_vit_priority,
    STRATEGY_VIT_ONLY,
    STRATEGY_MAJORITY_VOTE,
    STRATEGY_MEAN_PROBABILITY,
    STRATEGY_VIT_PRIORITY,
)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "phase6_test_manifest.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results")

OUTPUT_VIT_CSV = os.path.join(RESULTS_DIR, "phase6_vit_predictions.csv")
OUTPUT_UNIVFD_CSV = os.path.join(RESULTS_DIR, "phase6_univfd_predictions.csv")
OUTPUT_CONVNEXT_CSV = os.path.join(RESULTS_DIR, "phase6_convnext_predictions.csv")
OUTPUT_COMBINED_CSV = os.path.join(RESULTS_DIR, "phase6_combined_predictions.csv")
OUTPUT_METRICS_JSON = os.path.join(RESULTS_DIR, "phase6_combined_metrics.json")
OUTPUT_COMPARISON_CSV = os.path.join(RESULTS_DIR, "phase6_model_comparison.csv")

DETECTOR_KEYS = ["vit", "univfd", "chen2024_convnext"]


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> List[float]:
    """
    Computes Wilson score confidence interval for binomial proportion k / n.
    Standard z for 95% confidence is 1.959963984540054.
    Returns [lower_bound, upper_bound] rounded to 4 decimal places.
    """
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
    """
    Worker function executed in an isolated child process for a single detector.
    Evaluates all samples in input_json_path and writes results to output_json_path.
    """
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
    """
    Executes a detector in an isolated child process to ensure memory safety.
    Returns (results, load_time_sec, mean_latency_ms).
    """
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


def compute_binary_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """
    Computes confusion matrix, accuracy, balanced accuracy, precision, recall, specificity, F1, and Wilson 95% CIs.
    """
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
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Evaluation Pipeline")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode")
    parser.add_argument("--worker-detector", type=str, help="Detector key for worker")
    parser.add_argument("--worker-input", type=str, help="Input JSON for worker")
    parser.add_argument("--worker-output", type=str, help="Output JSON for worker")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip preliminary smoke tests")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker_detector, args.worker_input, args.worker_output)
        return

    print("================================================================================")
    print("PHASE 6: AUTONOMOUS EVALUATION PIPELINE & COMBINED CLASSIFIER")
    print("================================================================================")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load manifest
    print(f"Loading manifest from: {MANIFEST_PATH}")
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
                "ground_truth": row["ground_truth"],
                "category": row.get("category", ""),
                "domain": row.get("domain", ""),
            })

    total_samples = len(samples)
    real_count = sum(1 for s in samples if s["ground_truth"] == PREDICTION_REAL)
    fake_count = sum(1 for s in samples if s["ground_truth"] == PREDICTION_FAKE)
    print(f"Manifest verified: {total_samples} samples total (REAL={real_count}, FAKE={fake_count})")
    assert total_samples == 200, f"Expected 200 samples, got {total_samples}"
    assert real_count == 100, f"Expected 100 REAL samples, got {real_count}"
    assert fake_count == 100, f"Expected 100 FAKE samples, got {fake_count}"

    # 2. Preliminary Smoke Test
    if not args.skip_smoke:
        print("\n--- Step 1: Preliminary Single-Sample Smoke Tests ---")
        smoke_sample = [samples[0]]  # first real image
        for det_key in DETECTOR_KEYS:
            print(f"Testing detector '{det_key}' smoke execution...")
            smoke_res, load_t, lat_t = run_detector_subprocess(det_key, smoke_sample)
            print(f"  -> '{det_key}' smoke passed! Pred: {smoke_res[0].get('prediction')}, Latency: {lat_t}ms")

    # 3. Sequential Subprocess Inference across 200 samples
    print("\n--- Step 2: Sequential Detector Inference on 200 Samples ---")
    detector_outputs: Dict[str, List[Dict[str, Any]]] = {}
    detector_perf: Dict[str, Dict[str, float]] = {}

    for det_key in DETECTOR_KEYS:
        print(f"\nEvaluating detector '{det_key}' on 200 samples...")
        t_start = time.perf_counter()
        results, load_time, mean_lat = run_detector_subprocess(det_key, samples)
        t_total = time.perf_counter() - t_start
        print(f"Completed '{det_key}' in {t_total:.2f}s (Load: {load_time}s, Mean Sample Latency: {mean_lat}ms)")
        detector_outputs[det_key] = results
        detector_perf[det_key] = {
            "load_time_sec": load_time,
            "mean_latency_ms": mean_lat,
            "total_time_sec": round(t_total, 2),
        }

    # 4. Save Individual Prediction CSVs
    print("\n--- Step 3: Exporting Individual Model Predictions ---")
    pred_fieldnames = [
        "sample_id", "original_filename", "ground_truth", "detector_key",
        "detector_name", "model_name", "prediction", "predicted_class",
        "fake_probability", "real_probability", "confidence", "raw_logit",
        "threshold", "status", "error_message", "latency_ms", "file_path"
    ]

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
                    "detector_key": det_key,
                    "detector_name": r.get("detector_name", ""),
                    "model_name": r.get("model_name", ""),
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
    print("\n--- Step 4: Running Combined Classifier & Fusion Strategies ---")
    ground_truths = [s["ground_truth"] for s in samples]

    vit_preds = [r.get("prediction") for r in detector_outputs["vit"]]
    univfd_preds = [r.get("prediction") for r in detector_outputs["univfd"]]
    convnext_preds = [r.get("prediction") for r in detector_outputs["chen2024_convnext"]]

    vit_probs = [r.get("fake_probability") for r in detector_outputs["vit"]]
    univfd_probs = [r.get("fake_probability") for r in detector_outputs["univfd"]]
    convnext_probs = [r.get("fake_probability") for r in detector_outputs["chen2024_convnext"]]

    combined_rows = []
    maj_preds = []
    maj_probs = []
    mean_preds = []
    mean_probs = []
    vit_priority_preds = []
    vit_priority_probs = []

    agree_3_count = 0
    agree_maj_count = 0

    for i, s in enumerate(samples):
        # Prepare dictionary for fusion
        record_map = {
            "vit": detector_outputs["vit"][i],
            "univfd": detector_outputs["univfd"][i],
            "chen2024_convnext": detector_outputs["chen2024_convnext"][i],
        }

        # Run strategies
        res_maj = strategy_majority_vote(record_map, tie_breaker="vit_or_real")
        res_mean = strategy_mean_fake_probability(record_map, threshold=0.50)
        res_prio = strategy_vit_priority(record_map, vit_key="vit")

        maj_preds.append(res_maj.prediction)
        maj_probs.append(res_maj.fake_probability)

        mean_preds.append(res_mean.prediction)
        mean_probs.append(res_mean.fake_probability)

        vit_priority_preds.append(res_prio.prediction)
        vit_priority_probs.append(res_prio.fake_probability)

        all_three_agree = (vit_preds[i] == univfd_preds[i] == convnext_preds[i])
        if all_three_agree:
            agree_3_count += 1

        combined_rows.append({
            "sample_id": s["sample_id"],
            "original_filename": s["original_filename"],
            "ground_truth": s["ground_truth"],
            "vit_pred": vit_preds[i],
            "vit_fake_prob": vit_probs[i],
            "univfd_pred": univfd_preds[i],
            "univfd_fake_prob": univfd_probs[i],
            "convnext_pred": convnext_preds[i],
            "convnext_fake_prob": convnext_probs[i],
            "majority_vote_pred": res_maj.prediction,
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

    # Write combined predictions CSV
    comb_fieldnames = list(combined_rows[0].keys())
    with open(OUTPUT_COMBINED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comb_fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"  Written combined predictions ({len(combined_rows)} rows) to: {OUTPUT_COMBINED_CSV}")

    # 6. Compute Comprehensive Metrics
    print("\n--- Step 5: Computing Statistical Metrics & Cross-Model Agreement ---")
    metrics_report: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest_path": os.path.relpath(MANIFEST_PATH, PROJECT_ROOT).replace("\\", "/"),
            "dataset_source": "Kaggle 140k Real and Fake Faces (FFHQ / StyleGAN)",
            "total_test_samples": total_samples,
            "class_distribution": {"REAL": real_count, "FAKE": fake_count},
            "environment": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "platform": sys.platform,
            }
        },
        "model_performance": {},
        "agreement_analysis": {},
    }

    # Evaluate each individual detector and strategy
    eval_targets = {
        "vit": {"name": "ViT (dima806)", "preds": vit_preds, "lat": detector_perf["vit"]["mean_latency_ms"]},
        "univfd": {"name": "UnivFD (CLIP ViT-L/14)", "preds": univfd_preds, "lat": detector_perf["univfd"]["mean_latency_ms"]},
        "chen2024_convnext": {"name": "ConvNeXt (DRCT / RAID)", "preds": convnext_preds, "lat": detector_perf["chen2024_convnext"]["mean_latency_ms"]},
        "ensemble_majority_vote": {"name": "Ensemble (Majority Vote)", "preds": maj_preds, "lat": sum(detector_perf[k]["mean_latency_ms"] for k in DETECTOR_KEYS)},
        "ensemble_mean_prob": {"name": "Ensemble (Mean Fake Probability)", "preds": mean_preds, "lat": sum(detector_perf[k]["mean_latency_ms"] for k in DETECTOR_KEYS)},
        "ensemble_vit_priority": {"name": "Ensemble (ViT Priority Fallback)", "preds": vit_priority_preds, "lat": detector_perf["vit"]["mean_latency_ms"]},
    }

    comparison_rows = []

    for key, target in eval_targets.items():
        m = compute_binary_metrics(target["preds"], ground_truths)
        m["mean_latency_ms"] = target["lat"]
        m["display_name"] = target["name"]
        metrics_report["model_performance"][key] = m

        print(f"  {target['name']:35s} | Acc: {m['accuracy']*100:6.2f}% ({m['accuracy_ci_95'][0]*100:.1f}-{m['accuracy_ci_95'][1]*100:.1f}%) | "
              f"BalAcc: {m['balanced_accuracy']*100:6.2f}% | Prec: {m['precision']:5.3f} | Rec: {m['recall']:5.3f} | "
              f"Spec: {m['specificity']:5.3f} | F1: {m['f1_score']:5.3f}")

        comparison_rows.append({
            "Model / Strategy": target["name"],
            "Accuracy (%)": f"{m['accuracy'] * 100:.2f}",
            "95% CI (Accuracy)": f"[{m['accuracy_ci_95'][0]*100:.2f}%, {m['accuracy_ci_95'][1]*100:.2f}%]",
            "Balanced Acc (%)": f"{m['balanced_accuracy'] * 100:.2f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "Specificity": f"{m['specificity']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "TP": m["confusion_matrix"]["tp"],
            "FP": m["confusion_matrix"]["fp"],
            "TN": m["confusion_matrix"]["tn"],
            "FN": m["confusion_matrix"]["fn"],
            "Mean Latency (ms)": f"{target['lat']:.1f}",
        })

    # Agreement matrices & Cohen's Kappa
    def get_pairwise_kappa(p1, p2):
        both_fake = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_FAKE and b == PREDICTION_FAKE)
        both_real = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_REAL and b == PREDICTION_REAL)
        m1_f_m2_r = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_FAKE and b == PREDICTION_REAL)
        m1_r_m2_f = sum(1 for a, b in zip(p1, p2) if a == PREDICTION_REAL and b == PREDICTION_FAKE)
        raw_agree = (both_fake + both_real) / len(p1)
        kappa = compute_cohen_kappa(both_fake, both_real, m1_f_m2_r, m1_r_m2_f)
        return {
            "raw_agreement_rate": round(raw_agree, 4),
            "cohen_kappa": kappa,
            "both_fake": both_fake,
            "both_real": both_real,
            "m1_fake_m2_real": m1_f_m2_r,
            "m1_real_m2_fake": m1_r_m2_f,
        }

    metrics_report["agreement_analysis"] = {
        "all_three_agree_count": agree_3_count,
        "all_three_agree_fraction": round(agree_3_count / total_samples, 4),
        "vit_vs_univfd": get_pairwise_kappa(vit_preds, univfd_preds),
        "vit_vs_convnext": get_pairwise_kappa(vit_preds, convnext_preds),
        "univfd_vs_convnext": get_pairwise_kappa(univfd_preds, convnext_preds),
    }

    # Save metrics JSON
    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"\n  Saved comprehensive metrics JSON to: {OUTPUT_METRICS_JSON}")

    # Save comparison CSV
    with open(OUTPUT_COMPARISON_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"  Saved model comparison summary CSV to: {OUTPUT_COMPARISON_CSV}")

    print("\n================================================================================")
    print("PHASE 6 PIPELINE EVALUATION COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()

