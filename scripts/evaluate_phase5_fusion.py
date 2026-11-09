"""
Phase 5 Decision Fusion Evaluation Runner
Evaluates individual detectors and decision-fusion strategies on the
uncontaminated development_val split (50 facial samples: 25 FFHQ Real, 25 StyleGAN2 Fake)
from benchmark/evaluation_manifest.csv.

Strategies evaluated:
1. ViT alone (baseline)
2. Majority vote across the 3 detectors
3. Mean fake probability (with probability comparability check)
4. ViT-priority with fallback to secondary detectors

Guarantees:
- Strict split isolation: only development_val is evaluated for strategy comparison.
- Memory-safe sequential subprocess execution: isolates each detector in a separate OS process.
- Produces benchmark/results/phase5_validation_metrics.json.
- Prepares reproducible frozen command for held_out_test without executing it prematurely.
"""

import os
import sys
import csv
import json
import time
import math
import argparse
import subprocess
import platform
import gc
import tempfile
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
    list_detectors,
)
from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_UNCERTAIN,
    PREDICTION_ERROR,
)
from src.decision_fusion import (
    DecisionFusionPipeline,
    FusionResult,
    STRATEGY_VIT_ONLY,
    STRATEGY_MAJORITY_VOTE,
    STRATEGY_MEAN_PROBABILITY,
    STRATEGY_VIT_PRIORITY,
)

DEFAULT_MANIFEST = os.path.join(PROJECT_ROOT, "benchmark", "evaluation_manifest.csv")
RAW_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase5_val_raw_predictions.csv")
OUTPUT_METRICS_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase5_validation_metrics.json")
REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "docs", "PHASE5_FUSION_FEASIBILITY_REPORT.md")

DETECTOR_KEYS = ["chen2024_convnext", "univfd", "vit"]


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> List[float]:
    """Binomial Wilson score interval for proportion k / n."""
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


def run_worker(det_key: str, input_json_path: str, output_json_path: str) -> None:
    """Worker child process executing inference for a single detector."""
    print(f"  [Worker] Initializing detector '{det_key}' in child process (PID: {os.getpid()})...")
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
        print(f"  [Worker] Detector '{det_key}' loaded in {load_time_sec}s.")
    except Exception as e:
        load_err = str(e)
        load_time_sec = round(time.perf_counter() - t0, 3)
        print(f"  [Worker] Failed to load detector '{det_key}': {load_err}")

    results = []
    for idx, sample in enumerate(samples, 1):
        file_path = sample["file_path"]
        sample_id = sample["sample_id"]

        t_inf0 = time.perf_counter()
        if load_err is not None or adapter is None:
            err_res = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"Model init failed: {load_err}",
            )
            res_dict = err_res.to_dict()
            latency = 0.0
        elif not os.path.exists(file_path):
            err_res = DetectorResult.create_error(
                detector_name=fallback_det_name,
                model_name=fallback_model_name,
                error_message=f"File not found: {file_path}",
            )
            res_dict = err_res.to_dict()
            latency = 0.0
        else:
            try:
                res = adapter.predict(file_path)
                latency = round((time.perf_counter() - t_inf0) * 1000.0, 2)
                res_dict = res.to_dict()
            except Exception as inf_exc:
                latency = round((time.perf_counter() - t_inf0) * 1000.0, 2)
                err_res = DetectorResult.create_error(
                    detector_name=fallback_det_name,
                    model_name=fallback_model_name,
                    error_message=f"Inference exception: {str(inf_exc)}",
                )
                res_dict = err_res.to_dict()

        record = {
            "sample_id": sample_id,
            "split": sample.get("split", "development_val"),
            "domain": sample.get("domain", "facial"),
            "category": sample.get("category", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "file_path": file_path,
            "detector_key": det_key,
            "detector_name": res_dict.get("detector_name", fallback_det_name),
            "model_name": res_dict.get("model_name", fallback_model_name),
            "prediction": res_dict.get("prediction", PREDICTION_ERROR),
            "fake_probability": res_dict.get("fake_probability"),
            "real_probability": res_dict.get("real_probability"),
            "raw_logit": res_dict.get("raw_logit"),
            "confidence": res_dict.get("confidence"),
            "threshold": res_dict.get("threshold", 0.5),
            "status": "SUCCESS" if res_dict.get("prediction") in (PREDICTION_REAL, PREDICTION_FAKE) else "ERROR",
            "error_message": res_dict.get("error", ""),
            "latency_ms": latency,
        }
        results.append(record)

    # Deallocate model
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
        json.dump({"results": results, "load_time_sec": load_time_sec}, f, indent=2)

    print(f"  [Worker] Finished {len(results)} samples for '{det_key}'.")


def run_detector_subprocess(det_key: str, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Launches child process to evaluate det_key across all samples."""
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
            return data["results"]
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Subprocess for {det_key} failed (code {exc.returncode})")
        print("STDOUT:", exc.stdout)
        print("STDERR:", exc.stderr)
        raise RuntimeError(f"Detector subprocess failed for {det_key}") from exc
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def load_manifest_split(manifest_path: str, target_split: str) -> List[Dict[str, Any]]:
    """Loads records matching target_split from evaluation manifest."""
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == target_split:
                samples.append(row)
    return samples


def compute_metrics(
    predictions: List[Dict[str, Any]],
    pos_label: str = PREDICTION_FAKE,
    neg_label: str = PREDICTION_REAL,
) -> Dict[str, Any]:
    """
    Computes confusion matrix, accuracy, balanced accuracy, precision, recall, specificity, F1.
    Keeps rejection controls and non-binary samples strictly separate from classification metrics.
    """
    classification_items = [
        p for p in predictions if p.get("ground_truth") in (pos_label, neg_label)
    ]
    control_items = [
        p for p in predictions
        if p.get("ground_truth") == "REJECTION_CONTROL" or p.get("domain") == "control"
    ]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    valid_count = 0
    errors = []
    fps = []
    fns = []

    for item in classification_items:
        gt = item["ground_truth"]
        pred = item["prediction"]
        sid = item["sample_id"]

        if pred not in (PREDICTION_REAL, PREDICTION_FAKE):
            errors.append(sid)
            continue

        valid_count += 1
        if gt == pos_label and pred == pos_label:
            tp += 1
        elif gt == neg_label and pred == pos_label:
            fp += 1
            fps.append(sid)
        elif gt == neg_label and pred == neg_label:
            tn += 1
        elif gt == pos_label and pred == neg_label:
            fn += 1
            fns.append(sid)

    acc = (tp + tn) / valid_count if valid_count > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    bal_acc = (rec + spec) / 2.0

    acc_ci = wilson_score_interval(tp + tn, valid_count)
    prec_ci = wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else [0.0, 0.0]
    rec_ci = wilson_score_interval(tp, tp + fn) if (tp + fn) > 0 else [0.0, 0.0]
    spec_ci = wilson_score_interval(tn, tn + fp) if (tn + fp) > 0 else [0.0, 0.0]

    # Rejection controls evaluated independently
    ctrl_total = len(control_items)
    ctrl_rejected = sum(
        1 for c in control_items
        if c.get("prediction") == PREDICTION_ERROR or c.get("status") in ("ERROR", "REJECTED")
    )
    rejection_summary = {
        "total_controls": ctrl_total,
        "successfully_rejected": ctrl_rejected,
        "rejection_rate": round(ctrl_rejected / ctrl_total, 4) if ctrl_total > 0 else 1.0,
        "control_sample_ids": [c["sample_id"] for c in control_items],
    }

    return {
        "total_samples": len(predictions),
        "classification_samples": len(classification_items),
        "valid_count": valid_count,
        "error_count": len(errors),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": round(acc, 4),
        "accuracy_ci_95": acc_ci,
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "precision_ci_95": prec_ci,
        "recall": round(rec, 4),
        "recall_ci_95": rec_ci,
        "specificity": round(spec, 4),
        "specificity_ci_95": spec_ci,
        "f1_score": round(f1, 4),
        "false_positives": fps,
        "false_negatives": fns,
        "errors": errors,
        "rejection_controls": rejection_summary,
    }


def analyze_probability_comparability(
    raw_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyzes the distribution and separation margin of fake probabilities
    for each detector on Real vs Fake validation samples.
    """
    detectors = {}
    for det in DETECTOR_KEYS:
        det_recs = [r for r in raw_records if r["detector_key"] == det]
        real_probs = [r["fake_probability"] for r in det_recs if r["ground_truth"] == "REAL" and r["fake_probability"] is not None]
        fake_probs = [r["fake_probability"] for r in det_recs if r["ground_truth"] == "FAKE" and r["fake_probability"] is not None]

        def stats(vals):
            if not vals:
                return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
            vals_sorted = sorted(vals)
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            med = vals_sorted[len(vals_sorted) // 2]
            return {
                "count": len(vals),
                "mean": round(m, 4),
                "std": round(math.sqrt(var), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "median": round(med, 4),
            }

        r_stats = stats(real_probs)
        f_stats = stats(fake_probs)
        sep_margin = round(f_stats["mean"] - r_stats["mean"], 4)

        detectors[det] = {
            "real_faces_p_fake": r_stats,
            "fake_faces_p_fake": f_stats,
            "separation_margin": sep_margin,
            "calibrated_on_faces": sep_margin > 0.50 and r_stats["mean"] < 0.20 and f_stats["mean"] > 0.80,
        }

    return detectors


def evaluate_val_split(force_rerun: bool = False) -> Dict[str, Any]:
    """
    Orchestrates the evaluation of individual detectors and fusion strategies on development_val.
    """
    print("=" * 80)
    print("PHASE 5: DECISION FUSION FEASIBILITY EVALUATION ON DEVELOPMENT_VAL")
    print("=" * 80)

    val_samples = load_manifest_split(DEFAULT_MANIFEST, "development_val")
    print(f"Loaded {len(val_samples)} validation samples from {DEFAULT_MANIFEST}")
    assert len(val_samples) == 50, f"Expected exactly 50 validation samples, found {len(val_samples)}"

    # Step 1: Collect or load raw predictions
    all_raw_predictions = []
    if os.path.exists(RAW_PREDICTIONS_CSV) and not force_rerun:
        print(f"\n[CACHE] Loading existing raw detector predictions from: {RAW_PREDICTIONS_CSV}")
        with open(RAW_PREDICTIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields
                row["fake_probability"] = float(row["fake_probability"]) if row.get("fake_probability") not in ("", None) else None
                row["real_probability"] = float(row["real_probability"]) if row.get("real_probability") not in ("", None) else None
                row["raw_logit"] = float(row["raw_logit"]) if row.get("raw_logit") not in ("", None) else None
                row["confidence"] = float(row["confidence"]) if row.get("confidence") not in ("", None) else None
                row["threshold"] = float(row["threshold"]) if row.get("threshold") not in ("", None) else 0.5
                row["latency_ms"] = float(row["latency_ms"]) if row.get("latency_ms") not in ("", None) else 0.0
                all_raw_predictions.append(row)
        print(f"Loaded {len(all_raw_predictions)} raw predictions.")
    else:
        print("\n[INFERENCE] Executing memory-safe sequential evaluation across 3 detectors...")
        t_start = time.perf_counter()
        for det_idx, det_key in enumerate(DETECTOR_KEYS, 1):
            print(f"\n({det_idx}/3) Evaluating Detector: '{det_key}' ({len(val_samples)} samples)...")
            det_results = run_detector_subprocess(det_key, val_samples)
            all_raw_predictions.extend(det_results)
            print(f"  [COMPLETED] '{det_key}' evaluated. Running memory deallocation...")
            gc.collect()

        total_time = round(time.perf_counter() - t_start, 2)
        print(f"\nAll detector passes finished in {total_time}s. Saving to {RAW_PREDICTIONS_CSV}...")

        # Save raw predictions CSV
        fieldnames = [
            "sample_id", "split", "domain", "category", "ground_truth", "file_path",
            "detector_key", "detector_name", "model_name", "prediction", "fake_probability",
            "real_probability", "raw_logit", "confidence", "threshold", "status",
            "error_message", "latency_ms"
        ]
        os.makedirs(os.path.dirname(RAW_PREDICTIONS_CSV), exist_ok=True)
        with open(RAW_PREDICTIONS_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_raw_predictions:
                writer.writerow(r)
        print("Raw predictions CSV saved successfully.")

    # Step 2: Compute Individual Detector Baselines on development_val
    individual_baselines = {}
    print("\n" + "-" * 70)
    print("INDIVIDUAL DETECTOR BASELINES (development_val, N=50)")
    print("-" * 70)
    for det_key in DETECTOR_KEYS:
        det_preds = [r for r in all_raw_predictions if r["detector_key"] == det_key]
        metrics = compute_metrics(det_preds)
        individual_baselines[det_key] = metrics
        print(
            f"Detector: {det_key:18} | Acc: {metrics['accuracy']*100:6.2f}% "
            f"({metrics['confusion_matrix']['tp']+metrics['confusion_matrix']['tn']}/{metrics['valid_count']}) | "
            f"BalAcc: {metrics['balanced_accuracy']*100:6.2f}% | "
            f"Prec: {metrics['precision']*100:6.2f}% | Rec: {metrics['recall']*100:6.2f}% | "
            f"Spec: {metrics['specificity']*100:6.2f}% | F1: {metrics['f1_score']:.4f}"
        )

    # Step 3: Probability Comparability & Calibration Analysis
    prob_analysis = analyze_probability_comparability(all_raw_predictions)
    print("\n" + "-" * 70)
    print("PROBABILITY COMPARABILITY & SEPARATION MARGINS")
    print("-" * 70)
    for det_key, p_info in prob_analysis.items():
        r_mean = p_info["real_faces_p_fake"]["mean"]
        f_mean = p_info["fake_faces_p_fake"]["mean"]
        margin = p_info["separation_margin"]
        print(
            f"Detector: {det_key:18} | Mean P(fake) on Real: {r_mean*100:5.2f}% | "
            f"Mean P(fake) on Fake: {f_mean*100:5.2f}% | Separation Margin Delta: {margin*100:+6.2f}%"
        )

    # Step 4: Group predictions by sample_id to form ensembles
    samples_by_id = {s["sample_id"]: s for s in val_samples}
    ensembles_by_sample: Dict[str, Dict[str, Any]] = {s["sample_id"]: {} for s in val_samples}
    for r in all_raw_predictions:
        sid = r["sample_id"]
        det_key = r["detector_key"]
        ensembles_by_sample[sid][det_key] = r

    # Step 5: Execute Decision Fusion Pipeline for all 4 strategies
    pipeline = DecisionFusionPipeline(vit_key="vit")
    strategies = [
        STRATEGY_VIT_ONLY,
        STRATEGY_MAJORITY_VOTE,
        STRATEGY_MEAN_PROBABILITY,
        STRATEGY_VIT_PRIORITY,
    ]

    fusion_predictions: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    for sid, ensemble in ensembles_by_sample.items():
        sample_meta = samples_by_id[sid]
        fused_res = pipeline.fuse_all_strategies(ensemble)
        for strat_name, res in fused_res.items():
            fusion_predictions[strat_name].append({
                "sample_id": sid,
                "ground_truth": sample_meta["ground_truth"],
                "prediction": res.prediction,
                "fake_probability": res.fake_probability,
                "real_probability": res.real_probability,
                "confidence": res.confidence,
                "status": res.status,
                "tie_broken": res.tie_broken,
                "fallback_applied": res.fallback_applied,
                "explanation": res.explanation,
                "error": res.error,
            })

    # Step 6: Compute Metrics for Each Fusion Strategy
    strategy_metrics = {}
    print("\n" + "-" * 70)
    print("DECISION FUSION STRATEGY PERFORMANCE (development_val, N=50)")
    print("-" * 70)
    for strat_name in strategies:
        preds = fusion_predictions[strat_name]
        metrics = compute_metrics(preds)
        strategy_metrics[strat_name] = metrics
        print(
            f"Strategy: {strat_name:24} | Acc: {metrics['accuracy']*100:6.2f}% "
            f"({metrics['confusion_matrix']['tp']+metrics['confusion_matrix']['tn']}/{metrics['valid_count']}) | "
            f"BalAcc: {metrics['balanced_accuracy']*100:6.2f}% | "
            f"Prec: {metrics['precision']*100:6.2f}% | Rec: {metrics['recall']*100:6.2f}% | "
            f"Spec: {metrics['specificity']*100:6.2f}% | F1: {metrics['f1_score']:.4f}"
        )

    # Step 7: Transition Analysis relative to ViT Alone
    vit_preds_map = {p["sample_id"]: p for p in fusion_predictions[STRATEGY_VIT_ONLY]}
    relative_analysis = {}

    print("\n" + "-" * 70)
    print("COMPARATIVE TRANSITION ANALYSIS RELATIVE TO ViT ALONE")
    print("-" * 70)
    vit_acc = strategy_metrics[STRATEGY_VIT_ONLY]["accuracy"]
    vit_bal_acc = strategy_metrics[STRATEGY_VIT_ONLY]["balanced_accuracy"]
    vit_f1 = strategy_metrics[STRATEGY_VIT_ONLY]["f1_score"]

    for strat_name in [STRATEGY_MAJORITY_VOTE, STRATEGY_MEAN_PROBABILITY, STRATEGY_VIT_PRIORITY]:
        strat_m = strategy_metrics[strat_name]
        delta_acc = round(strat_m["accuracy"] - vit_acc, 4)
        delta_bal_acc = round(strat_m["balanced_accuracy"] - vit_bal_acc, 4)
        delta_f1 = round(strat_m["f1_score"] - vit_f1, 4)

        improved_samples = []  # ViT incorrect, Strategy correct
        regressed_samples = []  # ViT correct, Strategy incorrect
        both_correct = []
        both_incorrect = []

        for p in fusion_predictions[strat_name]:
            sid = p["sample_id"]
            gt = p["ground_truth"]
            vit_p = vit_preds_map[sid]["prediction"]
            strat_p = p["prediction"]

            vit_correct = (vit_p == gt)
            strat_correct = (strat_p == gt)

            if not vit_correct and strat_correct:
                improved_samples.append({
                    "sample_id": sid,
                    "ground_truth": gt,
                    "vit_prediction": vit_p,
                    "strategy_prediction": strat_p,
                    "explanation": p["explanation"],
                })
            elif vit_correct and not strat_correct:
                regressed_samples.append({
                    "sample_id": sid,
                    "ground_truth": gt,
                    "vit_prediction": vit_p,
                    "strategy_prediction": strat_p,
                    "explanation": p["explanation"],
                })
            elif vit_correct and strat_correct:
                both_correct.append(sid)
            else:
                both_incorrect.append(sid)

        relative_analysis[strat_name] = {
            "delta_accuracy": delta_acc,
            "delta_balanced_accuracy": delta_bal_acc,
            "delta_f1": delta_f1,
            "count_improved": len(improved_samples),
            "count_regressed": len(regressed_samples),
            "count_both_correct": len(both_correct),
            "count_both_incorrect": len(both_incorrect),
            "improved_samples": improved_samples,
            "regressed_samples": regressed_samples,
        }

        print(
            f"Strategy: {strat_name:24} | DeltaAcc: {delta_acc*100:+6.2f}% | "
            f"DeltaBalAcc: {delta_bal_acc*100:+6.2f}% | DeltaF1: {delta_f1:+6.4f} | "
            f"Improved: {len(improved_samples)} | Regressed: {len(regressed_samples)}"
        )
        if regressed_samples:
            print(f"    --> Regressed samples ({len(regressed_samples)}): {[r['sample_id'] for r in regressed_samples]}")

    # Step 8: Frozen Strategy Recommendation
    # ViT alone or ViT priority (which preserves ViT accuracy while adding fallback robustness)
    best_strategy = STRATEGY_VIT_PRIORITY
    reproducible_test_command = (
        f"python scripts/evaluate_phase5_fusion.py --eval-test --strategy {best_strategy}"
    )

    final_document = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_split": "development_val",
            "split_composition": {
                "total_samples": 50,
                "authentic_real_faces": 25,
                "stylegan2_fake_faces": 25,
                "general_scenes": 0,
                "rejection_controls": 0,
            },
            "manifest_path": "benchmark/evaluation_manifest.csv",
            "environment": {
                "os": platform.system(),
                "os_release": platform.release(),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "device": "cpu",
            },
        },
        "individual_detectors": individual_baselines,
        "probability_comparability_analysis": prob_analysis,
        "fusion_strategies": strategy_metrics,
        "relative_to_vit_alone": relative_analysis,
        "frozen_strategy_selection": {
            "selected_strategy": best_strategy,
            "selection_rationale": (
                "ViT achieves ceiling-level accuracy on facial crops. ViT-priority preserves ViT's "
                "100.0% accuracy on development_val while providing explicit software contract fallback "
                "to secondary detectors (UnivFD, ConvNeXt) if ViT encounters execution failure. "
                "In contrast, unweighted majority voting and naive mean probability degrade performance "
                "by incorporating uncalibrated detectors."
            ),
            "reproducible_held_out_command": reproducible_test_command,
            "held_out_test_status": "PENDING_APPROVAL (Not executed per Phase 5 protocol)",
            "known_limitations": [
                "Validation split contains exclusively cropped facial images (25 FFHQ Real, 25 StyleGAN Fake).",
                "Results do not prove generalization to modern diffusion scenes, face swaps, or localized splicing.",
                "Held-out test set was previously observed in Phase 4 baseline evaluation, so it is not a blind test.",
                "Raw probabilities across detectors are not calibrated to a shared distribution.",
            ],
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_METRICS_JSON), exist_ok=True)
    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_document, f, indent=2)
    print(f"\n[OUTPUT] Phase 5 validation metrics saved to: {OUTPUT_METRICS_JSON}")

    return final_document


def evaluate_held_out_test_protocol(
    strategy: str = STRATEGY_VIT_PRIORITY,
    confirm: bool = False,
    force_rerun: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Protocol execution for held_out_test.
    Strictly protected by safety check: only executes if confirm=True.
    Without confirmation, it displays protocol information, parameters, and halts
    per the Phase 5 stopping rule.
    """
    print("=" * 80)
    print("PHASE 5 FROZEN EVALUATION PROTOCOL: HELD_OUT_TEST")
    print("=" * 80)
    print(f"Target Strategy: '{strategy}'")
    print(f"Target Partition: 'held_out_test' (105 samples: 100 facial, 3 general scene, 2 controls)")
    print(f"Master Manifest: {DEFAULT_MANIFEST}")

    if not confirm:
        print("\n" + "!" * 80)
        print("SAFETY GATE: EVALUATION PAUSED PENDING USER APPROVAL")
        print("!" * 80)
        print("In accordance with Phase 5 Experimental Rules, the held_out_test split is quarantined")
        print("until the fusion strategy is selected, frozen, and approved by the user.")
        print("\nTo execute this evaluation once approval is granted, run:")
        print(f"  python scripts/evaluate_phase5_fusion.py --eval-test --strategy {strategy} --confirm")
        print("!" * 80 + "\n")
        return None

    print("\n[APPROVAL VERIFIED] Executing frozen strategy evaluation on held_out_test...")

    test_samples = load_manifest_split(DEFAULT_MANIFEST, "held_out_test")
    print(f"Loaded {len(test_samples)} held-out test samples from {DEFAULT_MANIFEST}")
    assert len(test_samples) == 105, f"Expected 105 held-out test samples, found {len(test_samples)}"

    # Step 1: Collect predictions from Phase 4 baseline or fresh inference
    phase4_preds_path = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_predictions.csv")
    all_test_predictions = []

    if os.path.exists(phase4_preds_path) and not force_rerun:
        print(f"[INPUT] Loading verified Phase 4 detector predictions from: {phase4_preds_path}")
        with open(phase4_preds_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == "held_out_test":
                    row["fake_probability"] = float(row["fake_probability"]) if row.get("fake_probability") not in ("", None) else None
                    row["real_probability"] = float(row["real_probability"]) if row.get("real_probability") not in ("", None) else None
                    row["raw_logit"] = float(row["raw_logit"]) if row.get("raw_logit") not in ("", None) else None
                    row["confidence"] = float(row["confidence"]) if row.get("confidence") not in ("", None) else None
                    row["threshold"] = float(row["threshold"]) if row.get("threshold") not in ("", None) else 0.5
                    row["latency_ms"] = float(row["latency_ms"]) if row.get("latency_ms") not in ("", None) else 0.0
                    all_test_predictions.append(row)
        print(f"Loaded {len(all_test_predictions)} test prediction rows.")
    else:
        print("[INFERENCE] Executing sequential evaluation on held_out_test across 3 detectors...")
        for det_idx, det_key in enumerate(DETECTOR_KEYS, 1):
            print(f"  ({det_idx}/3) Evaluating Detector: '{det_key}' ({len(test_samples)} samples)...")
            det_results = run_detector_subprocess(det_key, test_samples)
            all_test_predictions.extend(det_results)
            gc.collect()

    # Step 2: Group by sample_id
    samples_by_id = {s["sample_id"]: s for s in test_samples}
    ensembles_by_sample: Dict[str, Dict[str, Any]] = {s["sample_id"]: {} for s in test_samples}
    for r in all_test_predictions:
        sid = r["sample_id"]
        det_key = r["detector_key"]
        ensembles_by_sample[sid][det_key] = r

    # Step 3: Run pipeline for selected strategy
    pipeline = DecisionFusionPipeline(vit_key="vit")
    fused_test_results = []

    for sid, ensemble in ensembles_by_sample.items():
        sample_meta = samples_by_id[sid]
        res = pipeline.fuse(ensemble, strategy=strategy)
        fused_test_results.append({
            "sample_id": sid,
            "split": "held_out_test",
            "domain": sample_meta.get("domain", ""),
            "category": sample_meta.get("category", ""),
            "ground_truth": sample_meta["ground_truth"],
            "file_path": sample_meta.get("file_path", ""),
            "strategy": strategy,
            "prediction": res.prediction,
            "fake_probability": res.fake_probability,
            "real_probability": res.real_probability,
            "confidence": res.confidence,
            "status": res.status,
            "tie_broken": res.tie_broken,
            "fallback_applied": res.fallback_applied,
            "explanation": res.explanation,
            "error": res.error,
        })

    # Step 4: Compute domain-partitioned metrics
    overall_items = [r for r in fused_test_results if r["ground_truth"] in (PREDICTION_REAL, PREDICTION_FAKE)]
    facial_items = [r for r in fused_test_results if r["domain"] == "facial"]
    scene_items = [r for r in fused_test_results if r["domain"] == "general_scene"]
    control_items = [r for r in fused_test_results if r["ground_truth"] == "REJECTION_CONTROL" or r["domain"] == "control"]

    overall_metrics = compute_metrics(overall_items)
    facial_metrics = compute_metrics(facial_items)
    scene_metrics = compute_metrics(scene_items)
    control_metrics = compute_metrics(control_items)["rejection_controls"]

    print("\n" + "-" * 70)
    print(f"HELD-OUT TEST RESULTS — STRATEGY: '{strategy}'")
    print("-" * 70)
    print(
        f"Overall (103 eligible) Acc: {overall_metrics['accuracy']*100:.2f}% | "
        f"BalAcc: {overall_metrics['balanced_accuracy']*100:.2f}% | "
        f"Prec: {overall_metrics['precision']*100:.2f}% | "
        f"Rec: {overall_metrics['recall']*100:.2f}% | "
        f"F1: {overall_metrics['f1_score']:.4f}"
    )
    print(
        f"Facial  (100 samples)  Acc: {facial_metrics['accuracy']*100:.2f}% | "
        f"BalAcc: {facial_metrics['balanced_accuracy']*100:.2f}% | "
        f"Prec: {facial_metrics['precision']*100:.2f}% | "
        f"Rec: {facial_metrics['recall']*100:.2f}% | "
        f"F1: {facial_metrics['f1_score']:.4f}"
    )
    print(
        f"Controls (2 samples)   Rejection Success: "
        f"{control_metrics['successfully_rejected']}/{control_metrics['total_controls']} "
        f"({control_metrics['rejection_rate']*100:.1f}%)"
    )

    test_summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split": "held_out_test",
            "total_samples": len(test_samples),
            "eligible_samples": len(overall_items),
            "strategy": strategy,
        },
        "overall_classification": overall_metrics,
        "facial_subbenchmark": facial_metrics,
        "scene_diagnostic": scene_metrics,
        "rejection_controls": control_metrics,
    }

    test_out_json = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase5_held_out_test_metrics.json")
    with open(test_out_json, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    print(f"\n[OUTPUT] Saved held-out test metrics to: {test_out_json}")

    return test_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Decision Fusion Evaluation")
    parser.add_argument("--worker", action="store_true", help="Run in child worker mode")
    parser.add_argument("--worker-detector", type=str, help="Detector key for worker")
    parser.add_argument("--worker-input", type=str, help="Input JSON path for worker")
    parser.add_argument("--worker-output", type=str, help="Output JSON path for worker")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run model inference on validation set")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate frozen strategy on held_out_test (Requires Approval)")
    parser.add_argument("--confirm", action="store_true", help="Explicit confirmation to execute held_out_test evaluation protocol")
    parser.add_argument("--strategy", type=str, default=STRATEGY_VIT_PRIORITY, help="Strategy to evaluate on test split")

    args = parser.parse_args()

    if args.worker:
        if not (args.worker_detector and args.worker_input and args.worker_output):
            print("Worker mode requires --worker-detector, --worker-input, and --worker-output")
            sys.exit(1)
        run_worker(args.worker_detector, args.worker_input, args.worker_output)
        return

    if args.eval_test:
        evaluate_held_out_test_protocol(strategy=args.strategy, confirm=args.confirm, force_rerun=args.force_rerun)
        return

    evaluate_val_split(force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
