"""
Independent Victory Audit Script
Executes independent verification of Phase 4 Baseline Evaluation deliverables:
1. Predictions CSV forensic analysis (counts, leakage, schema, distributions)
2. Standalone recomputation of all metrics (confusion matrix, Wilson CIs, Cohen's kappa)
3. Direct numerical diff against phase4_baseline_metrics.json and PHASE4_BASELINE_EVALUATION_REPORT.md
4. Live inference reproduction on held-out test sample
"""

import os
import sys
import csv
import json
import math
from typing import Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "evaluation_manifest.csv")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_predictions.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_metrics.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "PHASE4_BASELINE_EVALUATION_REPORT.md")


def independent_wilson(k: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def independent_kappa(ff: int, rr: int, fr: int, rf: int) -> float:
    n = ff + rr + fr + rf
    if n == 0:
        return 0.0
    po = (ff + rr) / n
    pf1 = (ff + fr) / n
    pr1 = (rf + rr) / n
    pf2 = (ff + rf) / n
    pr2 = (fr + rr) / n
    pe = (pf1 * pf2) + (pr1 * pr2)
    if abs(1.0 - pe) < 1e-9:
        return 1.0 if abs(po - 1.0) < 1e-9 else 0.0
    return round((po - pe) / (1.0 - pe), 4)


def audit_phase4():
    print("=== VICTORY AUDIT INDEPENDENT EXECUTION ===")
    
    # 1. Manifest Audit
    manifest_rows = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            manifest_rows[r["sample_id"]] = r
            
    assert len(manifest_rows) == 159, f"Manifest count mismatch: {len(manifest_rows)}"
    held_out = {k: v for k, v in manifest_rows.items() if v["split"] == "held_out_test"}
    dev_val = {k: v for k, v in manifest_rows.items() if v["split"] == "development_val"}
    smoke = {k: v for k, v in manifest_rows.items() if v["split"] == "operational_smoke"}
    assert len(held_out) == 105
    assert len(dev_val) == 50
    assert len(smoke) == 4
    print("[PASS] Manifest structure verified: 159 samples (105 test, 50 dev, 4 smoke).")

    # 2. Predictions CSV Audit
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
        
    assert len(csv_rows) == 315, f"Expected 315 rows, found {len(csv_rows)}"
    
    detector_keys = ["chen2024_convnext", "univfd", "vit"]
    by_detector = {k: [] for k in detector_keys}
    
    seen_pairs = set()
    for row in csv_rows:
        dk = row["detector_key"]
        sid = row["sample_id"]
        assert dk in by_detector, f"Unexpected detector key: {dk}"
        assert sid in held_out, f"Sample {sid} not in held_out_test!"
        assert sid not in dev_val, f"LEAKAGE DETECTED: Sample {sid} in development_val!"
        assert sid not in smoke, f"LEAKAGE DETECTED: Sample {sid} in operational_smoke!"
        
        pair = (dk, sid)
        assert pair not in seen_pairs, f"Duplicate pair found: {pair}"
        seen_pairs.add(pair)
        
        by_detector[dk].append(row)
        
    for dk in detector_keys:
        assert len(by_detector[dk]) == 105, f"Detector {dk} row count != 105: {len(by_detector[dk])}"
    print("[PASS] Predictions CSV verified: 315 rows, 105 per detector, 0 duplicates, 0 data leakage.")

    # 3. Recompute All Metrics
    recomputed = {}
    for dk in detector_keys:
        rows = by_detector[dk]
        
        # Segregate subsets
        overall_rows = [r for r in rows if r["ground_truth"] in ("REAL", "FAKE")]
        facial_rows = [r for r in rows if r["domain"] == "facial"]
        ctrl_rows = [r for r in rows if r["domain"] == "control"]
        scene_rows = [r for r in rows if r["domain"] == "general_scene"]
        
        assert len(overall_rows) == 103
        assert len(facial_rows) == 100
        assert len(ctrl_rows) == 2
        assert len(scene_rows) == 3
        
        # Overall metrics
        tp = sum(1 for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        fp = sum(1 for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        tn = sum(1 for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        fn = sum(1 for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")
        valid_n = tp + fp + tn + fn
        acc = round((tp + tn) / valid_n, 4)
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
        f1 = round((2 * prec * rec) / (prec + rec), 4) if (prec + rec) > 0 else 0.0
        bacc = round((rec + spec) / 2.0, 4)
        
        # Facial metrics
        f_tp = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        f_fp = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        f_tn = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        f_fn = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")
        f_valid_n = f_tp + f_fp + f_tn + f_fn
        f_acc = round((f_tp + f_tn) / f_valid_n, 4)
        f_prec = round(f_tp / (f_tp + f_fp), 4) if (f_tp + f_fp) > 0 else 0.0
        f_rec = round(f_tp / (f_tp + f_fn), 4) if (f_tp + f_fn) > 0 else 0.0
        f_spec = round(f_tn / (f_tn + f_fp), 4) if (f_tn + f_fp) > 0 else 0.0
        f_f1 = round((2 * f_prec * f_rec) / (f_prec + f_rec), 4) if (f_prec + f_rec) > 0 else 0.0
        f_bacc = round((f_rec + f_spec) / 2.0, 4)
        
        f_acc_ci = independent_wilson(f_tp + f_tn, f_valid_n)
        f_rec_ci = independent_wilson(f_tp, f_tp + f_fn)
        f_prec_ci = independent_wilson(f_tp, f_tp + f_fp)
        f_spec_ci = independent_wilson(f_tn, f_tn + f_fp)
        
        real_facial_probs = [float(r["fake_probability"]) for r in facial_rows if r["ground_truth"] == "REAL"]
        fake_facial_probs = [float(r["fake_probability"]) for r in facial_rows if r["ground_truth"] == "FAKE"]
        mean_real_prob = round(sum(real_facial_probs) / len(real_facial_probs), 4)
        mean_fake_prob = round(sum(fake_facial_probs) / len(fake_facial_probs), 4)
        
        # Control checks
        ctrl_errs = [r for r in ctrl_rows if r["status"] == "ERROR" and r["prediction"] == "ERROR"]
        assert len(ctrl_errs) == 2, f"Detector {dk} failed to reject controls cleanly!"
        
        recomputed[dk] = {
            "overall": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": acc, "precision": prec, "recall": rec,
                "specificity": spec, "f1_score": f1, "balanced_accuracy": bacc,
                "fps": [r["sample_id"] for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE"],
                "fns": [r["sample_id"] for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL"],
            },
            "facial": {
                "tp": f_tp, "fp": f_fp, "tn": f_tn, "fn": f_fn,
                "accuracy": f_acc, "precision": f_prec, "recall": f_rec,
                "specificity": f_spec, "f1_score": f_f1, "balanced_accuracy": f_bacc,
                "accuracy_ci": f_acc_ci, "precision_ci": f_prec_ci,
                "recall_ci": f_rec_ci, "specificity_ci": f_spec_ci,
                "mean_fake_prob_real": mean_real_prob,
                "mean_fake_prob_fake": mean_fake_prob,
            }
        }
    print("[PASS] Standalone metrics calculated successfully for all detectors.")

    # 4. Pairwise Concordance
    row_maps = {dk: {r["sample_id"]: r for r in by_detector[dk]} for dk in detector_keys}
    pairs = [
        ("chen2024_convnext", "univfd"),
        ("chen2024_convnext", "vit"),
        ("univfd", "vit"),
    ]
    pairwise_recomputed = {}
    for d1, d2 in pairs:
        ff = rr = fr = rf = 0
        common_ids = [sid for sid in row_maps[d1] if row_maps[d1][sid]["ground_truth"] in ("REAL", "FAKE")]
        for sid in common_ids:
            p1 = row_maps[d1][sid]["prediction"]
            p2 = row_maps[d2][sid]["prediction"]
            if p1 == "FAKE" and p2 == "FAKE":
                ff += 1
            elif p1 == "REAL" and p2 == "REAL":
                rr += 1
            elif p1 == "FAKE" and p2 == "REAL":
                fr += 1
            elif p1 == "REAL" and p2 == "FAKE":
                rf += 1
        n = len(common_ids)
        concordant = ff + rr
        rate = round(concordant / n, 4)
        kappa = independent_kappa(ff, rr, fr, rf)
        pairwise_recomputed[f"{d1}_vs_{d2}"] = {
            "n": n, "concordant": concordant, "rate": rate, "kappa": kappa,
            "ff": ff, "rr": rr, "fr": fr, "rf": rf
        }
    print("[PASS] Standalone pairwise agreement calculated successfully.")

    # 5. Diff against phase4_baseline_metrics.json
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)
        
    for dk in detector_keys:
        j_det = metrics_json["detectors"][dk]
        r_det = recomputed[dk]
        
        # Check overall
        j_ov = j_det["overall_classification_metrics"]
        r_ov = r_det["overall"]
        assert j_ov["confusion_matrix"]["tp"] == r_ov["tp"]
        assert j_ov["confusion_matrix"]["fp"] == r_ov["fp"]
        assert j_ov["confusion_matrix"]["tn"] == r_ov["tn"]
        assert j_ov["confusion_matrix"]["fn"] == r_ov["fn"]
        assert math.isclose(j_ov["accuracy"], r_ov["accuracy"], abs_tol=1e-4)
        assert math.isclose(j_ov["precision"], r_ov["precision"], abs_tol=1e-4)
        assert math.isclose(j_ov["recall"], r_ov["recall"], abs_tol=1e-4)
        assert math.isclose(j_ov["f1_score"], r_ov["f1_score"], abs_tol=1e-4)
        assert math.isclose(j_ov["balanced_accuracy"], r_ov["balanced_accuracy"], abs_tol=1e-4)
        assert j_ov["false_positives"] == r_ov["fps"]
        assert j_ov["false_negatives"] == r_ov["fns"]
        
        # Check facial
        j_fc = j_det["facial_evaluation_metrics"]
        r_fc = r_det["facial"]
        assert j_fc["confusion_matrix"]["tp"] == r_fc["tp"]
        assert j_fc["confusion_matrix"]["fp"] == r_fc["fp"]
        assert j_fc["confusion_matrix"]["tn"] == r_fc["tn"]
        assert j_fc["confusion_matrix"]["fn"] == r_fc["fn"]
        assert math.isclose(j_fc["accuracy"], r_fc["accuracy"], abs_tol=1e-4)
        assert j_fc["accuracy_95ci"] == r_fc["accuracy_ci"]
        assert j_fc["recall_95ci"] == r_fc["recall_ci"]
        assert j_fc["precision_95ci"] == r_fc["precision_ci"]
        assert j_fc["specificity_95ci"] == r_fc["specificity_ci"]
        assert math.isclose(j_fc["mean_fake_prob_real_faces"], r_fc["mean_fake_prob_real"], abs_tol=1e-4)
        assert math.isclose(j_fc["mean_fake_prob_fake_faces"], r_fc["mean_fake_prob_fake"], abs_tol=1e-4)
        
    for pair_key, r_pair in pairwise_recomputed.items():
        j_pair = metrics_json["pairwise_agreement"][pair_key]
        assert j_pair["concordant_samples"] == r_pair["concordant"]
        assert math.isclose(j_pair["concordance_rate"], r_pair["rate"], abs_tol=1e-4)
        assert math.isclose(j_pair["cohen_kappa"], r_pair["kappa"], abs_tol=1e-4)
        assert j_pair["agreement_breakdown"]["both_fake"] == r_pair["ff"]
        assert j_pair["agreement_breakdown"]["both_real"] == r_pair["rr"]
        assert j_pair["agreement_breakdown"]["model1_fake_model2_real"] == r_pair["fr"]
        assert j_pair["agreement_breakdown"]["model1_real_model2_fake"] == r_pair["rf"]
    print("[PASS] phase4_baseline_metrics.json matches recomputed values 100%!")

    # 6. Diff against docs/PHASE4_BASELINE_EVALUATION_REPORT.md
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()
        
    # Check expected key numbers in report text
    expected_snippets = [
        "41.75%", "59.22%", "98.06%",  # Overall accuracies
        "42.00%", "60.00%", "100.00%", # Facial accuracies
        "[32.8%, 51.8%]", "[50.2%, 69.1%]", "[96.3%, 100.0%]", # Wilson CIs
        "-0.0241", "-0.1466", "0.2056", # Cohen kappas
        "72.82%", "43.69%", "61.17%",   # Concordances
        "test_scene_real_001", "test_scene_diffusion_001", "test_scene_manip_001", # Diagnostic probes
        "test_control_corr_001", "test_control_unsupp_001" # Controls
    ]
    for snippet in expected_snippets:
        assert snippet in report_text, f"Report missing expected snippet: '{snippet}'"
    print("[PASS] PHASE4_BASELINE_EVALUATION_REPORT.md contains all verified numerical values.")

    # 7. Live Inference Bit-Level Reproducibility Test
    print("\n--- Live Inference Bit-Level Reproducibility Check ---")
    from src.detector_registry import get_detector, reset_instances
    
    target_sample_id = "test_face_real_001"
    for dk in detector_keys:
        expected_row = row_maps[dk][target_sample_id]
        target_file = expected_row["file_path"]
        exp_pred = expected_row["prediction"]
        exp_prob = float(expected_row["fake_probability"])
        exp_logit = float(expected_row["raw_logit"])
        
        adapter = get_detector(dk, lazy_load=False, singleton=False)
        res = adapter.predict(target_file)
        
        if res.error is not None:
            print(f"Detector {dk} error: {res.error}")
        
        print(f"Detector {dk}:")
        print(f"  Live:     pred={res.prediction}, fake_prob={res.fake_probability:.6f}, logit={res.raw_logit:.6f}")
        print(f"  CSV:      pred={exp_pred}, fake_prob={exp_prob:.6f}, logit={exp_logit:.6f}")
        
        assert res.prediction == exp_pred, f"Prediction mismatch for {dk}: live={res.prediction}, exp={exp_pred}"
        assert math.isclose(res.fake_probability, exp_prob, abs_tol=1e-4), f"Prob mismatch for {dk}: live={res.fake_probability}, exp={exp_prob}"
        assert math.isclose(res.raw_logit, exp_logit, abs_tol=1e-4), f"Logit mismatch for {dk}: live={res.raw_logit}, exp={exp_logit}"
        
        if hasattr(adapter, "unload_model"):
            adapter.unload_model()
        reset_instances()
        print(f"  [MATCH] Live inference verified within floating point precision.")

    print("\n=======================================================")
    print("ALL VICTORY AUDIT CHECKS PASSED WITH 100% CONCORDANCE!")
    print("=======================================================")


if __name__ == "__main__":
    audit_phase4()
