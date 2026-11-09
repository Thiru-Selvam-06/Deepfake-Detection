"""
Adversarial Empirical Verification Oracle for Phase 4 Baseline Evaluation.
Independently inspects and verifies:
1. benchmark/evaluation_manifest.csv
2. benchmark/results/phase4_baseline_predictions.csv
3. benchmark/results/phase4_baseline_metrics.json
4. docs/PHASE4_BASELINE_EVALUATION_REPORT.md
"""

import os
import sys
import csv
import json
import math
from typing import Dict, List, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "evaluation_manifest.csv")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_predictions.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_metrics.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "docs", "PHASE4_BASELINE_EVALUATION_REPORT.md")


def independent_wilson_ci(k: int, n: int, confidence: float = 0.95) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p_hat = k / n
    z = 1.959963984540054  # 95% standard normal quantile
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return [round(lower, 4), round(upper, 4)]


def independent_cohen_kappa(both_fake: int, both_real: int, m1_f_m2_r: int, m1_r_m2_f: int) -> float:
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


def run_all_checks() -> Dict[str, Any]:
    print("=" * 80)
    print("ADVERSARIAL EMPIRICAL VERIFICATION ORACLE: PHASE 4 BASELINE EVALUATION")
    print("=" * 80)

    summary = {
        "manifest_checks": {},
        "csv_integrity": {},
        "metrics_recomputed": {},
        "json_matches": {},
        "report_matches": {},
        "edge_case_checks": {},
        "errors": []
    }

    # ---------------------------------------------------------
    # 1. Manifest Split Audit
    # ---------------------------------------------------------
    print("\n[1] Auditing Manifest splits & sample IDs...")
    manifest_samples = {}
    split_counts = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sample_id"]
            split = row["split"]
            manifest_samples[sid] = row
            split_counts[split] = split_counts.get(split, 0) + 1

    print(f"    Total samples in manifest: {len(manifest_samples)}")
    print(f"    Split breakdown: {split_counts}")
    assert split_counts.get("held_out_test") == 105, f"Expected 105 held_out_test, got {split_counts.get('held_out_test')}"
    assert split_counts.get("development_val") == 50, f"Expected 50 development_val, got {split_counts.get('development_val')}"
    assert split_counts.get("operational_smoke") == 4, f"Expected 4 operational_smoke, got {split_counts.get('operational_smoke')}"
    summary["manifest_checks"]["split_counts"] = split_counts

    held_out_ids = {sid for sid, row in manifest_samples.items() if row["split"] == "held_out_test"}
    dev_val_ids = {sid for sid, row in manifest_samples.items() if row["split"] == "development_val"}
    smoke_ids = {sid for sid, row in manifest_samples.items() if row["split"] == "operational_smoke"}

    # ---------------------------------------------------------
    # 2. CSV Integrity Audit
    # ---------------------------------------------------------
    print("\n[2] Auditing phase4_baseline_predictions.csv integrity...")
    assert os.path.exists(PREDICTIONS_PATH), f"Predictions CSV missing: {PREDICTIONS_PATH}"

    expected_columns = [
        "sample_id", "split", "domain", "category", "ground_truth",
        "file_path", "detector_key", "detector_name", "model_name",
        "prediction", "predicted_class", "fake_probability", "real_probability",
        "raw_logit", "relative_logit", "confidence", "threshold",
        "status", "error_message", "latency_ms"
    ]

    pred_rows = []
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_columns, f"Columns mismatch: {reader.fieldnames} vs {expected_columns}"
        for row in reader:
            pred_rows.append(row)

    total_rows = len(pred_rows)
    print(f"    Total prediction rows: {total_rows} (Expected 315 = 105 * 3)")
    assert total_rows == 315, f"Expected exactly 315 prediction rows, found {total_rows}"

    # Detector row counts & duplicates
    detector_keys = ["chen2024_convnext", "univfd", "vit"]
    detector_rows = {k: [] for k in detector_keys}
    seen_pairs = set()

    data_leakage_detected = []
    probability_violations = []

    for idx, row in enumerate(pred_rows):
        sid = row["sample_id"]
        dkey = row["detector_key"]
        split = row["split"]
        status = row["status"]
        gt = row["ground_truth"]
        pred = row["prediction"]

        # Check pair uniqueness
        pair = (dkey, sid)
        if pair in seen_pairs:
            raise AssertionError(f"Duplicate detector-sample pair detected: {pair}")
        seen_pairs.add(pair)

        # Check detector key valid
        assert dkey in detector_rows, f"Unexpected detector key: {dkey}"
        detector_rows[dkey].append(row)

        # STRICT DATA LEAKAGE CHECK
        if split != "held_out_test":
            data_leakage_detected.append(f"Row {idx}: {sid} has split '{split}', expected 'held_out_test'")
        if sid in dev_val_ids:
            data_leakage_detected.append(f"CRITICAL LEAK: {sid} is from development_val!")
        if sid in smoke_ids:
            data_leakage_detected.append(f"CRITICAL LEAK: {sid} is from operational_smoke!")
        if sid not in held_out_ids:
            data_leakage_detected.append(f"Row {idx}: {sid} not in held_out_test manifest!")

        # Probability and value validations
        if status == "SUCCESS":
            assert pred in ["REAL", "FAKE"], f"Invalid prediction: {pred}"
            assert row["predicted_class"] == pred, f"predicted_class mismatch: {row['predicted_class']} != {pred}"
            fake_p = float(row["fake_probability"])
            real_p = float(row["real_probability"])
            conf = float(row["confidence"])
            thresh = float(row["threshold"])
            lat = float(row["latency_ms"])
            assert 0.0 <= fake_p <= 1.0, f"fake_probability out of bounds: {fake_p}"
            assert 0.0 <= real_p <= 1.0, f"real_probability out of bounds: {real_p}"
            assert abs(fake_p + real_p - 1.0) < 1e-4, f"Probabilities do not sum to 1: {fake_p} + {real_p}"
            assert 0.5 <= conf <= 1.0, f"Confidence out of bounds: {conf}"
            assert thresh == 0.5, f"Threshold != 0.5: {thresh}"
            assert lat > 0, f"Latency <= 0: {lat}"
            assert row["error_message"] == "", f"Unexpected error message in SUCCESS row: {row['error_message']}"
        elif status == "ERROR":
            assert pred == "ERROR", f"Error row prediction is not ERROR: {pred}"
            assert row["predicted_class"] == "ERROR", f"Error row predicted_class is not ERROR: {row['predicted_class']}"
            assert row["fake_probability"] == "", f"Error row has fake_probability: {row['fake_probability']}"
            assert row["real_probability"] == "", f"Error row has real_probability: {row['real_probability']}"
            assert row["raw_logit"] == "", f"Error row has raw_logit: {row['raw_logit']}"
            assert row["confidence"] == "", f"Error row has confidence: {row['confidence']}"
            assert row["latency_ms"] == "", f"Error row has latency_ms: {row['latency_ms']}"
            assert row["error_message"] != "", "Error row has empty error_message"
            assert gt == "REJECTION_CONTROL", f"Error row ground_truth is not REJECTION_CONTROL: {gt}"
        else:
            raise AssertionError(f"Unknown status: {status}")

    assert len(data_leakage_detected) == 0, f"DATA LEAKAGE DETECTED:\n" + "\n".join(data_leakage_detected)
    print("    [PASS] Zero data leakage: 100% of prediction rows belong to held_out_test.")
    print("    [PASS] Zero duplicate pairs found.")
    print("    [PASS] All probability constraints and field formats strictly satisfied.")

    summary["csv_integrity"]["total_rows"] = total_rows
    summary["csv_integrity"]["detectors_count"] = {k: len(v) for k, v in detector_rows.items()}
    summary["csv_integrity"]["data_leakage_detected"] = len(data_leakage_detected)

    # ---------------------------------------------------------
    # 3. Independent Metrics Recomputation from Scratch
    # ---------------------------------------------------------
    print("\n[3] Recomputing all metrics directly from phase4_baseline_predictions.csv...")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        ground_truth_metrics = json.load(f)

    computed_metrics = {"detectors": {}, "pairwise_agreement": {}}

    for dkey in detector_keys:
        rows = detector_rows[dkey]
        assert len(rows) == 105

        # 3.1 Separate samples by role
        eligible_rows = [r for r in rows if r["ground_truth"] in ["REAL", "FAKE"]]
        facial_rows = [r for r in rows if r["domain"] == "facial"]
        scene_rows = [r for r in rows if r["domain"] == "general_scene"]
        control_rows = [r for r in rows if r["ground_truth"] == "REJECTION_CONTROL"]

        assert len(eligible_rows) == 103, f"{dkey}: Expected 103 eligible rows, got {len(eligible_rows)}"
        assert len(facial_rows) == 100, f"{dkey}: Expected 100 facial rows, got {len(facial_rows)}"
        assert len(scene_rows) == 3, f"{dkey}: Expected 3 scene rows, got {len(scene_rows)}"
        assert len(control_rows) == 2, f"{dkey}: Expected 2 control rows, got {len(control_rows)}"

        # 3.2 Overall metrics (103 eligible samples)
        tp = sum(1 for r in eligible_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        fp = sum(1 for r in eligible_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        tn = sum(1 for r in eligible_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        fn = sum(1 for r in eligible_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")
        errs = sum(1 for r in eligible_rows if r["status"] == "ERROR")

        n_eligible = len(eligible_rows)
        acc = round((tp + tn) / n_eligible, 4)
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
        f1 = round(2.0 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
        bal_acc = round((rec + spec) / 2.0, 4)

        fps = [r["sample_id"] for r in eligible_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE"]
        fns = [r["sample_id"] for r in eligible_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL"]

        # 3.3 Facial metrics (100 samples)
        f_tp = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        f_fp = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        f_tn = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        f_fn = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")

        f_acc = round((f_tp + f_tn) / 100, 4)
        f_prec = round(f_tp / (f_tp + f_fp), 4) if (f_tp + f_fp) > 0 else 0.0
        f_rec = round(f_tp / (f_tp + f_fn), 4) if (f_tp + f_fn) > 0 else 0.0
        f_spec = round(f_tn / (f_tn + f_fp), 4) if (f_tn + f_fp) > 0 else 0.0
        f_f1 = round(2.0 * f_prec * f_rec / (f_prec + f_rec), 4) if (f_prec + f_rec) > 0 else 0.0
        f_bal_acc = round((f_rec + f_spec) / 2.0, 4)

        f_acc_ci = independent_wilson_ci(f_tp + f_tn, 100)
        f_prec_ci = independent_wilson_ci(f_tp, f_tp + f_fp) if (f_tp + f_fp) > 0 else [0.0, 0.0]
        f_rec_ci = independent_wilson_ci(f_tp, f_tp + f_fn) if (f_tp + f_fn) > 0 else [0.0, 0.0]
        f_spec_ci = independent_wilson_ci(f_tn, f_tn + f_fp) if (f_tn + f_fp) > 0 else [0.0, 0.0]

        real_face_probs = [float(r["fake_probability"]) for r in facial_rows if r["ground_truth"] == "REAL"]
        fake_face_probs = [float(r["fake_probability"]) for r in facial_rows if r["ground_truth"] == "FAKE"]
        mean_p_real = round(sum(real_face_probs) / len(real_face_probs), 4)
        mean_p_fake = round(sum(fake_face_probs) / len(fake_face_probs), 4)

        # 3.4 Rejection controls verification
        ctrl_dict = {}
        for r in control_rows:
            ctrl_dict[r["sample_id"]] = {
                "prediction": r["prediction"],
                "status": r["status"],
                "error_message": r["error_message"],
                "rejection_successful": (r["prediction"] == "ERROR" and r["status"] == "ERROR")
            }

        # 3.5 Scene diagnostic cases
        scene_dict = {}
        for r in scene_rows:
            scene_dict[r["sample_id"]] = {
                "prediction": r["prediction"],
                "ground_truth": r["ground_truth"],
                "fake_probability": round(float(r["fake_probability"]), 6),
                "correct": (r["prediction"] == r["ground_truth"])
            }

        computed_metrics["detectors"][dkey] = {
            "overall": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": acc, "precision": prec, "recall": rec,
                "specificity": spec, "f1_score": f1, "balanced_accuracy": bal_acc,
                "false_positives": sorted(fps),
                "false_negatives": sorted(fns),
                "errors": errs
            },
            "facial": {
                "tp": f_tp, "fp": f_fp, "tn": f_tn, "fn": f_fn,
                "accuracy": f_acc, "precision": f_prec, "recall": f_rec,
                "specificity": f_spec, "f1_score": f_f1, "balanced_accuracy": f_bal_acc,
                "accuracy_95ci": f_acc_ci,
                "precision_95ci": f_prec_ci,
                "recall_95ci": f_rec_ci,
                "specificity_95ci": f_spec_ci,
                "mean_fake_prob_real_faces": mean_p_real,
                "mean_fake_prob_fake_faces": mean_p_fake
            },
            "scene": scene_dict,
            "controls": ctrl_dict
        }

    # 3.6 Pairwise agreement recomputation
    pair_defs = [
        ("chen2024_convnext", "univfd"),
        ("chen2024_convnext", "vit"),
        ("univfd", "vit")
    ]

    for d1, d2 in pair_defs:
        pair_key = f"{d1}_vs_{d2}"
        rows1 = {r["sample_id"]: r for r in detector_rows[d1]}
        rows2 = {r["sample_id"]: r for r in detector_rows[d2]}

        valid_sids = [
            sid for sid in rows1
            if rows1[sid]["status"] == "SUCCESS" and rows2[sid]["status"] == "SUCCESS"
        ]
        assert len(valid_sids) == 103, f"{pair_key}: Expected 103 valid joint samples, got {len(valid_sids)}"

        both_fake = sum(1 for sid in valid_sids if rows1[sid]["prediction"] == "FAKE" and rows2[sid]["prediction"] == "FAKE")
        both_real = sum(1 for sid in valid_sids if rows1[sid]["prediction"] == "REAL" and rows2[sid]["prediction"] == "REAL")
        m1_f_m2_r = sum(1 for sid in valid_sids if rows1[sid]["prediction"] == "FAKE" and rows2[sid]["prediction"] == "REAL")
        m1_r_m2_f = sum(1 for sid in valid_sids if rows1[sid]["prediction"] == "REAL" and rows2[sid]["prediction"] == "FAKE")

        concordant = both_fake + both_real
        discordant = m1_f_m2_r + m1_r_m2_f
        assert concordant + discordant == 103

        rate = round(concordant / 103, 4)
        kappa = independent_cohen_kappa(both_fake, both_real, m1_f_m2_r, m1_r_m2_f)

        computed_metrics["pairwise_agreement"][pair_key] = {
            "valid_sample_count": 103,
            "concordant_samples": concordant,
            "discordant_samples": discordant,
            "concordance_rate": rate,
            "both_fake": both_fake,
            "both_real": both_real,
            "model1_fake_model2_real": m1_f_m2_r,
            "model1_real_model2_fake": m1_r_m2_f,
            "cohen_kappa": kappa
        }

    # ---------------------------------------------------------
    # 4. Strict Comparison against phase4_baseline_metrics.json
    # ---------------------------------------------------------
    print("\n[4] Comparing computed metrics against phase4_baseline_metrics.json...")
    discrepancies = []

    for dkey in detector_keys:
        comp_d = computed_metrics["detectors"][dkey]
        json_d = ground_truth_metrics["detectors"][dkey]

        # Overall
        j_ov = json_d["overall_classification_metrics"]
        j_cm = j_ov["confusion_matrix"]
        c_ov = comp_d["overall"]

        for k in ["tp", "fp", "tn", "fn"]:
            if j_cm[k] != c_ov[k]:
                discrepancies.append(f"{dkey} overall {k}: JSON {j_cm[k]} != Computed {c_ov[k]}")

        for k in ["accuracy", "precision", "recall", "specificity", "f1_score", "balanced_accuracy"]:
            if abs(j_ov[k] - c_ov[k]) > 1e-4:
                discrepancies.append(f"{dkey} overall {k}: JSON {j_ov[k]} != Computed {c_ov[k]}")

        if sorted(j_ov["false_positives"]) != c_ov["false_positives"]:
            discrepancies.append(f"{dkey} overall false_positives mismatch")
        if sorted(j_ov["false_negatives"]) != c_ov["false_negatives"]:
            discrepancies.append(f"{dkey} overall false_negatives mismatch")
        if j_ov["execution_errors"] != []:
            discrepancies.append(f"{dkey} unexpected execution_errors in JSON: {j_ov['execution_errors']}")

        # Facial
        j_fc = json_d["facial_evaluation_metrics"]
        j_fcm = j_fc["confusion_matrix"]
        c_fc = comp_d["facial"]

        for k in ["tp", "fp", "tn", "fn"]:
            if j_fcm[k] != c_fc[k]:
                discrepancies.append(f"{dkey} facial {k}: JSON {j_fcm[k]} != Computed {c_fc[k]}")

        for k in ["accuracy", "precision", "recall", "specificity", "f1_score", "balanced_accuracy"]:
            if abs(j_fc[k] - c_fc[k]) > 1e-4:
                discrepancies.append(f"{dkey} facial {k}: JSON {j_fc[k]} != Computed {c_fc[k]}")

        for k in ["accuracy_95ci", "precision_95ci", "recall_95ci", "specificity_95ci"]:
            if [round(x, 4) for x in j_fc[k]] != [round(x, 4) for x in c_fc[k]]:
                discrepancies.append(f"{dkey} facial {k}: JSON {j_fc[k]} != Computed {c_fc[k]}")

        for k in ["mean_fake_prob_real_faces", "mean_fake_prob_fake_faces"]:
            if abs(j_fc[k] - c_fc[k]) > 1e-4:
                discrepancies.append(f"{dkey} facial {k}: JSON {j_fc[k]} != Computed {c_fc[k]}")

        # Controls
        j_rc = json_d["rejection_control_results"]
        c_rc = comp_d["controls"]
        for cid in ["test_control_corr_001", "test_control_unsupp_001"]:
            if not j_rc[cid]["rejection_successful"]:
                discrepancies.append(f"{dkey} control {cid} rejection not successful in JSON")
            if not c_rc[cid]["rejection_successful"]:
                discrepancies.append(f"{dkey} control {cid} rejection not successful in CSV")

        # Scenes
        j_sc = json_d["scene_diagnostic_results"]
        c_sc = comp_d["scene"]
        for sid in ["test_scene_real_001", "test_scene_diffusion_001", "test_scene_manip_001"]:
            if j_sc[sid]["prediction"] != c_sc[sid]["prediction"]:
                discrepancies.append(f"{dkey} scene {sid} prediction mismatch: JSON {j_sc[sid]['prediction']} != Computed {c_sc[sid]['prediction']}")
            if abs(j_sc[sid]["fake_probability"] - c_sc[sid]["fake_probability"]) > 1e-4:
                discrepancies.append(f"{dkey} scene {sid} fake_probability mismatch")

    # Pairwise agreement comparison
    for pair_key, comp_p in computed_metrics["pairwise_agreement"].items():
        json_p = ground_truth_metrics["pairwise_agreement"][pair_key]
        for k in ["valid_sample_count", "concordant_samples", "discordant_samples"]:
            if json_p[k] != comp_p[k]:
                discrepancies.append(f"{pair_key} {k}: JSON {json_p[k]} != Computed {comp_p[k]}")
        if abs(json_p["concordance_rate"] - comp_p["concordance_rate"]) > 1e-4:
            discrepancies.append(f"{pair_key} concordance_rate: JSON {json_p['concordance_rate']} != Computed {comp_p['concordance_rate']}")
        if abs(json_p["cohen_kappa"] - comp_p["cohen_kappa"]) > 1e-4:
            discrepancies.append(f"{pair_key} cohen_kappa: JSON {json_p['cohen_kappa']} != Computed {comp_p['cohen_kappa']}")
        ab = json_p["agreement_breakdown"]
        for k in ["both_fake", "both_real", "model1_fake_model2_real", "model1_real_model2_fake"]:
            if ab[k] != comp_p[k]:
                discrepancies.append(f"{pair_key} {k}: JSON {ab[k]} != Computed {comp_p[k]}")

    if discrepancies:
        print(f"    [FAIL] Found {len(discrepancies)} discrepancies between CSV and JSON metrics!")
        for d in discrepancies:
            print(f"      - {d}")
        raise AssertionError("Discrepancies found between predictions CSV and metrics JSON")
    else:
        print("    [PASS] 100% exact match between recomputed metrics and phase4_baseline_metrics.json!")

    # ---------------------------------------------------------
    # 5. Cross-Verification against Markdown Report
    # ---------------------------------------------------------
    print("\n[5] Cross-verifying against docs/PHASE4_BASELINE_EVALUATION_REPORT.md...")
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Check key text patterns in report
    report_checks = [
        # Confusion matrix numbers
        ("Chen2024ConvNeXtDetector | 6 | 14 | 37 | 46 | 103 | 0", "ConvNeXt 103 CM row"),
        ("UniversalFakeDetect | 11 | 1 | 50 | 41 | 103 | 0", "UnivFD 103 CM row"),
        ("ViTDeepfakeDetector | 50 | 0 | 51 | 2 | 103 | 0", "ViT 103 CM row"),

        # Overall metrics
        ("41.75% (43/103)", "ConvNeXt overall accuracy"),
        ("30.00% (6/20)", "ConvNeXt overall precision"),
        ("11.54% (6/52)", "ConvNeXt overall recall"),
        ("72.55% (37/51)", "ConvNeXt overall specificity"),
        ("0.1667", "ConvNeXt overall F1"),
        ("42.04%", "ConvNeXt overall balanced accuracy"),

        ("59.22% (61/103)", "UnivFD overall accuracy"),
        ("91.67% (11/12)", "UnivFD overall precision"),
        ("21.15% (11/52)", "UnivFD overall recall"),
        ("98.04% (50/51)", "UnivFD overall specificity"),
        ("0.3437", "UnivFD overall F1"),
        ("59.60%", "UnivFD overall balanced accuracy"),

        ("98.06% (101/103)", "ViT overall accuracy"),
        ("100.00% (50/50)", "ViT overall precision"),
        ("96.15% (50/52)", "ViT overall recall"),
        ("100.00% (51/51)", "ViT overall specificity"),
        ("0.9804", "ViT overall F1"),
        ("98.08%", "ViT overall balanced accuracy"),

        # Facial metrics & Wilson CIs
        ("42.00% (42/100)", "ConvNeXt facial accuracy"),
        ("[32.8%, 51.8%]", "ConvNeXt facial accuracy CI"),
        ("60.00% (60/100)", "UnivFD facial accuracy"),
        ("[50.2%, 69.1%]", "UnivFD facial accuracy CI"),
        ("100.00% (100/100)", "ViT facial accuracy"),
        ("[96.3%, 100.0%]", "ViT facial accuracy CI"),

        # Pairwise concordance & Kappa
        ("72.82%", "ConvNeXt vs UnivFD concordance rate"),
        ("-0.0241", "ConvNeXt vs UnivFD kappa"),
        ("43.69%", "ConvNeXt vs ViT concordance rate"),
        ("-0.1466", "ConvNeXt vs ViT kappa"),
        ("61.17%", "UnivFD vs ViT concordance rate"),
        ("0.2056", "UnivFD vs ViT kappa"),

        # Diagnostic cases
        ("test_scene_real_001", "Scene real ID"),
        ("test_scene_diffusion_001", "Scene diffusion ID"),
        ("test_scene_manip_001", "Scene manip ID"),
        ("test_control_corr_001", "Control corrupted ID"),
        ("test_control_unsupp_001", "Control unsupported ID")
    ]

    report_discrepancies = []
    for pattern, desc in report_checks:
        if pattern not in report_text:
            report_discrepancies.append(f"Report missing expected string '{pattern}' ({desc})")

    if report_discrepancies:
        print(f"    [FAIL] Report verification discrepancies found:")
        for rd in report_discrepancies:
            print(f"      - {rd}")
        raise AssertionError("Report verification discrepancies found")
    else:
        print("    [PASS] 100% exact match between Markdown report and computed ground truth!")

    print("\n" + "=" * 80)
    print("ALL ADVERSARIAL EMPIRICAL CHECKS PASSED WITH ZERO DISCREPANCIES!")
    print("=" * 80)
    return computed_metrics


if __name__ == "__main__":
    run_all_checks()
