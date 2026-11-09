"""
Independent Adversarial Empirical Verification Suite for Milestone 1: Phase 4 Baseline Evaluation.
Verifies data integrity, partition isolation, statistical metrics, Wilson CIs, non-degeneracy, and report consistency.
"""

import os
import sys
import json
import csv
import math
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_CSV = os.path.join(PROJECT_ROOT, "benchmark", "evaluation_manifest.csv")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_predictions.csv")
METRICS_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_metrics.json")
REPORT_MD = os.path.join(PROJECT_ROOT, "docs", "PHASE4_BASELINE_EVALUATION_REPORT.md")


def independent_wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """
    Independent implementation of Wilson score interval:
    center = (p + z^2 / (2n)) / (1 + z^2 / n)
    margin = (z / (1 + z^2 / n)) * sqrt(p(1-p)/n + z^2 / (4n^2))
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p + (z2 / (2.0 * n))) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def independent_cohen_kappa(both_fake: int, both_real: int, m1_f_m2_r: int, m1_r_m2_f: int) -> float:
    """Independent implementation of Cohen's Kappa for 2x2 agreement matrix."""
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


def run_verification():
    findings = []
    successes = []

    print("=" * 80)
    print("INDEPENDENT ADVERSARIAL VERIFICATION: PHASE 4 BASELINE EVALUATION")
    print("=" * 80)

    # 1. Load Manifest
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")

    manifest_rows = {}
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_rows[row["sample_id"]] = row

    held_out_ids = {sid for sid, r in manifest_rows.items() if r["split"] == "held_out_test"}
    val_ids = {sid for sid, r in manifest_rows.items() if r["split"] == "development_val"}
    smoke_ids = {sid for sid, r in manifest_rows.items() if r["split"] == "operational_smoke"}

    print(f"Manifest loaded: Total={len(manifest_rows)}, held_out_test={len(held_out_ids)}, development_val={len(val_ids)}, operational_smoke={len(smoke_ids)}")

    # 2. Load Predictions CSV
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(f"Predictions CSV not found: {PREDICTIONS_CSV}")

    with open(PREDICTIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prediction_rows = list(reader)

    print(f"Predictions CSV loaded: Total rows = {len(prediction_rows)}")

    # Check 1: Partition Leakage & Contamination
    print("\n--- CHECK 1: DATA CONTAMINATION & PARTITION LEAKAGE ---")
    expected_detectors = {"chen2024_convnext", "univfd", "vit"}
    detector_samples = {det: set() for det in expected_detectors}
    detector_rows = {det: [] for det in expected_detectors}

    for row in prediction_rows:
        sid = row["sample_id"]
        det = row["detector_key"]
        split = row["split"]

        if det not in expected_detectors:
            findings.append(f"Unexpected detector_key in predictions: {det}")
            continue

        detector_samples[det].add(sid)
        detector_rows[det].append(row)

        # Check contamination
        if sid in val_ids or "val_face_" in sid:
            findings.append(f"CRITICAL CONTAMINATION: development_val sample {sid} found in predictions under {det}!")
        if sid in smoke_ids or "smoke_" in sid:
            findings.append(f"CRITICAL CONTAMINATION: operational_smoke sample {sid} found in predictions under {det}!")
        if sid not in held_out_ids:
            findings.append(f"CRITICAL CONTAMINATION: sample {sid} is not in held_out_test!")
        if split != "held_out_test":
            findings.append(f"Split column value '{split}' is not 'held_out_test' for sample {sid}")

        # Check metadata consistency against manifest
        m_row = manifest_rows.get(sid)
        if m_row:
            if m_row["ground_truth"] != row["ground_truth"]:
                findings.append(f"Ground truth mismatch for {sid}: manifest={m_row['ground_truth']} vs pred_csv={row['ground_truth']}")
            if m_row["domain"] != row["domain"]:
                findings.append(f"Domain mismatch for {sid}: manifest={m_row['domain']} vs pred_csv={row['domain']}")
            if m_row["category"] != row["category"]:
                findings.append(f"Category mismatch for {sid}: manifest={m_row['category']} vs pred_csv={row['category']}")

    for det in expected_detectors:
        count = len(detector_samples[det])
        total_rows = len(detector_rows[det])
        if count != 105:
            findings.append(f"Detector {det} evaluated {count} distinct samples, expected exactly 105!")
        else:
            successes.append(f"Detector {det} evaluated exactly 105 distinct samples.")

        if total_rows != 105:
            findings.append(f"Detector {det} has {total_rows} rows (duplicate sample IDs detected!)")
        else:
            successes.append(f"Detector {det} has exactly 105 rows (zero duplicate rows).")

        # Check set difference between evaluated samples and held_out_ids
        missing = held_out_ids - detector_samples[det]
        if missing:
            findings.append(f"Detector {det} is missing held_out samples: {missing}")
        extra = detector_samples[det] - held_out_ids
        if extra:
            findings.append(f"Detector {det} has extra non-held_out samples: {extra}")

    # Check 2: Row-level consistency in predictions.csv
    print("\n--- CHECK 2: ROW-LEVEL CONSISTENCY & NUMERICAL VALIDITY ---")
    for row in prediction_rows:
        sid = row["sample_id"]
        det = row["detector_key"]
        status = row["status"]
        pred = row["prediction"]
        pred_class = row["predicted_class"]
        gt = row["ground_truth"]

        # Check status and prediction consistency
        if gt == "REJECTION_CONTROL":
            # Controls are expected to return ERROR prediction
            if pred != "ERROR":
                findings.append(f"Rejection control {sid} under {det} has prediction={pred}, expected ERROR")
            if pred_class != "ERROR":
                findings.append(f"Rejection control {sid} under {det} has predicted_class={pred_class}, expected ERROR")
            if not row["error_message"]:
                findings.append(f"Rejection control {sid} under {det} missing error_message")
            if status != "ERROR":
                findings.append(f"Rejection control {sid} under {det} has unexpected status: {status}")
            successes.append(f"Rejection control {sid} gracefully rejected under {det} with prediction=ERROR.")
        else:
            if status != "SUCCESS":
                findings.append(f"Sample {sid} under {det} has non-SUCCESS status: {status}")
            if pred not in ("REAL", "FAKE"):
                findings.append(f"Sample {sid} under {det} has non-standard prediction: {pred}")
            if pred != pred_class:
                findings.append(f"Sample {sid} under {det} prediction ({pred}) != predicted_class ({pred_class})")

            # Check probability bounds and threshold consistency
            try:
                p_fake = float(row["fake_probability"])
                p_real = float(row["real_probability"])
                conf = float(row["confidence"])
                thresh = float(row["threshold"])
                latency = float(row["latency_ms"])

                if not (0.0 <= p_fake <= 1.0):
                    findings.append(f"Sample {sid} under {det} p_fake out of bounds: {p_fake}")
                if not (0.0 <= p_real <= 1.0):
                    findings.append(f"Sample {sid} under {det} p_real out of bounds: {p_real}")
                if abs((p_fake + p_real) - 1.0) > 1e-4:
                    findings.append(f"Sample {sid} under {det} p_fake + p_real != 1.0: {p_fake + p_real}")

                # Threshold decision consistency
                expected_pred = "FAKE" if p_fake >= thresh else "REAL"
                if pred != expected_pred:
                    findings.append(f"Sample {sid} under {det} prediction {pred} contradicts p_fake={p_fake} with threshold={thresh}")

                expected_conf = max(p_fake, p_real)
                if abs(conf - expected_conf) > 1e-4:
                    findings.append(f"Sample {sid} under {det} confidence {conf} != max prob {expected_conf}")

                # Latency check: non-zero, positive, realistic
                if latency <= 0:
                    findings.append(f"Sample {sid} under {det} has non-positive latency: {latency}")
            except ValueError as ve:
                findings.append(f"Sample {sid} under {det} numerical parse error: {ve}")

    # Check 3: Model Non-Degeneracy
    print("\n--- CHECK 3: MODEL DIVERSITY & NON-DEGENERACY ---")
    for det in expected_detectors:
        preds = [r["prediction"] for r in detector_rows[det] if r["ground_truth"] != "REJECTION_CONTROL"]
        unique_preds = set(preds)
        fake_probs = [float(r["fake_probability"]) for r in detector_rows[det] if r["ground_truth"] != "REJECTION_CONTROL"]
        prob_variance = sum((p - (sum(fake_probs)/len(fake_probs)))**2 for p in fake_probs) / len(fake_probs)

        print(f"Detector {det}: unique predictions = {unique_preds}, prob variance = {prob_variance:.6f}")
        if len(unique_preds) < 2:
            findings.append(f"DEGENERATE MODEL WARNING: Detector {det} produced only one prediction class: {unique_preds}")
        else:
            successes.append(f"Detector {det} produced diverse predictions: {unique_preds} with variance {prob_variance:.6f}")

    # Check 4: Independent Metric Calculation vs JSON vs Report Markdown
    print("\n--- CHECK 4: INDEPENDENT METRIC CALCULATION & CROSS-CHECK ---")
    if not os.path.exists(METRICS_JSON):
        raise FileNotFoundError(f"Metrics JSON not found: {METRICS_JSON}")

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    with open(REPORT_MD, "r", encoding="utf-8") as f:
        report_md = f.read()

    for det in expected_detectors:
        rows = detector_rows[det]

        # 4a: Overall Eligible Samples (103 samples: 51 REAL, 52 FAKE)
        overall_rows = [r for r in rows if r["ground_truth"] in ("REAL", "FAKE")]
        if len(overall_rows) != 103:
            findings.append(f"Detector {det} has {len(overall_rows)} overall eligible samples, expected 103")

        tp = sum(1 for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        fp = sum(1 for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        tn = sum(1 for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        fn = sum(1 for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")

        n_valid = tp + fp + tn + fn
        acc = (tp + tn) / n_valid if n_valid > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0

        fps = [r["sample_id"] for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE"]
        fns = [r["sample_id"] for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL"]

        # Cross-check with JSON
        json_overall = metrics_json["detectors"][det]["overall_classification_metrics"]
        cm_json = json_overall["confusion_matrix"]

        if (cm_json["tp"], cm_json["fp"], cm_json["tn"], cm_json["fn"]) != (tp, fp, tn, fn):
            findings.append(f"Overall confusion matrix mismatch for {det}: Calc={(tp, fp, tn, fn)} vs JSON={(cm_json['tp'], cm_json['fp'], cm_json['tn'], cm_json['fn'])}")
        else:
            successes.append(f"Overall confusion matrix matches for {det}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        if round(acc, 4) != json_overall["accuracy"]:
            findings.append(f"Overall accuracy mismatch for {det}: Calc={round(acc, 4)} vs JSON={json_overall['accuracy']}")
        if round(prec, 4) != json_overall["precision"]:
            findings.append(f"Overall precision mismatch for {det}: Calc={round(prec, 4)} vs JSON={json_overall['precision']}")
        if round(rec, 4) != json_overall["recall"]:
            findings.append(f"Overall recall mismatch for {det}: Calc={round(rec, 4)} vs JSON={json_overall['recall']}")
        if round(spec, 4) != json_overall["specificity"]:
            findings.append(f"Overall specificity mismatch for {det}: Calc={round(spec, 4)} vs JSON={json_overall['specificity']}")
        if round(f1, 4) != json_overall["f1_score"]:
            findings.append(f"Overall F1 mismatch for {det}: Calc={round(f1, 4)} vs JSON={json_overall['f1_score']}")
        if round(bal_acc, 4) != json_overall["balanced_accuracy"]:
            findings.append(f"Overall balanced accuracy mismatch for {det}: Calc={round(bal_acc, 4)} vs JSON={json_overall['balanced_accuracy']}")

        if fps != json_overall["false_positives"]:
            findings.append(f"Overall FP list mismatch for {det}")
        if fns != json_overall["false_negatives"]:
            findings.append(f"Overall FN list mismatch for {det}")

        # 4b: Facial Evaluation (100 samples: 50 REAL, 50 FAKE)
        facial_rows = [r for r in rows if r["domain"] == "facial"]
        if len(facial_rows) != 100:
            findings.append(f"Detector {det} has {len(facial_rows)} facial samples, expected 100")

        f_tp = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        f_fp = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        f_tn = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        f_fn = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")
        f_valid = f_tp + f_fp + f_tn + f_fn

        f_acc = (f_tp + f_tn) / f_valid if f_valid > 0 else 0.0
        f_prec = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0.0
        f_rec = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0.0
        f_spec = f_tn / (f_tn + f_fp) if (f_tn + f_fp) > 0 else 0.0
        f_f1 = (2 * f_prec * f_rec) / (f_prec + f_rec) if (f_prec + f_rec) > 0 else 0.0
        f_bal_acc = (f_rec + f_spec) / 2.0

        # Wilson Confidence Intervals for facial
        f_acc_ci = independent_wilson_ci(f_tp + f_tn, f_valid)
        f_prec_ci = independent_wilson_ci(f_tp, f_tp + f_fp)
        f_rec_ci = independent_wilson_ci(f_tp, f_tp + f_fn)
        f_spec_ci = independent_wilson_ci(f_tn, f_tn + f_fp)

        json_facial = metrics_json["detectors"][det]["facial_evaluation_metrics"]
        f_cm_json = json_facial["confusion_matrix"]

        if (f_cm_json["tp"], f_cm_json["fp"], f_cm_json["tn"], f_cm_json["fn"]) != (f_tp, f_fp, f_tn, f_fn):
            findings.append(f"Facial confusion matrix mismatch for {det}: Calc={(f_tp, f_fp, f_tn, f_fn)} vs JSON={(f_cm_json['tp'], f_cm_json['fp'], f_cm_json['tn'], f_cm_json['fn'])}")
        else:
            successes.append(f"Facial confusion matrix matches for {det}: TP={f_tp}, FP={f_fp}, TN={f_tn}, FN={f_fn}")

        if round(f_acc, 4) != json_facial["accuracy"]:
            findings.append(f"Facial accuracy mismatch for {det}: Calc={round(f_acc, 4)} vs JSON={json_facial['accuracy']}")
        if list(f_acc_ci) != json_facial["accuracy_95ci"]:
            findings.append(f"Facial accuracy 95% CI mismatch for {det}: Calc={list(f_acc_ci)} vs JSON={json_facial['accuracy_95ci']}")
        else:
            successes.append(f"Facial accuracy 95% CI matches for {det}: {list(f_acc_ci)}")

        if round(f_prec, 4) != json_facial["precision"]:
            findings.append(f"Facial precision mismatch for {det}: Calc={round(f_prec, 4)} vs JSON={json_facial['precision']}")
        if list(f_prec_ci) != json_facial["precision_95ci"]:
            findings.append(f"Facial precision 95% CI mismatch for {det}: Calc={list(f_prec_ci)} vs JSON={json_facial['precision_95ci']}")

        if round(f_rec, 4) != json_facial["recall"]:
            findings.append(f"Facial recall mismatch for {det}: Calc={round(f_rec, 4)} vs JSON={json_facial['recall']}")
        if list(f_rec_ci) != json_facial["recall_95ci"]:
            findings.append(f"Facial recall 95% CI mismatch for {det}: Calc={list(f_rec_ci)} vs JSON={json_facial['recall_95ci']}")

        if round(f_spec, 4) != json_facial["specificity"]:
            findings.append(f"Facial specificity mismatch for {det}: Calc={round(f_spec, 4)} vs JSON={json_facial['specificity']}")
        if list(f_spec_ci) != json_facial["specificity_95ci"]:
            findings.append(f"Facial specificity 95% CI mismatch for {det}: Calc={list(f_spec_ci)} vs JSON={json_facial['specificity_95ci']}")

        if round(f_f1, 4) != json_facial["f1_score"]:
            findings.append(f"Facial F1 mismatch for {det}: Calc={round(f_f1, 4)} vs JSON={json_facial['f1_score']}")
        if round(f_bal_acc, 4) != json_facial["balanced_accuracy"]:
            findings.append(f"Facial balanced accuracy mismatch for {det}: Calc={round(f_bal_acc, 4)} vs JSON={json_facial['balanced_accuracy']}")

        # Separation margin
        real_probs = [float(r["fake_probability"]) for r in facial_rows if r["ground_truth"] == "REAL"]
        fake_probs = [float(r["fake_probability"]) for r in facial_rows if r["ground_truth"] == "FAKE"]
        mean_real = sum(real_probs) / len(real_probs) if real_probs else 0.0
        mean_fake = sum(fake_probs) / len(fake_probs) if fake_probs else 0.0
        sep_margin = mean_fake - mean_real

        if round(mean_real, 4) != json_facial["mean_fake_prob_real_faces"]:
            findings.append(f"Mean real prob mismatch for {det}: Calc={round(mean_real, 4)} vs JSON={json_facial['mean_fake_prob_real_faces']}")
        if round(mean_fake, 4) != json_facial["mean_fake_prob_fake_faces"]:
            findings.append(f"Mean fake prob mismatch for {det}: Calc={round(mean_fake, 4)} vs JSON={json_facial['mean_fake_prob_fake_faces']}")

        # Verify separation margin in markdown report table 6.2
        margin_str = f"**{sep_margin*100:+.2f}%**" if sep_margin >= 0 else f"**{sep_margin*100:.2f}%**"
        if f"{sep_margin*100:.2f}%" not in report_md:
            findings.append(f"Separation margin {sep_margin*100:.2f}% not found in report markdown for {det}")
        else:
            successes.append(f"Separation margin for {det} ({sep_margin*100:.2f}%) verified in markdown report.")

    # Check 5: Pairwise Concordance
    print("\n--- CHECK 5: PAIRWISE AGREEMENT & COHEN'S KAPPA ---")
    pairs = [
        ("chen2024_convnext", "univfd"),
        ("chen2024_convnext", "vit"),
        ("univfd", "vit")
    ]

    for det1, det2 in pairs:
        pair_key = f"{det1}_vs_{det2}"
        rows1 = {r["sample_id"]: r for r in detector_rows[det1] if r["ground_truth"] != "REJECTION_CONTROL"}
        rows2 = {r["sample_id"]: r for r in detector_rows[det2] if r["ground_truth"] != "REJECTION_CONTROL"}

        common_ids = sorted(list(set(rows1.keys()) & set(rows2.keys())))
        if len(common_ids) != 103:
            findings.append(f"Pair {pair_key} evaluated {len(common_ids)} common valid samples, expected 103")

        both_fake = sum(1 for sid in common_ids if rows1[sid]["prediction"] == "FAKE" and rows2[sid]["prediction"] == "FAKE")
        both_real = sum(1 for sid in common_ids if rows1[sid]["prediction"] == "REAL" and rows2[sid]["prediction"] == "REAL")
        m1_f_m2_r = sum(1 for sid in common_ids if rows1[sid]["prediction"] == "FAKE" and rows2[sid]["prediction"] == "REAL")
        m1_r_m2_f = sum(1 for sid in common_ids if rows1[sid]["prediction"] == "REAL" and rows2[sid]["prediction"] == "FAKE")

        concordant = both_fake + both_real
        discordant = m1_f_m2_r + m1_r_m2_f
        total_p = concordant + discordant
        conc_rate = concordant / total_p if total_p > 0 else 0.0
        kappa = independent_cohen_kappa(both_fake, both_real, m1_f_m2_r, m1_r_m2_f)

        json_pair = metrics_json["pairwise_agreement"][pair_key]
        if json_pair["concordant_samples"] != concordant:
            findings.append(f"Concordant samples mismatch for {pair_key}: Calc={concordant} vs JSON={json_pair['concordant_samples']}")
        if json_pair["discordant_samples"] != discordant:
            findings.append(f"Discordant samples mismatch for {pair_key}: Calc={discordant} vs JSON={json_pair['discordant_samples']}")
        if round(conc_rate, 4) != json_pair["concordance_rate"]:
            findings.append(f"Concordance rate mismatch for {pair_key}: Calc={round(conc_rate, 4)} vs JSON={json_pair['concordance_rate']}")
        if round(kappa, 4) != json_pair["cohen_kappa"]:
            findings.append(f"Cohen Kappa mismatch for {pair_key}: Calc={round(kappa, 4)} vs JSON={json_pair['cohen_kappa']}")
        else:
            successes.append(f"Pairwise agreement matches for {pair_key}: Concordant={concordant}/{total_p} ({conc_rate*100:.2f}%), Kappa={kappa}")

    # Check 6: Markdown Report vs JSON cross-check
    print("\n--- CHECK 6: MARKDOWN REPORT NARRATIVE & TABLES CONSISTENCY ---")
    for det_key, det_metrics in metrics_json["detectors"].items():
        det_name = det_metrics["model_info"]["detector_name"]
        o_acc = det_metrics["overall_classification_metrics"]["accuracy_fraction"]
        f_acc = det_metrics["facial_evaluation_metrics"]["accuracy_fraction"]

        if o_acc not in report_md:
            findings.append(f"Markdown report missing overall accuracy fraction {o_acc} for {det_name}")
        if f_acc not in report_md:
            findings.append(f"Markdown report missing facial accuracy fraction {f_acc} for {det_name}")

    # Check 7: Latency Summary
    print("\n--- CHECK 7: LATENCY STATISTICS ---")
    for det in expected_detectors:
        latencies = [float(r["latency_ms"]) for r in detector_rows[det] if r["latency_ms"] != ""]
        min_lat = min(latencies)
        max_lat = max(latencies)
        avg_lat = sum(latencies) / len(latencies)
        print(f"Detector {det} latency: count={len(latencies)}, min={min_lat:.2f}ms, avg={avg_lat:.2f}ms, max={max_lat:.2f}ms")
        if min_lat <= 0:
            findings.append(f"Detector {det} has non-positive min latency: {min_lat}")
        else:
            successes.append(f"Detector {det} latency is realistic: min={min_lat:.2f}ms, avg={avg_lat:.2f}ms, max={max_lat:.2f}ms")

    # Check 8: Single-example Scene Diagnostics (3 cases)
    print("\n--- CHECK 8: SCENE & COMPOSITE DIAGNOSTICS ---")
    scene_cases = ["test_scene_real_001", "test_scene_diffusion_001", "test_scene_manip_001"]
    for sc in scene_cases:
        for det in expected_detectors:
            row = next((r for r in detector_rows[det] if r["sample_id"] == sc), None)
            if not row:
                findings.append(f"Scene case {sc} missing for {det}")
            else:
                successes.append(f"Scene case {sc} evaluated under {det}: pred={row['prediction']}, p_fake={row['fake_probability']}")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total Successes Checked: {len(successes)}")
    print(f"Total Findings / Discrepancies: {len(findings)}")

    if findings:
        print("\nDISCREPANCIES FOUND:")
        for f in findings:
            print(f"  [X] {f}")
        return False, findings, successes
    else:
        print("\nALL ADVERSARIAL CHECKS PASSED PERFECTLY!")
        return True, findings, successes


if __name__ == "__main__":
    passed, findings, successes = run_verification()
    sys.exit(0 if passed else 1)
