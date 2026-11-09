import csv
import json
import math
import sys

def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
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

def compute_cohen_kappa(both_fake: int, both_real: int, m1_f_m2_r: int, m1_r_m2_f: int):
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

def run_audit():
    csv_path = "benchmark/results/phase4_baseline_predictions.csv"
    json_path = "benchmark/results/phase4_baseline_metrics.json"
    report_path = "docs/PHASE4_BASELINE_EVALUATION_REPORT.md"

    with open(csv_path, "r", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    with open(json_path, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    print(f"Loaded {len(csv_rows)} CSV rows.")
    print("Checking splits and sample leakage...")
    for r in csv_rows:
        if r["split"] != "held_out_test":
            print(f"LEAKAGE DETECTED: Row has split {r['split']}")
            return False

    detectors = ["chen2024_convnext", "univfd", "vit"]
    for det in detectors:
        d_rows = [r for r in csv_rows if r["detector_key"] == det]
        valid_rows = [r for r in d_rows if r["status"] == "SUCCESS"]
        err_rows = [r for r in d_rows if r["status"] == "ERROR"]
        probs = [float(r["fake_probability"]) for r in valid_rows]
        logits = [float(r["raw_logit"]) for r in valid_rows]
        latencies = [float(r["latency_ms"]) for r in valid_rows]

        print(f"\n--- {det} ---")
        print(f"Total: {len(d_rows)}, Valid: {len(valid_rows)}, Errors: {len(err_rows)}")
        print(f"Unique probs: {len(set(probs))}/{len(valid_rows)}")
        print(f"Unique logits: {len(set(logits))}/{len(valid_rows)}")
        print(f"Unique latencies: {len(set(latencies))}/{len(valid_rows)}")
        print(f"Prob min/max: [{min(probs):.6f}, {max(probs):.6f}]")
        print(f"Logit min/max: [{min(logits):.4f}, {max(logits):.4f}]")
        print(f"Latency min/max: [{min(latencies):.1f}ms, {max(latencies):.1f}ms]")

        # Verify error rows
        assert len(err_rows) == 2, f"Expected 2 error rows for {det}, got {len(err_rows)}"
        for er in err_rows:
            assert er["ground_truth"] == "REJECTION_CONTROL", f"Unexpected error ground truth: {er['ground_truth']}"
            assert er["domain"] == "control", f"Unexpected error domain: {er['domain']}"
            print(f"  Captured Error sample {er['sample_id']}: {er['error_message'][:70]}")

        # Independent metrics calculation for overall (103 samples)
        overall_rows = [r for r in d_rows if r["ground_truth"] in ("REAL", "FAKE")]
        assert len(overall_rows) == 103, f"Expected 103 overall rows, got {len(overall_rows)}"
        tp = sum(1 for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        fp = sum(1 for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        tn = sum(1 for r in overall_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        fn = sum(1 for r in overall_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")
        v_count = tp + fp + tn + fn
        
        acc = round((tp + tn) / v_count, 4)
        prec_raw = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_raw = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_raw = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_raw = (2.0 * prec_raw * rec_raw) / (prec_raw + rec_raw) if (prec_raw + rec_raw) > 0 else 0.0
        bal_acc_raw = (rec_raw + spec_raw) / 2.0

        prec = round(prec_raw, 4)
        rec = round(rec_raw, 4)
        spec = round(spec_raw, 4)
        f1 = round(f1_raw, 4)
        bal_acc = round(bal_acc_raw, 4)

        json_overall = metrics_json["detectors"][det]["overall_classification_metrics"]
        assert json_overall["confusion_matrix"]["tp"] == tp
        assert json_overall["confusion_matrix"]["fp"] == fp
        assert json_overall["confusion_matrix"]["tn"] == tn
        assert json_overall["confusion_matrix"]["fn"] == fn
        assert json_overall["accuracy"] == acc
        assert json_overall["precision"] == prec
        assert json_overall["recall"] == rec
        assert json_overall["specificity"] == spec
        assert json_overall["f1_score"] == f1
        assert json_overall["balanced_accuracy"] == bal_acc
        print(f"Overall metrics match JSON perfectly: Acc={acc}, F1={f1}, BalAcc={bal_acc}")

        # Facial metrics (100 samples)
        facial_rows = [r for r in d_rows if r["domain"] == "facial"]
        assert len(facial_rows) == 100
        tp_f = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "FAKE")
        fp_f = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "FAKE")
        tn_f = sum(1 for r in facial_rows if r["ground_truth"] == "REAL" and r["prediction"] == "REAL")
        fn_f = sum(1 for r in facial_rows if r["ground_truth"] == "FAKE" and r["prediction"] == "REAL")
        acc_f = round((tp_f + tn_f) / 100, 4)
        ci_f = wilson_score_interval(tp_f + tn_f, 100)

        json_facial = metrics_json["detectors"][det]["facial_evaluation_metrics"]
        assert json_facial["confusion_matrix"]["tp"] == tp_f
        assert json_facial["confusion_matrix"]["fp"] == fp_f
        assert json_facial["confusion_matrix"]["tn"] == tn_f
        assert json_facial["confusion_matrix"]["fn"] == fn_f
        assert json_facial["accuracy"] == acc_f
        assert json_facial["accuracy_95ci"] == ci_f
        print(f"Facial metrics match JSON perfectly: Acc={acc_f}, 95% CI={ci_f}")

    # Check pairwise concordance
    pairs = [
        ("chen2024_convnext", "univfd"),
        ("chen2024_convnext", "vit"),
        ("univfd", "vit")
    ]
    for d1, d2 in pairs:
        pair_key = f"{d1}_vs_{d2}"
        r1 = {r["sample_id"]: r for r in csv_rows if r["detector_key"] == d1}
        r2 = {r["sample_id"]: r for r in csv_rows if r["detector_key"] == d2}
        
        both_f = sum(1 for sid in r1 if r1[sid]["prediction"] == "FAKE" and r2[sid]["prediction"] == "FAKE")
        both_r = sum(1 for sid in r1 if r1[sid]["prediction"] == "REAL" and r2[sid]["prediction"] == "REAL")
        d1_f_d2_r = sum(1 for sid in r1 if r1[sid]["prediction"] == "FAKE" and r2[sid]["prediction"] == "REAL")
        d1_r_d2_f = sum(1 for sid in r1 if r1[sid]["prediction"] == "REAL" and r2[sid]["prediction"] == "FAKE")
        v_both = both_f + both_r + d1_f_d2_r + d1_r_d2_f
        conc = both_f + both_r
        conc_rate = round(conc / v_both, 4)
        kappa = compute_cohen_kappa(both_f, both_r, d1_f_d2_r, d1_r_d2_f)

        json_pair = metrics_json["pairwise_agreement"][pair_key]
        assert json_pair["valid_sample_count"] == v_both == 103
        assert json_pair["concordant_samples"] == conc
        assert json_pair["concordance_rate"] == conc_rate
        assert json_pair["cohen_kappa"] == kappa
        print(f"Pairwise {pair_key} matches JSON perfectly: Concordance={conc_rate} ({conc}/103), Kappa={kappa}")

    # Check report content matches
    print("\nVerifying Markdown report concordance...")
    for det in detectors:
        om = metrics_json["detectors"][det]["overall_classification_metrics"]
        fm = metrics_json["detectors"][det]["facial_evaluation_metrics"]
        acc_str = f"{om['accuracy']*100:.2f}%"
        acc_f_str = f"{fm['accuracy']*100:.2f}%"
        assert acc_str in report_text, f"Overall accuracy {acc_str} not in report"
        assert acc_f_str in report_text, f"Facial accuracy {acc_f_str} not in report"

    print("\nALL NUMERICAL CHECKS PASSED 100%!")
    return True

if __name__ == "__main__":
    if not run_audit():
        sys.exit(1)
