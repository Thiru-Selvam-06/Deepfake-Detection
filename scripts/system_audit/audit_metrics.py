"""
Comprehensive Diagnostic and Metrics Audit Script for Multi-Model Connection
Computes all metrics and diagnostics required for Step 5, Step 6, and Step 7.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

# Project paths
PROJECT_ROOT = r"C:\Users\user\Desktop\multi model connection"
sys.path.insert(0, PROJECT_ROOT)

from src.decision_fusion import DecisionFusionPipeline, STRATEGY_VIT_PRIORITY, STRATEGY_MEAN_PROBABILITY
from src.detector_schema import DetectorResult

def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER) and the optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    # Find the point where fpr == fnr
    try:
        eer = brentq(lambda x: 1.0 - interp1d(fpr, tpr)(x) - x, 0.0, 1.0)
    except Exception:
        # Fallback to closest point
        diff = np.abs(fpr - fnr)
        min_idx = np.argmin(diff)
        eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
    return float(eer)

def compute_ece(y_true, y_score, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    bin_details = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Bin inclusion rule: [lower, upper] for last bin, [lower, upper) for others
        if i == n_bins - 1:
            in_bin = (y_score >= bin_lower) & (y_score <= bin_upper)
        else:
            in_bin = (y_score >= bin_lower) & (y_score < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_score[in_bin])
            bin_error = np.abs(bin_acc - bin_conf)
            ece += bin_error * prop_in_bin
            bin_details.append({
                "bin": i,
                "range": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "samples": int(np.sum(in_bin)),
                "acc": float(bin_acc),
                "conf": float(bin_conf),
                "abs_diff": float(bin_error),
            })
    return float(ece), bin_details

def compute_metrics_table(y_true, y_pred, y_score):
    """Compute standard evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Check if multiple classes are present in y_true for AUC
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        eer = compute_eer(y_true, y_score)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
        eer = float('nan')
        
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "EER": float(eer),
    }

def analyze_dataset_predictions(df, ground_truth_col, score_cols, pred_cols):
    """Analyze predictions across models on a dataset."""
    results = {}
    y_true = (df[ground_truth_col] == "FAKE").astype(int).values
    
    for entity_name in score_cols:
        y_score = df[score_cols[entity_name]].values
        if entity_name in pred_cols:
            y_pred = (df[pred_cols[entity_name]] == "FAKE").astype(int).values
        else:
            y_pred = (y_score >= 0.50).astype(int)
        
        m = compute_metrics_table(y_true, y_pred, y_score)
        ece, _ = compute_ece(y_true, y_score)
        m["ECE"] = ece
        results[entity_name] = m
    return results

def main():
    fusion_pipe = DecisionFusionPipeline()
    output_dir = os.path.join(PROJECT_ROOT, "scripts", "system_audit")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Step 5 & 6 Evaluation Audit Runner ===")
    
    # 1. HELD-OUT TEST SET (Phase 4 predictions, 103 classification samples)
    p4_file = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_predictions.csv")
    df_p4_long = pd.read_csv(p4_file)
    # Filter to eligible samples
    df_p4_long = df_p4_long[df_p4_long["status"] == "SUCCESS"].copy()
    
    # Pivot to wide format: one row per sample_id
    pivoted_p4 = []
    sample_ids = df_p4_long["sample_id"].unique()
    for sid in sample_ids:
        rows = df_p4_long[df_p4_long["sample_id"] == sid]
        sample_meta = rows.iloc[0]
        gt = sample_meta["ground_truth"]
        if gt not in ("REAL", "FAKE"):
            continue
            
        c_row = rows[rows["detector_key"] == "chen2024_convnext"]
        u_row = rows[rows["detector_key"] == "univfd"]
        v_row = rows[rows["detector_key"] == "vit"]
        
        if c_row.empty or u_row.empty or v_row.empty:
            continue
            
        c_prob = float(c_row["fake_probability"].iloc[0])
        u_prob = float(u_row["fake_probability"].iloc[0])
        v_prob = float(v_row["fake_probability"].iloc[0])
        
        c_pred = c_row["prediction"].iloc[0]
        u_pred = u_row["prediction"].iloc[0]
        v_pred = v_row["prediction"].iloc[0]
        
        # Simple average
        mean_p = round((c_prob + u_prob + v_prob) / 3.0, 6)
        mean_pred = "FAKE" if mean_p >= 0.50 else "REAL"
        
        # Central analyzer (ViT Priority)
        det_dict = {
            "chen2024_convnext": {"prediction": c_pred, "fake_probability": c_prob},
            "univfd": {"prediction": u_pred, "fake_probability": u_prob},
            "vit": {"prediction": v_pred, "fake_probability": v_prob},
        }
        fused = fusion_pipe.fuse(det_dict, strategy=STRATEGY_VIT_PRIORITY)
        
        pivoted_p4.append({
            "sample_id": sid,
            "ground_truth": gt,
            "category": sample_meta["category"],
            "domain": sample_meta["domain"],
            "file_path": sample_meta["file_path"],
            "convnext_prob": c_prob,
            "convnext_pred": c_pred,
            "univfd_prob": u_prob,
            "univfd_pred": u_pred,
            "vit_prob": v_prob,
            "vit_pred": v_pred,
            "simple_avg_prob": mean_p,
            "simple_avg_pred": mean_pred,
            "analyzer_prob": fused.fake_probability,
            "analyzer_pred": fused.prediction,
        })
    df_p4 = pd.DataFrame(pivoted_p4)
    print(f"Held-out test set processed: {len(df_p4)} samples")
    
    score_cols_p4 = {
        "Model A (ConvNeXt)": "convnext_prob",
        "Model B (UnivFD)": "univfd_prob",
        "Model C (ViT)": "vit_prob",
        "Simple Average": "simple_avg_prob",
        "Central Analyzer (ViT-Priority)": "analyzer_prob",
    }
    pred_cols_p4 = {
        "Model A (ConvNeXt)": "convnext_pred",
        "Model B (UnivFD)": "univfd_pred",
        "Model C (ViT)": "vit_pred",
        "Simple Average": "simple_avg_pred",
        "Central Analyzer (ViT-Priority)": "analyzer_pred",
    }
    metrics_held_out = analyze_dataset_predictions(df_p4, "ground_truth", score_cols_p4, pred_cols_p4)
    
    # 2. VALIDATION SET (Phase 5, 50 samples)
    p5_val_file = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase5_val_raw_predictions.csv")
    df_p5_long = pd.read_csv(p5_val_file)
    pivoted_p5 = []
    for sid in df_p5_long["sample_id"].unique():
        rows = df_p5_long[df_p5_long["sample_id"] == sid]
        sample_meta = rows.iloc[0]
        gt = sample_meta["ground_truth"]
        
        c_row = rows[rows["detector_key"] == "chen2024_convnext"]
        u_row = rows[rows["detector_key"] == "univfd"]
        v_row = rows[rows["detector_key"] == "vit"]
        
        c_prob = float(c_row["fake_probability"].iloc[0])
        u_prob = float(u_row["fake_probability"].iloc[0])
        v_prob = float(v_row["fake_probability"].iloc[0])
        
        c_pred = c_row["prediction"].iloc[0]
        u_pred = u_row["prediction"].iloc[0]
        v_pred = v_row["prediction"].iloc[0]
        
        mean_p = round((c_prob + u_prob + v_prob) / 3.0, 6)
        mean_pred = "FAKE" if mean_p >= 0.50 else "REAL"
        
        det_dict = {
            "chen2024_convnext": {"prediction": c_pred, "fake_probability": c_prob},
            "univfd": {"prediction": u_pred, "fake_probability": u_prob},
            "vit": {"prediction": v_pred, "fake_probability": v_prob},
        }
        fused = fusion_pipe.fuse(det_dict, strategy=STRATEGY_VIT_PRIORITY)
        
        pivoted_p5.append({
            "sample_id": sid,
            "ground_truth": gt,
            "convnext_prob": c_prob,
            "convnext_pred": c_pred,
            "univfd_prob": u_prob,
            "univfd_pred": u_pred,
            "vit_prob": v_prob,
            "vit_pred": v_pred,
            "simple_avg_prob": mean_p,
            "simple_avg_pred": mean_pred,
            "analyzer_prob": fused.fake_probability,
            "analyzer_pred": fused.prediction,
        })
    df_p5 = pd.DataFrame(pivoted_p5)
    print(f"Validation set processed: {len(df_p5)} samples")
    metrics_val = analyze_dataset_predictions(df_p5, "ground_truth", score_cols_p4, pred_cols_p4)
    
    # 3. PHASE 6 TEST SET (200 samples)
    p6_file = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase6_combined_predictions.csv")
    df_p6 = pd.read_csv(p6_file)
    score_cols_p6 = {
        "Model A (ConvNeXt)": "convnext_fake_prob",
        "Model B (UnivFD)": "univfd_fake_prob",
        "Model C (ViT)": "vit_fake_prob",
        "Simple Average": "mean_prob_fake_prob",
        "Central Analyzer (ViT-Priority)": "vit_priority_fake_prob",
    }
    pred_cols_p6 = {
        "Model A (ConvNeXt)": "convnext_pred",
        "Model B (UnivFD)": "univfd_pred",
        "Model C (ViT)": "vit_pred",
        "Simple Average": "mean_prob_pred",
        "Central Analyzer (ViT-Priority)": "vit_priority_pred",
    }
    metrics_p6 = analyze_dataset_predictions(df_p6, "ground_truth", score_cols_p6, pred_cols_p6)
    print(f"Phase 6 test set processed: {len(df_p6)} samples")

    # 4. PHASE 7 TEST SET (200 samples)
    p7_file = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase7_combined_predictions.csv")
    df_p7 = pd.read_csv(p7_file)
    metrics_p7 = analyze_dataset_predictions(df_p7, "ground_truth", score_cols_p6, pred_cols_p6)
    print(f"Phase 7 test set processed: {len(df_p7)} samples")

    # Save all tables to JSON
    audit_results = {
        "held_out_test_103": metrics_held_out,
        "development_val_50": metrics_val,
        "phase6_test_200": metrics_p6,
        "phase7_test_200": metrics_p7,
    }
    
    with open(os.path.join(output_dir, "step5_metrics.json"), "w") as f:
        json.dump(audit_results, f, indent=2)
    print("Saved step5_metrics.json")
    
    # Save formatted markdown tables for easy reading
    print("\n--- HELD-OUT TEST (103 samples) ---")
    print(pd.DataFrame(metrics_held_out).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "EER", "TP", "FP", "TN", "FN"]])
    
    print("\n--- PHASE 6 TEST (200 samples) ---")
    print(pd.DataFrame(metrics_p6).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "EER", "TP", "FP", "TN", "FN"]])

    print("\n--- VALIDATION SET (50 samples) ---")
    print(pd.DataFrame(metrics_val).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "EER", "TP", "FP", "TN", "FN"]])

    print("\n--- PHASE 7 TEST (200 samples) ---")
    print(pd.DataFrame(metrics_p7).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "EER", "TP", "FP", "TN", "FN"]])

if __name__ == "__main__":
    main()
