"""
Step 6 & 7 Diagnostics and Failure Analysis Script
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from PIL import Image

PROJECT_ROOT = r"C:\Users\user\Desktop\multi model connection"
sys.path.insert(0, PROJECT_ROOT)

def compute_distribution_stats(series):
    """Compute summary statistics and percentiles for a series of scores."""
    arr = series.dropna().values
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50_median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "near_0_pct": float(np.mean(arr < 0.05) * 100.0),
        "near_1_pct": float(np.mean(arr > 0.95) * 100.0),
    }

def run_diagnostics():
    output_dir = os.path.join(PROJECT_ROOT, "scripts", "system_audit")
    os.makedirs(output_dir, exist_ok=True)

    print("=== Step 6: Running Diagnostics ===")
    
    # We will run diagnostics primarily on Phase 6 (in-domain faces, 200 samples)
    # and Phase 7 (out-of-domain GenImage, 200 samples), as well as held-out test (103 samples).
    
    # 1. Load Phase 6
    df6 = pd.read_csv(os.path.join(PROJECT_ROOT, "benchmark", "results", "phase6_combined_predictions.csv"))
    
    # 2. Load Phase 7
    df7 = pd.read_csv(os.path.join(PROJECT_ROOT, "benchmark", "results", "phase7_combined_predictions.csv"))
    
    # 3. Load Phase 4 held-out test (103 samples)
    p4_file = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase4_baseline_predictions.csv")
    df_p4_long = pd.read_csv(p4_file)
    df_p4_long = df_p4_long[df_p4_long["status"] == "SUCCESS"].copy()
    
    pivoted_p4 = []
    for sid in df_p4_long["sample_id"].unique():
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
        c_p = float(c_row["fake_probability"].iloc[0])
        u_p = float(u_row["fake_probability"].iloc[0])
        v_p = float(v_row["fake_probability"].iloc[0])
        c_pred = c_row["prediction"].iloc[0]
        u_pred = u_row["prediction"].iloc[0]
        v_pred = v_row["prediction"].iloc[0]
        mean_p = round((c_p + u_p + v_p) / 3.0, 6)
        mean_pred = "FAKE" if mean_p >= 0.50 else "REAL"
        
        # analyzer
        analyzer_pred = v_pred
        analyzer_prob = v_p
        
        pivoted_p4.append({
            "sample_id": sid,
            "original_filename": sample_meta["file_path"].replace("\\", "/").split("/")[-1],
            "ground_truth": gt,
            "category": sample_meta["category"],
            "convnext_fake_prob": c_p,
            "convnext_pred": c_pred,
            "univfd_fake_prob": u_p,
            "univfd_pred": u_pred,
            "vit_fake_prob": v_p,
            "vit_pred": v_pred,
            "mean_prob_fake_prob": mean_p,
            "mean_prob_pred": mean_pred,
            "vit_priority_fake_prob": analyzer_prob,
            "vit_priority_pred": analyzer_pred,
        })
    df_p4 = pd.DataFrame(pivoted_p4)

    diagnostics_result = {}

    for name, df in [("held_out_test_103", df_p4), ("phase6_test_200", df6), ("phase7_test_200", df7)]:
        print(f"\n--- Processing {name} ---")
        real_df = df[df["ground_truth"] == "REAL"]
        fake_df = df[df["ground_truth"] == "FAKE"]
        
        # a) Score distributions
        score_dist = {}
        for model_col, model_label in [
            ("convnext_fake_prob", "Model A (ConvNeXt)"),
            ("univfd_fake_prob", "Model B (UnivFD)"),
            ("vit_fake_prob", "Model C (ViT)"),
            ("mean_prob_fake_prob", "Simple Average"),
            ("vit_priority_fake_prob", "Central Analyzer"),
        ]:
            score_dist[model_label] = {
                "REAL": compute_distribution_stats(real_df[model_col]),
                "FAKE": compute_distribution_stats(fake_df[model_col]),
            }
            
        # b) Correlation matrix between 3 models' scores
        scores = df[["convnext_fake_prob", "univfd_fake_prob", "vit_fake_prob"]].values
        pearson_mat = np.corrcoef(scores.T)
        
        # Spearman correlation
        spearman_mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                spearman_mat[i, j] = spearmanr(scores[:, i], scores[:, j])[0]
                
        # Error correlation
        y_true = (df["ground_truth"] == "FAKE").astype(int).values
        c_err = ((df["convnext_pred"] == "FAKE").astype(int).values != y_true).astype(int)
        u_err = ((df["univfd_pred"] == "FAKE").astype(int).values != y_true).astype(int)
        v_err = ((df["vit_pred"] == "FAKE").astype(int).values != y_true).astype(int)
        
        err_matrix = np.column_stack([c_err, u_err, v_err])
        # Error correlation
        err_corr = np.corrcoef(err_matrix.T)
        
        # Error co-occurrence counts
        total_errs_per_sample = c_err + u_err + v_err
        all_3_correct = int(np.sum(total_errs_per_sample == 0))
        only_1_wrong = int(np.sum(total_errs_per_sample == 1))
        only_2_wrong = int(np.sum(total_errs_per_sample == 2))
        all_3_wrong = int(np.sum(total_errs_per_sample == 3))
        
        breakdown_1_wrong = {
            "convnext_only": int(np.sum((c_err == 1) & (u_err == 0) & (v_err == 0))),
            "univfd_only": int(np.sum((c_err == 0) & (u_err == 1) & (v_err == 0))),
            "vit_only": int(np.sum((c_err == 0) & (u_err == 0) & (v_err == 1))),
        }
        breakdown_2_wrong = {
            "convnext_and_univfd": int(np.sum((c_err == 1) & (u_err == 1) & (v_err == 0))),
            "convnext_and_vit": int(np.sum((c_err == 1) & (u_err == 0) & (v_err == 1))),
            "univfd_and_vit": int(np.sum((c_err == 0) & (u_err == 1) & (v_err == 1))),
        }

        diagnostics_result[name] = {
            "score_distributions": score_dist,
            "pearson_correlation": {
                "models": ["ConvNeXt", "UnivFD", "ViT"],
                "matrix": pearson_mat.tolist(),
            },
            "spearman_correlation": {
                "models": ["ConvNeXt", "UnivFD", "ViT"],
                "matrix": spearman_mat.tolist(),
            },
            "error_correlation": {
                "models": ["ConvNeXt", "UnivFD", "ViT"],
                "matrix": err_corr.tolist(),
            },
            "error_cooccurrence": {
                "total_samples": len(df),
                "all_3_correct": all_3_correct,
                "only_1_wrong": only_1_wrong,
                "only_2_wrong": only_2_wrong,
                "all_3_wrong": all_3_wrong,
                "breakdown_1_wrong": breakdown_1_wrong,
                "breakdown_2_wrong": breakdown_2_wrong,
            }
        }
        
    # Save diagnostics
    with open(os.path.join(output_dir, "step6_diagnostics.json"), "w") as f:
        json.dump(diagnostics_result, f, indent=2)
    print("Saved step6_diagnostics.json")

    # Save 10 misclassified examples per type (FP and FN) for Step 7
    # We will pull from Phase 6 (StyleGAN faces) and Phase 7 (GenImage scenes)
    failures = {}
    for dname, df in [("phase6_facial", df6), ("phase7_general_scene", df7), ("held_out_test", df_p4)]:
        y_true = (df["ground_truth"] == "FAKE").astype(int).values
        # Analyzer predictions
        ana_pred = (df["vit_priority_pred"] == "FAKE").astype(int).values
        
        fp_indices = np.where((y_true == 0) & (ana_pred == 1))[0]
        fn_indices = np.where((y_true == 1) & (ana_pred == 0))[0]
        
        # Also check where ANY base model failed
        c_pred = (df["convnext_pred"] == "FAKE").astype(int).values
        u_pred = (df["univfd_pred"] == "FAKE").astype(int).values
        v_pred = (df["vit_pred"] == "FAKE").astype(int).values
        
        # Gather samples
        fp_samples = []
        for idx in fp_indices[:10]:
            row = df.iloc[idx]
            fp_samples.append({
                "sample_id": str(row["sample_id"]),
                "filename": str(row.get("original_filename", row["sample_id"])),
                "ground_truth": "REAL",
                "convnext_prob": float(row["convnext_fake_prob"]),
                "convnext_pred": str(row["convnext_pred"]),
                "univfd_prob": float(row["univfd_fake_prob"]),
                "univfd_pred": str(row["univfd_pred"]),
                "vit_prob": float(row["vit_fake_prob"]),
                "vit_pred": str(row["vit_pred"]),
                "simple_avg_prob": float(row["mean_prob_fake_prob"]),
                "analyzer_pred": str(row["vit_priority_pred"]),
                "analyzer_prob": float(row["vit_priority_fake_prob"]),
            })
            
        fn_samples = []
        for idx in fn_indices[:10]:
            row = df.iloc[idx]
            fn_samples.append({
                "sample_id": str(row["sample_id"]),
                "filename": str(row.get("original_filename", row["sample_id"])),
                "ground_truth": "FAKE",
                "generator": str(row.get("generator_family", "StyleGAN")),
                "convnext_prob": float(row["convnext_fake_prob"]),
                "convnext_pred": str(row["convnext_pred"]),
                "univfd_prob": float(row["univfd_fake_prob"]),
                "univfd_pred": str(row["univfd_pred"]),
                "vit_prob": float(row["vit_fake_prob"]),
                "vit_pred": str(row["vit_pred"]),
                "simple_avg_prob": float(row["mean_prob_fake_prob"]),
                "analyzer_pred": str(row["vit_priority_pred"]),
                "analyzer_prob": float(row["vit_priority_fake_prob"]),
            })
            
        failures[dname] = {
            "false_positives_count": len(fp_indices),
            "false_negatives_count": len(fn_indices),
            "fp_examples_top10": fp_samples,
            "fn_examples_top10": fn_samples,
        }
        
    with open(os.path.join(output_dir, "step7_failures.json"), "w") as f:
        json.dump(failures, f, indent=2)
    print("Saved step7_failures.json")

if __name__ == "__main__":
    run_diagnostics()
