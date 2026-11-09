"""
Robustness Quick Test for Step 6g
Evaluates 20 test samples (10 Real, 10 Fake from Phase 6 manifest) under:
1. Baseline clean images
2. JPEG compression (quality = 75)
3. Downscaling (50% scale, resized back via bicubic)
Runs sequentially with memory-safe model loading/unloading.
"""

import os
import io
import sys
import gc
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = r"C:\Users\user\Desktop\multi model connection"
sys.path.insert(0, PROJECT_ROOT)

from src.detector_registry import get_detector
from src.decision_fusion import DecisionFusionPipeline, STRATEGY_VIT_PRIORITY

def main():
    print("=== Step 6g: Robustness Evaluation Runner ===")
    manifest_path = os.path.join(PROJECT_ROOT, "benchmark", "phase6_test_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # Select first 10 Real and first 10 Fake
    real_subset = df[df["ground_truth"] == "REAL"].head(10)
    fake_subset = df[df["ground_truth"] == "FAKE"].head(10)
    eval_df = pd.concat([real_subset, fake_subset], ignore_index=True)
    
    perturbations = ["clean", "jpeg_75", "downscale_half"]
    
    # Store predictions: {det_key: {perturbation: [{sample_id, gt, prob, pred}]}}
    raw_results = {k: {p: [] for p in perturbations} for k in ["chen2024_convnext", "univfd", "vit"]}
    
    for det_key in ["chen2024_convnext", "univfd", "vit"]:
        print(f"\nEvaluating detector: {det_key}")
        adapter = get_detector(det_key, lazy_load=False, singleton=False)
        
        for p in perturbations:
            print(f"  Condition: {p}")
            for _, row in eval_df.iterrows():
                img_path = row["absolute_path"]
                gt = row["ground_truth"]
                sid = row["sample_id"]
                
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    
                    if p == "clean":
                        test_img = img_rgb
                    elif p == "jpeg_75":
                        buf = io.BytesIO()
                        img_rgb.save(buf, format="JPEG", quality=75)
                        buf.seek(0)
                        test_img = Image.open(buf).convert("RGB")
                    elif p == "downscale_half":
                        w, h = img_rgb.size
                        # Downscale by 50% then upscale back to original size
                        small = img_rgb.resize((w // 2, h // 2), Image.Resampling.BILINEAR)
                        test_img = small.resize((w, h), Image.Resampling.BICUBIC)
                        
                    res = adapter.predict(test_img)
                    raw_results[det_key][p].append({
                        "sample_id": sid,
                        "ground_truth": gt,
                        "prediction": res.prediction,
                        "fake_probability": res.fake_probability,
                    })
                    
        # Explicitly unload and free memory
        adapter.unload_model()
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Now compute metrics across perturbations including simple average and central analyzer
    fusion_pipe = DecisionFusionPipeline()
    metrics_summary = {}
    
    for p in perturbations:
        metrics_summary[p] = {}
        # Reconstruct rows for this perturbation
        fused_rows = []
        for i in range(len(eval_df)):
            sid = eval_df.iloc[i]["sample_id"]
            gt = eval_df.iloc[i]["ground_truth"]
            
            c_res = raw_results["chen2024_convnext"][p][i]
            u_res = raw_results["univfd"][p][i]
            v_res = raw_results["vit"][p][i]
            
            c_prob = c_res["fake_probability"]
            u_prob = u_res["fake_probability"]
            v_prob = v_res["fake_probability"]
            
            mean_p = round((c_prob + u_prob + v_prob) / 3.0, 6)
            mean_pred = "FAKE" if mean_p >= 0.50 else "REAL"
            
            det_dict = {
                "chen2024_convnext": {"prediction": c_res["prediction"], "fake_probability": c_prob},
                "univfd": {"prediction": u_res["prediction"], "fake_probability": u_prob},
                "vit": {"prediction": v_res["prediction"], "fake_probability": v_prob},
            }
            fused = fusion_pipe.fuse(det_dict, strategy=STRATEGY_VIT_PRIORITY)
            
            fused_rows.append({
                "sample_id": sid,
                "ground_truth": gt,
                "convnext_pred": c_res["prediction"],
                "univfd_pred": u_res["prediction"],
                "vit_pred": v_res["prediction"],
                "simple_avg_pred": mean_pred,
                "analyzer_pred": fused.prediction,
            })
            
        df_p = pd.DataFrame(fused_rows)
        y_true = (df_p["ground_truth"] == "FAKE").astype(int).values
        
        for ent_name, col in [
            ("Model A (ConvNeXt)", "convnext_pred"),
            ("Model B (UnivFD)", "univfd_pred"),
            ("Model C (ViT)", "vit_pred"),
            ("Simple Average", "simple_avg_pred"),
            ("Central Analyzer (ViT-Priority)", "analyzer_pred"),
        ]:
            y_pred = (df_p[col] == "FAKE").astype(int).values
            acc = float(np.mean(y_true == y_pred))
            real_acc = float(np.mean((y_true == 0) == (y_pred == 0)))
            fake_acc = float(np.mean((y_true == 1) == (y_pred == 1)))
            metrics_summary[p][ent_name] = {
                "accuracy": acc,
                "real_recall": real_acc,
                "fake_recall": fake_acc,
            }
            
    out_file = os.path.join(PROJECT_ROOT, "scripts", "system_audit", "robustness_results.json")
    with open(out_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nRobustness test finished! Results saved to {out_file}")
    print(json.dumps(metrics_summary, indent=2))

if __name__ == "__main__":
    main()
