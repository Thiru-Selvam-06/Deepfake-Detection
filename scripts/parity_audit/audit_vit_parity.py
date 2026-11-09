"""
Audit Phase 4 — ViT Reference-Inference Parity Test
Compares the official Hugging Face ViT pipeline against ViTAdapter across Phase 7 images.
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vit_adapter import ViTAdapter

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "deepfake_vs_real_image_detection")
OUT_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "parity_audit_vit_results.json")

def main():
    print("=" * 80)
    print("AUDIT PHASE 4: ViT REFERENCE PARITY TEST")
    print("=" * 80)

    # 1. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"Loaded {len(manifest_rows)} manifest samples.")

    # 2. Select 30 deterministic samples (10 Real, 10 Midjourney, 10 ADM) + full run
    reals = [r for r in manifest_rows if r["label"] == "REAL"][:10]
    mjs = [r for r in manifest_rows if r["generator_family"] == "midjourney"][:10]
    adms = [r for r in manifest_rows if r["generator_family"] == "adm"][:10]
    sample_subset = reals + mjs + adms
    print(f"Deterministic subset: {len(sample_subset)} samples (10 Real, 10 MJ, 10 ADM)")

    # 3. Setup Official Pipeline
    print("\n[Step 1] Initializing Official Hugging Face ViT Implementation...")
    t0 = time.perf_counter()
    official_processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    official_model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    official_model.eval()
    print(f"  Official model loaded in {time.perf_counter() - t0:.2f}s.")
    print(f"  Official id2label: {official_model.config.id2label}")
    print(f"  Official label2id: {official_model.config.label2id}")

    # 4. Setup Adapter Pipeline
    print("\n[Step 2] Initializing Project ViTAdapter...")
    t0 = time.perf_counter()
    adapter = ViTAdapter(device="cpu", lazy_load=False)
    print(f"  Adapter loaded in {time.perf_counter() - t0:.2f}s.")

    # 5. Check Architecture & Checkpoint Parity
    print("\n[Step 3] Checking Architecture and Weights Parity...")
    off_params = dict(official_model.named_parameters())
    ad_params = dict(adapter.model.named_parameters())
    
    assert set(off_params.keys()) == set(ad_params.keys()), "Parameter key mismatch!"
    max_weight_diff = 0.0
    for k in off_params:
        diff = (off_params[k] - ad_params[k]).abs().max().item()
        if diff > max_weight_diff:
            max_weight_diff = diff
    print(f"  Parameter count: {len(off_params)} tensors.")
    print(f"  Max parameter weight difference: {max_weight_diff:.10f}")

    # 6. Run Parity on All 200 Samples
    print("\n[Step 4] Executing Parity Comparison across all 200 samples...")
    results = []
    max_tensor_diff = 0.0
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    prediction_mismatches = 0

    for idx, row in enumerate(manifest_rows):
        sample_id = row["sample_id"]
        rel_path = row["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        ground_truth = row["label"]
        gen_family = row["generator_family"]

        # Stage A: Image loading
        with Image.open(abs_path) as opened_img:
            img_rgb = opened_img.convert("RGB")

        # Stage B: Preprocessing
        # Official
        off_inputs = official_processor(images=img_rgb, return_tensors="pt")
        off_tensor = off_inputs["pixel_values"]
        # Adapter (internal processor)
        ad_inputs = adapter.processor(images=img_rgb, return_tensors="pt")
        ad_tensor = ad_inputs["pixel_values"]

        tensor_diff = (off_tensor - ad_tensor).abs().max().item()
        if tensor_diff > max_tensor_diff:
            max_tensor_diff = tensor_diff

        # Stage C: Raw logits
        with torch.no_grad():
            off_out = official_model(**off_inputs)
            off_logits = off_out.logits.squeeze(0)  # [2]
            off_probs = torch.softmax(off_logits, dim=0)

        # Adapter inference
        ad_result = adapter.predict(abs_path)

        # Logit and prob extraction
        # In official model: id2label 0=Real, 1=Fake
        off_real_logit = float(off_logits[0].item())
        off_fake_logit = float(off_logits[1].item())
        off_rel_logit = off_fake_logit - off_real_logit
        off_fake_prob = float(off_probs[1].item())
        off_pred = "FAKE" if off_fake_prob >= 0.50 else "REAL"

        logit_diff = abs(off_rel_logit - ad_result.raw_logit)
        prob_diff = abs(off_fake_prob - ad_result.fake_probability)
        if logit_diff > max_logit_diff:
            max_logit_diff = logit_diff
        if prob_diff > max_prob_diff:
            max_prob_diff = prob_diff

        if off_pred != ad_result.prediction:
            prediction_mismatches += 1

        results.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth,
            "generator": gen_family,
            "tensor_shape": list(off_tensor.shape),
            "tensor_diff": float(tensor_diff),
            "official_fake_prob": round(off_fake_prob, 6),
            "adapter_fake_prob": round(ad_result.fake_probability, 6),
            "prob_diff": round(prob_diff, 8),
            "official_rel_logit": round(off_rel_logit, 6),
            "adapter_rel_logit": round(ad_result.raw_logit, 6),
            "logit_diff": round(logit_diff, 8),
            "official_pred": off_pred,
            "adapter_pred": ad_result.prediction,
            "pred_match": off_pred == ad_result.prediction,
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/200 samples...")

    print(f"\nPARITY RESULTS FOR ViT ({len(results)} samples):")
    print(f"  Max Preprocessing Tensor Diff: {max_tensor_diff:.10f}")
    print(f"  Max Relative Logit Diff:       {max_logit_diff:.10f}")
    print(f"  Max Fake Probability Diff:     {max_prob_diff:.10f}")
    print(f"  Prediction Mismatches:         {prediction_mismatches} / {len(results)}")

    # Print first 5 from each slice (Real, Midjourney, ADM)
    print("\nSAMPLE PARITY COMPARISON TABLE (First 3 per group):")
    print(f"{'Sample ID':<28} {'GT':<5} {'Gen':<12} {'Off Prob':<10} {'Ad Prob':<10} {'Diff':<10} {'Off Pred':<8} {'Ad Pred':<8}")
    print("-" * 95)
    for sample in results[:3] + results[100:103] + results[150:153]:
        print(f"{sample['sample_id']:<28} {sample['ground_truth']:<5} {sample['generator']:<12} "
              f"{sample['official_fake_prob']:<10.6f} {sample['adapter_fake_prob']:<10.6f} "
              f"{sample['prob_diff']:<10.8f} {sample['official_pred']:<8} {sample['adapter_pred']:<8}")

    # Save output json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "model": "ViT (dima806/deepfake_vs_real_image_detection)",
            "total_samples": len(results),
            "max_tensor_diff": max_tensor_diff,
            "max_logit_diff": max_logit_diff,
            "max_prob_diff": max_prob_diff,
            "prediction_mismatches": prediction_mismatches,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved ViT parity results to: {OUT_JSON}")

if __name__ == "__main__":
    main()
