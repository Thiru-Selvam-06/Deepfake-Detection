"""
Audit Phase 4 — UnivFD Reference-Inference Parity Test
Compares the official UniversalFakeDetect implementation against UnivFDAdapter across Phase 7 images.
Evaluates:
  1. Pipeline A: Official validate.py (CenterCrop(224), no resize)
  2. Pipeline B: Official datasets.py (Resize(256) -> CenterCrop(224))
  3. Pipeline C: Official model.preprocess (Resize(224, bicubic) -> CenterCrop(224)) [Adapter baseline]
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UFD_DIR = os.path.join(PROJECT_ROOT, "UniversalFakeDetect")
if UFD_DIR not in sys.path:
    sys.path.insert(0, UFD_DIR)

from src.univfd_adapter import UnivFDAdapter

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
CKPT_PATH = os.path.join(UFD_DIR, "pretrained_weights", "fc_weights.pth")
OUT_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "parity_audit_univfd_results.json")

# Normalization constants from UniversalFakeDetect/validate.py
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Pipeline A: validate.py (Direct CenterCrop without resizing)
transform_validate = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# Pipeline B: datasets.py (Resize(256) -> CenterCrop(224))
transform_datasets = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

def main():
    print("=" * 80)
    print("AUDIT PHASE 4: UnivFD REFERENCE PARITY TEST")
    print("=" * 80)

    # 1. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"Loaded {len(manifest_rows)} manifest samples.")

    # 2. Select 30 deterministic samples (10 Real, 10 MJ, 10 ADM)
    reals = [r for r in manifest_rows if r["label"] == "REAL"][:10]
    mjs = [r for r in manifest_rows if r["generator_family"] == "midjourney"][:10]
    adms = [r for r in manifest_rows if r["generator_family"] == "adm"][:10]
    eval_samples = reals + mjs + adms
    print(f"Evaluating {len(eval_samples)} deterministic samples (10 Real, 10 MJ, 10 ADM).")

    # 3. Setup Official Pipeline
    print("\n[Step 1] Loading Official UnivFD Model from repository...")
    t0 = time.perf_counter()
    from models import get_model
    official_model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    official_model.fc.load_state_dict(state_dict)
    official_model.eval()
    print(f"  Official model loaded in {time.perf_counter() - t0:.2f}s.")
    print(f"  fc_weights keys: {list(state_dict.keys())}")
    print(f"  fc weight shape: {state_dict['weight'].shape}, bias: {state_dict['bias']}")

    # 4. Setup Adapter Pipeline
    print("\n[Step 2] Initializing Project UnivFDAdapter...")
    t0 = time.perf_counter()
    adapter = UnivFDAdapter(device="cpu", lazy_load=False)
    print(f"  Adapter loaded in {time.perf_counter() - t0:.2f}s.")

    # 5. Check Checkpoint & Weights Parity
    print("\n[Step 3] Verifying Checkpoint & Weights Parity between Official and Adapter...")
    diff_fc_w = (official_model.fc.weight - adapter.model.fc.weight).abs().max().item()
    diff_fc_b = (official_model.fc.bias - adapter.model.fc.bias).abs().max().item()
    print(f"  fc weight diff: {diff_fc_w:.10f}")
    print(f"  fc bias diff:   {diff_fc_b:.10f}")
    assert diff_fc_w == 0.0 and diff_fc_b == 0.0, "Weight mismatch in fc layer!"

    # 6. Execute Parity Comparison
    print("\n[Step 4] Executing Inference Comparison across the 30 deterministic samples...")
    results = []
    
    for idx, row in enumerate(eval_samples, 1):
        sample_id = row["sample_id"]
        rel_path = row["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        ground_truth = row["label"]
        gen_family = row["generator_family"]

        with Image.open(abs_path) as opened_img:
            img_rgb = opened_img.convert("RGB")

        # Tensor A: validate.py (CenterCrop(224))
        tensor_val = transform_validate(img_rgb).unsqueeze(0)
        # Tensor B: datasets.py (Resize(256) -> CenterCrop(224))
        tensor_ds = transform_datasets(img_rgb).unsqueeze(0)
        # Tensor C: adapter / model.preprocess (Resize(224) -> CenterCrop(224))
        tensor_ad = adapter.preprocess(img_rgb).unsqueeze(0)

        with torch.no_grad():
            # Pipeline A: validate.py
            feat_val = official_model.model.encode_image(tensor_val)
            logit_val = official_model.fc(feat_val.float()).item()
            prob_val = float(torch.sigmoid(torch.tensor(logit_val)).item())
            pred_val = "FAKE" if prob_val >= 0.50 else "REAL"

            # Pipeline B: datasets.py
            feat_ds = official_model.model.encode_image(tensor_ds)
            logit_ds = official_model.fc(feat_ds.float()).item()
            prob_ds = float(torch.sigmoid(torch.tensor(logit_ds)).item())
            pred_ds = "FAKE" if prob_ds >= 0.50 else "REAL"

            # Pipeline C: official model with adapter preprocessing
            feat_c = official_model.model.encode_image(tensor_ad)
            logit_c = official_model.fc(feat_c.float()).item()
            prob_c = float(torch.sigmoid(torch.tensor(logit_c)).item())
            pred_c = "FAKE" if prob_c >= 0.50 else "REAL"

        # Adapter inference
        ad_res = adapter.predict(abs_path)

        # Tensor differences
        tensor_diff_c_vs_adapter = (tensor_ad - adapter.preprocess(img_rgb).unsqueeze(0)).abs().max().item()
        logit_diff_c_vs_adapter = abs(logit_c - ad_res.raw_logit)
        prob_diff_c_vs_adapter = abs(prob_c - ad_res.fake_probability)

        results.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth,
            "generator": gen_family,
            "image_size": [img_rgb.width, img_rgb.height],
            # Adapter vs Official Pipeline C (exact same preprocessing)
            "adapter_fake_prob": round(ad_res.fake_probability, 6),
            "adapter_raw_logit": round(ad_res.raw_logit, 6),
            "adapter_pred": ad_res.prediction,
            "pipeline_c_fake_prob": round(prob_c, 6),
            "pipeline_c_raw_logit": round(logit_c, 6),
            "pipeline_c_pred": pred_c,
            "prob_diff_c_vs_adapter": round(prob_diff_c_vs_adapter, 8),
            "logit_diff_c_vs_adapter": round(logit_diff_c_vs_adapter, 8),
            # Pipeline A (validate.py CenterCrop 224 without resize)
            "pipeline_a_validate_prob": round(prob_val, 6),
            "pipeline_a_validate_logit": round(logit_val, 6),
            "pipeline_a_validate_pred": pred_val,
            # Pipeline B (datasets.py Resize 256 -> CenterCrop 224)
            "pipeline_b_datasets_prob": round(prob_ds, 6),
            "pipeline_b_datasets_logit": round(logit_ds, 6),
            "pipeline_b_datasets_pred": pred_ds,
        })
        print(f"  [{idx:02d}/30] {sample_id} ({gen_family:10s}): Ad Prob={ad_res.fake_probability:.4f}, "
              f"Pipe C (CLIP Preproc)={prob_c:.4f}, Pipe A (validate.py)={prob_val:.4f}, Pipe B (datasets.py)={prob_ds:.4f}")

    print("\n" + "=" * 80)
    print("UnivFD PARITY SUMMARY ACROSS PREPROCESSING REGIMES")
    print("=" * 80)
    # Check Pipeline C vs Adapter
    max_c_diff = max(r["prob_diff_c_vs_adapter"] for r in results)
    pred_c_matches = sum(1 for r in results if r["adapter_pred"] == r["pipeline_c_pred"])
    print(f"Adapter vs Pipeline C (Same Preprocessing):")
    print(f"  Max probability difference: {max_c_diff:.10f}")
    print(f"  Prediction match:           {pred_c_matches} / {len(results)} ({pred_c_matches/len(results)*100:.1f}%)")

    # Check Pipeline A (validate.py CenterCrop) vs Adapter
    pred_a_matches = sum(1 for r in results if r["adapter_pred"] == r["pipeline_a_validate_pred"])
    print(f"\nAdapter vs Pipeline A (validate.py CenterCrop(224) without resize):")
    print(f"  Prediction match:           {pred_a_matches} / {len(results)} ({pred_a_matches/len(results)*100:.1f}%)")

    # Mean fake probs across slices
    for g in ["none", "midjourney", "adm"]:
        g_rows = [r for r in results if r["generator"] == g]
        lbl = "REAL (ImageNet)" if g == "none" else f"FAKE ({g.upper()})"
        mean_ad = np.mean([r["adapter_fake_prob"] for r in g_rows])
        mean_a = np.mean([r["pipeline_a_validate_prob"] for r in g_rows])
        mean_b = np.mean([r["pipeline_b_datasets_prob"] for r in g_rows])
        print(f"  {lbl:<20}: Adapter Mean={mean_ad:.4f} | Pipe A (Crop224) Mean={mean_a:.4f} | Pipe B (Rz256) Mean={mean_b:.4f}")

    # Save output json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "model": "UnivFD (UniversalFakeDetect CLIP ViT-L/14)",
            "samples_audited": len(results),
            "max_prob_diff_c_vs_adapter": max_c_diff,
            "pred_c_matches": pred_c_matches,
            "pred_a_matches": pred_a_matches,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved UnivFD parity results to: {OUT_JSON}")

if __name__ == "__main__":
    main()
