"""
Audit Phase 4 — ConvNeXt DRCT Reference-Inference Parity Test
Compares the official DRCT implementation (Chen et al. 2024 / ICML 2024) against ConvNeXtDetectorAdapter across Phase 7 images.
Evaluates:
  1. Pipeline A: Official test_GenImage.sh (--is_crop: cv2 BGR->RGB, CenterCrop(224) at native resolution)
  2. Pipeline B: Official DRCT without --is_crop (cv2 BGR->RGB, LongestMaxSize(224) -> Pad(224))
  3. Pipeline C: Adapter transform (PIL RGB, Resize(224, bicubic) -> CenterCrop(224))
"""

import os
import sys
import csv
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UFD_DIR = os.path.join(PROJECT_ROOT, "UniversalFakeDetect")
if os.path.join(UFD_DIR, "models") not in sys.path:
    sys.path.insert(0, os.path.join(UFD_DIR, "models"))

DRCT_DIR = os.path.join(PROJECT_ROOT, "temp_DRCT_reference")
if DRCT_DIR not in sys.path:
    sys.path.insert(0, DRCT_DIR)

from src.convnext_adapter import ConvNeXtDetectorAdapter

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "chen2024_convnext", "16_acc0.9993.pth")
OUT_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "parity_audit_convnext_results.json")

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def drct_val_transform_is_crop(img_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Exact implementation of DRCT create_val_transforms(size=224, is_crop=True):
      - PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0)
      - CenterCrop(height=224, width=224)
      - Normalize(mean=ImageNet, std=ImageNet)
      - ToTensorV2()
    """
    h, w, c = img_rgb.shape
    # Pad if smaller than size
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img_rgb = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        h, w, c = img_rgb.shape

    # Center crop
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    cropped = img_rgb[start_y:start_y + size, start_x:start_x + size]

    # Normalize: (img / 255.0 - mean) / std
    norm = (cropped.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

def drct_val_transform_no_crop(img_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Exact implementation of DRCT create_val_transforms(size=224, is_crop=False):
      - LongestMaxSize(max_size=224)
      - PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0)
      - Normalize(mean=ImageNet, std=ImageNet)
      - ToTensorV2()
    """
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to size x size
    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    norm = (padded.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

def main():
    print("=" * 80)
    print("AUDIT PHASE 4: ConvNeXt DRCT REFERENCE PARITY TEST")
    print("=" * 80)

    # 1. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    print(f"Loaded {len(manifest_rows)} manifest samples.")

    # 2. Select 30 deterministic samples
    reals = [r for r in manifest_rows if r["label"] == "REAL"][:10]
    mjs = [r for r in manifest_rows if r["generator_family"] == "midjourney"][:10]
    adms = [r for r in manifest_rows if r["generator_family"] == "adm"][:10]
    eval_samples = reals + mjs + adms
    print(f"Evaluating {len(eval_samples)} deterministic samples (10 Real, 10 MJ, 10 ADM).")

    # 3. Setup Official DRCT Pipeline
    print("\n[Step 1] Initializing Official DRCT Architecture and Loading Checkpoint...")
    t0 = time.perf_counter()
    from network.models import get_models
    official_model = get_models(
        model_name="convnext_base_in22k",
        num_classes=2,
        pretrained=False,
        embedding_size=1024,
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    official_model.load_state_dict(ckpt, strict=True)
    official_model.eval()
    print(f"  Official DRCT model loaded strictly in {time.perf_counter() - t0:.2f}s.")

    # 4. Setup Adapter Pipeline
    print("\n[Step 2] Initializing Project ConvNeXtDetectorAdapter...")
    t0 = time.perf_counter()
    adapter = ConvNeXtDetectorAdapter(checkpoint_path=CKPT_PATH, device="cpu", lazy_load=False)
    print(f"  Adapter loaded in {time.perf_counter() - t0:.2f}s.")

    # 5. Weights and State-Dict Comparison
    print("\n[Step 3] Verifying Parameter-by-Parameter State-Dict Parity...")
    off_dict = dict(official_model.named_parameters())
    ad_dict = dict(adapter.model.named_parameters())
    
    # Check matching keys
    print(f"  Official model parameter tensors: {len(off_dict)}")
    print(f"  Adapter model parameter tensors:  {len(ad_dict)}")
    assert set(off_dict.keys()) == set(ad_dict.keys()), "Parameter key mismatch!"

    max_param_diff = 0.0
    for name in off_dict:
        diff = (off_dict[name] - ad_dict[name]).abs().max().item()
        if diff > max_param_diff:
            max_param_diff = diff
    print(f"  Max parameter weight difference: {max_param_diff:.10f}")

    # 6. Execute Parity Comparison
    print("\n[Step 4] Executing Parity Inference across the 30 deterministic samples...")
    results = []

    for idx, row in enumerate(eval_samples, 1):
        sample_id = row["sample_id"]
        rel_path = row["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        ground_truth = row["label"]
        gen_family = row["generator_family"]

        # Stage A: Image decoding
        # Official DRCT uses cv2.imread and BGR->RGB
        img_bgr = cv2.imread(abs_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image with cv2: {abs_path}")
        img_rgb_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb_cv2.shape[:2]

        # Stage B: Preprocessing pipelines
        # Pipeline A: Official test_GenImage.sh (--is_crop: CenterCrop 224 at native resolution)
        tensor_is_crop = drct_val_transform_is_crop(img_rgb_cv2, size=224)
        # Pipeline B: Official DRCT without --is_crop (LongestMaxSize 224 -> Pad 224)
        tensor_no_crop = drct_val_transform_no_crop(img_rgb_cv2, size=224)
        # Pipeline C: Adapter transform (PIL Resize 224 bicubic -> CenterCrop 224)
        with Image.open(abs_path) as opened_img:
            img_pil_rgb = opened_img.convert("RGB")
        tensor_adapter = adapter.transform(img_pil_rgb).unsqueeze(0)

        # Forward passes
        with torch.no_grad():
            # Pipeline A: Official is_crop
            out_is_crop = official_model(tensor_is_crop)
            prob_is_crop = torch.softmax(out_is_crop, dim=1).squeeze(0)
            fake_prob_is_crop = float(prob_is_crop[1].item())
            logit_is_crop = float((out_is_crop[0, 1] - out_is_crop[0, 0]).item())
            pred_is_crop = "FAKE" if fake_prob_is_crop >= 0.50 else "REAL"

            # Pipeline B: Official no_crop
            out_no_crop = official_model(tensor_no_crop)
            prob_no_crop = torch.softmax(out_no_crop, dim=1).squeeze(0)
            fake_prob_no_crop = float(prob_no_crop[1].item())
            logit_no_crop = float((out_no_crop[0, 1] - out_no_crop[0, 0]).item())
            pred_no_crop = "FAKE" if fake_prob_no_crop >= 0.50 else "REAL"

            # Official model with Pipeline C (Adapter tensor)
            out_c = official_model(tensor_adapter)
            prob_c = torch.softmax(out_c, dim=1).squeeze(0)
            fake_prob_c = float(prob_c[1].item())
            logit_c = float((out_c[0, 1] - out_c[0, 0]).item())
            pred_c = "FAKE" if fake_prob_c >= 0.50 else "REAL"

        # Adapter inference
        ad_res = adapter.predict(abs_path)

        # Difference between official model on adapter tensor vs adapter output
        diff_prob_c_vs_ad = abs(fake_prob_c - ad_res.fake_probability)
        diff_logit_c_vs_ad = abs(logit_c - ad_res.raw_logit)

        results.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth,
            "generator": gen_family,
            "image_dims": [w, h],
            # Adapter output
            "adapter_fake_prob": round(ad_res.fake_probability, 6),
            "adapter_raw_logit": round(ad_res.raw_logit, 6),
            "adapter_pred": ad_res.prediction,
            # Official model with adapter tensor (Pipeline C)
            "official_pipe_c_prob": round(fake_prob_c, 6),
            "official_pipe_c_logit": round(logit_c, 6),
            "official_pipe_c_pred": pred_c,
            "diff_prob_c_vs_ad": round(diff_prob_c_vs_ad, 8),
            # Pipeline A (Official GenImage script with --is_crop: CenterCrop 224 native)
            "official_pipe_a_is_crop_prob": round(fake_prob_is_crop, 6),
            "official_pipe_a_is_crop_logit": round(logit_is_crop, 6),
            "official_pipe_a_is_crop_pred": pred_is_crop,
            # Pipeline B (Official without --is_crop: LongestMaxSize 224)
            "official_pipe_b_no_crop_prob": round(fake_prob_no_crop, 6),
            "official_pipe_b_no_crop_logit": round(logit_no_crop, 6),
            "official_pipe_b_no_crop_pred": pred_no_crop,
        })
        print(f"  [{idx:02d}/30] {sample_id} ({gen_family:10s}): Ad Prob={ad_res.fake_probability:.4f} | "
              f"Pipe C (Model on Ad Tensor)={fake_prob_c:.4f} | "
              f"Pipe A (GenImage is_crop)={fake_prob_is_crop:.4f} | "
              f"Pipe B (no_crop)={fake_prob_no_crop:.4f}", flush=True)

    print("\n" + "=" * 80)
    print("ConvNeXt DRCT PARITY SUMMARY ACROSS PREPROCESSING REGIMES")
    print("=" * 80)
    max_c_diff = max(r["diff_prob_c_vs_ad"] for r in results)
    pred_c_matches = sum(1 for r in results if r["adapter_pred"] == r["official_pipe_c_pred"])
    print(f"Adapter vs Official Model on Same Input Tensor (Pipeline C):")
    print(f"  Max probability difference: {max_c_diff:.10f}")
    print(f"  Prediction match:           {pred_c_matches} / {len(results)} ({pred_c_matches/len(results)*100:.1f}%)")

    pred_a_matches = sum(1 for r in results if r["adapter_pred"] == r["official_pipe_a_is_crop_pred"])
    print(f"\nAdapter vs Pipeline A (Official GenImage test_GenImage.sh --is_crop):")
    print(f"  Prediction match:           {pred_a_matches} / {len(results)} ({pred_a_matches/len(results)*100:.1f}%)")

    for g in ["none", "midjourney", "adm"]:
        g_rows = [r for r in results if r["generator"] == g]
        lbl = "REAL (ImageNet)" if g == "none" else f"FAKE ({g.upper()})"
        mean_ad = np.mean([r["adapter_fake_prob"] for r in g_rows])
        mean_a = np.mean([r["official_pipe_a_is_crop_prob"] for r in g_rows])
        mean_b = np.mean([r["official_pipe_b_no_crop_prob"] for r in g_rows])
        print(f"  {lbl:<20}: Adapter Mean={mean_ad:.4f} | Pipe A (is_crop) Mean={mean_a:.4f} | Pipe B (no_crop) Mean={mean_b:.4f}")

    # Save output json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "model": "ConvNeXt (Chen et al. 2024 / DRCT 16_acc0.9993.pth)",
            "samples_audited": len(results),
            "max_prob_diff_c_vs_ad": max_c_diff,
            "pred_c_matches": pred_c_matches,
            "pred_a_matches": pred_a_matches,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved ConvNeXt parity results to: {OUT_JSON}")

if __name__ == "__main__":
    main()
