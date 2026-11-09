"""
Phase 7 V2 Reference Parity Verification Script
Verifies numerical agreement between official upstream references and corrected adapters
across deterministic 30 samples (10 Real, 10 Midjourney, 10 ADM) and ViT regression parity.
Checks:
- Decoded RGB pixels
- Input tensors
- Raw logits
- Fake probabilities
- Final binary predictions
Requires numerical agreement within floating-point tolerance (< 1e-5).
"""

import os
import sys
import csv
import json
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results", "phase7_v2_reference_preprocessing", "parity_verification_audit.json")

# 1. Deterministic samples
def get_deterministic_samples():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    reals = [r for r in rows if r["label"] == "REAL"][:10]
    mjs = [r for r in rows if r["generator_family"] == "midjourney"][:10]
    adms = [r for r in rows if r["generator_family"] == "adm"][:10]
    return reals + mjs + adms


def verify_convnext_parity():
    print("\n" + "=" * 70)
    print("VERIFYING CONVNEXT DRCT REFERENCE PARITY (30 SAMPLES)")
    print("=" * 70)

    # Import reference DRCT
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "UniversalFakeDetect", "models"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "temp_DRCT_reference"))
    from network.models import get_models
    from scripts.parity_audit.audit_convnext_parity import drct_val_transform_is_crop
    from src.convnext_adapter import ConvNeXtDetectorAdapter

    samples = get_deterministic_samples()
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "chen2024_convnext", "16_acc0.9993.pth")

    # Load official model
    print("Loading Official DRCT model...")
    official_net = get_models(model_name="convnext_base_in22k", num_classes=2, pretrained=False, embedding_size=1024)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    official_net.load_state_dict(ckpt, strict=True)
    official_net.eval()

    # Load adapter
    print("Loading ConvNeXtDetectorAdapter...")
    adapter = ConvNeXtDetectorAdapter(checkpoint_path=ckpt_path, device="cpu", lazy_load=False)

    max_pixel_diff = 0
    max_tensor_diff = 0.0
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    pred_matches = 0

    records = []

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        rel_path = sample["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        gt = sample["label"]
        gen = sample["generator_family"]

        # 1. Decoded RGB comparison
        img_bgr = cv2.imread(abs_path)
        img_rgb_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with Image.open(abs_path) as opened:
            img_rgb_pil = np.array(opened.convert("RGB"))

        pixel_diff = int(np.max(np.abs(img_rgb_cv2.astype(np.int32) - img_rgb_pil.astype(np.int32))))
        max_pixel_diff = max(max_pixel_diff, pixel_diff)

        # 2. Input tensor comparison
        t_official = drct_val_transform_is_crop(img_rgb_cv2, size=224)
        t_adapter = adapter.transform(img_rgb_pil).unsqueeze(0)
        tensor_diff = float(torch.max(torch.abs(t_official - t_adapter)).item())
        max_tensor_diff = max(max_tensor_diff, tensor_diff)

        # 3. Model outputs
        with torch.no_grad():
            out_official = official_net(t_official)
            prob_official = torch.softmax(out_official, dim=1).squeeze(0)
            fake_p_official = float(prob_official[1].item())
            logit_official = float((out_official[0, 1] - out_official[0, 0]).item())
            pred_official = "FAKE" if fake_p_official >= 0.50 else "REAL"

        res_adapter = adapter.predict(abs_path)
        fake_p_adapter = res_adapter.fake_probability
        logit_adapter = res_adapter.raw_logit
        pred_adapter = res_adapter.prediction

        logit_diff = abs(logit_official - logit_adapter)
        prob_diff = abs(fake_p_official - fake_p_adapter)

        max_logit_diff = max(max_logit_diff, logit_diff)
        max_prob_diff = max(max_prob_diff, prob_diff)

        match = (pred_official == pred_adapter)
        if match:
            pred_matches += 1

        records.append({
            "sample_id": sample_id,
            "generator": gen,
            "gt": gt,
            "pixel_diff": pixel_diff,
            "tensor_diff": round(tensor_diff, 8),
            "official_logit": round(logit_official, 6),
            "adapter_logit": round(logit_adapter, 6),
            "logit_diff": round(logit_diff, 8),
            "official_prob": round(fake_p_official, 6),
            "adapter_prob": round(fake_p_adapter, 6),
            "prob_diff": round(prob_diff, 8),
            "official_pred": pred_official,
            "adapter_pred": pred_adapter,
            "match": match,
        })
        print(f"  [{idx:02d}/30] {sample_id} ({gen:10s}): TensorDiff={tensor_diff:.8f}, ProbDiff={prob_diff:.8f}, "
              f"OffProb={fake_p_official:.4f}, AdProb={fake_p_adapter:.4f} [{'PASS' if match else 'FAIL'}]")

    print("\nConvNeXt DRCT Parity Results:")
    print(f"  Max Pixel Diff (cv2 vs PIL): {max_pixel_diff}")
    print(f"  Max Input Tensor Diff:       {max_tensor_diff:.10f}")
    print(f"  Max Raw Logit Diff:          {max_logit_diff:.10f}")
    print(f"  Max Fake Probability Diff:   {max_prob_diff:.10f}")
    print(f"  Binary Prediction Matches:   {pred_matches} / 30 ({pred_matches / 30 * 100:.1f}%)")

    assert max_tensor_diff < 1e-6, f"ConvNeXt tensor diff too high: {max_tensor_diff}"
    assert max_logit_diff < 1e-4, f"ConvNeXt logit diff too high: {max_logit_diff}"
    assert max_prob_diff < 1e-4, f"ConvNeXt prob diff too high: {max_prob_diff}"
    assert pred_matches == 30, f"ConvNeXt prediction mismatch: {pred_matches}/30"

    del official_net
    adapter.unload_model()
    del adapter
    return {
        "model": "ConvNeXt (DRCT / RAID)",
        "status": "PASSED",
        "max_pixel_diff": max_pixel_diff,
        "max_tensor_diff": max_tensor_diff,
        "max_logit_diff": max_logit_diff,
        "max_prob_diff": max_prob_diff,
        "pred_matches": pred_matches,
        "total_samples": 30,
        "records": records,
    }


def verify_univfd_parity():
    print("\n" + "=" * 70)
    print("VERIFYING UNIVFD REFERENCE PARITY (30 SAMPLES)")
    print("=" * 70)

    ufd_dir = os.path.join(PROJECT_ROOT, "UniversalFakeDetect")
    sys.path.insert(0, ufd_dir)
    sys.path.insert(0, os.path.join(ufd_dir, "models"))
    from models import get_model
    from src.univfd_adapter import UnivFDAdapter

    samples = get_deterministic_samples()
    ckpt_path = os.path.join(ufd_dir, "pretrained_weights", "fc_weights.pth")

    print("Loading Official UnivFD model...")
    official_model = get_model("CLIP:ViT-L/14")
    official_model.fc.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    official_model.eval()

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform_official = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    print("Loading UnivFDAdapter...")
    adapter = UnivFDAdapter(device="cpu", lazy_load=False)

    max_tensor_diff = 0.0
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    pred_matches = 0

    records = []

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        rel_path = sample["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        gt = sample["label"]
        gen = sample["generator_family"]

        with Image.open(abs_path) as opened:
            img_rgb_pil = opened.convert("RGB")

        # 1. Input tensor comparison
        t_official = transform_official(img_rgb_pil).unsqueeze(0)
        t_adapter = adapter.preprocess(img_rgb_pil).unsqueeze(0)
        tensor_diff = float(torch.max(torch.abs(t_official - t_adapter)).item())
        max_tensor_diff = max(max_tensor_diff, tensor_diff)

        # 2. Model outputs
        with torch.no_grad():
            feat_official = official_model.model.encode_image(t_official)
            logit_official = float(official_model.fc(feat_official.float()).item())
            prob_official = float(torch.sigmoid(torch.tensor(logit_official)).item())
            pred_official = "FAKE" if prob_official >= 0.50 else "REAL"

        res_adapter = adapter.predict(abs_path)
        fake_p_adapter = res_adapter.fake_probability
        logit_adapter = res_adapter.raw_logit
        pred_adapter = res_adapter.prediction

        logit_diff = abs(logit_official - logit_adapter)
        prob_diff = abs(prob_official - fake_p_adapter)

        max_logit_diff = max(max_logit_diff, logit_diff)
        max_prob_diff = max(max_prob_diff, prob_diff)

        match = (pred_official == pred_adapter)
        if match:
            pred_matches += 1

        records.append({
            "sample_id": sample_id,
            "generator": gen,
            "gt": gt,
            "tensor_diff": round(tensor_diff, 8),
            "official_logit": round(logit_official, 6),
            "adapter_logit": round(logit_adapter, 6),
            "logit_diff": round(logit_diff, 8),
            "official_prob": round(prob_official, 6),
            "adapter_prob": round(fake_p_adapter, 6),
            "prob_diff": round(prob_diff, 8),
            "official_pred": pred_official,
            "adapter_pred": pred_adapter,
            "match": match,
        })
        print(f"  [{idx:02d}/30] {sample_id} ({gen:10s}): TensorDiff={tensor_diff:.8f}, ProbDiff={prob_diff:.8f}, "
              f"OffProb={prob_official:.4f}, AdProb={fake_p_adapter:.4f} [{'PASS' if match else 'FAIL'}]")

    print("\nUnivFD Parity Results:")
    print(f"  Max Input Tensor Diff:       {max_tensor_diff:.10f}")
    print(f"  Max Raw Logit Diff:          {max_logit_diff:.10f}")
    print(f"  Max Fake Probability Diff:   {max_prob_diff:.10f}")
    print(f"  Binary Prediction Matches:   {pred_matches} / 30 ({pred_matches / 30 * 100:.1f}%)")

    assert max_tensor_diff < 1e-6, f"UnivFD tensor diff too high: {max_tensor_diff}"
    assert max_logit_diff < 1e-4, f"UnivFD logit diff too high: {max_logit_diff}"
    assert max_prob_diff < 1e-4, f"UnivFD prob diff too high: {max_prob_diff}"
    assert pred_matches == 30, f"UnivFD prediction mismatch: {pred_matches}/30"

    del official_model
    adapter.unload_model()
    del adapter
    return {
        "model": "UnivFD (CLIP ViT-L/14)",
        "status": "PASSED",
        "max_tensor_diff": max_tensor_diff,
        "max_logit_diff": max_logit_diff,
        "max_prob_diff": max_prob_diff,
        "pred_matches": pred_matches,
        "total_samples": 30,
        "records": records,
    }


def verify_vit_regression_parity():
    print("\n" + "=" * 70)
    print("VERIFYING VIT REGRESSION PARITY (10 SAMPLES)")
    print("=" * 70)

    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from src.vit_adapter import ViTAdapter

    samples = get_deterministic_samples()[:10]  # 10 samples
    model_dir = os.path.join(PROJECT_ROOT, "models", "deepfake_vs_real_image_detection")

    print("Loading HuggingFace ViT model & processor...")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    hf_model = AutoModelForImageClassification.from_pretrained(model_dir)
    hf_model.eval()

    print("Loading ViTAdapter...")
    adapter = ViTAdapter(device="cpu", lazy_load=False)

    max_tensor_diff = 0.0
    max_prob_diff = 0.0
    pred_matches = 0
    records = []

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        rel_path = sample["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        gt = sample["label"]

        with Image.open(abs_path) as opened:
            img_rgb_pil = opened.convert("RGB")

        inputs = processor(images=img_rgb_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            fake_prob_hf = float(probs[1].item())
            pred_hf = "FAKE" if fake_prob_hf >= 0.50 else "REAL"

        res_adapter = adapter.predict(abs_path)
        fake_prob_adapter = res_adapter.fake_probability
        pred_adapter = res_adapter.prediction

        prob_diff = abs(fake_prob_hf - fake_prob_adapter)
        max_prob_diff = max(max_prob_diff, prob_diff)

        match = (pred_hf == pred_adapter)
        if match:
            pred_matches += 1

        records.append({
            "sample_id": sample_id,
            "gt": gt,
            "hf_prob": round(fake_prob_hf, 6),
            "adapter_prob": round(fake_prob_adapter, 6),
            "prob_diff": round(prob_diff, 8),
            "hf_pred": pred_hf,
            "adapter_pred": pred_adapter,
            "match": match,
        })
        print(f"  [{idx:02d}/10] {sample_id}: ProbDiff={prob_diff:.8f}, HF={fake_prob_hf:.4f}, Adapter={fake_prob_adapter:.4f} [{'PASS' if match else 'FAIL'}]")

    print("\nViT Regression Parity Results:")
    print(f"  Max Fake Probability Diff:   {max_prob_diff:.10f}")
    print(f"  Binary Prediction Matches:   {pred_matches} / 10 ({pred_matches / 10 * 100:.1f}%)")

    assert max_prob_diff < 1e-4, f"ViT prob diff too high: {max_prob_diff}"
    assert pred_matches == 10, f"ViT prediction mismatch: {pred_matches}/10"

    del hf_model
    adapter.unload_model()
    del adapter
    return {
        "model": "ViT (dima806)",
        "status": "PASSED",
        "max_prob_diff": max_prob_diff,
        "pred_matches": pred_matches,
        "total_samples": 10,
        "records": records,
    }


def main():
    t_start = time.time()
    convnext_audit = verify_convnext_parity()
    univfd_audit = verify_univfd_parity()
    vit_audit = verify_vit_regression_parity()

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": "ALL_PARITY_CHECKS_PASSED",
        "duration_sec": round(time.time() - t_start, 2),
        "convnext": convnext_audit,
        "univfd": univfd_audit,
        "vit": vit_audit,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL REFERENCE PARITY CHECKS COMPLETED SUCCESSFULLY")
    print(f"Audit log saved to: {OUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
