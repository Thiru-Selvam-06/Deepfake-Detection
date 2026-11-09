"""
Audit Phase 4 Full 200-Sample Evaluation:
Runs official UnivFD (CLIP ViT-L/14 + fc_weights.pth) with validate.py transform (CenterCrop 224, no resize)
across all 200 Phase 7 external generalization samples.
Saves results to benchmark/results/parity_audit_univfd_full_200.json.
"""

import os
import sys
import csv
import time
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "UniversalFakeDetect"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "UniversalFakeDetect", "models"))

from models import get_model

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
CKPT_PATH = os.path.join(PROJECT_ROOT, "UniversalFakeDetect", "pretrained_weights", "fc_weights.pth")
OUT_JSON = os.path.join(PROJECT_ROOT, "benchmark/results/parity_audit_univfd_full_200.json")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

transform_validate = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

def main():
    print("=" * 80)
    print("FULL 200-SAMPLE UNIVFD VALIDATE.PY AUDIT (CENTERCROP 224)")
    print("=" * 80)

    # 1. Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} samples from manifest.")

    # 2. Load model
    print("Loading UnivFD CLIP:ViT-L/14 and fc_weights.pth...")
    t0 = time.time()
    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s.")

    # 3. Inference loop
    records = []
    y_true = []
    y_score = []

    real_correct = 0
    real_total = 0
    mj_correct = 0
    mj_total = 0
    adm_correct = 0
    adm_total = 0

    print("Running inference across 200 samples...")
    t_start = time.time()

    for idx, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        rel_path = row["relative_path"]
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        gt = row["label"]
        gen = row["generator_family"]

        with Image.open(abs_path) as opened_img:
            img_rgb = opened_img.convert("RGB")

        tensor = transform_validate(img_rgb).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            fake_p = float(torch.sigmoid(out).item())
            pred = "FAKE" if fake_p >= 0.50 else "REAL"

        is_fake_gt = 1 if gt == "FAKE" else 0
        y_true.append(is_fake_gt)
        y_score.append(fake_p)

        if gt == "REAL":
            real_total += 1
            if pred == "REAL":
                real_correct += 1
        elif gen == "midjourney":
            mj_total += 1
            if pred == "FAKE":
                mj_correct += 1
        elif gen == "adm":
            adm_total += 1
            if pred == "FAKE":
                adm_correct += 1

        records.append({
            "sample_id": sample_id,
            "label": gt,
            "generator": gen,
            "fake_prob": fake_p,
            "pred": pred,
            "correct": pred == gt
        })

        if idx % 50 == 0:
            print(f"  Processed {idx}/200 samples ({time.time() - t_start:.1f}s)...")

    t_total = time.time() - t_start
    print(f"Inference completed in {t_total:.1f}s ({t_total/len(rows):.3f}s per sample).")

    # 4. Metrics
    total_acc = (real_correct + mj_correct + adm_correct) / len(rows)
    specificity = real_correct / real_total if real_total > 0 else 0
    mj_recall = mj_correct / mj_total if mj_total > 0 else 0
    adm_recall = adm_correct / adm_total if adm_total > 0 else 0
    fake_recall = (mj_correct + adm_correct) / (mj_total + adm_total)
    auc = roc_auc_score(y_true, y_score)

    print("\n" + "=" * 80)
    print("UNIVFD VALIDATE.PY AUDIT RESULTS (ALL 200 SAMPLES):")
    print("=" * 80)
    print(f"  Total Accuracy:    {total_acc * 100:.2f}% ({real_correct + mj_correct + adm_correct} / {len(rows)})")
    print(f"  Real Specificity:  {specificity * 100:.2f}% ({real_correct} / {real_total})")
    print(f"  Midjourney Recall: {mj_recall * 100:.2f}% ({mj_correct} / {mj_total})")
    print(f"  ADM Recall:        {adm_recall * 100:.2f}% ({adm_correct} / {adm_total})")
    print(f"  Fake Recall (Agg): {fake_recall * 100:.2f}% ({mj_correct + adm_correct} / {mj_total + adm_total})")
    print(f"  ROC-AUC:           {auc:.4f}")

    # Percentiles
    real_probs = [r["fake_prob"] for r in records if r["label"] == "REAL"]
    mj_probs = [r["fake_prob"] for r in records if r["generator"] == "midjourney"]
    adm_probs = [r["fake_prob"] for r in records if r["generator"] == "adm"]

    def calc_stats(probs):
        arr = np.array(probs)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    stats = {
        "real": calc_stats(real_probs),
        "midjourney": calc_stats(mj_probs),
        "adm": calc_stats(adm_probs),
    }

    out_data = {
        "model": "UnivFD (CLIP ViT-L/14 + fc_weights.pth)",
        "transform": "validate.py CenterCrop(224) without resize",
        "total_samples": len(rows),
        "accuracy": total_acc,
        "specificity": specificity,
        "fake_recall": fake_recall,
        "midjourney_recall": mj_recall,
        "adm_recall": adm_recall,
        "roc_auc": auc,
        "score_distributions": stats,
        "records": records,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved full results to: {OUT_JSON}")

if __name__ == "__main__":
    main()
