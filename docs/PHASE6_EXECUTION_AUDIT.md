# Phase 6 Execution Audit: Project State & Model Verification

**Project:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Working Directory:** `C:\Users\user\Desktop\multi model connection`  
**Date:** November 2026  
**Document:** `docs/PHASE6_EXECUTION_AUDIT.md`  

---

## 1. Executive Summary & Objective

In accordance with Phase A of the Autonomous Master Goal for Phase 6, this document establishes the verified technical state of the repository, inspects all integrated detector adapters and schemas, verifies checkpoint files on disk, audits existing dataset structures and evaluation results, and confirms the explicit scope for independent image-based multi-model detection.

---

## 2. Model & Checkpoint Verification

### 2.1 Integrated 2D Image Detectors

| Detector Key | Registry Identifier | Architecture Family | Checkpoint File Path | File Size (Bytes) | SHA-256 Digest | Status & Contract |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`vit`** | `ViTDeepfakeDetector` | Vision Transformer (`ViT-Base/16`) | `models/deepfake_vs_real_image_detection/model.safetensors` | 343,223,968 | Verified in Phase 1 Audit | **ACTIVE / LOADABLE**<br>`DetectorResult` compliant |
| **`univfd`** | `UniversalFakeDetect` | CLIP `ViT-L/14` + Linear Probe | `UniversalFakeDetect/pretrained_weights/fc_weights.pth` | 4,083 (head) + ~1.7 GB (CLIP) | Verified in Phase 1 Audit | **ACTIVE / LOADABLE**<br>`DetectorResult` compliant |
| **`chen2024_convnext`** | `Chen2024ConvNeXtDetector` | ConvNeXt-Base | `models/chen2024_convnext/16_acc0.9993.pth` | 354,605,453 | Verified in Phase 1 Audit | **ACTIVE / LOADABLE**<br>`DetectorResult` compliant |

### 2.2 Exclusion of Temporal Video Model (`df_model.pt`)
- **Physical Location:** `C:\Users\user\Desktop\currently working\model\df_model.pt` (226,597,567 bytes). Completely absent from `multi model connection`.
- **Architecture:** ResNeXt-50 (32x4d) + LSTM recurrent temporal layer (2048 hidden units) + linear classification head.
- **Input Contract:** Requires a 5D temporal video tensor `[Batch, 20, 3, 112, 112]`.
- **Finding:** Evaluating `df_model.pt` on single 2D images forces the LSTM to evaluate at timestep $t=1$ rather than sequence end $t=20$, collapsing accuracy to ~36% (documented in `currently working/model_accuracy_diagnosis.txt`).
- **Conclusion:** Strictly excluded from 2D image benchmarking in compliance with task instructions.

---

## 3. Detector Input/Output Contracts & Semantics

| Detector | Preprocessing Steps | Output Format | Label Mapping | Decision Threshold | Direction Semantics |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **ViT (`dima806`)** | RGB PIL $\to$ Resize to $224 \times 224$ (Bicubic) $\to$ Rescale $[0, 1] \to$ Normalize (Mean: `[0.5, 0.5, 0.5]`, Std: `[0.5, 0.5, 0.5]`). | 2 logits $\to$ Softmax $\to [P(\text{real}), P(\text{fake})]$ | Index 0: Real<br>Index 1: Fake | $P(\text{fake}) \ge 0.50 \implies \text{FAKE}$ | Larger index 1 score $\implies$ FAKE |
| **UnivFD** | RGB PIL $\to$ Resize to 224 (Bicubic) $\to$ CenterCrop $224 \times 224 \to$ OpenAI CLIP Normalization. | Single logit $\to$ Sigmoid $\to P(\text{fake})$ | 0: Real<br>1: Fake | $P(\text{fake}) \ge 0.50 \implies \text{FAKE}$ | Positive logit / larger sigmoid $\implies$ FAKE |
| **ConvNeXt/DRCT** | RGB PIL $\to$ Resize/CenterCrop to $224 \times 224 \to$ Rescale $[0, 1] \to$ ImageNet Normalization. | 2 logits $\to$ Softmax $\to [P(\text{real}), P(\text{fake})]$ | Index 0: Real<br>Index 1: Fake | $P(\text{fake}) \ge 0.50 \implies \text{FAKE}$ | Larger index 1 score $\implies$ FAKE |

All adapters conform to the standard `DetectorResult` dataclass defined in `src/detector_schema.py`.

---

## 4. Dataset Locations & Prior Result Archives

### 4.1 Master Dataset Path
- **Root:** `C:\Users\user\Desktop\currently working\Dataset`
- **Partitions:**
  - `Train`: 140,002 images (70,001 Real, 70,001 Fake)
  - `Validation`: 39,428 images (19,787 Real, 19,641 Fake)
  - `Test`: 10,905 images (5,413 Real, 5,492 Fake)
- **Image Specifications:** $256 \times 256$ RGB JPEGs.
- **Integrity:** Zero corrupted images, zero zero-byte files, zero label contradictions.

### 4.2 Protected Historical Benchmark Artifacts
- **Phase 4 Deliverables (FROZEN - MUST NOT OVERWRITE):**
  - `benchmark/results/phase4_baseline_predictions.csv` (315 rows, SHA-256 preserved)
  - `benchmark/results/phase4_baseline_metrics.json`
  - `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`
- **Phase 5 Deliverables (FROZEN - MUST NOT OVERWRITE):**
  - `benchmark/results/phase5_val_raw_predictions.csv`
  - `benchmark/results/phase5_validation_metrics.json`
  - `benchmark/results/phase5_held_out_test_metrics.json`
  - `docs/PHASE5_FUSION_FEASIBILITY_REPORT.md`
- **Active Manifest:**
  - `benchmark/evaluation_manifest.csv` (159 samples: 105 `held_out_test`, 50 `development_val`, 4 `operational_smoke`).

---

## 5. Hardware & Runtime Constraints

- **System Environment:** Windows 11 Home (64-bit), AMD Ryzen 5 5600H, 8.0 GB RAM, NVIDIA RTX 3050 Laptop GPU (4.0 GB VRAM), Python 3.13.13, PyTorch 2.13.0+cpu.
- **Known Memory Limit:** Concurrently instantiating multiple detector backbones inside a single Python process triggers Windows virtual memory paging exhaustion (`os error 1455`).
- **Required Mitigation:** Strict sequential execution with subprocess isolation and explicit garbage collection between models.

---
*Phase A Audit complete. Moving directly to Phase B: Selection of 200 Labeled Test Images.*

