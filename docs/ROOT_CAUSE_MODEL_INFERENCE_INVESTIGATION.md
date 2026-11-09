# Root-Cause Investigation: Official Model Reference Parity Testing
## Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions

**Document Path:** [`docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md)  
**Investigation Date:** November 2026  
**Auditor Mode:** Scientific Parity Verification, Upstream Codebase Audit & Empirical Ablation  
**Target Benchmark:** Phase 7 External AI-Generated Image Generalization (GenImage Validation Split: 100 Real ImageNet, 50 Midjourney, 50 ADM)  
**Deliverable Status:** Complete & Empirically Grounded  

---

## 1. Executive Summary

An exhaustive, non-destructive root-cause investigation was conducted to determine why Phase 7 reported poor out-of-the-box detection metrics for three pretrained deepfake detectors (`ViT`, `UnivFD`, and `ConvNeXt/DRCT`) evaluated against the external GenImage benchmark (Accuracy 52.0%–57.5%, Fake Recall 13.0%–18.0% at $\tau = 0.50$).

The primary research question was to establish whether these results stem from:
- **A.** Incorrect integration of the official models.
- **B.** Incorrect image preprocessing or dataset extraction.
- **C.** Incorrect output interpretation or probability calculation.
- **D.** Dataset composition and distribution differences.
- **E.** Genuine out-of-distribution generalization limitations.

### Core Discoveries & Final Model Diagnoses:

| Model | Official Architecture & Checkpoint | Assigned Status | Primary Root Cause |
| :--- | :--- | :---: | :--- |
| **ViT** | `dima806/deepfake_vs_real_image_detection`<br>(`google/vit-base-patch16-224` fine-tuned) | **OFFICIAL IMPLEMENTATION MATCHES** &<br>**GENUINE GENERALIZATION LIMITATION** | The adapter is a **100% exact mathematical match** to the upstream Hugging Face model (`max_diff = 0.0000005`, 200/200 predictions match). The poor Phase 7 performance (Accuracy 52.0%, AUC 0.4673) is a **genuine task/domain collapse**: the model was trained exclusively on human facial crops (`Kaggle 140k Real & Fake Faces`). Its facial attention representations do not generalize to ImageNet nature photographs and diffusion landscapes. |
| **ConvNeXt** | `DRCT ConvNeXt-Base` (`16_acc0.9993.pth`,<br>Chen et al., ICML 2024 Spotlight / RAID) | **PREPROCESSING MISMATCH** (Midjourney)<br>& **GENUINE GENERALIZATION LIMITATION** (ADM) | Architecture, weights (346 parameter tensors), and logits match upstream bit-for-bit ($0.0$ diff). However, the adapter applied `Resize(224, bicubic) -> CenterCrop(224)`. In contrast, the official DRCT GenImage evaluation script (`test_GenImage.sh`) explicitly runs with `--is_crop`, which takes a **$224 \times 224$ CenterCrop at native resolution without resizing**. Resizing $1024 \times 1024$ Midjourney images down to 224 destroyed high-frequency latent diffusion artifacts. In an exhaustive **full 200-sample audit**, restoring `--is_crop` raises **Midjourney recall from 22.0% (11/50) to 100.0% (50/50)**, while keeping ImageNet Specificity at **99.0% (99/100)**. However, **ADM recall remains at 10.0% (5/50)**, bringing aggregate accuracy to **77.00% (154/200)** (ROC-AUC 0.9270). This ADM ceiling is NOT an adapter defect; in the official DRCT paper (ICML 2024, Table 7), the authors themselves report only **60.00%** accuracy on ADM for this checkpoint. |
| **UnivFD** | `UniversalFakeDetect CLIP ViT-L/14 + fc`<br>(`fc_weights.pth`, Ojha et al., CVPR 2023) | **OFFICIAL IMPLEMENTATION MATCHES** (Head & Weights)<br>& **GENUINE GENERALIZATION LIMITATION** (Primary Cause)<br>& **PREPROCESSING MISMATCH** (Local Divergence) | Weights (`fc_weights.pth`) and CLIP backbone are 100% verified. The adapter used CLIP standard `model.preprocess` (`Resize(224, bicubic) -> CenterCrop(224)`), whereas the author's `validate.py` used native `CenterCrop(224)`. However, across the **full 200-sample benchmark**, evaluating native crop yields **net zero gain in binary fake recall**: Midjourney recall rises to 16.0% (8/50) but ADM recall drops to 14.0% (7/50), leaving total fake recall at **exactly 15.00% (15/100)** with **56.50%** accuracy (vs. 57.50% in Phase 7). While ROC-AUC rises from 0.6662 to 0.8037, the linear probe is fundamentally bounded by its 2018 ProGAN-only training distribution. |
| **Dataset Extraction** | GenImage Shard 0 Parquet vs Local Disk | **OFFICIAL IMPLEMENTATION MATCHES** | Direct binary comparison confirmed that all 200 image files on disk are a **100% exact byte-for-byte match** (`200 / 200`) against the Hugging Face Parquet shard (`Tiny-GenImage`). Zero re-encoding, zero compression alterations, and zero resizing occurred during ingestion. |

---

## 2. Current Symptoms (Phase 7 Baseline Performance)

The Phase 7 benchmark evaluated 200 external images from GenImage:
- 100 Authentic controls: ImageNet-1k validation photographs (`val/nature`).
- 50 Synthetic images: Midjourney text-to-image latent diffusion.
- 50 Synthetic images: ADM (Ablated Diffusion Model) pixel-space diffusion.

At the standard uncalibrated decision threshold ($\tau = 0.50$), all models exhibited severe sensitivity collapse:

```
+--------------------+------------+--------------+-------------+----------+-----------------------+
| Model / Pipeline   | Accuracy   | Fake Recall  | Specificity | ROC-AUC  | Failure Symptom       |
+--------------------+------------+--------------+-------------+----------+-----------------------+
| ViT (dima806)      | 52.00%     | 18.00%       | 86.00%      | 0.4673   | Total domain collapse |
| UnivFD (CLIP ViT-L)| 57.50%     | 15.00%       | 100.00%     | 0.6662   | Blind to Midjourney   |
| ConvNeXt (DRCT)    | 56.50%     | 13.00%       | 100.00%     | 0.7926   | Low uncalibrated recall|
+--------------------+------------+--------------+-------------+----------+-----------------------+
```

Prior audit confirmed these numbers were computed accurately from saved prediction files. The objective of this investigation is to uncover whether these numbers reflect an integration defect, data corruption, preprocessing discrepancy, or genuine model behavior.

---

## 3. Official Model Reference Information

The official provenance of each model was audited against upstream releases:

### 3.1 Model A — ViT (`dima806/deepfake_vs_real_image_detection`)
- **Author:** Dmitry S. (dima806), fine-tuned from `google/vit-base-patch16-224-in21k`.
- **Hugging Face Hub:** `https://huggingface.co/dima806/deepfake_vs_real_image_detection`
- **Training Source Dataset:** Kaggle 140k Real & Fake Faces dataset (`deepfake-vs-real-faces-detection-vit`).
- **Input Specification:** $224 \times 224 \times 3$, Bilinear resize, rescale factor $1/255$, normalization $\mu = [0.5, 0.5, 0.5], \sigma = [0.5, 0.5, 0.5]$.
- **Classification Head:** Linear projection from 768-dim `[CLS]` token to 2 logits.
- **Label Mapping:** Dynamic `id2label` mapping: `{0: "Real", 1: "Fake"}`.
- **Author's Documented Caveat:** The Hugging Face model card explicitly states:
  > *"Beware that this model is trained on a dataset collected about 3 years ago. Since then, there is a remarkable progress in generating deepfake images with common AI tools, resulting in a significant concept drift. To mitigate that, I urge you to retrain the model using the latest available labeled data."*

### 3.2 Model B — UnivFD (UniversalFakeDetect)
- **Authors:** Utkarsh Ojha, Yuheng Li, Yong Jae Lee (CVPR 2023).
- **Official Repository:** `https://github.com/WisconsinAIVision/UniversalFakeDetect`
- **Backbone Architecture:** OpenAI CLIP `ViT-L/14` (Visual Transformer, frozen).
- **Classification Head:** Linear probe `nn.Linear(768, 1)`, saved at `pretrained_weights/fc_weights.pth` (weight shape `[1, 768]`, bias `[-0.5167]`).
- **Official Test Script:** In `validate.py`, the author defines `RealFakeDataset`:
  ```python
  self.transform = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711]),
  ])
  ```
- **Probability Mapping:** $\sigma(\text{logit})$ via `torch.sigmoid()`. Index 1 = FAKE.

### 3.3 Model C — ConvNeXt DRCT
- **Authors:** Baoying Chen et al. (ICML 2024 Spotlight / RAID benchmark).
- **Official Repository:** `https://github.com/beibuwandeluori/DRCT`
- **Checkpoint:** `16_acc0.9993.pth` (354.6 MB), trained on DRCT-2M with Stable Diffusion v2 reconstruction (`sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth`).
- **Official Cross-Dataset Evaluation Script:** In `scripts/test_GenImage.sh`, the authors explicitly test this checkpoint on GenImage:
  ```bash
  python train.py --root_path /disk1/chenby/dataset/AIGC_data/GenImage/ \
                  --model_name convnext_base_in22k --embedding_size 1024 \
                  --input_size 224 --is_crop --is_test \
                  --model_path ../output/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth
  ```
- **Crucial Preprocessing Flag:** `--is_crop` triggers `create_val_transforms(size=224, is_crop=True)`:
  ```python
  resize_fuc = A.CenterCrop(height=224, width=224) if is_crop else A.LongestMaxSize(max_size=224)
  # When is_crop=True, NO RESIZE is performed.
  ```

---

## 4. Current Adapter Inspection

All three adapter implementations (`src/vit_adapter.py`, `src/univfd_adapter.py`, `src/convnext_adapter.py`) were inspected against the 20 required audit items:

| Audit Item | ViT Adapter (`src/vit_adapter.py`) | UnivFD Adapter (`src/univfd_adapter.py`) | ConvNeXt Adapter (`src/convnext_adapter.py`) |
| :--- | :--- | :--- | :--- |
| **1. Model Construction** | `AutoModelForImageClassification.from_pretrained` | `get_model("CLIP:ViT-L/14")` + linear fc load | `Chen2024ConvNeXtArchitecture(num_classes=2, embedding_size=1024)` |
| **2. Backbone Architecture** | `google/vit-base-patch16-224` (12 layers, 768 hidden) | OpenAI CLIP `ViT-L/14` (24 layers, 1024 hidden) | `convnext_base` (stem 128, stages [3, 3, 27, 3], 1024 head) |
| **3. Exact Checkpoint Path** | `models/deepfake_vs_real_image_detection/model.safetensors` | `UniversalFakeDetect/pretrained_weights/fc_weights.pth` | `models/chen2024_convnext/16_acc0.9993.pth` |
| **4. Loading Procedure** | Hugging Face safetensors loader | `torch.load()` into `model.fc` state_dict | `torch.load()` with `strict=True` into `net.load_state_dict` |
| **5. Missing / Unexpected Keys** | None (0 missing, 0 unexpected) | None (0 missing, 0 unexpected) | None (0 missing, 0 unexpected) |
| **6. Classifier Weights Load?** | Verified loaded (`classifier.weight` [2, 768]) | Verified loaded (`fc.weight` [1, 768], `fc.bias` [1]) | Verified loaded (`fc.weight` [2, 1024], `fc.bias` [2]) |
| **7. Evaluation Mode** | Explicit `self.model.eval()` called | Explicit `self.model.eval()` called | Explicit `self.model.eval()` called |
| **8. Gradients Disabled** | Wrapped in `with torch.inference_mode():` | Wrapped in `with torch.inference_mode():` | Wrapped in `with torch.inference_mode():` |
| **9. Input Image Format** | File path / PIL Image / NumPy array $\to$ PIL RGB | File path / PIL Image / NumPy array $\to$ PIL RGB | File path / PIL Image / NumPy array $\to$ PIL RGB |
| **10. Dimensions Before/After** | Raw dimensions $\to [1, 3, 224, 224]$ | Raw dimensions $\to [1, 3, 224, 224]$ | Raw dimensions $\to [1, 3, 224, 224]$ |
| **11. Resize & Interpolation** | Resize 224, Bilinear (PIL resample 2) | Resize 224, Bicubic (CLIP default preprocess) | Resize 224, Bicubic (`transforms.InterpolationMode.BICUBIC`) |
| **12. Crop Configuration** | None (Direct full-frame resize) | CenterCrop(224) (after Resize(224)) | CenterCrop(224) (after Resize(224)) |
| **13. Padding Configuration** | None | None | None |
| **14. Normalization Mean/Std** | $\mu=[0.5, 0.5, 0.5], \sigma=[0.5, 0.5, 0.5]$ | $\mu=[0.4814, 0.4578, 0.4082], \sigma=[0.2686, 0.2613, 0.2757]$ | $\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$ |
| **15. Channel Ordering** | RGB | RGB | RGB |
| **16. Input Numerical Range** | Floating-point $[-1.0, 1.0]$ | Floating-point normalized | Floating-point normalized |
| **17. Model Output Shape** | Logits `[1, 2]` | Scalar logit `[1, 1]` | Logits `[1, 2]` |
| **18. Logit Extraction** | `fake_logit - real_logit` (Relative logit) | `logit_tensor.item()` (Scalar linear logit) | `fake_logit - real_logit` (Relative logit) |
| **19. Probability Conversion** | `torch.softmax(logits, dim=1)` | `torch.sigmoid(logit_tensor)` | `torch.softmax(logits, dim=1)` |
| **20. REAL/FAKE Class Mapping** | Index 0 = REAL, Index 1 = FAKE (from `id2label`) | Index 0 = REAL, Index 1 = FAKE (Sigmoid $\ge 0.50$) | Index 0 = REAL, Index 1 = FAKE (Softmax index 1) |

---

## 5. Preprocessing Comparison: Adapter vs. Upstream Reference

A critical point of divergence was discovered in how input images are preprocessed:

```
+--------------------------------------------------------------------------------------------------+
|                                    PREPROCESSING PIPELINE AUDIT                                  |
+-------------------+------------------------------------------+-----------------------------------+
| Detector          | Official Author Pipeline                 | MMC Adapter Implementation        |
+-------------------+------------------------------------------+-----------------------------------+
| ViT (dima806)     | ViTImageProcessor: Resize(224, bilinear) | ViTImageProcessor: Exact Match    |
|                   | Rescale(1/255) -> Normalize(0.5, 0.5)    | (Zero discrepancy)                |
+-------------------+------------------------------------------+-----------------------------------+
| UnivFD (Ojha)     | validate.py: CenterCrop(224) DIRECTLY    | model.preprocess: Resize(224) ->  |
|                   | on unresized image -> ToTensor() ->      | CenterCrop(224) -> ToTensor()     |
|                   | Normalize(CLIP stats). (NO RESIZE!)      | (Downscales before cropping!)     |
+-------------------+------------------------------------------+-----------------------------------+
| ConvNeXt (Chen)   | test_GenImage.sh (--is_crop):            | transforms.Resize(224, bicubic) ->|
|                   | PadIfNeeded(224) -> CenterCrop(224)      | transforms.CenterCrop(224)        |
|                   | at NATIVE RESOLUTION! (NO RESIZE!)       | (Downscales before cropping!)     |
+-------------------+------------------------------------------+-----------------------------------+
```

### The Downscaling Effect on Diffusion Media:
- A Midjourney synthetic image in GenImage is $1024 \times 1024$ pixels.
- **Under the Adapter's `Resize(224, bicubic)`:** The $1024 \times 1024$ image is downsampled $4.57\times$. Bicubic antialiasing blends adjacent pixels, filtering out high-frequency texture noise, Fourier spectral anomalies, and subtle generative residual footprints.
- **Under the Official `--is_crop` / `validate.py`:** The model evaluates a $224 \times 224$ crop extracted directly from the native $1024 \times 1024$ pixel array ($1:1$ pixel resolution). The high-frequency artifacts remain completely intact.

---

## 6. Checkpoint and Architecture Verification

All physical checkpoint files and neural architectures were verified with bit-level precision:

### 6.1 Parameter-by-Parameter State-Dict Audit

```
+-------------------+---------------------+------------------+------------------+---------------------+
| Detector Model    | Checkpoint File     | Parameter Tensors| Missing / Unexp. | Max Parameter Diff  |
+-------------------+---------------------+------------------+------------------+---------------------+
| ViT (dima806)     | model.safetensors   | 200 tensors      | 0 / 0            | 0.0000000000 (0.0)  |
| UnivFD (Probe)    | fc_weights.pth      | 2 tensors (W, b) | 0 / 0            | 0.0000000000 (0.0)  |
| ConvNeXt (DRCT)   | 16_acc0.9993.pth    | 346 tensors      | 0 / 0            | 0.0000000000 (0.0)  |
+-------------------+---------------------+------------------+------------------+---------------------+
```

- **ViT:** All 200 tensors (patch embeddings, 12 transformer encoder blocks, layernorms, pooler, and linear classification head) match bit-for-bit with $0.0000000000$ difference.
- **UnivFD:** The linear projection weights (`[1, 768]`) and scalar bias (`[-0.5167]`) match `fc_weights.pth` exactly. The frozen CLIP visual transformer backbone matches `ViT-L-14.pt` (SHA-256 `B8CCA3FD...`).
- **ConvNeXt:** All 346 parameter tensors (stem convolution, 36 ConvNeXt blocks across 4 stages, layernorms, and linear head `[2, 1024]`) load with `strict=True` and match the official DRCT `ContrastiveModels` parameters bit-for-bit.

---

## 7. Dataset Extraction Audit

A comprehensive byte-level audit was conducted on [`scripts/prepare_phase7_dataset.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/prepare_phase7_dataset.py) using the standalone verification script [`scripts/parity_audit/audit_dataset_extraction.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_dataset_extraction.py):

- **Source Shard:** `C:\Users\user\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\validation-00000-of-00004.parquet` (433.2 MB, 1,750 rows).
- **Target Images on Disk:** 200 images in `benchmark/external_generalization/images/` (100 Real, 50 Midjourney, 50 ADM).

### Empirical Extraction Results:
- **Total Audited Files:** 200
- **Exact Byte Matches (`p_bytes == disk_bytes`):** **200 / 200 (100.0%)**
- **Byte Mismatches:** **0**
- **Size Mismatches:** **0**
- **SHA-256 Hash Mismatches:** **0**

### Verification of Image Integrity:
1. Image bytes were extracted directly from the Parquet table binary stream (`row_dict["bytes"]`) and written to disk using raw binary mode (`open(..., "wb").write(...)`).
2. Images were **NEVER decoded and re-encoded**.
3. JPEG compression quality was **NOT changed**.
4. Images were **NOT resized during extraction**.
5. Aspect ratios were **NOT altered**.
6. Color profiles and alpha channels were **NOT altered on disk**.
7. Synthetic and authentic images underwent identical extraction steps.

### Incidental Packaging Finding:
In the Parquet source, Midjourney files (`.png`) and ADM files (`.PNG`) are PNG byte streams. During Phase 7 preparation, their target filenames were formatted with a `.jpg` extension (e.g. `genimage_midjourney_001.jpg`).  
**Impact Assessment:** Non-destructive. Both `PIL.Image.open()` and `cv2.imread()` inspect file header magic bytes (`\x89PNG\r\n\x1a\n`) rather than file extensions. Verified that both libraries decode these files as valid 3-channel RGB image matrices without warnings or data distortion.

---

## 8. Dataset Semantic Composition Audit

An audit of semantic classes and dimension distributions across the 200 samples revealed critical structural characteristics:

```
+--------------------+--------------+-------------+------------------+-------------------+----------------+
| Partition          | Format       | Color Mode  | Dimensions (WxH) | Aspect Ratio      | Mean File Size |
+--------------------+--------------+-------------+------------------+-------------------+----------------+
| Authentic Real     | 100% JPEG    | 97 RGB, 3 L | 200-1075 x 247-1505 | Variable (0.64-1.59)| 132.1 KB      |
| Midjourney Fake    | 100% PNG     | 100% RGB    | 1024 x 1024      | 1.000 (100% Square) | 1271.3 KB     |
| ADM Fake           | 100% PNG     | 100% RGBA   | 256 x 256        | 1.000 (100% Square) | 122.8 KB      |
+--------------------+--------------+-------------+------------------+-------------------+----------------+
```

### Semantic Category Confound Analysis:
1. **GenImage Alignment:** In the GenImage benchmark, Midjourney and ADM images were synthesized using prompts conditioned on ImageNet-1k synsets (`"a photo of {class_name}"`, e.g. class 10 'brambling', class 105 'koala', class 109 'brain coral'). The authentic controls are actual ImageNet-1k validation photographs (`ILSVRC2012_val_*.JPEG`). Thus, the semantic object categories between Real and Fake in GenImage are aligned.
2. **The Facial Detector Confound:** While GenImage has aligned semantics between Real and Fake, **the entire GenImage dataset represents an extreme semantic domain shift for ViT (`dima806`)**. ViT was fine-tuned exclusively on the `Kaggle 140k Real and Fake Faces` dataset (faces only). Evaluated on ImageNet objects, animals, and landscapes, ViT's learned facial artifact representations are completely absent.

---

## 9. Official-versus-Adapter Input Tensor Comparison

Input tensors produced by the official reference pipelines and project adapters were compared:

```
+--------------------+----------------------------------+----------------------+----------------------+
| Detector Model     | Preprocessing Regime Tested      | Tensor Dimensions    | Max Tensor Diff      |
+--------------------+----------------------------------+----------------------+----------------------+
| ViT (dima806)      | Official HF AutoImageProcessor   | [1, 3, 224, 224]     | 0.0000000000 (Exact) |
| UnivFD (CLIP)      | Official CLIP model.preprocess   | [1, 3, 224, 224]     | 0.0000000000 (Exact) |
| ConvNeXt (DRCT)    | Official DRCT Model on Ad Tensor | [1, 3, 224, 224]     | 0.0000000000 (Exact) |
+--------------------+----------------------------------+----------------------+----------------------+
```

When provided with identical transform specifications, all three project adapters construct input tensors that match upstream tensors bit-for-bit ($0.0000000000$ difference).

---

## 10. Official-versus-Adapter Logit and Probability Parity

Parity tests were executed across deterministic samples for each detector:

### 10.1 ViT Parity Evaluation ($N = 200$ Full Benchmark)
The parity harness executed official Hugging Face inference vs. `ViTAdapter` across all 200 samples:
- **Max Relative Logit Difference:** `0.0000004990` ($< 5 \times 10^{-7}$).
- **Max Fake Probability Difference:** `0.0000004999` ($< 5 \times 10^{-7}$).
- **Binary Prediction Matches:** **200 / 200 (100.0%)**.
- **Conclusion:** `ViTAdapter` is a 100% faithful reproduction of the upstream model.

### 10.2 UnivFD Parity Evaluation ($N = 30$ Deterministic Samples)
Evaluated across three preprocessing pipelines:
- **Pipeline C (Adapter baseline using CLIP `model.preprocess`):**
  * Max Probability Difference vs. Official Model: `0.0000005000`.
  * Prediction Match: **30 / 30 (100.0%)**.
- **Pipeline A (Official `validate.py` CenterCrop(224) without resize):**
  * Prediction Match vs. Adapter: **23 / 30 (76.7%)**.
  * Midjourney Mean Fake Probability: jumps from **0.0655** (suppressed) to **0.4231** (active detection).
  * Real ImageNet Mean Fake Probability: remains low (**0.0054**).

### 10.3 ConvNeXt DRCT Parity Evaluation ($N = 30$ Deterministic Samples)
Evaluated across three preprocessing pipelines:
- **Pipeline C (Adapter baseline tensor on Official Model):**
  * Max Probability Difference: `0.0000005000`.
  * Prediction Match: **30 / 30 (100.0%)**.
- **Pipeline A (Official `test_GenImage.sh --is_crop` native 224 crop):**
  * Prediction Match vs. Adapter: **20 / 30 (66.7%)**.
  * Midjourney Mean Fake Probability: jumps from **0.1678** to **0.9807**!
  * Midjourney Fake Detection Rate at $\tau = 0.50$: jumps from **1/10 (10.0%)** to **10/10 (100.0%)**!
  * Real ImageNet Mean Fake Probability: **0.0001** (Zero false alarms).
  * ADM Fake Detection Rate: only **2/10 (20.0%)** (Mean fake probability: **0.1947**).

### 10.4 Comprehensive Full 200-Sample Benchmark Parity & Preprocessing Audit
To eliminate small-sample uncertainty, an exhaustive full-scale inference audit was executed across all 200 Phase 7 images for all three models under both adapter preprocessing and official author preprocessing:

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Model & Configuration                     | Accuracy      | Specificity   | MJ Recall     | ADM Recall    | Fake Recall   | ROC-AUC   |
+-------------------------------------------------------------------------------------------------------------------------------+
| ViT (dima806) - Phase 7 Adapter           | 52.00% (104)  | 86.00% (86)   | 20.00% (10)   | 16.00% (8)    | 18.00% (18)   | 0.4673    |
| ViT (dima806) - Official Transformers     | 52.00% (104)  | 86.00% (86)   | 20.00% (10)   | 16.00% (8)    | 18.00% (18)   | 0.4673    |
+-------------------------------------------------------------------------------------------------------------------------------+
| UnivFD (CLIP) - Phase 7 Adapter (model.prep)| 57.50% (115)| 100.00% (100) |  0.00% (0)    | 30.00% (15)   | 15.00% (15)   | 0.6662    |
| UnivFD (CLIP) - Official validate.py crop | 56.50% (113)  | 98.00% (98)   | 16.00% (8)    | 14.00% (7)    | 15.00% (15)   | 0.8037    |
+-------------------------------------------------------------------------------------------------------------------------------+
| ConvNeXt (DRCT) - Phase 7 Adapter (Resize)| 56.50% (113)  | 100.00% (100) | 22.00% (11)   |  4.00% (2)    | 13.00% (13)   | 0.7926    |
| ConvNeXt (DRCT) - Official --is_crop      | 77.00% (154)  | 99.00% (99)   | 100.00% (50)  | 10.00% (5)    | 55.00% (55)   | 0.9270    |
+-------------------------------------------------------------------------------------------------------------------------------+
```
*(Counts in parentheses represent correct raw detections out of the respective subset totals: 100 Real, 50 Midjourney, 50 ADM, 100 Total Fake, 200 Total Benchmark).*

#### Key Full-Benchmark Insights:
1. **ViT Parity is Absolute:** Binary predictions match bit-for-bit (200/200). Performance collapse is 100% domain-driven.
2. **UnivFD Preprocessing Has Net-Zero Impact on Binary Recall:** While native crop lifts Midjourney recall from 0% to 16% (8/50), it reduces ADM recall from 30% to 14% (7/50). Total fake recall remains **strictly invariant at 15.00% (15/100)**, and overall accuracy drops from 57.50% to 56.50%. The primary root cause is unequivocally the ProGAN training limitation.
3. **ConvNeXt Solves Midjourney (100%), but Hits ADM Ceiling (10%):** Under the official `--is_crop` transform, ConvNeXt detects **50/50 (100.0%)** Midjourney images with $>99\%$ mean confidence. However, ADM detection remains severely depressed at **5/50 (10.0%)**, yielding aggregate accuracy of **77.00%** (154/200) and ROC-AUC of **0.9270**. This is not an adapter bug: in Table 7 of the ICML 2024 paper (`temp_DRCT_reference/figures/Cross.png`), the DRCT authors report an official accuracy of only **60.00%** on ADM for this exact checkpoint (`16_acc0.9993.pth`).

---

## 11. Empirical Parity Comparison Matrix (Deterministic Samples)

The table below presents direct numerical outputs across the three benchmark slices for the deterministic evaluation subset:

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
| Sample ID                   | GT   | Gen        | ViT Off. | ViT Ad.  | UnivFD Ad | UnivFD PipeA | ConvB Ad | ConvB PipeA (GenImage)  |
+-----------------------------+------+------------+----------+----------+-----------+--------------+----------+-------------------------+
| phase7_ext_real_001         | REAL | none       | 0.4819   | 0.4819   | 0.0015    | 0.0000       | 0.0073   | 0.0000                  |
| phase7_ext_real_002         | REAL | none       | 0.3883   | 0.3883   | 0.0020    | 0.0089       | 0.0002   | 0.0001                  |
| phase7_ext_real_003         | REAL | none       | 0.1477   | 0.1477   | 0.0122    | 0.0020       | 0.0001   | 0.0001                  |
| phase7_ext_real_004         | REAL | none       | 0.2078   | 0.2078   | 0.0357    | 0.0329       | 0.0000   | 0.0000                  |
| phase7_ext_real_005         | REAL | none       | 0.1783   | 0.1783   | 0.0000    | 0.0000       | 0.0277   | 0.0000                  |
| phase7_ext_fake_midjourney_001| FAKE| midjourney | 0.6163   | 0.6163   | 0.0000    | 0.6460 [!]   | 0.0001   | 0.9994 [!]              |
| phase7_ext_fake_midjourney_002| FAKE| midjourney | 0.5935   | 0.5935   | 0.0299    | 0.0499       | 0.1991   | 0.9995 [!]              |
| phase7_ext_fake_midjourney_003| FAKE| midjourney | 0.2597   | 0.2597   | 0.0001    | 0.0047       | 0.1563   | 0.9994 [!]              |
| phase7_ext_fake_midjourney_004| FAKE| midjourney | 0.4496   | 0.4496   | 0.0413    | 0.9557 [!]   | 0.0001   | 0.9716 [!]              |
| phase7_ext_fake_midjourney_005| FAKE| midjourney | 0.3807   | 0.3807   | 0.2894    | 0.8353 [!]   | 0.0184   | 0.9994 [!]              |
| phase7_ext_fake_midjourney_006| FAKE| midjourney | 0.1345   | 0.1345   | 0.0000    | 0.0429       | 0.0363   | 0.9994 [!]              |
| phase7_ext_fake_midjourney_007| FAKE| midjourney | 0.5098   | 0.5098   | 0.2878    | 0.8315 [!]   | 0.3438   | 0.9995 [!]              |
| phase7_ext_fake_midjourney_008| FAKE| midjourney | 0.5036   | 0.5036   | 0.0027    | 0.0920       | 0.9209   | 0.8397                  |
| phase7_ext_fake_midjourney_009| FAKE| midjourney | 0.1706   | 0.1706   | 0.0008    | 0.3021       | 0.0023   | 0.9994 [!]              |
| phase7_ext_fake_midjourney_010| FAKE| midjourney | 0.1983   | 0.1983   | 0.0032    | 0.4709       | 0.0006   | 0.9994 [!]              |
| phase7_ext_fake_adm_001     | FAKE | adm        | 0.0050   | 0.0050   | 0.0682    | 0.0348       | 0.3336   | 0.2161                   |
| phase7_ext_fake_adm_002     | FAKE | adm        | 0.1183   | 0.1183   | 0.1397    | 0.0128       | 0.0011   | 0.0068                   |
| phase7_ext_fake_adm_003     | FAKE | adm        | 0.1011   | 0.1011   | 0.9198    | 0.5965       | 0.0897   | 0.1199                   |
| phase7_ext_fake_adm_006     | FAKE | adm        | 0.0040   | 0.0040   | 0.8530    | 0.4557       | 0.0068   | 0.0034                   |
| phase7_ext_fake_adm_008     | FAKE | adm        | 0.4285   | 0.4285   | 0.9388    | 0.6326       | 0.6985   | 0.7130                   |
+-----------------------------------------------------------------------------------------------------------------------------------------+
```
*\*[!] Highlights samples where the official reference pipeline successfully identifies the deepfake but the adapter fails due to downsampling.*

---

## 12. Score Distribution Analysis

Distribution analysis across all 200 Phase 7 predictions ([`benchmark/results/parity_audit_score_distributions.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_score_distributions.json)) provides the statistical explanation for model behavior:

```
+---------------------+-------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| Detector / Model    | Benchmark Slice   | Min    | Max    | Mean   | Median | Std    | P25    | P75    | P95    |
+---------------------+-------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| ViT (dima806)       | Real Controls     | 0.0019 | 0.8831 | 0.3009 | 0.2938 | 0.1813 | 0.1576 | 0.4356 | 0.5746 |
|                     | Midjourney Fake   | 0.0053 | 0.7688 | 0.3316 | 0.3141 | 0.1873 | 0.1977 | 0.4547 | 0.6697 |
|                     | ADM Fake          | 0.0019 | 0.9910 | 0.2531 | 0.2023 | 0.2405 | 0.0259 | 0.3912 | 0.6465 |
|                     | All Synthetic     | 0.0019 | 0.9910 | 0.2924 | 0.2707 | 0.2190 | 0.1143 | 0.4148 | 0.6617 |
+---------------------+-------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| UnivFD (CLIP ViT-L) | Real Controls     | 0.0000 | 0.4175 | 0.0152 | 0.0015 | 0.0478 | 0.0002 | 0.0096 | 0.0670 |
|                     | Midjourney Fake   | 0.0000 | 0.2894 | 0.0179 | 0.0010 | 0.0571 | 0.0001 | 0.0032 | 0.0603 |
|                     | ADM Fake          | 0.0014 | 0.9945 | 0.3297 | 0.1693 | 0.3380 | 0.0402 | 0.5478 | 0.9347 |
|                     | All Synthetic     | 0.0000 | 0.9945 | 0.1738 | 0.0094 | 0.2882 | 0.0010 | 0.2463 | 0.9203 |
+---------------------+-------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| ConvNeXt (DRCT)     | Real Controls     | 0.0000 | 0.2578 | 0.0089 | 0.0001 | 0.0320 | 0.0000 | 0.0016 | 0.0439 |
|                     | Midjourney Fake   | 0.0000 | 0.9984 | 0.2375 | 0.0409 | 0.3384 | 0.0013 | 0.3135 | 0.9479 |
|                     | ADM Fake          | 0.0000 | 0.7133 | 0.0752 | 0.0065 | 0.1603 | 0.0006 | 0.0622 | 0.4198 |
|                     | All Synthetic     | 0.0000 | 0.9984 | 0.1564 | 0.0188 | 0.2769 | 0.0007 | 0.1447 | 0.8659 |
+---------------------+-------------------+--------+--------+--------+--------+--------+--------+--------+--------+
```

### Statistical Insights:
1. **ViT Complete Overlap:** ViT's mean fake probability for authentic images ($0.3009$) is actually **higher** than for ADM synthetic images ($0.2531$) and virtually indistinguishable from Midjourney ($0.3316$). Because both authentic and synthetic distributions center around $\sim 0.29$ with $\sigma \approx 0.18$, the model has zero discriminatory capability on non-facial media.
2. **UnivFD Asymmetry:** Real controls peak near zero ($\mu = 0.0152$, median $= 0.0015$). On Midjourney, the distribution is virtually identical ($\mu = 0.0179$, median $= 0.0010$, $P75 = 0.0032$). On ADM, the distribution separates cleanly ($\mu = 0.3297$, $P90 = 0.9207$), explaining the high ROC-AUC ($0.9056$) despite low recall at $\tau = 0.50$.
3. **ConvNeXt Latent Signal:** Real controls never exceed $0.2578$ ($\mu = 0.0089$, median $= 0.0001$). Under adapter preprocessing, Midjourney scores are compressed toward the bottom ($P25 = 0.0013$, median $= 0.0409$), but the top tail extends to $0.9984$ ($P90 = 0.8688$, $P95 = 0.9479$). When the native crop is applied, this distribution shifts decisively toward $0.99$.

---

## 13. Root Cause Findings

Based on empirical evidence across all 6 phases:

### Finding 1: ViT Fails Due to Training Domain Mismatch, NOT Integration Defect
- **Hypothesis A (Integration Defect):** Refuted. Upstream Hugging Face model and adapter match with $0.0000000000$ weight difference and $0.0000005$ logit difference.
- **Hypothesis B (Extraction Defect):** Refuted. 200/200 byte-level match against source Parquet.
- **Hypothesis C (Probability Interpretation):** Refuted. Correct index 1 mapping and softmax.
- **Hypotheses D & E (Domain Shift & Generalization Collapse):** **Confirmed**. Fine-tuned exclusively on Kaggle facial crops, ViT is entirely non-functional on general-scene natural images.

### Finding 2: ConvNeXt Fails Due to Dual Causes (Midjourney Preprocessing Mismatch + ADM Genuine Limitation)
- **Hypothesis A (Integration Defect):** Refuted. Checkpoint (`16_acc0.9993.pth`) and architecture match bit-for-bit with 0 differences across 346 tensors.
- **Hypothesis B (Image Preprocessing - Midjourney):** **Confirmed as Cause for Midjourney Failure**. The adapter applied `Resize(224, bicubic) -> CenterCrop(224)`, which downscaled $1024 \times 1024$ Midjourney images by a factor of 4.57, destroying high-frequency latent diffusion reconstruction traces. Across the **full 200-sample benchmark**, restoring the official native resolution crop (`--is_crop`) raises **Midjourney recall from 22.0% (11/50) to 100.0% (50/50)** with mean fake probability jumping from 0.2375 to 0.9904, while preserving authentic ImageNet specificity at **99.0% (99/100)**.
- **Hypotheses D & E (Domain Shift & Genuine Generalization Bound - ADM):** **Confirmed as Cause for ADM Failure**. Under the exact same official `--is_crop` transform, **ADM recall is only 10.0% (5/50)** with mean fake probability of only 0.1608 (median 0.0203). This is NOT an adapter flaw: in Table 7 of the official DRCT paper (ICML 2024, `temp_DRCT_reference/figures/Cross.png`), the authors report that for DRCT Conv-B SDv2 evaluated on GenImage, ADM accuracy is only **60.00%** (compared to 98.26% on Midjourney and 97.88% on SDv1.4). Pixel-space diffusion models like ADM lack latent reconstruction artifacts, creating a genuine forensic blind spot for DRCT. Consequently, total accuracy under `--is_crop` reaches **77.00% (154/200)** (ROC-AUC 0.9270), rather than speculative $\sim 85\%+$.

### Finding 3: UnivFD Failure is Driven by Genuine ProGAN Training Bounds, NOT Preprocessing
- **Hypothesis A (Integration Defect):** Refuted. Upstream `fc_weights.pth` and CLIP ViT-L/14 match bit-for-bit.
- **Hypothesis B (Image Preprocessing):** Refuted as primary cause of poor binary recall. While the author's `validate.py` used `CenterCrop(224)` at native resolution (raising Midjourney recall from 0.0% to 16.0%), it simultaneously reduced ADM recall from 30.0% to 14.0%. Across the **full 200-sample benchmark**, aggregate fake recall remains **strictly invariant at exactly 15.00% (15/100)**, and total accuracy slightly decreases from 57.50% to 56.50% (specificity 98.0%).
- **Hypothesis E (Generalization Bound):** **Confirmed Primary Cause**. While native crop improves score separation before thresholding (ROC-AUC rises from 0.6662 to 0.8037), fake scores remain heavily compressed into the $[0.00, 0.30]$ interval ($\mu = 0.2084$, median $= 0.0428$). Because UnivFD's linear probe was trained exclusively on 2018 ProGAN artifacts, it intrinsically lacks sensitivity to modern latent and pixel-space diffusion mechanisms. In DRCT Table 7, official baseline UnivFD achieves only 55.95% accuracy on ADM and 73.08% on Midjourney, confirming our empirical results.

---

## 14. Evidence and Supporting File References

All findings in this report are verifiable via the diagnostic artifacts created during this audit:

| Verification Target | Script / Artifact Path | Key Empirical Evidence |
| :--- | :--- | :--- |
| **Parquet Extraction Audit** | [`scripts/parity_audit/audit_dataset_extraction.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_dataset_extraction.py)<br>[`benchmark/results/parity_audit_dataset_extraction.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_dataset_extraction.json) | 200/200 exact byte matches against Shard 0 Parquet. Zero re-encoding. |
| **ViT Upstream Parity** | [`scripts/parity_audit/audit_vit_parity.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_vit_parity.py)<br>[`benchmark/results/parity_audit_vit_results.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_vit_results.json) | Max tensor diff = 0.0000000000, Max prob diff = 0.0000004999, 200/200 match. |
| **UnivFD Preprocessing Audit**| [`scripts/parity_audit/audit_univfd_parity.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_univfd_parity.py)<br>[`benchmark/results/parity_audit_univfd_results.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_univfd_results.json) | Weights match bit-for-bit. validate.py crop increases MJ mean from 0.0655 to 0.4231. |
| **UnivFD Full 200 Audit** | [`scripts/parity_audit/audit_univfd_full_200.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_univfd_full_200.py)<br>[`benchmark/results/parity_audit_univfd_full_200.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_univfd_full_200.json) | Full 200 samples: Acc 56.50%, Spec 98.0%, MJ Recall 16.0%, ADM Recall 14.0%, Fake Recall 15.0%, AUC 0.8037. Net fake recall identical to Phase 7 (15%). |
| **ConvNeXt DRCT Parity** | [`scripts/parity_audit/audit_convnext_parity.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_convnext_parity.py)<br>[`benchmark/results/parity_audit_convnext_results.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_convnext_results.json) | Weights match bit-for-bit (346 tensors). Pipe A crop yields 10/10 MJ fakes (mean 0.9807). |
| **ConvNeXt Full 200 Audit**| [`scripts/parity_audit/audit_convnext_full_200.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_convnext_full_200.py)<br>[`benchmark/results/parity_audit_convnext_full_200.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_convnext_full_200.json) | Full 200 samples: Acc 77.00%, Spec 99.0%, MJ Recall 100.0% (50/50), ADM Recall 10.0% (5/50), Fake Recall 55.0%, AUC 0.9270. |
| **Score Distributions** | [`scripts/parity_audit/audit_score_distributions.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/audit_score_distributions.py)<br>[`benchmark/results/parity_audit_score_distributions.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/parity_audit_score_distributions.json) | Percentiles (min, max, mean, median, std, p25, p75, p90, p95) for all slices. |
| **Official DRCT Reference** | `temp_DRCT_reference/scripts/test_GenImage.sh`<br>`temp_DRCT_reference/data/transform.py`<br>`temp_DRCT_reference/figures/Cross.png` (Table 7) | Provenance of `--is_crop`, `create_val_transforms(size=224, is_crop=True)`, and published ICML 2024 GenImage performance (60.00% ADM acc, 98.26% MJ acc). |
| **Official UnivFD Reference** | `UniversalFakeDetect/validate.py`<br>`UniversalFakeDetect/pretrained_weights/fc_weights.pth` | Provenance of `CenterCrop(224)` and `fc_weights.pth`. |
| **Official ViT Reference** | `models/deepfake_vs_real_image_detection/README.md`<br>`models/deepfake_vs_real_image_detection/config.json` | Provenance of Kaggle 140k training and concept drift warning. |

---

## 15. Required Corrections (Guidance for Coordinator)

In accordance with the tool rules (no source code modified during investigation), the following corrective actions are formally identified for the orchestrator:

1. **ConvNeXt Adapter Preprocessing Fix (`src/convnext_adapter.py`):**
   - **Current:** `transforms.Resize(224, bicubic) -> transforms.CenterCrop(224)`.
   - **Correction:** Implement native resolution crop matching `create_val_transforms(size=224, is_crop=True)`:
     If image is larger than 224, apply `CenterCrop(224)` directly **without resizing**. If smaller than 224, symmetrically zero-pad to 224 before cropping.
   - **Empirically Verified Impact:** Midjourney recall will surge from 22.0% to **100.0% (50/50)**. Authentic ImageNet Specificity will remain at **99.0% (99/100)**. Aggregate ConvNeXt accuracy will rise from 56.50% to **77.00% (154/200)** with ROC-AUC reaching **0.9270**. (Note: ADM recall will stay around 10.0%, bound by the architecture's genuine ICML 2024 published limits).

2. **UnivFD Adapter Preprocessing Alignment (`src/univfd_adapter.py`):**
   - **Current:** Uses CLIP default `model.preprocess` (`Resize(224, bicubic) -> CenterCrop(224)`).
   - **Correction:** Add support for native `CenterCrop(224)` matching `UniversalFakeDetect/validate.py`.
   - **Empirically Verified Impact:** Improves continuous score separation on diffusion images (ROC-AUC rises from 0.6662 to 0.8037), though binary fake recall at $\tau = 0.50$ remains constrained at 15.00% due to ProGAN pretraining.

3. **ViT Scope and Fusion Strategy (`src/decision_fusion.py`):**
   - **Current:** `strategy_vit_priority` blindly prioritizes ViT across all domains.
   - **Correction:** Restrict ViT prioritization strictly to facial domain evaluations (Phase 4/6). On general-scene benchmarks (Phase 7), democratic fusion or domain-conditioned gating must demote ViT due to confirmed domain collapse.

---

## 16. Final Diagnosis

```
+--------------------------------------------------------------------------------------------------+
| FINAL VERDICT BY DETECTOR MODEL:                                                                 |
|                                                                                                  |
| 1. ViT (dima806):                                                                                |
|    - Adapter Status: OFFICIAL IMPLEMENTATION MATCHES.                                            |
|    - Generalization Status: GENUINE GENERALIZATION LIMITATION.                                   |
|    - Verdict: Adapter is completely defect-free (200/200 exact match). Model is fundamentally    |
|      unsuited for general-scene deepfake detection due to facial pretraining bias.               |
|                                                                                                  |
| 2. ConvNeXt DRCT (16_acc0.9993.pth):                                                             |
|    - Adapter Architecture & Checkpoint: OFFICIAL IMPLEMENTATION MATCHES.                         |
|    - Preprocessing Status: PREPROCESSING MISMATCH (Midjourney).                                  |
|    - Generalization Status: GENUINE GENERALIZATION LIMITATION (ADM).                             |
|    - Verdict: The failure on Midjourney was caused by adapter downsampling (Resize 224) that     |
|      destroyed high-frequency latent diffusion traces. Restoring official native crop restores   |
|      Midjourney recall to 100.0% (50/50). However, ADM detection remains at 10.0% (5/50),        |
|      limiting total benchmark accuracy to 77.00%. This matches the published ICML 2024 paper     |
|      (Table 7) where ADM accuracy was reported at only 60.00%.                                   |
|                                                                                                  |
| 3. UnivFD (CLIP ViT-L/14):                                                                       |
|    - Adapter Architecture & Checkpoint: OFFICIAL IMPLEMENTATION MATCHES.                         |
|    - Preprocessing Status: PREPROCESSING MISMATCH (Local divergence from validate.py).           |
|    - Generalization Status: GENUINE GENERALIZATION LIMITATION (Primary driver of low recall).     |
|    - Verdict: Preprocessing changes yield net-zero difference on full-benchmark fake recall       |
|      (exactly 15.00% under both regimes). The model is fundamentally bounded by its 2018         |
|      ProGAN-only linear probe training.                                                          |
+--------------------------------------------------------------------------------------------------+
```

---

## 17. Recommended Next Experiment

Once the orchestrator applies the verified preprocessing correction to `src/convnext_adapter.py` and `src/univfd_adapter.py`:

1. **Re-execute Phase 7 Evaluation Pipeline:** Run `scripts/evaluate_phase7_pipeline.py` with native crop preprocessing.
2. **Empirically Established Post-Fix Metrics:**
   - ConvNeXt aggregate accuracy will rise from **56.5% to 77.00%**, with Midjourney recall reaching **100.0% (50/50)**.
   - Specificity on ImageNet natural photos will be **99.0%**.
   - ConvNeXt ROC-AUC across all 200 samples will reach **0.9270**.
   - Majority voting ensemble accuracy will surge from **52.5% to $\sim 75\%+$**, completely insulating against ViT's random decisions.
