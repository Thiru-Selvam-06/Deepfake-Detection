# Model Status Audit and Project Progression Report

**Project Title:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Audit Date:** 2026-10-30  
**Audit Scope:** Read-only inspection of Desktop workspaces (`multi model connection`, `currently working`, `Deepfake-Detection-main`, `file 1`, `PG final year Project`), model caches, source code, adapters, test suites, and benchmark outputs.  
**Auditor Mode:** Safe diagnostic audit — zero modifications to source code, checkpoints, configurations, or packages.

---

## 1. Executive Summary

A comprehensive, non-destructive audit of all project workspaces on the Desktop was conducted to uncover every detector model, establish exact provenance and physical condition, evaluate implementation completeness, and reconstruct historical project progression.

### Key Audit Findings

1. **Total Distinct Detectors Discovered:** Five (5) distinct model architectures across eight (8) physical weight checkpoints.
2. **`df_model.pt` Reality:** `df_model.pt` is **NOT** present in `multi model connection`. It resides in `currently working\model\df_model.pt` (and older repositories `file 1` and `Deepfake-Detection-main`). Crucially, `df_model.pt` is **not an image detector**; it is a **temporal video deepfake detector** (ResNeXt-50-32x4d + LSTM) trained on 20-frame FaceForensics++ video sequences. When evaluated on static images, the LSTM evaluates at timestep $t=1$ instead of $t=20$, collapsing accuracy to ~36%. It was deliberately set aside during the transition to image-based multi-model detection.
3. **Primary Image Detector Suite in `multi model connection`:** Three (3) image detector models are integrated into a standardized, decoupled adapter layer (`src/`):
   - **`chen2024_convnext` (ConvNeXt-Base):** Diffusion Reconstruction Contrastive Training (DRCT, ICML 2024 Spotlight / RAID). Checkpoint: `16_acc0.9993.pth` (354.6 MB). Fully operational adapter, 13/13 diagnostic checks passed.
   - **`vit` (ViT-Base/16):** Fine-tuned Vision Transformer (`dima806/deepfake_vs_real_image_detection`). Checkpoint: `model.safetensors` (343.2 MB). Highly accurate on facial manipulations (99.30% measured on 1,000 images in `currently working`), fully operational adapter.
   - **`univfd` (CLIP ViT-L/14 + Linear Probe):** UniversalFakeDetect (CVPR 2023). Checkpoint: `fc_weights.pth` (4.1 KB) + cached CLIP backbone `ViT-L-14.pt` (932.8 MB). Fully operational adapter, verified against official reference implementations.
4. **Additional Discovered Model in `currently working`:** `CommunityForensics-DeepfakeDet-ViT` (CVPR 2025, ViT-Small 384x384). Checkpoints: `model.safetensors` (87.3 MB) and legacy weights. Evaluated in `currently working` where it achieved 52.0% on face crops (indicating general scene training does not transfer directly to isolated faces). Not yet ported to `multi model connection`.
5. **Folder Completeness Comparison:** 
   - `C:\Users\user\Desktop\multi model connection` contains the **most advanced, clean, and extensible multi-model architecture** for the research objective ("Cross-Model Evaluation of Deepfake Image Detection"). It implements a formal dataclass schema (`DetectorResult`), dynamic registry with lazy loading (`detector_registry.py`), isolated adapters for all three core image detectors, and diagnostic benchmark scripts.
   - `C:\Users\user\Desktop\currently working` contains the **most comprehensive empirical evaluation datasets and results** (10,905 labeled Kaggle face images, 1,000-sample balanced benchmarks, robustness ablations, video pipeline, and MTCNN face detection), but is architecturally a monolithic Flask/Gradio web deployment.
6. **Current Primary Blocker:** In `multi model connection`, when `run_detector_benchmark.py` runs all three detectors in a single Python process on CPU, the sequential retention of large model weights (CLIP ViT-L/14 + ConvNeXt-Base + ViT-Base) exhausts the Windows virtual memory paging file (`os error 1455`), crashing ViT inference. Additionally, no decision-fusion layer exists yet (currently only observational agreement tracking is implemented), and the local benchmark dataset contains only 5 diagnostic smoke-test samples.

---

## 2. Complete Model Inventory

| Model Identifier | Architecture | Checkpoint Path | Size (Bytes) | SHA-256 Hash | Provenance & Source | Intended Task |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`df_model.pt`** | ResNeXt-50-32x4d + LSTM(2048, 2048) + Linear(2048, 2) | `c:\Users\user\Desktop\currently working\model\df_model.pt` *(also in `file 1` and `Deepfake-Detection-main`; pointer in `PG final year Project`)* | 226,597,567 (~216.1 MB) | `B893DE920A78A23F23F56F5EDADFD21D0217DC0C5BF15691EA6EE13CB9AFCD3C` | Pretrained on FaceForensics++ video sequences (PyTorch state_dict). Derived from original video deepfake project. | Sequential Video Deepfake Detection (20-frame temporal sequence) |
| **`chen2024_convnext`** | ConvNeXt-Base (1024 embedding) + Linear(1024, 2) | `c:\Users\user\Desktop\multi model connection\models\chen2024_convnext\16_acc0.9993.pth` | 354,605,453 (~338.2 MB) | `F3638355676C0C07BC0B0E1082EBA698314971ED4771D0B55B207E946A2D74E0` | ICML 2024 Spotlight: *DRCT* (Chen et al., BokingChen) trained on DRCT-2M with SDv2 reconstruction; mirrored in RAID benchmark (`aimagelab/RAID_ckpt`). | Image Deepfake Detection (Diffusion & Synthetic Media) |
| **`vit` (dima806)** | `ViTForImageClassification` (`google/vit-base-patch16-224`) | `c:\Users\user\Desktop\multi model connection\models\deepfake_vs_real_image_detection\model.safetensors` *(identical copy in `currently working`)* | 343,223,968 (~327.3 MB) | `B38E7D3DE2BE44B224CB4CFFE881A0C4B75F7283C5D4A2017FBADD58BD41FDEF` | Fine-tuned from Google ViT-Base on Kaggle 140k Real & Fake Faces dataset (`dima806/deepfake_vs_real_image_detection`). SafeTensors. | Static Image Facial Deepfake Detection |
| **`univfd` (Probe)** | Linear Layer (`nn.Linear(768, 1)`) | `c:\Users\user\Desktop\multi model connection\UniversalFakeDetect\pretrained_weights\fc_weights.pth` | 4,083 (~4.0 KB) | `477100745713BCC957BEB2B40859536859B6483FD6301B3B9293151B194C7847` | CVPR 2023: *UniversalFakeDetect* (Ojha et al., WisconsinAIVision). Trained on ProGAN artifacts. | General Synthetic Image Detection (Linear Probe Head) |
| **`univfd` (Backbone)** | OpenAI CLIP `ViT-L/14` (Visual Transformer, frozen) | `C:\Users\user\.cache\clip\ViT-L-14.pt` | 932,768,134 (~889.6 MB) | `B8CCA3FD41AE0C99BA7E8951ADF17D267CDB84CD88BE6F7C2E0ECA1737A03836` | Official OpenAI CLIP ViT-L/14 release, auto-downloaded by `clip.load()` in `UniversalFakeDetect`. | Feature Representation Backbone for UnivFD |
| **`CommunityForensics ViT` (Fixed)** | `ViTForImageClassification` (`timm/vit_small_patch16_384`, ViT-Small) | `c:\Users\user\Desktop\currently working\deepfake-detector-model-v1-main\models\CommunityForensics-DeepfakeDet-ViT\model.safetensors` *(and `model_fixed.safetensors`)* | 87,270,764 (~83.2 MB) | `275BA982236DDD6AFDDF7131F8133E89F537574B964CF8FA5825B4956D741692` | CVPR 2025: *Community Forensics* (buildborderless/OpenSight). Trained on 2.7M samples across 4,803 generators. | Multi-Generator Scene Deepfake Detection |
| **`CommunityForensics ViT` (Legacy)** | Pre-fix ViT configuration checkpoint | `c:\Users\user\Desktop\currently working\deepfake-detector-model-v1-main\models\CommunityForensics-DeepfakeDet-ViT\model_legacy.safetensors` | 143,969,256 (~137.3 MB) | `91A3C4D1B7D453B4B74E377C8E079662A6CC2C97F8CA448DAEDDA8864EC54605` | Pre-July 2026 legacy release from CommunityForensics (superseded by fixed checkpoint). | Archived/Historical Checkpoint |

---

## 3. Model Status Matrix

Each discovered model is evaluated against the 8 required lifecycle stages. Statuses are assigned based on strict empirical evidence.

| Status Stage | `df_model.pt` | `chen2024_convnext` | `vit` (dima806) | `univfd` | `CommunityForensics` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Discovered** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **VERIFIED** |
| **2. Identified** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **VERIFIED** |
| **3. Loadable** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **VERIFIED** |
| **4. Integrated** | **PARTIAL** *(Video only in `currently working`; NOT in `multi model connection`)* | **VERIFIED** *(in `multi model connection`)* | **VERIFIED** *(in both folders)* | **VERIFIED** *(in `multi model connection`)* | **NOT VERIFIED** *(Only in `currently working` eval script)* |
| **5. Smoke-tested** | **VERIFIED** *(Video passed; static image failed)* | **VERIFIED** *(13/13 checks passed in `scripts/test_convnext_adapter.py`)* | **VERIFIED** *(5/5 checks passed in `scripts/test_vit_adapter.py`)* | **VERIFIED** *(5/5 checks passed in `scripts/test_univfd_adapter.py`)* | **VERIFIED** *(in `currently working` test suite)* |
| **6. Inference-validated** | **VERIFIED** *(as video model; INVALIDATED as image detector)* | **VERIFIED** *(Validated against DRCT reference code)* | **VERIFIED** *(Config label mapping & ViTProcessor verified)* | **VERIFIED** *(Validated against `validate.py` & `datasets.py`)* | **VERIFIED** *(384x384 sigmoid output verified)* |
| **7. Benchmark-evaluated**| **VERIFIED** *(4 videos passed; 100 static images failed ~36%)* | **PARTIAL** *(5 diagnostic smoke images in MMC; NOT on full DRCT-2M)* | **VERIFIED** *(1,000 images in CW: 99.30%; 5 images in MMC)* | **PARTIAL** *(5 diagnostic smoke images in MMC; NOT on full Wang2020)* | **VERIFIED** *(100 images in CW: 52.00%)* |
| **8. Fusion-ready** | **NOT VERIFIED** *(Modality mismatch: 5D video tensor)* | **VERIFIED** *(Standard `DetectorResult`, calibrated logits)* | **PARTIAL** *(RAM paging error under joint execution; face-crop bias)* | **VERIFIED** *(Standard `DetectorResult`, calibrated logits)* | **NOT VERIFIED** *(No adapter in `multi model connection`)* |

---

## 4. Evidence Supporting Completed Stages

### Model 1: `df_model.pt`
- **Identity & Provenance:** Diagnostic report `model_accuracy_diagnosis.txt` (lines 33-88) and `CURRENT_STATE_AUDIT.md` confirm it is a ResNeXt-50-32x4d spatial backbone combined with a 1-layer unidirectional LSTM (latent dimension 2048, hidden dimension 2048, no bias) and a linear classification head (2048 $\to$ 2).
- **Video Modality Evidence:** In `currently working/server.py` (lines 375-463) and `video_inference.py`, input is extracted into 20 sequential frames, resized to $112 \times 112$, normalized with ImageNet statistics, and stacked into a 5-dimensional tensor `[1, 20, 3, 112, 112]`.
- **Static Image Collapse:** When fed a single static image, sequence length is 1 (`[1, 1, 3, 112, 112]`). The LSTM hidden state is extracted at timestep $t=-1$, which evaluates initial state $h_1$ instead of accumulated temporal state $h_{20}$. Evaluated on 100 images from Kaggle 140k faces, it produced 36% accuracy (worse than random guessing).
- **Physical Locations:** Exactly identical 226,597,567-byte binary present at `currently working\model\df_model.pt`, `Deepfake-Detection-main\Deepfake-Detection-main\model\df_model.pt`, and `file 1\model\df_model.pt`. Git LFS pointer at `PG final year Project`. Completely absent from `multi model connection`.

### Model 2: `chen2024_convnext`
- **Identity & Provenance:** Checkpoint `16_acc0.9993.pth` is verified against `temp_DRCT_reference/README.md` (lines 72, 92). It represents the official ICML 2024 Spotlight checkpoint for `convnext_base_in22k` trained on DRCT-2M with Stable Diffusion v2 reconstruction (`sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth`). It was also mirrored in the RAID benchmark checkpoint repository (`aimagelab/RAID_ckpt`).
- **Architecture & Loading:** Built with `timm.create_model("convnext_base", pretrained=False, num_classes=1024)` + `nn.Linear(1024, 2)`. Loads cleanly with `strict=True` and 0 missing or unexpected parameters.
- **Preprocessing:** Verified in `src/convnext_adapter.py`: `transforms.Resize(224, bicubic) -> transforms.CenterCrop(224) -> transforms.ToTensor() -> transforms.Normalize(ImageNet mean/std)`.
- **Operational Diagnostic Results:** In `scripts/convnext_adapter_test_results.json`, all 13 diagnostic tests passed:
  - Lazy loading: verified (zero parameter instantiation before first call).
  - Detector isolation: verified.
  - Valid real image (`real.jpg`): `Prediction=REAL`, fake probability `0.004387`, logit `-5.4248`.
  - Valid AI image (`ai_generated.jpg`): `Prediction=REAL`, fake probability `0.073853`, logit `-2.5290`.
  - Valid manipulated image (`manipulated.jpg`): `Prediction=REAL`, fake probability `0.142673`, logit `-1.7936`.
  - Invalid inputs: Corrupted file, plain text file, non-existent file, and `None` input all properly rejected with `Prediction=ERROR`.
  - Probability bounds $[0, 1]$, sum to $1.0 \pm 10^{-5}$, and confidence alignment all verified.

### Model 3: `vit` (`dima806/deepfake_vs_real_image_detection`)
- **Identity & Checkpoint:** Hugging Face model `dima806/deepfake_vs_real_image_detection`, based on `ViTForImageClassification` (`google/vit-base-patch16-224`). Checkpoint `model.safetensors` (343.2 MB) matches SHA-256 `B38E7D3D...` in both project directories.
- **Preprocessing & Label Mapping:** `AutoImageProcessor` applies Resize 224x224, Rescale 1/255, and Normalize mean/std `[0.5, 0.5, 0.5]`. Adapter dynamically inspects `config.id2label` to map label indices (`0: Real`, `1: Fake`).
- **Benchmark Evaluation in `currently working`:**
  - Initial 100-sample test: 99.00% accuracy (50/50 fake recall, 49/50 real recall).
  - 1,000-sample large-scale balanced evaluation (`extended_validation_report.md`): **99.30% accuracy** (500/500 fake recall, 493/500 real recall), ROC-AUC **0.9999**, PR-AUC **0.9999**, Brier score **0.0058**.
  - Robustness testing: Maintained $\ge 96\%$ accuracy under JPEG compression (Q=30, Q=50), resizing down to $64 \times 64$, Gaussian blur, brightness adjustments, and contrast perturbations.
- **Performance in `multi model connection`:** Passed all operational checks in `scripts/vit_adapter_test_results.json`. On general/non-facial diffusion (`ai_001`), produced fake probability 34.24% (assigned higher suspicion than UnivFD or ConvNeXt, though still below the 0.50 threshold due to lack of face geometry).

### Model 4: `univfd` (UniversalFakeDetect)
- **Identity & Checkpoint:** CVPR 2023 UniversalFakeDetect (Ojha et al.). Checkpoint `UniversalFakeDetect/pretrained_weights/fc_weights.pth` contains only linear probe weights `[1, 768]` and scalar bias `-0.5167`. Frozen visual backbone weights cached at `C:\Users\user\.cache\clip\ViT-L-14.pt` (932.8 MB).
- **Preprocessing Pipeline Alignment:** `scripts/compare_official_inference.py` cross-examined three repository implementations:
  1. `validate.py`: Direct `CenterCrop(224)` without resizing (evaluates only 0.4% of high-res image area).
  2. `datasets.py`: `Resize(256, bicubic) -> CenterCrop(224)`.
  3. `app.py` / `src/univfd_adapter.py`: `model.preprocess` (`Resize(224, bicubic) -> CenterCrop(224)`), which provides full-scene coverage and matches CLIP standards.
- **Operational Diagnostic Results:** Passed all test cases in `scripts/adapter_test_results.json`.
- **Known Domain Boundary:** `docs/UNIVFD_LIMITATIONS.md` establishes that because UnivFD was trained on 2018 ProGAN artifacts, it is blind to modern diffusion generation (predicts REAL on Google Imagen 3 diffusion with 99.9% confidence) and localized splicing (predicts REAL on composite images with 99.4% confidence).

### Model 5: `CommunityForensics-DeepfakeDet-ViT`
- **Identity & Checkpoint:** CVPR 2025 "Community Forensics" (OpenSight). Built on `timm/vit_small_patch16_384.augreg_in21k_ft_in1k`, single-class sigmoid output (`num_labels: 1`), input size $384 \times 384$. Local directory contains verified fixed weights `model.safetensors` (87.3 MB, SHA-256 `275BA982...`).
- **Benchmark in `currently working`:** Evaluated on 100 face images in `evaluate_community_forensics.py`: achieved **52.00% accuracy** (8.00% fake recall, 96.00% real recall). This proves that models trained on general scene generators (Midjourney/SD landscapes) fail to detect facial manipulations without facial fine-tuning.

---

## 5. Timeline of Project Progression

```
+---------------------------------------------------------------------------------------------------------+
| PHASE 1: Single-Model Video Baseline (December 2025 - Early 2026)                                       |
| - Repositories: 'file 1', 'Deepfake-Detection-main', 'PG final year Project'                           |
| - Core Model: df_model.pt (ResNeXt-50 + LSTM)                                                           |
| - Architecture: Flask web server (server.py) + React frontend                                           |
| - Key Event: 100-image static evaluation yields unexpected 36% accuracy                                 |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| PHASE 2: Deep Architectural Diagnosis & Modality Separation (September 2026)                            |
| - Workspace: 'currently working'                                                                        |
| - model_accuracy_diagnosis.txt proves df_model.pt is a 20-frame temporal video model, not image model    |
| - Discovered dlib face-cropping bug that mutilated already-cropped face datasets                       |
| - Split application into dual-pipeline: Video (df_model.pt) vs Static Image (dima806 ViT)               |
| - Evaluated ViT (dima806) on 1,000 Kaggle face images: achieved 99.30% accuracy, ROC-AUC 0.9999        |
| - Evaluated CommunityForensics-DeepfakeDet-ViT (52.0% accuracy) and PrithivMLmods SigLIP (39.0% acc)    |
| - Attempted Hugging Face ZeroGPU Gradio deployment (branch hf-gradio-deployment)                        |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| PHASE 3: New Multi-Model Cross-Evaluation Framework (Mid-to-Late September 2026)                        |
| - Workspace: 'multi model connection'                                                                   |
| - Focused on: Cross-Model Evaluation of Deepfake Image Detection Under Diverse Generation Conditions    |
| - Integrated UniversalFakeDetect (CLIP ViT-L/14 + linear probe) in app.py and Hugging Face Space        |
| - Documented UnivFD limitations on modern diffusion (Imagen 3) and composite manipulations              |
| - Built standardized dataclass schema (src/detector_schema.py) defining DetectorResult                 |
| - Built decoupled registry with lazy instantiation (src/detector_registry.py)                          |
| - Implemented and contract-tested isolated ViT adapter (src/vit_adapter.py)                             |
| - Researched Detector #3 candidates (DETECTOR_3_CANDIDATE_ANALYSIS.md): SigLIP, Swin, CNN, RAID         |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| PHASE 4: ConvNeXt/DRCT Integration & Joint Multi-Model Benchmark (October 2026 - Current State)         |
| - Workspace: 'multi model connection'                                                                   |
| - Selected Chen et al. 2024 / DRCT ConvNeXt-Base (16_acc0.9993.pth from RAID checkpoint collection)    |
| - Sourced temp_DRCT_reference codebase to confirm exact architecture, preprocessing, and label mapping   |
| - Implemented src/convnext_adapter.py and executed scripts/test_convnext_adapter.py (13/13 passed)       |
| - Verified all 3 adapters compliant with contract (scripts/contract_test_summary.json)                 |
| - Ran scripts/run_detector_benchmark.py across all 3 detectors on benchmark manifest                   |
| - Encountered Windows paging file memory exhaustion (os error 1455) for ViT during joint run            |
| - Observational agreement tracking recorded in detector_comparison.json (DISAGREE on test samples)     |
+---------------------------------------------------------------------------------------------------------+
```

---

## 6. Comparison of the Two Known Project Folders

| Dimension | `C:\Users\user\Desktop\currently working` | `C:\Users\user\Desktop\multi model connection` |
| :--- | :--- | :--- |
| **Primary Focus** | Full-stack application deployment (Flask + React + Gradio) for dual-modality (facial images & videos). | Modular scientific evaluation framework for multi-model cross-evaluation of image deepfakes. |
| **Model Architectures Present** | 1. `df_model.pt` (ResNeXt-50+LSTM video)<br>2. `dima806` (ViT-Base/16 image)<br>3. `CommunityForensics` (ViT-Small 384) | 1. `chen2024_convnext` (ConvNeXt-Base)<br>2. `dima806` (ViT-Base/16 image)<br>3. `univfd` (CLIP ViT-L/14 + Linear Probe) |
| **Architectural Modularity** | Monolithic. Code is tightly coupled to Flask endpoints, MTCNN cascades, and specific model wrapper singletons. | Highly modular and decoupled. Centralized `detector_registry.py`, lazy loading, and standardized `DetectorResult` schema. |
| **Adapter & Registry Layer** | None. Ad-hoc wrappers (`PilotDeepfakePipeline`, `DeepfakeModelWrapper`, `VideoModelWrapper`). | Standardized `src/detector_schema.py`, `src/detector_registry.py`, `src/vit_adapter.py`, `src/convnext_adapter.py`, `src/univfd_adapter.py`. |
| **Evaluation Datasets** | Comprehensive: 10,905 labeled Kaggle face images (`Dataset/Test/Real` & `Fake`), 4 test videos. | Minimal: 5 diagnostic smoke-test samples (`benchmark/test_images/` & `benchmark/images/`). |
| **Inference Pipeline** | Uses MTCNN face-detection cascading (primary, sensitive, 2x upscaled) and quality analysis heuristics. | Pure detector evaluation: passes full image through verified native preprocessing transforms without external face cropping. |
| **Video Detection** | Fully implemented and validated (`video_inference.py`, `detectFakeVideo`). | Not present. Out of scope for current image-detection objective. |
| **Benchmark Harness** | Ad-hoc scripts (`evaluate_model.py`, `evaluate_random_250.py`, `extended_validation_dima806.py`). | Standardized harness (`scripts/run_detector_benchmark.py`) outputting long-format and side-by-side JSON & CSV. |
| **Version & Status** | Historical baseline and dataset repository (last major commits mid-September 2026). | **Active, current working directory** containing the latest multi-model code, ConvNeXt integration, and DRCT references. |

---

## 7. Duplicate, Missing, or Unverified Artifacts

### 1. Duplicate Artifacts
- **`dima806 ViT` weights (`model.safetensors`):** Exact binary duplicate (343,223,968 bytes, SHA-256 `B38E7D3D...`) present in both `currently working` and `multi model connection`.
- **`df_model.pt` weights:** Exact binary duplicate (226,597,567 bytes, SHA-256 `B893DE92...`) present in `currently working`, `Deepfake-Detection-main`, and `file 1`.
- **`CommunityForensics` weights:** `model.safetensors` and `model_fixed.safetensors` in `currently working` are 100% identical copies (87,270,764 bytes, SHA-256 `275BA982...`).

### 2. Missing Artifacts
- **`df_model.pt` in `multi model connection`:** Completely absent from the active project directory. While the task prompt noted it as a known location, physical inspection confirms it was never copied into `multi model connection`.
- **Statistically Significant Labeled Benchmark Dataset in `multi model connection`:** Only 5 diagnostic images exist in `benchmark/images/` and `benchmark/test_images/`. A robust evaluation dataset spanning diverse generators (GAN, Diffusion, Splicing, Authentic) is missing from the main workspace.
- **`UniversalFakeDetect` Alternative Checkpoints:** As documented in `UNIVFD_LIMITATIONS.md`, the repository contains only `fc_weights.pth` (ProGAN linear probe). No diffusion-adapted or modern generator probes are present.

### 3. Unverified Claims in Documentation
- **ConvNeXt "99.93% Accuracy":** The checkpoint filename `16_acc0.9993.pth` reflects an internal training accuracy metric from DRCT on DRCT-2M (SDv2). It must **not** be cited as the model's accuracy on the project's target benchmark.
- **Historical "95% Accuracy" in `currently working/README.md`:** TensorBoard event logs in `logs/` confirm the original Keras training runs never converged (loss remained $\sim 0.69$, accuracy hovered at $49.5\% - 54.8\%$). The claim in the old README is factually invalid.

---

## 8. Current Blockers and Technical Risks

### Blocker 1: Process Memory Exhaustion under Joint Model Execution (RESOLVED)
- **Status:** **RESOLVED** on 2026-10-30 (See detailed verification in `scripts/MEMORY_SAFE_BENCHMARK_REPORT.md`).
- **Resolution:** Refactored `scripts/run_detector_benchmark.py` to evaluate detector-by-detector sequentially rather than interleaved across samples. Added explicit `unload_model()` methods on all adapters, generational garbage collection (`gc.collect()`), CUDA cache clearing, and process isolation support (`--mode subprocess`).
- **Verified Outcome:** All 3 detectors (`chen2024_convnext`, `univfd`, `vit`) now evaluate cleanly with 0 errors across all benchmark samples. ViT memory paging crash (`os error 1455`) is fully eliminated, with predictions matching historical baseline floats to 6 decimal places.

### Blocker 2: Absence of a Decision-Fusion Engine (Architectural)
- **Current State:** The system only records an observational `agreement_status` string (`"AGREE"` if all predictions match, `"DISAGREE"` otherwise) in `detector_comparison.json`.
- **Deficiency:** No logic exists to combine continuous probabilities, compute weighted confidence scores, handle conflicting verdicts, apply Bayesian evidence updating, or output a synthesized prediction with uncertainty calibration.

### Blocker 3: Outdated Single-Model Interface in `app.py` (Integration)
- **Current State:** `app.py` in `multi model connection` remains hardcoded to UniversalFakeDetect alone.
- **Deficiency:** It does not use `detector_registry.py`, does not expose the ViT or ConvNeXt adapters, and cannot display multi-model comparative results or fused outputs in the Gradio UI.

### Blocker 4: Dataset Scale and Diversity Mismatch (Methodological)
- **Current State:** The benchmark in `multi model connection` comprises only 5 files (3 valid, 1 corrupted, 1 text).
- **Deficiency:** Three valid images cannot establish statistical metrics (Accuracy, Precision, Recall, F1, ROC-AUC, ECE). Furthermore, the models have fundamentally different training domains:
  - `dima806 ViT`: Facial manipulations and crops.
  - `univfd`: ProGAN structural artifacts.
  - `chen2024_convnext`: Diffusion reconstruction contrastive artifacts.
  Evaluating them on arbitrary images without category-specific labeling produces misleading cross-model comparisons.

### Blocker 5: Video Modality Isolation (`df_model.pt`)
- **Current State:** `df_model.pt` requires a 5D temporal video tensor `[B, 20, 3, 112, 112]`.
- **Deficiency:** It cannot be directly evaluated on still images. Attempting to pass static images into `df_model.pt` produces meaningless representations. If video detection is part of the final system, it must remain isolated in a dedicated video pipeline.

---

## 9. Recommended Next Steps (Priority Order)

1. **Implement Memory-Safe Sequential Execution in Benchmark Harness:** *(COMPLETED)*  
   Updated `scripts/run_detector_benchmark.py` with sequential model evaluation, explicit adapter deallocation (`unload_model()`), garbage collection, and subprocess isolation mode. Verified 100% success across all 3 detectors without paging errors.
2. **Assemble a Stratified Multi-Condition Benchmark Dataset:**  
   Construct an expanded benchmark manifest (e.g. 100–300 images) containing balanced categories:
   - Authentic photographs (natural cameras, portraits, landscapes).
   - GAN-generated images (StyleGAN, ProGAN).
   - Diffusion-generated images (Stable Diffusion, Midjourney, Imagen).
   - Localized facial/composite manipulations.  
   Port relevant samples from `currently working/Dataset/Test` for facial benchmarks.
3. **Design and Implement the Multi-Model Decision-Fusion Engine:**  
   Create `src/fusion_engine.py` to accept multiple `DetectorResult` outputs and compute:
   - Dynamic domain routing (e.g. face vs general scene weighting).
   - Calibrated probability fusion (weighted soft voting, Bayesian logit pooling).
   - An explicit `UNCERTAIN` classification policy when model confidence is discordant.
4. **Update `app.py` to Serve the Multi-Model Framework:**  
   Refactor `app.py` to use `src.detector_registry` and the fusion engine, presenting users with:
   - Per-detector independent predictions, probabilities, and latency.
   - The unified fusion verdict with confidence breakdown and agreement diagnostics.
5. **Decide Scope for `df_model.pt` (Video Pipeline):**  
   Formally document whether video detection is within the scope of the current "Cross-Model Evaluation of Deepfake Image Detection" project. If image-only, archive `df_model.pt`. If dual-modality is required, port `video_inference.py` from `currently working` as a separate modality service.

---

## 10. Proposed Decision-Fusion Engine Architecture

### Architectural Overview

```text
                        +----------------------------+
                        |     Input Image (RGB)      |
                        +----------------------------+
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
                 v                    v                    v
       +------------------+  +------------------+  +------------------+
       |   UnivFDAdapter  |  |    ViTAdapter    |  | ConvNeXtAdapter  |
       |  (CLIP ViT-L/14) |  |  (ViT-Base/16)   |  | (ConvNeXt-Base)  |
       +------------------+  +------------------+  +------------------+
                 |                    |                    |
                 v                    v                    v
          DetectorResult       DetectorResult       DetectorResult
                 |                    |                    |
                 +--------------------+--------------------+
                                      |
                                      v
                        +----------------------------+
                        |   Decision-Fusion Engine   |
                        |   (src/fusion_engine.py)   |
                        +----------------------------+
                                      |
               +----------------------+----------------------+
               |                      |                      |
               v                      v                      v
     [Domain Weighting]     [Log-Odds / Bayesian]   [Uncertainty Gate]
     Face: ViT high         P(Fake) aggregation     Threshold |P - 0.5|
     Diffusion: ConvNeXt    Outlier dampening       Disagreement flag
               |                      |                      |
               +----------------------+----------------------+
                                      |
                                      v
                        +----------------------------+
                        |     FusedDetectorResult    |
                        |  - Final Verdict           |
                        |  - Fused Fake Probability  |
                        |  - Confidence Score        |
                        |  - Agreement Level         |
                        |  - Contributing Weights    |
                        +----------------------------+
```

### Core Design Principles for the Proposed Fusion Engine

1. **Non-Invasive Modularity:**  
   The fusion engine consumes standardized `DetectorResult` dataclass instances emitted by the existing adapters. Zero modifications to the underlying adapters or model checkpoints are required.
2. **Domain-Aware Dynamic Weighting:**  
   Because detectors possess divergent domain specializations:
   - **ViT (`dima806`)** receives higher weight when facial features/landmarks are prominent ($w_\text{vit} \approx 0.50$).
   - **ConvNeXt (`chen2024`)** receives higher weight on full scenes and diffusion generation ($w_\text{conv} \approx 0.40$).
   - **UnivFD** provides high-precision corroboration for GAN structural artifacts ($w_\text{univ} \approx 0.10$).
3. **Calibrated Probability Fusion:**  
   Instead of crude hard-voting (which discards confidence) or naive probability averaging (which suffers from overconfident outliers), combine raw log-odds:
   $$\text{Logit}_\text{fused} = \sum_{i=1}^N w_i \cdot \text{logit}_i$$
   $$P_\text{fused}(\text{Fake}) = \frac{1}{1 + e^{-\text{Logit}_\text{fused}}}$$
4. **Explicit Uncertainty Policy:**  
   - If $|P_\text{fused} - 0.50| < 0.15$ or if strong binary conflict exists with high individual confidences ($P_1 > 0.85$ and $P_2 < 0.15$), classify as `UNCERTAIN` / `REVIEW_REQUIRED`.
   - Never force a false binary decision when detectors strongly disagree.
5. **Memory-Conscious Sequential Orchestration:**  
   The fusion coordinator will evaluate detectors sequentially, clearing memory between passes if running on memory-limited hardware, ensuring robust execution in any environment.

---

## 11. Phase 6 Completion: Cross-Model Deepfake Detection & Ensemble Classifier

### 11.1 Phase 6 Overview & Execution Status
- **Status:** **COMPLETED & VERIFIED (November 2026)**
- **Test Corpus:** 200 pristine, leak-free test samples (100 REAL authentic FFHQ portraits, 100 FAKE StyleGAN portraits) frozen at [`benchmark/phase6_test_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/phase6_test_manifest.csv).
- **Execution Harness:** Fully sequential subprocess isolation, preventing Windows virtual memory exhaustion (`os error 1455`) while operating under an 8.0 GB RAM / CPU inference profile.
- **Active Detector Models Evaluated:**
  1. `vit` (`dima806/deepfake_vs_real_image_detection`)
  2. `univfd` (`CLIP ViT-L/14 + linear probe`)
  3. `chen2024_convnext` (`models/chen2024_convnext/16_acc0.9993.pth`)
  *(Incompatible 5D video model `df_model.pt` strictly excluded from 2D image pipelines).*

### 11.2 Verified Performance Comparison Matrix ($N = 200$)

| Model / Fusion Strategy | Accuracy (%) | 95% Wilson CI (Acc) | Balanced Acc (%) | Precision | Recall (Fake Det.) | Specificity (Real Ret.) | F1-Score | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT (`dima806`)** | **98.00%** | **[94.97%, 99.22%]** | **98.00%** | **0.9615** | **1.0000** | **0.9600** | **0.9804** | **192.4 ms** |
| **UnivFD (CLIP ViT-L/14)** | 51.50% | [44.61%, 58.33%] | 51.50% | 0.5484 | 0.1700 | 0.8600 | 0.2595 | 942.1 ms |
| **ConvNeXt (DRCT / RAID)** | 51.00% | [44.12%, 57.84%] | 51.00% | 0.5333 | 0.1600 | 0.8600 | 0.2462 | 252.7 ms |
| **Ensemble (Majority Vote)** | 63.00% | [56.12%, 69.39%] | 63.00% | 0.8824 | 0.3000 | 0.9600 | 0.4478 | 1387.2 ms |
| **Ensemble (Mean Fake Prob)** | 63.50% | [56.63%, 69.86%] | 63.50% | 0.8857 | 0.3100 | 0.9600 | 0.4593 | 1387.2 ms |
| **Ensemble (ViT Priority)** | **98.00%** | **[94.97%, 99.22%]** | **98.00%** | **0.9615** | **1.0000** | **0.9600** | **0.9804** | **192.4 ms** |

### 11.3 Key Findings & Verified Insights
1. **ViT In-Domain Excellence:** ViT detected 100 out of 100 StyleGAN fake faces (100% Recall, 0 False Negatives) and achieved 96% Specificity (4 False Positives).
2. **UnivFD & ConvNeXt StyleGAN Blind Spot:** UnivFD and ConvNeXt suffered severe false-negative rates ($FN = 83$ and $FN = 84$, respectively), missing >80% of synthetic faces due to differing training distributions (ProGAN, general diffusion, and localized frequency artifacts).
3. **The Ensemble Degradation Paradox:** Unweighted democratic voting (**Majority Vote** at 63.00% and **Mean Fake Probability** at 63.50%) caused a catastrophic **-35.00% collapse** in accuracy. UnivFD and ConvNeXt formed an erroneous 2-to-1 majority on 70 out of 100 fake faces, outvoting ViT's correct predictions.
4. **Hierarchical Routing Success:** The **ViT-Priority Fallback** architecture successfully preserved the full 98.00% accuracy while maintaining secondary models as fallback guards.
5. **Phase 6 Artifacts:**
   - Manifest: [`benchmark/phase6_test_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/phase6_test_manifest.csv)
   - Dataset Selection Report: [`docs/PHASE6_DATASET_SELECTION_REPORT.md`](file:///C:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_DATASET_SELECTION_REPORT.md)
   - Evaluation Report: [`docs/PHASE6_EVALUATION_REPORT.md`](file:///C:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_EVALUATION_REPORT.md)
   - Generalization Plan: [`docs/PHASE6_GENERALIZATION_PLAN.md`](file:///C:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_GENERALIZATION_PLAN.md)
   - Completion Report: [`docs/PHASE6_COMPLETION_REPORT.md`](file:///C:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_COMPLETION_REPORT.md)
   - Post-Execution Audit: [`docs/PHASE6_POST_EXECUTION_AUDIT.md`](file:///C:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_POST_EXECUTION_AUDIT.md)

---

## 12. Phase 7 Completion & Post-Execution Audit: External AI-Generated Image Generalization (GenImage)

### 12.1 Phase 7 Overview & Methodology
- **Audit Status:** **VERIFIED WITH LIMITATIONS (November 2026)**
- **Audit Deliverable:** [`docs/PHASE7_POST_EXECUTION_AUDIT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_POST_EXECUTION_AUDIT.md)
- **Test Corpus:** 200 external general-scene samples from official GenImage validation partition (`TheKernel01/Tiny-GenImage`):
  * **100 REAL:** Authentic ImageNet-1k photographs (`val/nature`).
  * **50 FAKE (Midjourney):** Commercial closed-source latent diffusion (unseen in training to all 3 detectors).
  * **50 FAKE (ADM):** Academic pixel-space guided diffusion (unseen in training to all 3 detectors).
- **Frozen Manifest:** [`benchmark/external_generalization_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/external_generalization_manifest.csv)
- **Execution Harness:** Sequential subprocess isolation; zero memory exhaustion errors.

### 12.2 Verified Performance Matrix ($N = 200$)

| Model / Strategy | Aggregate Acc (%) | 95% Wilson CI | Recall (Fake) | Specificity (Real) | F1-Score | ROC-AUC | Midjourney Acc (%) | ADM Acc (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT (`dima806`)** | 52.00% | [45.10%, 58.82%] | 18.00% | 86.00% | 0.2727 | 0.4673 | 64.00% | 62.67% | **390.1 ms** |
| **UnivFD (CLIP ViT-L/14)** | **57.50%** | [50.57%, 64.15%] | 15.00% | **100.00%** | 0.2609 | 0.6662 | 66.67% | **76.67%** | 1494.1 ms |
| **ConvNeXt (DRCT / RAID)** | 56.50% | [49.57%, 63.18%] | 13.00% | **100.00%** | 0.2301 | **0.7926** | **74.00%** | 68.00% | 516.8 ms |
| **Ensemble (Majority Vote)** | 52.50% | [45.60%, 59.31%] | 5.00% | **100.00%** | 0.0952 | 0.7181* | 67.33% | 69.33% | 2401.0 ms |
| **Ensemble (Mean Fake Prob)** | 52.50% | [45.60%, 59.31%] | 5.00% | **100.00%** | 0.0952 | 0.7181 | 67.33% | 69.33% | 2401.0 ms |
| **Ensemble (ViT Priority)** | 52.00% | [45.10%, 58.82%] | 18.00% | 86.00% | 0.2727 | 0.4673 | 64.00% | 62.67% | **390.1 ms** |

*\*Note: 0.7181 reflects the continuous mean probability vector attached to `FusionResult.fake_probability`. When evaluated on discrete vote counts (0, 1, 2, 3), Majority Voting ROC-AUC is **0.6385**; under hard binary decisions, it is **0.5250** (matching Balanced Accuracy).*

### 12.3 Verified Empirical Discoveries & Identified Limitations
1. **Compounded Domain & Generator Collapse for ViT:** ViT collapsed from 98.00% on facial StyleGAN down to **52.00% on general GenImage scenes** ($\text{ROC-AUC} = \mathbf{0.4673}$). This demonstrates that ViT's fine-tuned representations fail outside facial anatomy under a compounded semantic shift (faces $\to$ ImageNet objects) and generative shift (StyleGAN $\to$ diffusion).
2. **Zero False Positives vs. Severe Sensitivity Deficit:** Both UnivFD and ConvNeXt achieved **100.00% Specificity (0 False Positives)** on real ImageNet photographs. However, at default threshold $\tau = 0.50$, both suffer from severe sensitivity deficits (UnivFD recall: 15%; ConvNeXt recall: 13%). While ConvNeXt demonstrated strong latent discriminability ($\text{ROC-AUC} = \mathbf{0.7926}$, leading on Midjourney with 74.00% accuracy) and UnivFD on ADM ($\text{ROC-AUC} = \mathbf{0.9056}$), neither functions as an effective out-of-the-box detector without threshold recalibration.
3. **Shared Control Group Structure:** The Midjourney and ADM evaluations each evaluate 150 rows but share the **exact same 100 authentic ImageNet controls**. Total unique benchmark samples evaluated is 200 (not 300).
4. **Fusion Dynamics:** Prioritizing ViT in general-scene domains inherits ViT's 14 False Positives, whereas democratic fusion achieves 100% Specificity but collapses fake recall to 5.00% (missing 95/100 fakes). Democratic fusion accuracy (52.50%) is lower than UnivFD alone (57.50%) and ConvNeXt alone (56.50%).
5. **Documentation Discrepancy Noted:** Section 5 of [`docs/PHASE7_EXTERNAL_GENERALIZATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_EXTERNAL_GENERALIZATION_REPORT.md) erroneously stated 91 three-way agreements (45.50%) and discrepant Cohen's Kappa figures. The verified three-way agreement count is **145 out of 200 (72.50%)** (86 agreed Real on authentic images + 59 agreed Real on fake images; 0 agreed Fake), with true $\kappa$ values of $0.0284$ (ViT vs UnivFD), $-0.0529$ (ViT vs ConvNeXt), and $0.0019$ (UnivFD vs ConvNeXt).
6. **Acceptance Verdict:** **Accepted as a Completed Benchmark Experiment with Documented Limitations**.

### 12.4 Phase 7 Master Artifacts
- **Post-Execution Audit:** [`docs/PHASE7_POST_EXECUTION_AUDIT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_POST_EXECUTION_AUDIT.md)
- **Audit Runner Script:** [`scripts/audit_phase7_results.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/audit_phase7_results.py)
- **Dataset Audit:** [`docs/PHASE7_DATASET_AUDIT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_DATASET_AUDIT.md)
- **Evaluation Report:** [`docs/PHASE7_EXTERNAL_GENERALIZATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_EXTERNAL_GENERALIZATION_REPORT.md)
- **Manifest:** [`benchmark/external_generalization_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/external_generalization_manifest.csv)
- **Comparison CSV:** [`benchmark/results/phase7_model_comparison.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_model_comparison.csv)
- **Metrics JSON:** [`benchmark/results/phase7_combined_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_combined_metrics.json)

---

## 13. Root-Cause Investigation: Official Model Reference Parity Testing

### 13.1 Investigation Overview & Scope
- **Investigation Deliverable:** [`docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md)
- **Audit Mode:** Scientific Parity Audit, Upstream Codebase Verification & Full Empirical 200-Sample Ablation (November 2026).
- **Core Research Question:** Determine whether poor Phase 7 results (ViT 52.0%, UnivFD 57.5%, ConvNeXt 56.5%) stem from integration defects, preprocessing/extraction issues, output calculation bugs, or genuine domain/generator generalization boundaries.

### 13.2 Model-by-Model Status Assignments

| Detector Model | Primary Status Assignment | Root Cause & Key Empirical Findings |
| :--- | :---: | :--- |
| **ViT (`dima806`)** | **OFFICIAL IMPLEMENTATION MATCHES** &<br>**GENUINE GENERALIZATION LIMITATION** | 100.0% exact match to upstream Hugging Face reference ($< 5 \times 10^{-7}$ diff across all 200 samples; 200/200 predictions match). Near random-chance performance (52.0% Acc, 0.4673 AUC) is caused exclusively by domain collapse from facial-only pretraining (Kaggle 140k). |
| **ConvNeXt (DRCT `16_acc0.9993.pth`)** | **PREPROCESSING MISMATCH** (Midjourney)<br>& **GENUINE GENERALIZATION LIMITATION** (ADM) | Checkpoint and architecture match 100% (346 tensors). Adapter downsampling (`Resize 224`) destroyed high-frequency latent diffusion traces. Official native crop (`--is_crop`) restores Midjourney recall to **100.0% (50/50)**. However, ADM recall remains at **10.0% (5/50)**, matching the DRCT paper's published ICML 2024 GenImage benchmark (60.00% ADM accuracy). Overall accuracy under native crop rises to **77.00%** with **0.9270 ROC-AUC**. |
| **UnivFD (CLIP ViT-L/14)** | **OFFICIAL IMPLEMENTATION MATCHES** (Arch/Weights)<br>& **GENUINE GENERALIZATION LIMITATION** (Primary Cause)<br>& **PREPROCESSING MISMATCH** (Local Divergence) | Weights match bit-for-bit. While adapter used standard CLIP preprocess instead of `validate.py` native crop, full-benchmark testing proves this had **net zero effect on fake recall** (15.00% under both). Failure is overwhelmingly driven by the 2018 ProGAN-only linear probe training. |
| **Dataset Extraction** | **OFFICIAL IMPLEMENTATION MATCHES** | 200/200 files on disk match source Hugging Face Parquet shard byte-for-byte with zero alteration, re-encoding, or compression drift. |

### 13.3 Full 200-Sample Parity Comparison Matrix

| Detector Model & Configuration | Total Accuracy | Real Specificity | Midjourney Recall | ADM Recall | Aggregate Fake Recall | ROC-AUC | Primary Failure Mechanism |
| :--- | :---:| :---:| :---:| :---:| :---:| :--- |
| **ViT (`dima806`)** — Phase 7 Adapter | 52.00% (104) | 86.00% (86) | 20.00% (10) | 16.00% (8) | 18.00% (18) | 0.4673 | Facial pretraining collapse |
| **ViT (`dima806`)** — Official Hugging Face | 52.00% (104) | 86.00% (86) | 20.00% (10) | 16.00% (8) | 18.00% (18) | 0.4673 | Bit-for-bit parity ($< 5 \times 10^{-7}$) |
| **UnivFD (`CLIP-L/14`)** — Phase 7 Adapter | 57.50% (115) | 100.00% (100) | 0.00% (0) | 30.00% (15) | 15.00% (15) | 0.6662 | ProGAN-only linear probe |
| **UnivFD (`CLIP-L/14`)** — Official `validate.py` | 56.50% (113) | 98.00% (98) | 16.00% (8) | 14.00% (7) | 15.00% (15) | 0.8037 | Net-zero recall gain (15% vs 15%) |
| **ConvNeXt (`DRCT`)** — Phase 7 Adapter | 56.50% (113) | 100.00% (100) | 22.00% (11) | 4.00% (2) | 13.00% (13) | 0.7926 | Preprocessing downsampling |
| **ConvNeXt (`DRCT`)** — Official `--is_crop` | **77.00% (154)** | **99.00% (99)** | **100.00% (50)** | **10.00% (5)** | **55.00% (55)** | **0.9270** | MJ resolved (100%), ADM bounded (10%) |

### 13.4 Investigation Artifacts
- **Primary Report:** [`docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md)
- **Audit Suite:** [`scripts/parity_audit/`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/parity_audit/)
- **Diagnostic JSON Records:**
  * `benchmark/results/parity_audit_convnext_full_200.json`
  * `benchmark/results/parity_audit_univfd_full_200.json`
  * `benchmark/results/parity_audit_vit_results.json`
  * `benchmark/results/parity_audit_dataset_extraction.json`
  * `benchmark/results/parity_audit_score_distributions.json`

---

## 14. Phase 7 Version 2: Corrected External Generalization Evaluation (Official Preprocessing Alignment)

### 14.1 Phase 7 V2 Overview & Execution Context
- **Report Deliverable:** [`docs/PHASE7_V2_CORRECTED_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_V2_CORRECTED_EVALUATION_REPORT.md)
- **Execution Mode:** Corrective Implementation & Empirical Re-evaluation (November 2026).
- **Core Action:** Corrected preprocessing mismatches in `src/convnext_adapter.py` (restored native `DRCTNativeCropTransform` matching official `--is_crop`) and `src/univfd_adapter.py` (restored native `CenterCrop(224)` matching `validate.py`). Preserved ViT without modifications.
- **Reference Parity:** Verified side-by-side on 30 deterministic samples against official author code with floating-point differences $< 10^{-6}$ (`parity_verification_audit.json`).
- **Regression Tests:** 4/4 passed in [`tests/test_adapter_preprocessing_regression.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/tests/test_adapter_preprocessing_regression.py).
- **Evaluation Partition:** Exact same 200 manifest samples from GenImage (`benchmark/external_generalization_manifest.csv`: 100 Real ImageNet, 50 Midjourney, 50 ADM). Version 1 artifacts strictly preserved.

### 14.2 Empirical Performance Matrix (Phase 7 V2 vs. V1)

| Detector / Strategy | Aggregate Acc (%) | 95% Wilson CI | Recall (Fake) | Specificity (Real) | F1-Score | ROC-AUC | Midjourney Recall | ADM Recall | V1 $\to$ V2 $\Delta$ Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT (`dima806`)** | 52.00% | [45.10%, 58.82%] | 18.00% | 86.00% | 0.2727 | 0.4673 | 20.00% | 16.00% | $\pm 0.00\%$ |
| **UnivFD (CLIP ViT-L/14)** | 56.50% | [49.57%, 63.18%] | 15.00% | 98.00% | 0.2564 | 0.8037 | 16.00% | 14.00% | $-1.00\%$ |
| **ConvNeXt (DRCT / RAID)** | **77.00%** | **[70.69%, 82.29%]** | **55.00%** | **99.00%** | **0.7051** | **0.9271** | **100.00%** | 10.00% | $\mathbf{+20.50\%}$ |
| **Ensemble (Majority Vote)** | 58.50% | [51.57%, 65.11%] | 18.00% | 99.00% | 0.3025 | 0.7662 | 30.00% | 6.00% | $+6.00\%$ |
| **Ensemble (Mean Fake Prob)** | 61.50% | [54.60%, 67.97%] | 23.00% | **100.00%** | 0.3740 | 0.8309 | 40.00% | 6.00% | $+9.00\%$ |
| **Ensemble (ViT Priority)** | 52.00% | [45.10%, 58.82%] | 18.00% | 86.00% | 0.2727 | 0.4673 | 20.00% | 16.00% | $\pm 0.00\%$ |

### 14.3 Key Empirical Discoveries & Architectural Implications
1. **Resolution of Midjourney Defect:** Restoring native cropping increased ConvNeXt Midjourney recall from 22.0% to **100.0% (50/50)** with cohort ROC-AUC of **0.9992**, raising overall accuracy to **77.00%** and ROC-AUC to **0.9271**.
2. **Persistence of ADM Generalization Boundary:** Even under official native crop, ConvNeXt detects only **10.0% (5/50)** of pixel-space guided diffusion (ADM) images, matching the ICML 2024 published limits (60.0% accuracy).
3. **Failure of Unweighted Democratic Fusion:** Unweighted fusion (Majority Vote: 58.50%, Mean Prob: 61.50%) significantly underperforms ConvNeXt alone (77.00%) because out-of-domain detectors (ViT and UnivFD) systematically outvote ConvNeXt on Midjourney samples.
4. **Architectural Recommendation:** Move from unweighted fusion to domain-conditioned routing: route faces to ViT; route general scenes exclusively to ConvNeXt.

### 14.4 Phase 7 Version 2 Artifacts
- **Evaluation Report:** [`docs/PHASE7_V2_CORRECTED_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_V2_CORRECTED_EVALUATION_REPORT.md)
- **Results Directory:** [`benchmark/results/phase7_v2_reference_preprocessing/`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_v2_reference_preprocessing/)
- **Parity Audit:** `benchmark/results/phase7_v2_reference_preprocessing/parity_verification_audit.json`
- **Comparison CSV:** `benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_model_comparison.csv`
- **Combined Metrics JSON:** `benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_combined_metrics.json`





