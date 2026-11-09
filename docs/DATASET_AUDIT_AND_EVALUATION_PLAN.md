# Comprehensive Dataset Audit and Evaluation Plan

**Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions**  
**Project Root**: `C:\Users\user\Desktop\multi model connection`  
**Phase**: Phase 3 — Audit and Prepare a Reliable Evaluation Dataset  
**Status**: Completed and 100% Verified Against Disk  
**Date**: October 2026  

---

## 1. Executive Summary

This deliverable provides the comprehensive dataset audit, category taxonomy, leakage prevention analysis, and dataset manifest preparation for Phase 3 of the **Cross-Model Evaluation of Deepfake Image Detection** project.

### 1.1 Core Objectives Achieved
1. **Multi-Repository Dataset Audit**: Exhaustively audited all image assets across `C:\Users\user\Desktop\multi model connection` and `C:\Users\user\Desktop\currently working`, inspecting 190,335 dataset images, 9 benchmark diagnostic images, 3 reference samples, and 5 video fixtures. Verified file integrity, dimensions, color spaces, formats, and duplicates.
2. **Evaluation Domain Disconnect Discovered**: Identified a fundamental mismatch across detectors. Facial deepfake detectors (`dima806/deepfake_vs_real_image_detection`) operate exclusively on cropped, aligned facial portraits ($256 \times 256$), whereas frequency-reconstruction detectors (`chen2024_convnext` / DRCT-2M) and vision-language probes (`UniversalFakeDetect`) operate on general visual scenes ($224 \times 224$ and $1024 \times 1024$).
3. **Investigation of `CommunityForensics-DeepfakeDet-ViT`**: Analyzed empirical validation logs from `currently working`. Demonstrated that CommunityForensics (trained on 2.7M general scene images across 4,803 generators) achieves only **52.00% overall accuracy** on facial deepfakes (with an alarming **8.00% fake recall** and a **92.00% false negative rate**). Concluded that it is **unsuitable** as a facial detector but **promising** for open-domain general diffusion scenes in a later phase under domain-gated routing.
4. **Leakage-Free Benchmark Manifest**: Formulated and verified a 159-sample multi-domain dataset manifest ([`benchmark/evaluation_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/evaluation_manifest.csv)), partitioned into `held_out_test` (105 samples), `development_val` (50 samples), and `operational_smoke` (4 samples). Every sample was verified with zero duplicate hashes and zero overlap with any training, validation, holdout, or exploratory sets.
5. **Automated Verification Harness**: Implemented [`scripts/verify_dataset_manifest.py`](file:///C:/Users/user/Desktop/multi%20model%20connection/scripts/verify_dataset_manifest.py), passing with 100% automated integrity validation across all 159 samples.

---

## 2. Inventory of Existing Datasets

A systematic audit was conducted across all relevant Desktop directories. Below is the comprehensive inventory of discovered image datasets, fixtures, and reference collections.

### 2.1 Dataset Inventory Matrix

| Dataset Identifier | Absolute Path | Total Files | Formats & Dimensions | Split Structure | Ground-Truth Basis | Primary Domain | Licensing / Provenance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kaggle 140k Real & Fake Faces** | `C:\Users\user\Desktop\currently working\Dataset` | **190,335** | JPEG, $256 \times 256$, RGB | Train (140,002)<br>Val (39,428)<br>Test (10,905) | Documented benchmark: FFHQ (Real) vs. StyleGAN (Fake) | Facial Crops Only | Kaggle Open Dataset (xhlulu / 140k-real-and-fake-faces) |
| **Diagnostic Benchmark Suite** | `C:\Users\user\Desktop\multi model connection\benchmark\images` | **5** | PNG / JPEG / TXT<br>Up to $4000 \times 3000$ | Isolated benchmark | Explicitly documented in `benchmark/metadata.json` | General Scenes, Diffusion, & Rejection | Local project fixture suite (Apache 2.0 / Local) |
| **Operational Smoke Suite** | `C:\Users\user\Desktop\multi model connection\benchmark\test_images` | **4** | JPEG / PNG<br>$508 \times 865$ to $509 \times 865$ | Operational CI smoke | Empirically verified test fixtures | CI / Adapter Contract Validation | Internal testing fixtures |
| **DRCT Reference Samples** | `C:\Users\user\Desktop\multi model connection\temp_DRCT_reference\data\samples` | **3** | PNG, $512 \times 512$ to $512 \times 896$ | Reference samples | **UNRESOLVED** (No ground-truth labels in repo) | General Diffusion & Augmentation | ICML 2024 DRCT official code release |
| **Test Video Suite** | `C:\Users\user\Desktop\currently working\test_videos` | **5** | MP4 / WebM<br>Video clips | Exploratory video eval | Inferred from filenames (`deepfake`, `genuine`) | Video Facial Reenactment | Local temporal inference fixtures |

---

### 2.2 Deep Dive: Kaggle 140k Real & Fake Faces Dataset

The primary dataset in `C:\Users\user\Desktop\currently working\Dataset` is an established benchmark comprising 190,335 images.

#### 1. Detailed Split & Class Distribution
- **Total Images**: 190,335 valid RGB JPEGs ($256 \times 256$ pixels).
- **Train Split**: 140,002 images (70,001 Real, 70,001 Fake) — 50.00% Real / 50.00% Fake.
- **Validation Split**: 39,428 images (19,787 Real, 19,641 Fake) — 50.18% Real / 49.82% Fake.
- **Test Split**: 10,905 images (5,413 Real, 5,492 Fake) — 49.64% Real / 50.36% Fake.
- **Integrity**: 0 corrupted files, 0 zero-byte files, 0 non-JPEG extensions. 100% readable via PIL.

#### 2. Provenance and Underlying Generation Methods
- **Real Images (Authentic)**: Sourced from the **Flickr-Faces-HQ (FFHQ)** dataset created by NVIDIA. These are real, high-resolution photographs of human faces crawled from Flickr, cropped and aligned to centered facial landmarks.
- **Fake Images (Synthetic)**: Sourced from **StyleGAN / StyleGAN2** generative adversarial networks (`thispersondoesnotexist.com`). These are fully synthetic facial portraits synthesized from random latent vectors $z \in \mathcal{Z}$.
- **Comparability**: Both classes underwent identical cropping, centering, and bicubic downsampling to $256 \times 256$ pixels. However, because Real originates from Flickr photography and Fake originates from a single StyleGAN generator checkpoint, models can learn generator-specific artifacts rather than generalized deepfake manipulation cues.

#### 3. Duplicate and Data Leakage Analysis (Audit Confirmation)
An exhaustive byte-for-byte and MD5 hash audit ([`duplicate_analysis_report.txt`](file:///C:/Users/user/Desktop/currently%20working/duplicate_analysis_report.txt)) established:
- **Total Duplicate Groups**: 17,135 duplicate groups across the entire dataset.
- **Train $\cap$ Validation Overlap**: 5,334 duplicate groups (10,668 files) exist between Train and Validation.
- **Test Split Cleanliness**:
  - Test images leaked into Train: **0 images (0.00%)**.
  - Test images leaked into Validation: **0 images (0.00%)**.
  - Internal duplicates within Test: **0 images (0.00%)**.
  - Label Conflicts across duplicates: **0 (Zero)**.
- **Critical Audit Conclusion**: The **Test split (10,905 images) is 100% clean and isolated** from training pixel data.

---

### 2.3 Diagnostic Benchmark Suite (`benchmark/images`)

Located at `C:\Users\user\Desktop\multi model connection\benchmark\images`, this directory serves as an independent diagnostic harness with documented provenance in `benchmark/metadata.json`:

1. **`real/real_phone_photo.png`**:
   - Format: PNG, RGB, $4000 \times 3000$ (11,063,620 bytes).
   - Label: `REAL`.
   - Evidence: Natural high-resolution photograph taken with a physical smartphone camera indoors. No generative synthesis or digital alteration.
2. **`ai_generated/ai_cyberpunk_city.jpg`**:
   - Format: JPEG, RGB, $1024 \times 1024$ (1,093,688 bytes).
   - Label: `FAKE`.
   - Evidence: Synthesized via **Google Imagen 3** text-to-image latent diffusion model. Depicts an urban cyberpunk scene with photorealistic lighting and complex geometry.
3. **`manipulated/manipulated_image.png`**:
   - Format: PNG, RGB, $800 \times 800$ (992,395 bytes).
   - Label: `FAKE`.
   - Evidence: Digital composite splicing photographic scene content with inserted synthetic elements.
4. **`corrupted/corrupted_image.png`**:
   - Format: Non-image, 20 bytes.
   - Label: `REJECTION_CONTROL` (`CORRUPTED`).
   - Evidence: Truncated byte stream missing valid PNG chunk headers, designed to verify input validation and graceful error handling.
5. **`corrupted/unsupported_doc.txt`**:
   - Format: Plain text, ASCII, 82 bytes.
   - Label: `REJECTION_CONTROL` (`UNSUPPORTED`).
   - Evidence: Plain text file designed to verify MIME-type and format rejection before neural inference.

---

### 2.4 Operational Smoke Suite (`benchmark/test_images`)

Located at `C:\Users\user\Desktop\multi model connection\benchmark\test_images`, these fixtures are used in automated contract and adapter test suites ([`test_detector_contract.py`](file:///C:/Users/user/Desktop/multi%20model%20connection/scripts/test_detector_contract.py)):
- **`real.jpg`**: $509 \times 865$, JPEG, RGB, 208,705 bytes. Natural photo.
- **`ai_generated.jpg`**: $508 \times 865$, JPEG, RGB, 227,341 bytes. Diffusion-generated image.
- **`manipulated.jpg`**: $509 \times 865$, JPEG, RGB, 200,922 bytes. Composite manipulation.
- **`corrupted_image.png`**: 8 bytes. Truncated malformed header.

---

### 2.5 DRCT Reference Samples & Desktop Negative Findings

1. **`temp_DRCT_reference/data/samples`**:
   - Files: `01.png` ($512 \times 896$), `02.png` ($768 \times 512$), `03.png` ($512 \times 512$).
   - Audit Status: **UNRESOLVED / UNANNOTATED**.
   - These files originate from the ICML 2024 DRCT reference repository. While used in `debug_code.py` to test image decoding and post-processing augmentations, the repository provides **no ground-truth label annotations or documentary proof** of whether they are authentic MSCOCO photos, reconstructed diffusion counterparts, or generated samples.
   - **Anti-Hallucination Policy Compliance**: In accordance with user rules, these samples are marked **UNRESOLVED** and excluded from ground-truth benchmark metrics.
2. **Other Desktop Folders Inspected**:
   - `C:\Users\user\Desktop\Deepfake-Detection-main`: Contains only 1 static web asset (`bgimage.14b90305.jpg`). No dataset.
   - `C:\Users\user\Desktop\file 1`: Static web bundle duplicate. No dataset.
   - `C:\Users\user\Desktop\Files`: UI icon assets (`icon16.png`, `icon48.png`, `icon128.png`). No dataset.
   - `C:\Users\user\Desktop\PG final year Project`: Duplicate web build. No dataset.
   - `C:\Users\user\Desktop\New`: Empty.
   - `C:\Users\user\Desktop\workflow`: Architectural diagrams only. No evaluation data.

---

## 3. Supported Evaluation Categories and Data Integrity

### 3.1 Genuinely Supported Categories

Based strictly on verified evidence and disk assets, the following categories are genuinely supported:

| Category Code | Category Name | Underlying Generator / Method | Verified Source | Ground Truth | Supported Image Resolution |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **CAT-A** | **Authentic Photographic Faces** | Photographic capture (FFHQ) | `Dataset/Test/Real` | `REAL` | $256 \times 256$ |
| **CAT-B** | **GAN-Generated Faces** | StyleGAN / StyleGAN2 synthesis | `Dataset/Test/Fake` | `FAKE` | $256 \times 256$ |
| **CAT-C** | **Authentic Natural Scenes / Objects** | Smartphone camera hardware capture | `benchmark/images/real` | `REAL` | $4000 \times 3000$ |
| **CAT-D** | **Modern Diffusion-Generated Scenes** | Google Imagen 3 text-to-image latent diffusion | `benchmark/images/ai_generated` | `FAKE` | $1024 \times 1024$ |
| **CAT-E** | **Digitally Manipulated / Spliced Composite** | Localized image editing and splicing | `benchmark/images/manipulated` | `FAKE` | $800 \times 800$ |
| **CAT-F** | **Format Rejection & Robustness Controls** | Byte truncation & plain ASCII text | `benchmark/images/corrupted` | `REJECTION_CONTROL` | 8–82 bytes |

> [!CAUTION]
> **Unsupported Categories (Not Inferred / Not Fabricated):**
> 1. **Face Swaps (DeepFaceLab, SimSwap, Roop, InsightFace)**: No verified face-swapped video or image datasets currently exist in local storage.
> 2. **Face Reenactment (First Order Motion, LivePortrait)**: No verified motion reenactment crops exist.
> 3. **Commercial Closed Generators (Midjourney v6, DALL-E 3, Flux.1)**: Only Google Imagen 3 is currently represented in verified local benchmark fixtures.
> 4. **Post-Processing Degradations (Heavy Social Compression, Low-Light Video)**: Currently unrepresented in ground truth.

---

### 3.2 Domain Disconnect & Cross-Model Behavioral Matrix

The 3 integrated detectors exhibit drastically different domain assumptions:

```mermaid
flowchart TD
    Input["Input Image"] --> Router{"Image Domain Inspection"}
    
    Router -->|"Cropped Face (256x256)"| FacePath["Facial Domain"]
    Router -->|"Full Scene / Objects"| ScenePath["General Scene Domain"]
    
    FacePath --> ViT["ViT Adapter (dima806)<br>Trained on 140k Faces<br>Accuracy: 99.3% on StyleGAN"]
    FacePath --> UnivFD_Face["UnivFD Adapter<br>Trained on ProGAN<br>Moderate Face Generalization"]
    FacePath --> ConvNeXt_Face["ConvNeXt/DRCT<br>Trained on Diffusion-2M<br>Untested on Pure GAN Faces"]
    
    ScenePath --> ConvNeXt_Scene["ConvNeXt/DRCT<br>Trained on SDv2 / COCO<br>High Sensitivity to Diffusion Artifacts"]
    ScenePath --> UnivFD_Scene["UnivFD Adapter<br>Trained on 2018 ProGAN<br>Fails on Modern Diffusion (0.08% Fake)"]
    ScenePath --> ViT_Scene["ViT Adapter (dima806)<br>Faces Only Backbone<br>High Uncertainty on Scenes (34% Fake)"]
```

#### Empirical Cross-Model Comparison on Diagnostic Suite:
- **`real_phone_photo.png` (Real Scene)**:
  - `UnivFD`: Correctly predicted REAL ($P_\text{fake} = 0.0018\%$).
  - `ViT (dima806)`: Correctly predicted REAL ($P_\text{fake} = 19.78\%$).
  - `ConvNeXt/DRCT`: Correctly predicted REAL ($P_\text{fake} = 4.89\%$).
- **`ai_cyberpunk_city.jpg` (Imagen 3 Diffusion)**:
  - `UnivFD`: **Misclassified as REAL** ($P_\text{fake} = 0.0828\%$, logit $-7.096$). Failed to detect modern diffusion.
  - `ViT (dima806)`: Predicted REAL at 0.50 threshold ($P_\text{fake} = 34.24\%$), but elevated fake probability reflected texture abnormality.
  - `ConvNeXt/DRCT`: Correctly identified subtle diffusion reconstruction discrepancy.
- **`manipulated_image.png` (Digital Splicing)**:
  - All whole-image classification backbones misclassify localized spatial splicing because global scene context remains visually coherent.

---

### 3.3 Investigation: CommunityForensics ViT Detector Suitability

The audit evaluated `buildborderless/CommunityForensics-DeepfakeDet-ViT` (stored in `C:\Users\user\Desktop\currently working\deepfake-detector-model-v1-main\models\CommunityForensics-DeepfakeDet-ViT`).

#### 1. Architecture and Training Provenance
- **Backbone**: Vision Transformer Small (`ViT-Small`, patch 16, $384 \times 384$ input resolution, 6 attention heads).
- **Parameters**: ~21.7M parameters (~87 MB checkpoint size).
- **Training Set**: CommunityForensics dataset, consisting of **2.7 million images** generated across **4,803 generative models** (primarily modern text-to-image diffusion models, Midjourney, Stable Diffusion, DALL-E, etc.).
- **Objective**: Open-domain synthetic media detection across diverse visual scene compositions.

#### 2. Empirical Performance on Facial Benchmark (from `results/CommunityForensics-DeepfakeDet-ViT/`)

```
========================================================================
COMMUNITY FORENSICS EVALUATION ON 100 SAMPLES (50 Real + 50 StyleGAN Fake)
========================================================================
Metric                       CommunityForensics-ViT    ViT (dima806)
------------------------------------------------------------------------
Overall Accuracy                     52.00%                99.00%
Real Recall (Specificity)            96.00% (48/50)        98.00% (49/50)
Fake Recall (Sensitivity)             8.00% ( 4/50)       100.00% (50/50)
False Alarms on Real (FP)             2 / 50                1 / 50
False Negatives on Fake (FN)         46 / 50                0 / 50
Fake Precision                       66.67%                98.04%
Fake F1 Score                        0.1429                0.9901
Mean Prediction Logit (Fake)         -7.82                 +6.84
Mean Prediction Logit (Real)         -9.73                 -5.12
Average CPU Latency                 108.39 ms             123.95 ms
========================================================================
```

#### 3. Root Cause Analysis
- **Severe Prediction Skew**: CommunityForensics predicted **94 of 100 samples as REAL** (94.0%).
- **Logit Distribution**: Even on synthetic StyleGAN faces, the average logit was $-7.82$, corresponding to $P(\text{fake}) \approx 0.04\%$.
- **Why did this occur?**
  1. The model was trained on high-resolution open-domain scenes ($384 \times 384$) focusing on global diffusion layout inconsistencies, semantic incoherence, and text-rendering errors.
  2. In tight $256 \times 256$ aligned face crops produced by StyleGAN, global scene layout artifacts do not exist. StyleGAN synthesizes plausible facial textures that completely fool CommunityForensics' general-scene attention heads.

#### 4. Explicit Recommendation for Later Phases
- **Suitability for Facial Deepfake Detection**: **STRICTLY UNSUITABLE**. With an 8% fake recall and 92% false negative rate, integrating CommunityForensics into a simple voting or averaging ensemble on facial crops would severely degrade overall detection accuracy.
- **Suitability for Open-Domain General AI Detection**: **POTENTIALLY VALUABLE**. For full-scene diffusion images (e.g. Midjourney, Imagen 3, Stable Diffusion), CommunityForensics provides complementary representation to `chen2024_convnext`.
- **Architectural Requirement**: Any future inclusion of CommunityForensics must use **conditional domain routing** (e.g., face detector gate) rather than naive multi-model averaging.

---

## 4. Proposed Evaluation Dataset Plan

### 4.1 Target Count and Composition

To provide rigorous, statistically sound cross-model evaluation while adhering to strict memory constraints, the proposed benchmark dataset comprises **159 verified samples**:

```
Total Manifest: 159 Samples
├── held_out_test (105 samples)  --> Strictly held out for cross-model benchmarking
│   ├── 50 Authentic Real Faces (FFHQ via Dataset/Test/Real)
│   ├── 50 Synthetic GAN Faces (StyleGAN2 via Dataset/Test/Fake)
│   ├── 1 Authentic Real Scene Photo (Natural smartphone capture)
│   ├── 1 Modern Diffusion Scene (Google Imagen 3)
│   ├── 1 Digitally Manipulated Scene (Composite Splicing)
│   ├── 1 Corrupted Byte Control (20-byte truncated file)
│   └── 1 Unsupported Format Control (82-byte ASCII text)
│
├── development_val (50 samples) --> Dedicated for fusion tuning & threshold calibration
│   ├── 25 Authentic Real Faces (FFHQ via Dataset/Test/Real)
│   └── 25 Synthetic GAN Faces (StyleGAN2 via Dataset/Test/Fake)
│
└── operational_smoke (4 samples) --> Reserved for CI/CD & adapter contract tests
    ├── 1 Authentic Real Scene (509x865 JPEG)
    ├── 1 Modern Diffusion Scene (508x865 JPEG)
    ├── 1 Digitally Manipulated Scene (509x865 JPEG)
    └── 1 Corrupted Control (8-byte malformed header)
```

### 4.2 Class and Domain Balance Statistics

| Split Name | Facial Domain | General Scene Domain | Control Domain | Total Real | Total Fake | Total Controls | Split Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`held_out_test`** | 100 | 3 | 2 | 51 (48.6%) | 52 (49.5%) | 2 (1.9%) | **105** |
| **`development_val`** | 50 | 0 | 0 | 25 (50.0%) | 25 (50.0%) | 0 (0.0%) | **50** |
| **`operational_smoke`**| 0 | 3 | 1 | 1 (25.0%) | 2 (50.0%) | 1 (25.0%) | **4** |
| **Overall Dataset** | **150** | **6** | **3** | **77 (48.4%)** | **79 (49.7%)** | **3 (1.9%)** | **159** |

---

### 4.3 Machine-Readable Manifest Schema

The manifest is saved at [`benchmark/evaluation_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/evaluation_manifest.csv). It contains 16 standardized columns:

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `sample_id` | `string` | Unique, deterministic sample identifier | `test_face_real_001` |
| `split` | `string` | Evaluation split (`held_out_test`, `development_val`, `operational_smoke`) | `held_out_test` |
| `domain` | `string` | High-level visual domain (`facial`, `general_scene`, `control`) | `facial` |
| `category` | `string` | Standardized category identifier (`authentic_face`, `gan_face`, etc.) | `authentic_face` |
| `ground_truth` | `string` | Authoritative ground truth (`REAL`, `FAKE`, `REJECTION_CONTROL`) | `REAL` |
| `source_dataset` | `string` | Underlying benchmark origin | `Kaggle 140k Real and Fake Faces (FFHQ)` |
| `generation_or_manipulation_method` | `string` | Precise generative or manipulation technique | `Photographic Capture (Flickr-Faces-HQ / FFHQ)` |
| `label_evidence` | `string` | Documentary proof citation supporting the label | `Documented FFHQ high-resolution photographic human portrait...` |
| `verification_status` | `string` | Disk and hash verification status | `VERIFIED` |
| `file_path` | `string` | Absolute path with forward slashes | `C:/Users/user/Desktop/currently working/Dataset/Test/Real/real_1500.jpg` |
| `relative_path` | `string` | Relative path from project or Desktop root | `currently_working/Dataset/Test/Real/real_1500.jpg` |
| `file_format` | `string` | Verified file container (`JPEG`, `PNG`, `TXT`) | `JPEG` |
| `width` | `integer` | Image horizontal resolution | `256` |
| `height` | `integer` | Image vertical resolution | `256` |
| `file_size_bytes` | `integer` | Exact file size in bytes | `8402` |
| `sha256` | `string` | Full 64-character SHA-256 cryptographic digest | `d577267882b8...` |

---

## 5. Leakage Prevention and Split Design

### 5.1 Historical Usage Audit

To guarantee zero data contamination, all past experiments and evaluation scripts across `currently working` were cross-referenced:
1. **Group A Exploration**: `real_0..49` and `fake_0..49` (100 samples) — used in initial ViT and CommunityForensics smoke tests.
2. **Threshold Calibration**: `real_400..499` and `fake_400..499` (200 samples) — used in `calibrate_thresholds.py`.
3. **Internal Holdout**: `real_500..699` and `fake_500..699` (400 samples) — used in `run_release_candidate_evaluation.py`.
4. **Random Sampling Experiment**: 250 samples evaluated via Flask API in `random_250_evaluation_results.csv`.
5. **Group B Large-Scale Validation**: 1,000 samples evaluated in `extended_validation_dima806.py`.

### 5.2 Strict Sample Isolation Protocol

```
Index 0 ─── 49    : Group A Smoke Test (EXCLUDED)
Index 50 ── 399   : Group B Lexicographical Sample (EXCLUDED)
Index 400 ─ 499   : Threshold Calibration Partition (EXCLUDED)
Index 500 ─ 699   : Release Candidate Holdout Partition (EXCLUDED)
Index 700 ─ 1447  : Exploratory & API Tested Samples (EXCLUDED)
───────────────────────────────────────────────────────────────────────
Index 1500 ─ 1549 : HELD-OUT TEST SPLIT (50 Real + 50 Fake)  <-- PRISTINE
Index 1550 ─ 1574 : DEVELOPMENT / VALIDATION SPLIT (25 Real + 25 Fake) <-- PRISTINE
───────────────────────────────────────────────────────────────────────
Index 1575 ─ 5412 : Untouched Test Reservoir (3,838 Real + 3,918 Fake)
```

- **Pristine Sample Guarantee**: Samples `real_1500.jpg` through `real_1574.jpg` and `fake_1500.jpg` through `fake_1574.jpg` have **never been accessed** by any tuning, calibration, or evaluation script in the project history.
- **Split Decoupling for Phase 4**: 
  - `development_val` (50 samples) is dedicated to calibrating decision fusion weights, temperature scaling, or threshold optimization.
  - `held_out_test` (105 samples) remains strictly held out and will only be evaluated **after** fusion logic is frozen.

---

## 6. Dataset Gaps and Expansion Strategy

### 6.1 Matrix of Supported vs. Unsupported Conditions

| Condition Dimension | Supported in Current Manifest | Status / Gaps |
| :--- | :--- | :--- |
| **Object Domain** | Facial Portraits (150 samples)<br>General Urban Scenes (6 samples) | **Gap**: No animal, macro, nature, text, or medical image benchmarks. |
| **Generative Architectures** | GAN (StyleGAN/StyleGAN2)<br>Diffusion (Google Imagen 3) | **Gap**: Missing Midjourney v6, Stable Diffusion 3.5, Flux.1, SDXL, Sora/video gen. |
| **Manipulation Methods** | Digital Splicing (Composite) | **Gap**: Missing Face Swapping (SimSwap, Roop), Face Morphing, Expression Reenactment. |
| **Image Resolutions** | $256 \times 256$, $800 \times 800$, $1024 \times 1024$, $4000 \times 3000$ | **Gap**: No standardized multi-scale pyramid on the same image content. |
| **Quality Perturbations** | Clean Uncompressed baseline | **Gap**: No social media recompression (JPEG Q=30, WebP, WhatsApp downsampling). |
| **Integrity Checks** | Corrupted headers, plain text | **Fully Supported**: Graceful `ERROR` / `REJECTION_CONTROL` verified. |

---

### 6.2 Safe Benchmark Expansion Strategy

When expanding the evaluation dataset in future phases, the following ingestion protocol must be followed:

1. **Attribution & Licensing**:
   - Only import images with open research licenses (Creative Commons, Apache 2.0, MIT) or explicit public release documentation.
   - For real images, prioritize camera-hardware verified sources (e.g., RAISE dataset, Dresden Image Database).
2. **Deterministic File Placement**:
   - Place new authentic photographs into `benchmark/images/real/`.
   - Place new diffusion/generative samples into `benchmark/images/ai_generated/<generator_name>/`.
   - Place face-swap manipulations into `benchmark/images/manipulated/face_swap/`.
3. **Metadata Registration**:
   - Compute SHA-256 immediately upon arrival.
   - Record exact prompt, seed, model checkpoint version, and generation software.
   - Do NOT overwrite existing IDs or files.

---

## 7. Verification and Ingestion Protocol

### 7.1 Automated Verification Script

The verification script [`scripts/verify_dataset_manifest.py`](file:///C:/Users/user/Desktop/multi%20model%20connection/scripts/verify_dataset_manifest.py) validates the complete manifest against disk in under 1 second.

```powershell
python scripts/verify_dataset_manifest.py
```

#### Execution Output:
```text
================================================================================
DATASET MANIFEST INTEGRITY VERIFICATION
================================================================================
Manifest Path: C:\Users\user\Desktop\multi model connection\benchmark\evaluation_manifest.csv
Total manifest entries: 159

[Split Breakdown]
  - development_val: 50 samples
  - held_out_test: 105 samples
  - operational_smoke: 4 samples

[Ground Truth Breakdown]
  - FAKE: 79 samples
  - REAL: 77 samples
  - REJECTION_CONTROL: 3 samples

[Domain Breakdown]
  - control: 3 samples
  - facial: 150 samples
  - general_scene: 6 samples

[SUCCESS] Manifest verification passed 100% with zero errors!
```

### 7.2 Verification Checklist & Results

- [x] **File Existence**: All 159 files verified present on disk at designated absolute and relative paths.
- [x] **Byte Integrity**: 100% of recorded `file_size_bytes` match exact filesystem size.
- [x] **Cryptographic Hash**: All 159 SHA-256 hashes match calculated digests byte-for-byte.
- [x] **Image Readability**: All 156 images load cleanly in PIL with verified dimensions and RGB color channels.
- [x] **Non-Image Rejection**: 3 corrupted/unsupported fixtures verified to reject gracefully without crashing.
- [x] **Duplicate Elimination**: Zero duplicate hashes exist across any sample in the manifest.
- [x] **Leakage Elimination**: Zero overlap with training splits, validation splits, or previous calibration subsets.
- [x] **Original Preservation**: Zero original files modified, renamed, moved, or deleted.

---

## 8. Summary Table of Artifacts Created

| Artifact File | Path | Purpose |
| :--- | :--- | :--- |
| **Dataset Manifest** | [`benchmark/evaluation_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/evaluation_manifest.csv) | Machine-readable 159-sample balanced evaluation dataset manifest |
| **Verification Tool** | [`scripts/verify_dataset_manifest.py`](file:///C:/Users/user/Desktop/multi%20model%20connection/scripts/verify_dataset_manifest.py) | Automated integrity, hash, and format verification harness |
| **Comprehensive Plan** | [`docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md`](file:///C:/Users/user/Desktop/multi%20model%20connection/docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md) | Authoritative dataset audit report and evaluation roadmap |

