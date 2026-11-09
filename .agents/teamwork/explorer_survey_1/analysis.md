# Phase 4 Baseline Evaluation: Manifest & Infrastructure Survey Analysis

**Author:** Teamwork Explorer (`explorer_survey_1`)  
**Date:** 2026-10-30  
**Context:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Working Directory:** `c:\Users\user\Desktop\multi model connection`

---

## 1. Executive Summary

This investigation surveys the dataset manifest, verification tools, detector adapters, and benchmark execution infrastructure required for **Phase 4: Baseline Evaluation**. The primary objective is to evaluate three integrated deepfake detectors—**ConvNeXt/DRCT** (`chen2024_convnext`), **UniversalFakeDetect** (`univfd`), and **Vision Transformer** (`vit` / `dima806`)—sequentially and memory-safely across the verified `held_out_test` partition (105 samples) of `benchmark/evaluation_manifest.csv`.

### Key Survey Findings:
1. **Manifest Integrity**: `benchmark/evaluation_manifest.csv` consists of exactly 159 verified rows across 16 columns. All 159 files exist on disk, have matching byte counts and SHA-256 hashes, load cleanly in PIL (for images), and contain zero duplicate hashes or IDs.
2. **Held-Out Test Composition**: The `held_out_test` partition contains exactly 105 samples:
   - **Facial Domain (100 samples)**: 50 authentic human faces (FFHQ) and 50 synthetic GAN faces (StyleGAN2), 100% balanced.
   - **General Scene Domain (3 samples)**: 1 authentic smartphone photo, 1 Google Imagen 3 diffusion scene, 1 composite digital splice.
   - **Control Domain (2 samples)**: 1 truncated corrupted image, 1 ASCII plain text file.
   - **Metric Eligibility**: 103 samples are eligible for REAL/FAKE classification metrics; 2 samples are rejection controls requiring separate reporting.
3. **Existing Runner Limitations**: `scripts/run_detector_benchmark.py` was designed for a 5-sample diagnostic suite. It reads hardcoded JSON (`benchmark/metadata.json`), assumes all files reside in `benchmark/images`, lacks CLI split filtering, computes no domain-specific classification metrics (accuracy, precision, recall, F1, balanced accuracy, confidence intervals), and would overwrite existing diagnostic benchmark outputs.
4. **Requirement R1 Recommendation**: Modifying `scripts/run_detector_benchmark.py` would introduce regression risks to existing diagnostic workflows. Creating a dedicated, narrowly scoped evaluation runner—`scripts/run_phase4_baseline_evaluation.py`—strictly complies with Requirement R1, preserves existing benchmark results, safely adopts the proven subprocess memory-management pattern, and generates the exact Phase 4 deliverables.

---

## 2. Dataset Manifest Deep Dive (`benchmark/evaluation_manifest.csv`)

### 2.1 Manifest Schema Specification (16 Columns)

The manifest is formatted as standard comma-separated values (CSV) with UTF-8 encoding. It defines 16 columns without missing values:

| # | Column Name | Data Type | Nullable | Permitted / Observed Values | Description |
|---|-------------|-----------|:--------:|-----------------------------|-------------|
| 1 | `sample_id` | `str` | No | 159 unique strings (`test_face_real_001`..`050`, `test_face_fake_001`..`050`, `test_scene_*`, `test_control_*`, `val_face_*`, `smoke_*`) | Unique deterministic identifier |
| 2 | `split` | `str` | No | `held_out_test` (105), `development_val` (50), `operational_smoke` (4) | Target evaluation partition |
| 3 | `domain` | `str` | No | `facial` (150), `general_scene` (6), `control` (3) | Visual problem domain |
| 4 | `category` | `str` | No | `authentic_face`, `gan_face`, `authentic_scene`, `diffusion_scene`, `manipulated_scene`, `corrupted_file`, `unsupported_format` | Specific visual category |
| 5 | `ground_truth` | `str` | No | `REAL` (77), `FAKE` (79), `REJECTION_CONTROL` (3) | Authoritative target ground truth |
| 6 | `source_dataset` | `str` | No | `Kaggle 140k Real and Fake Faces (FFHQ)`, `Kaggle 140k Real and Fake Faces (StyleGAN)`, `Diagnostic Benchmark (Local)`, `Operational Benchmark (Local)` | Upstream provenance |
| 7 | `generation_or_manipulation_method` | `str` | No | 10 distinct generative or capture method descriptions | Precise synthesis/capture technique |
| 8 | `label_evidence` | `str` | No | 11 distinct citation strings documenting ground-truth basis | Verification evidence citation |
| 9 | `verification_status` | `str` | No | `VERIFIED` (100% of samples) | Audit status |
| 10 | `file_path` | `str` | No | Absolute file paths with forward slashes (e.g. `C:/Users/user/Desktop/currently working/Dataset/Test/...`) | Target disk path |
| 11 | `relative_path` | `str` | No | Relative path from project or Desktop directory | Relative path reference |
| 12 | `file_format` | `str` | No | `JPEG` (152), `PNG` (4), `NON_IMAGE` (3) | File format / container type |
| 13 | `width` | `int` / `str` | No | `256`, `508`, `509`, `800`, `1024`, `4000`, or `N/A` | Horizontal pixel resolution |
| 14 | `height` | `int` / `str` | No | `256`, `800`, `865`, `1024`, `3000`, or `N/A` | Vertical pixel resolution |
| 15 | `file_size_bytes` | `int` | No | Positive integer (8 to 11,063,620 bytes) | Exact filesystem size in bytes |
| 16 | `sha256` | `str` | No | 64-character lowercase hex string | Cryptographic SHA-256 digest |

### 2.2 Manifest Partition Breakdown

| Split Name | Facial Domain | General Scene Domain | Control Domain | Total REAL | Total FAKE | Total REJECTION_CONTROL | Total Samples | Percent of Manifest |
|------------|:-------------:|:--------------------:|:--------------:|:----------:|:----------:|:-----------------------:|:-------------:|:-------------------:|
| **`held_out_test`** | 100 | 3 | 2 | 51 | 52 | 2 | **105** | 66.04% |
| **`development_val`** | 50 | 0 | 0 | 25 | 25 | 0 | **50** | 31.45% |
| **`operational_smoke`**| 0 | 3 | 1 | 1 | 2 | 1 | **4** | 2.52% |
| **Grand Total** | **150** | **6** | **3** | **77** | **79** | **3** | **159** | 100.0% |

### 2.3 Detailed Composition of `held_out_test` (105 Samples)

The 105 samples in `held_out_test` are strictly partitioned into three operational groups:

#### Group 1: Facial Classification Subset (100 samples)
- **Authentic Faces (`authentic_face`, REAL)**: 50 samples (`test_face_real_001` through `test_face_real_050`).
  - Source: FFHQ via `C:/Users/user/Desktop/currently working/Dataset/Test/Real/real_1500.jpg` to `real_1549.jpg`.
  - Specs: JPEG, $256 \times 256$, RGB, size range 7,549–16,357 bytes.
- **Synthetic GAN Faces (`gan_face`, FAKE)**: 50 samples (`test_face_fake_001` through `test_face_fake_050`).
  - Source: StyleGAN2 via `C:/Users/user/Desktop/currently working/Dataset/Test/Fake/fake_1500.jpg` to `fake_1549.jpg`.
  - Specs: JPEG, $256 \times 256$, RGB, size range 7,565–17,144 bytes.
- **Statistical Balance**: Exactly 50.0% Real / 50.0% Fake.
- **Leakage Status**: Pristine. Indexes 1500–1549 have never been exposed to prior training, validation, or threshold tuning.

#### Group 2: General Scene Diagnostic Cases (3 samples)
- **`test_scene_real_001`**: Natural camera photograph (`benchmark/images/real/real_phone_photo.png`).
  - Domain: `general_scene` | Category: `authentic_scene` | Ground Truth: `REAL` | PNG, $4000 \times 3000$, 11,063,620 bytes.
- **`test_scene_diffusion_001`**: Modern text-to-image synthesis (`benchmark/images/ai_generated/ai_cyberpunk_city.jpg`).
  - Domain: `general_scene` | Category: `diffusion_scene` | Ground Truth: `FAKE` | JPEG, $1024 \times 1024$, 1,093,688 bytes.
- **`test_scene_manip_001`**: Digital splicing composite (`benchmark/images/manipulated/manipulated_image.png`).
  - Domain: `general_scene` | Category: `manipulated_scene` | Ground Truth: `FAKE` | PNG, $800 \times 800$, 992,395 bytes.
- **Evaluation Rule**: Evaluated as **single-example qualitative diagnostics only** ($n=1$ each). Must not be aggregated into general scene benchmark claims.

#### Group 3: Rejection Controls (2 samples)
- **`test_control_corr_001`**: Corrupted 20-byte file (`benchmark/images/corrupted/corrupted_image.png`).
  - Domain: `control` | Category: `corrupted_file` | Ground Truth: `REJECTION_CONTROL` | `NON_IMAGE`.
  - Expected Behavior: Rejection (`PREDICTION_ERROR` via `DetectorResult.create_error`).
- **`test_control_unsupp_001`**: ASCII plain text file (`benchmark/images/corrupted/unsupported_doc.txt`).
  - Domain: `control` | Category: `unsupported_format` | Ground Truth: `REJECTION_CONTROL` | `NON_IMAGE`, 82 bytes.
  - Expected Behavior: Rejection (`PREDICTION_ERROR` via `DetectorResult.create_error`).
- **Evaluation Rule**: Must be isolated from classification accuracy metrics. Evaluated strictly on pipeline safety and error handling.

### 2.4 Disk File Existence Verification

Programmatic verification of all 159 rows confirmed:
- Files existing on disk: **159 / 159 (100.0%)**
- Missing files: **0**
- Accessible paths:
  - 150 facial samples at `C:\Users\user\Desktop\currently working\Dataset\Test\`
  - 5 diagnostic samples at `C:\Users\user\Desktop\multi model connection\benchmark\images\`
  - 4 smoke samples at `C:\Users\user\Desktop\multi model connection\benchmark\test_images\`

---

## 3. Dataset Manifest Verification Harness (`scripts/verify_dataset_manifest.py`)

### 3.1 Verification Mechanics
The verification script inspects:
1. **Manifest Presence**: Checks that `benchmark/evaluation_manifest.csv` exists.
2. **Row Count Enforcement**: Asserts that `total_entries == 159`.
3. **Uniqueness**: Asserts zero duplicate `sample_id` values and zero duplicate `sha256` hashes across the dataset.
4. **Filesystem Integrity**: Confirms each `file_path` exists, verifies `os.path.getsize(file_path) == file_size_bytes`, and computes SHA-256 to ensure exact cryptographic match.
5. **Image Parsing & Resolution**: For `JPEG` and `PNG`, opens images via PIL and asserts `img.size == (width, height)`. For non-images, asserts category is `corrupted_file` or `unsupported_format`.
6. **Partition & Domain Balances**: Asserts:
   - `held_out_test == 105`, `development_val == 50`, `operational_smoke == 4`
   - `REAL == 77`, `FAKE == 79`, `REJECTION_CONTROL == 3`
   - `facial == 150`, `general_scene == 6`, `control == 3`

### 3.2 Invocation and Results
- **Command**: `python scripts/verify_dataset_manifest.py`
- **Exit Code**: `0` (Success)
- **Execution Log**:
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

---

## 4. Benchmark Runner Assessment (`scripts/run_detector_benchmark.py`)

### 4.1 Implementation Overview
`scripts/run_detector_benchmark.py` was implemented during the memory-safe refactoring documented in `scripts/MEMORY_SAFE_BENCHMARK_REPORT.md`. Its architecture addresses the Windows virtual memory exhaustion error (`os error 1455`) that occurs when multiple heavy deep learning models reside concurrently in memory.

### 4.2 Ingestion & Argument Handling
- **Input Data**: Hardcodes `METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")`. Does not accept a CSV path or manifest CLI argument.
- **Image Resolution**: Hardcodes `IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")` and constructs paths via `os.path.join(IMAGES_DIR, rel_path)`. Fails if paths point outside `benchmark/images`.
- **CLI Arguments**: Only `--mode [inprocess|subprocess]` (default: `inprocess`).
- **Split Handling**: Has no concept of evaluation splits (`held_out_test`, `development_val`, `operational_smoke`). Evaluates 100% of rows present in `metadata.json`.

### 4.3 Memory Management Mechanics
The script implements a four-tier memory management pattern:
1. **Sequential Evaluation**: Evaluates one detector across all samples before initializing the next.
2. **Explicit Lifecycle Release**: Invokes `adapter.unload_model()`, deletes the adapter instance, clears registry caches via `reset_instances()`, triggers generational garbage collection (`gc.collect()`), and clears CUDA cache (`torch.cuda.empty_cache()`).
3. **Subprocess Isolation (`--mode subprocess`)**: Runs each detector in an independent Python subprocess. The OS unconditionally reclaims 100% of committed virtual memory and heap allocations upon child process exit.
4. **Fault Isolation**: Captures exceptions during model loading or inference and emits `DetectorResult.create_error(...)` records without terminating the run.

### 4.4 Outputs and Overwrite Risk
The script generates:
- `benchmark/results/{det_name}_results.json`
- `benchmark/results/all_detector_predictions.json`
- `benchmark/results/all_detector_predictions.csv`
- `benchmark/results/detector_comparison.json`
- `benchmark/results/detector_comparison.csv`

Running `scripts/run_detector_benchmark.py` directly overwrites existing diagnostic benchmark results stored in `benchmark/results/`.

---

## 5. Architectural Assessment for Requirement R1

### 5.1 Requirement R1 Analysis
Requirement R1 states:
> *"If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner."*

### 5.2 Comparative Analysis: Modification vs. Dedicated Evaluation Script

| Dimension | Option A: Modify `run_detector_benchmark.py` | Option B: Dedicated `run_phase4_baseline_evaluation.py` |
|---|---|---|
| **Impact on Existing Workflows** | High risk. Breaks backward compatibility for 5-sample diagnostic tests and existing CI scripts relying on `metadata.json`. | Zero risk. Preserves existing diagnostic runner and regression tests completely intact. |
| **Manifest Ingestion** | Requires complex refactoring to support both JSON (`metadata.json`) and CSV (`evaluation_manifest.csv`), differing schemas, and absolute vs relative path logic. | Clean, purpose-built CSV ingestion of `benchmark/evaluation_manifest.csv` with strict schema validation. |
| **Split Isolation** | Requires adding CLI split flags, default behaviors, and filtering logic into an already complex script (488 lines). | Built-in strict filtering on `split == 'held_out_test'` (105 samples), preventing contamination by design. |
| **Output Preservation** | Overwrites `benchmark/results/all_detector_predictions.*` and `detector_comparison.*`, violating R4 non-overwrite constraints. | Writes dedicated deliverables: `phase4_baseline_predictions.csv`, `phase4_baseline_metrics.json`, `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`. |
| **Metric Generation** | `run_detector_benchmark.py` computes zero classification metrics (only agreement status). Adding confusion matrices, Wilson CIs, domain splits, and diagnostic reporting would bloat the runner. | Encapsulates domain-specific metric computation, Wilson confidence intervals, diagnostic summaries, and pairwise analysis cleanly. |
| **Compliance with R1** | Direct violation of R1's conditional instruction. | Direct, explicit compliance with R1. |

### 5.3 Recommendation
**Implement a dedicated runner: `scripts/run_phase4_baseline_evaluation.py`.**

---

## 6. Functional Specification for `scripts/run_phase4_baseline_evaluation.py`

### 6.1 Core Responsibilities
1. **Manifest Loading & Verification**:
   - Ingest `benchmark/evaluation_manifest.csv`.
   - Filter rows strictly where `split == "held_out_test"` (105 samples).
   - Assert that sample count equals exactly 105:
     - 50 authentic facial (`authentic_face`, REAL)
     - 50 synthetic facial (`gan_face`, FAKE)
     - 1 authentic scene (`authentic_scene`, REAL)
     - 1 diffusion scene (`diffusion_scene`, FAKE)
     - 1 composite scene (`manipulated_scene`, FAKE)
     - 2 controls (`corrupted_file` and `unsupported_format`, REJECTION_CONTROL)
2. **Sequential Memory-Safe Execution**:
   - Detectors: `chen2024_convnext`, `univfd`, `vit`.
   - Execution Mode: Default to `--mode subprocess` for guaranteed OS-level memory reclamation on Windows, with fallback `--mode inprocess`.
   - In-process execution must invoke `adapter.unload_model()`, `del adapter`, `reset_instances()`, `gc.collect()`, and `torch.cuda.empty_cache()` between detectors.
3. **Robust Rejection Handling**:
   - For `REJECTION_CONTROL` samples (e.g. truncated PNG, plain text), adapters correctly catch parsing exceptions and emit `prediction == "ERROR"`.
   - Rejection controls are tracked separately and not counted as classification false positives/negatives.
4. **Domain-Specific Metrics Computation**:
   - **Eligible REAL/FAKE Set (103 samples)**:
     - Overall Confusion Matrix: TP, FP, TN, FN (treating FAKE as positive).
     - Metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score, Balanced Accuracy.
     - Report exact sample counts alongside all percentages.
   - **Facial Benchmark (100 samples: 50 Real, 50 Fake)**:
     - Dedicated confusion matrix and classification metrics.
     - 95% Confidence Intervals using the **Wilson score interval** for binomial proportions.
     - Explicit documentation of sample size limitations ($n=100$).
   - **General Scene Diagnostic Cases (3 samples)**:
     - Tabulate single-example predictions, probabilities, and logits for `test_scene_real_001`, `test_scene_diffusion_001`, and `test_scene_manip_001`.
     - Explicitly flag each as diagnostic and not statistically generalizable ($n=1$).
   - **Rejection Control Suite (2 samples)**:
     - Verify and report rejection outcomes (`ERROR` vs non-error) for `test_control_corr_001` and `test_control_unsupp_001`.
   - **Pairwise Agreement Analysis**:
     - Compute pairwise agreement strictly over samples where **both** detectors returned valid predictions (`prediction != "ERROR"`).
     - Exclude dual-error samples from agreement counts.
   - **Error & Failure Audit**:
     - Enumerate every False Positive, False Negative, and Execution Error with `sample_id`, domain, category, true label, predicted class, probability, and raw logit.
5. **Deliverables Generation**:
   - `benchmark/results/phase4_baseline_predictions.csv`: Standardized tabular log of all predictions and errors for all 105 samples across all 3 detectors (315 evaluation records).
   - `benchmark/results/phase4_baseline_metrics.json`: Structured JSON containing complete metrics, confusion matrices, confidence intervals, diagnostic outputs, and pairwise agreement stats.
   - `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`: Comprehensive markdown report detailing execution environment, exact commands, timestamps, model identifiers, sample counts, tables, and architectural implications for Phase 5 fusion.

---

## 7. Mathematical Formulations for Metrics & Confidence Intervals

### 7.1 Confusion Matrix Definitions (Positive Class = FAKE)
- **True Positive (TP)**: Ground Truth = `FAKE`, Prediction = `FAKE`
- **False Positive (FP)**: Ground Truth = `REAL`, Prediction = `FAKE`
- **True Negative (TN)**: Ground Truth = `REAL`, Prediction = `REAL`
- **False Negative (FN)**: Ground Truth = `FAKE`, Prediction = `REAL`

### 7.2 Core Classification Metrics
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad (\text{if } \text{TP} + \text{FP} = 0 \implies 0.0)$$

$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

$$\text{F1 Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (\text{if } \text{Precision} + \text{Recall} = 0 \implies 0.0)$$

$$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2}$$

### 7.3 Wilson Score Confidence Interval (95% CI)
For sample size $n$ and observed proportion $\hat{p} = \frac{k}{n}$, with standard normal quantile $z = 1.95996$ (for 95% confidence):

$$\hat{p}_{\text{center}} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}$$

$$\text{Margin} = \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}$$

$$\text{CI}_{95\%} = \left[ \max\left(0, \hat{p}_{\text{center}} - \text{Margin}\right), \min\left(1, \hat{p}_{\text{center}} + \text{Margin}\right) \right]$$

The Wilson score interval is preferred over the Wald interval ($\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$) because it provides accurate coverage probabilities even when proportions approach $0.0$ or $1.0$ and sample sizes are moderate ($n=50$ or $n=100$).

---

## 8. Summary of Findings & Next Steps

1. **Manifest Ready**: `benchmark/evaluation_manifest.csv` is 100% verified and disk-validated.
2. **Environment & Adapters Ready**: All 3 adapters (`ConvNeXtDetectorAdapter`, `UnivFDAdapter`, `ViTAdapter`) pass contract and registry tests, provide lazy loading, and implement `unload_model()`.
3. **Execution Script Required**: Create `scripts/run_phase4_baseline_evaluation.py` as specified to perform the evaluation safely, calculate all required metrics, and write Phase 4 deliverables.
