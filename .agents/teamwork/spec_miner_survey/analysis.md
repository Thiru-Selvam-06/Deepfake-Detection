# Phase 4 Baseline Evaluation: Technical Specification Analysis

**Document Status**: Final Specification Analysis  
**Author**: Specification Miner (`spec_miner_survey`)  
**Date**: 2026-10-30  
**Target Milestone**: Phase 4 — Baseline Independent Evaluation of Integrated Image Detectors  
**Working Directory**: `c:\Users\user\Desktop\multi model connection`  

---

## 1. Executive Summary & Mission Scope

This technical specification provides the definitive blueprint for executing the **Phase 4 Baseline Evaluation** across the three currently integrated deepfake image detectors:
1. **ConvNeXt-Base (Chen et al. 2024 / RAID)** (`chen2024_convnext`)
2. **UniversalFakeDetect (CVPR 2023)** (`univfd`)
3. **Vision Transformer (ViT-Base/16, dima806)** (`vit`)

### 1.1 Core Purpose
Before implementing any decision fusion layer, ensemble voting mechanism, or architectural alterations (Phase 5), the project requires a reproducible, isolated, and leak-free empirical baseline. Every model must be evaluated individually against the authoritative dataset manifest under strict memory safety constraints, producing standardized predictions, domain-stratified metrics, pairwise concordance statistics, and comprehensive diagnostic logs.

### 1.2 Authoritative Source Documents
- `c:\Users\user\Desktop\multi model connection\.agents\teamwork\ORIGINAL_REQUEST.md` (Authoritative Request & Acceptance Criteria)
- `docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md` (Dataset Audit, Taxonomy, and Manifest Plan)
- `scripts/MEMORY_SAFE_BENCHMARK_REPORT.md` (Memory Safety & Subprocess Execution Protocol)
- `benchmark/evaluation_manifest.csv` (159-Sample Master Benchmark Manifest)
- `scripts/verify_dataset_manifest.py` (Automated Manifest Verification Harness)
- `src/detector_schema.py` (`DetectorResult` Contract)
- `src/detector_registry.py` (Dynamic Discovery & Adapter Factory Registration)

---

## 2. Dataset Scope & Split Integrity Rules

### 2.1 The Master Manifest Composition
The master manifest located at `benchmark/evaluation_manifest.csv` contains **159 verified samples** partitioned into three disjoint splits:

| Split Name | Sample Range / Identifiers | Count | Ground-Truth Composition | Purpose & Integrity Rule |
| :--- | :--- | :---: | :--- | :--- |
| **`held_out_test`** | `test_face_real_001`–`test_face_real_050`<br>`test_face_fake_001`–`test_face_fake_050`<br>`test_scene_real_001`<br>`test_scene_diffusion_001`<br>`test_scene_manip_001`<br>`test_control_corr_001`<br>`test_control_unsupp_001` | **105** | 51 REAL<br>52 FAKE<br>2 REJECTION_CONTROL | **PRIMARY EVALUATION SPLIT**.<br>Exclusively used for Phase 4 baseline metrics. |
| **`development_val`** | `val_face_real_001`–`val_face_real_025`<br>`val_face_fake_001`–`val_face_fake_025` | **50** | 25 REAL<br>25 FAKE | **STRICTLY EXCLUDED** from Phase 4 metrics.<br>Reserved solely for Phase 5 fusion calibration. |
| **`operational_smoke`**| `smoke_scene_real_001`<br>`smoke_scene_diffusion_001`<br>`smoke_scene_manip_001`<br>`smoke_control_corr_001` | **4** | 1 REAL<br>2 FAKE<br>1 REJECTION_CONTROL | **STRICTLY EXCLUDED** from Phase 4 metrics.<br>Reserved for CI adapter contract smoke tests. |
| **Total Manifest** | | **159** | 77 REAL, 79 FAKE, 3 REJECTION_CONTROL | Verified with zero SHA-256 duplicate collisions. |

### 2.2 Split Isolation & Non-Contamination Rules
1. **Absolute Partition Boundary**: Samples belonging to `development_val` (50 samples) and `operational_smoke` (4 samples) must **NEVER** enter primary Phase 4 metric calculations. Their presence in confusion matrices, accuracy calculations, or summary statistics represents a catastrophic validation breach.
2. **Held-Out Test Sample Guarantee**: The 100 facial samples in `held_out_test` (`real_1500.jpg`–`real_1549.jpg` and `fake_1500.jpg`–`fake_1549.jpg`) were proven in Phase 3 to be completely untouched by any historical calibration, training, validation, or exploratory runs (which ended at index 1447).
3. **Execution Scope Filter**: The evaluation runner must explicitly filter rows where `split == "held_out_test"`, processing exactly 105 samples per detector.

### 2.3 Detailed Composition of `held_out_test` (105 Samples)

| Category Identifier | Domain | Sample Count | Ground Truth | Source Dataset & Method | Supported Dimensions |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `authentic_face` | `facial` | 50 | `REAL` | FFHQ photographic capture (`real_1500.jpg`–`real_1549.jpg`) | $256 \times 256$ JPEG |
| `gan_face` | `facial` | 50 | `FAKE` | StyleGAN / StyleGAN2 synthetic faces (`fake_1500.jpg`–`fake_1549.jpg`) | $256 \times 256$ JPEG |
| `authentic_scene` | `general_scene` | 1 | `REAL` | Natural smartphone camera photo (`real_phone_photo.png`) | $4000 \times 3000$ PNG |
| `diffusion_scene` | `general_scene` | 1 | `FAKE` | Google Imagen 3 text-to-image latent diffusion (`ai_cyberpunk_city.jpg`) | $1024 \times 1024$ JPEG |
| `manipulated_scene` | `general_scene` | 1 | `FAKE` | Spliced photographic/synthetic composite (`manipulated_image.png`) | $800 \times 800$ PNG |
| `corrupted_file` | `control` | 1 | `REJECTION_CONTROL` | Truncated 20-byte file missing valid PNG header (`corrupted_image.png`) | Non-image (20 bytes) |
| `unsupported_format`| `control` | 1 | `REJECTION_CONTROL` | ASCII plain text document (`unsupported_doc.txt`) | Non-image (82 bytes) |
| **Total** | | **105** | | | |

---

## 3. Rejection Controls Handling Specification

### 3.1 Identification in Manifest
Rejection controls are uniquely marked in `benchmark/evaluation_manifest.csv` by the following attributes:
- `split`: `"held_out_test"`
- `domain`: `"control"`
- `category`: `"corrupted_file"` or `"unsupported_format"`
- `ground_truth`: `"REJECTION_CONTROL"`
- `file_format`: `"NON_IMAGE"`
- `width` / `height`: `"N/A"`

### 3.2 Expected Detector Behavior
- When an adapter's `predict(file_path)` method receives a rejection control fixture, it must **never crash**, raise an unhandled exception, or default to an arbitrary probability.
- All three adapters (`ConvNeXtDetectorAdapter`, `UnivFDAdapter`, `ViTAdapter`) have verified image ingress validation:
  1. `PIL.Image.open()` fails or format validation triggers.
  2. The exception is caught internally.
  3. The adapter returns a well-formed `DetectorResult` with:
     * `prediction = "ERROR"`
     * `error = "<descriptive error string>"` (e.g. `Invalid or corrupted image: cannot identify image file...`)
     * `fake_probability = None`
     * `real_probability = None`
     * `raw_logit = None`

### 3.3 Evaluation & Segregation Protocol
- **Strict Metric Segregation**: Rejection controls must **NEVER** be counted as False Positives, False Negatives, True Positives, or True Negatives. They are NOT binary classification samples.
- **Scoring Logic**:
  * **PASS / REJECTED (Success)**: Detector returns `prediction == "ERROR"`. The system correctly refused to process an invalid input.
  * **FAIL (Error)**: Detector returns a valid classification (`REAL` or `FAKE`), or crashes the process.
- **Reporting Requirement**: Rejection controls must be evaluated and reported in an independent **Rejection Controls Verification Table** and in a dedicated section of the metrics JSON:
  * Number of control samples tested (2).
  * Number of successful rejections (expected: 2 / 2, 100.0%).
  * Verbatim error strings captured.

---

## 4. Primary Classification Metrics Specification (Eligible REAL/FAKE Samples)

### 4.1 Sample Eligibility
Within `held_out_test` (105 samples), exactly **103 samples** have binary ground truth (`REAL` or `FAKE`):
- **Total Eligible Ground-Truth Samples**: $N = 103$
- **Total Authentic (REAL)**: 51 samples (50 facial + 1 general scene)
- **Total Synthetic / Manipulated (FAKE)**: 52 samples (50 facial + 1 diffusion scene + 1 composite)

### 4.2 Definition of Positive and Negative Classes
In deepfake detection standards and throughout this project repository:
- **Positive Class = `FAKE`** (synthetic media, generative synthesis, digital manipulation)
- **Negative Class = `REAL`** (authentic camera capture, photographic portrait)

### 4.3 Metric Definitions & Formulas

#### 1. Confusion Matrix
- **True Positive (TP)**: Sample is `FAKE`, detector predicts `FAKE`.
- **False Positive (FP)**: Sample is `REAL`, detector predicts `FAKE` (*False Alarm / Flagged Authentic*).
- **True Negative (TN)**: Sample is `REAL`, detector predicts `REAL` (*Correct Authentic Pass*).
- **False Negative (FN)**: Sample is `FAKE`, detector predicts `REAL` (*Missed Deepfake / Missed AI*).

#### 2. Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{TP + TN}{N}$$

#### 3. Precision (Positive Predictive Value / Fake Precision)
$$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{if } TP + FP = 0, \text{ defined as } 0.0)$$

#### 4. Recall (Sensitivity / True Positive Rate / Fake Recall)
$$\text{Recall} = \frac{TP}{TP + FN} \quad (\text{if } TP + FN = 0, \text{ defined as } 0.0)$$

#### 5. Specificity (True Negative Rate / Real Recall)
$$\text{Specificity} = \frac{TN}{TN + FP} \quad (\text{if } TN + FP = 0, \text{ defined as } 0.0)$$

#### 6. F1 Score
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \quad (\text{if denominator } = 0, \text{ defined as } 0.0)$$

#### 7. Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$$

### 4.4 Mandatory Reporting Rule: Exact Sample Counts
Every percentage reported in tables, text, or JSON must include its exact integer fraction:
- *Example*: `"98.06% (101/103)"`, `"False Positives: 1 (1/51, 1.96%)"`.
- Never report floating-point percentages in isolation without the supporting integer numerator and denominator.

---

## 5. Facial Evaluation Specification (100 Samples)

### 5.1 Facial Domain Sub-Benchmark Scope
The 100 facial samples in `held_out_test` represent a balanced, controlled facial benchmark:
- **50 Authentic Facial Portraits** (`test_face_real_001`–`test_face_real_050`): FFHQ photographic captures, cropped and centered to $256 \times 256$.
- **50 Synthetic GAN Faces** (`test_face_fake_001`–`test_face_fake_050`): StyleGAN / StyleGAN2 synthetic generations from `thispersondoesnotexist.com`, cropped and centered to $256 \times 256$.

### 5.2 Metrics & Confidence Interval Methodology
Because $N=100$ is a finite benchmark sample, reporting point estimates of accuracy, recall, and precision alone does not reflect statistical uncertainty. The evaluation must compute **95% Confidence Intervals** using either:

#### Method A: Wilson Score Interval (Recommended / Closed Form)
For a binomial proportion $\hat{p} = \frac{k}{n}$ where $k$ is successes and $n$ is total trials, with significance level $\alpha = 0.05$ ($z_{1-\alpha/2} = 1.96$):

$$w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

*Advantages*:
- Guarantees lower bound $\ge 0.0$ and upper bound $\le 1.0$.
- Does not degenerate to zero width when $\hat{p} = 1.0$ (e.g. 50/50 recall gives interval $[0.9289, 1.0000]$).
- Statistically optimal for sample sizes $n \in [50, 100]$.

#### Method B: Non-Parametric Bootstrap Percentile Interval
- Resample with replacement $B = 1000$ iterations over the 100 samples.
- Compute metric $\hat{\theta}^{*b}$ for each replicate.
- Compute 2.5th and 97.5th percentiles: $[\hat{\theta}_{0.025}^*, \hat{\theta}_{0.975}^*]$.

### 5.3 Mandatory Stated Limitations of Facial Sub-Benchmark
Every report section discussing facial metrics must explicitly state the following scientific limitations:
1. **Resolution & Crop Bias**: All images are tightly cropped, centered $256 \times 256$ portraits. Detectors relying on facial bounding boxes or alignment may perform differently on in-the-wild full frames.
2. **Generator Homogeneity**: Synthetic faces originate solely from StyleGAN / StyleGAN2 architectures. The facial benchmark does not measure generalization against:
   * Latent diffusion face generators (e.g., Stable Diffusion 3, Flux.1 portraits)
   * Face-swap manipulations (DeepFaceLab, SimSwap, Roop)
   * Motion/reenactment deepfakes (LivePortrait, First Order Motion)
3. **Photographic Source Homogeneity**: Real portraits originate exclusively from FFHQ Flickr photography with high-end camera sensors, studio or natural outdoor lighting, and high visual clarity.

---

## 6. Diagnostic Reporting for Individual Scene & Composite Samples

### 6.1 Diagnostic Samples Inventory
The general scene subset of `held_out_test` contains exactly 3 individual samples:
1. **`test_scene_real_001`**:
   - File: `benchmark/images/real/real_phone_photo.png`
   - Description: Authentic natural photo captured indoors with smartphone camera hardware ($4000 \times 3000$ PNG).
   - Ground Truth: `REAL`
2. **`test_scene_diffusion_001`**:
   - File: `benchmark/images/ai_generated/ai_cyberpunk_city.jpg`
   - Description: Photorealistic urban cityscape generated via Google Imagen 3 latent diffusion ($1024 \times 1024$ JPEG).
   - Ground Truth: `FAKE`
3. **`test_scene_manip_001`**:
   - File: `benchmark/images/manipulated/manipulated_image.png`
   - Description: Digitally spliced composite combining authentic scene background with inserted synthetic elements ($800 \times 800$ PNG).
   - Ground Truth: `FAKE`

### 6.2 Strict Diagnostic Rules
1. **Flagged as Diagnostic Only**: These 3 samples MUST be explicitly labeled as **qualitative case studies**.
2. **Prohibition of Statistical Extrapolation**: Statistical metrics (e.g., "100% diffusion accuracy" or "0% composite recall") MUST NOT be claimed from an $N=1$ sample size.
3. **Structured Diagnostic Table Format**:
   The report must display each sample in a detailed comparative case-study matrix:

| Sample ID | Category | Ground Truth | ConvNeXt Prediction ($P_{\text{fake}}$, logit) | UnivFD Prediction ($P_{\text{fake}}$, logit) | ViT Prediction ($P_{\text{fake}}$, logit) | Concordance Status | Qualitative Forensic Insight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |

4. **Forensic Insight Requirements**:
   - Document whether the detector's continuous fake probability reflects artifact sensitivity even if discrete threshold ($0.50$) is not crossed.
   - Contrast UnivFD's failure on modern diffusion ($P_{\text{fake}} < 0.1\%$) against ViT's elevated probability and ConvNeXt's frequency response.
   - Explain why whole-image classification networks miss localized spatial splicing.

---

## 7. Pairwise Agreement & Concordance Specification

### 7.1 Objective & Boundary
Pairwise agreement measures the empirical consistency of classifications between detector pairs:
1. `chen2024_convnext` vs `univfd`
2. `chen2024_convnext` vs `vit`
3. `univfd` vs `vit`

### 7.2 Strict Eligibility Condition (Valid Predictions Only)
- Concordance is evaluated **STRICTLY over samples where BOTH models returned valid predictions** (`REAL` or `FAKE`).
- Samples where either model produced `prediction == "ERROR"` (e.g., rejection controls, decode failures) are **excluded** from the denominator $N_{\text{valid\_both}}$.
- For eligible `held_out_test` samples where both detectors succeed, $N_{\text{valid\_both}} \le 103$.

### 7.3 Formulas & Statistics

#### 1. Observed Concordance ($P_o$)
$$P_o = \frac{\sum_{i=1}^{N_{\text{valid\_both}}} \mathbb{I}(\text{pred}_{A, i} = \text{pred}_{B, i})}{N_{\text{valid\_both}}}$$

#### 2. Disagreement Rate ($P_{\text{dis}}$)
$$P_{\text{dis}} = 1.0 - P_o = \frac{\sum_{i=1}^{N_{\text{valid\_both}}} \mathbb{I}(\text{pred}_{A, i} \ne \text{pred}_{B, i})}{N_{\text{valid\_both}}}$$

#### 3. 2x2 Agreement Matrix
$$\begin{pmatrix} N_{FF} & N_{FR} \\ N_{RF} & N_{RR} \end{pmatrix}$$
- $N_{FF}$: Both detectors predict `FAKE`
- $N_{RR}$: Both detectors predict `REAL`
- $N_{FR}$: Model A predicts `FAKE`, Model B predicts `REAL`
- $N_{RF}$: Model A predicts `REAL`, Model B predicts `FAKE`

#### 4. Chance-Corrected Agreement (Cohen's Kappa $\kappa$)
$$P_{\text{fake}, A} = \frac{N_{FF} + N_{FR}}{N_{\text{valid\_both}}}, \quad P_{\text{fake}, B} = \frac{N_{FF} + N_{RF}}{N_{\text{valid\_both}}}$$
$$P_{\text{real}, A} = \frac{N_{RF} + N_{RR}}{N_{\text{valid\_both}}}, \quad P_{\text{real}, B} = \frac{N_{FR} + N_{RR}}{N_{\text{valid\_both}}}$$
$$P_e = (P_{\text{fake}, A} \cdot P_{\text{fake}, B}) + (P_{\text{real}, A} \cdot P_{\text{real}, B})$$
$$\kappa = \frac{P_o - P_e}{1 - P_e} \quad (\text{if } P_e = 1.0, \kappa = 1.0)$$

### 7.4 Critical Authenticity Disclaimer
> **MANDATORY NOTICE:**  
> High pairwise agreement between two detectors does **NOT** indicate ground-truth authenticity or accuracy. Two detectors can exhibit 100% concordance while being 100% incorrect (e.g., both failing to detect a diffusion image). Concordance is an observational metric of model behavioral alignment, not a surrogate for verification.

---

## 8. Error Handling & Failure Retention Specification

### 8.1 Retention Policy: Zero Sample Dropping
Under no circumstances may an evaluation script discard, drop, or ignore an image because of an execution failure, out-of-memory error, or decode exception.
- If an image cannot be processed, its row must be recorded with:
  * `prediction = "ERROR"`
  * `status = "ERROR"`
  * `error_message = "<full error text>"`
- The sample must remain present in `phase4_baseline_predictions.csv` and documented in `phase4_baseline_metrics.json`.

### 8.2 Execution Status Taxonomy
Every prediction row must have a `status` field taking one of three values:
1. `SUCCESS`: Valid model forward pass yielding continuous probabilities and discrete label (`REAL` or `FAKE`).
2. `REJECTED`: Verified rejection of an intentional control fixture (e.g. corrupted file or text document).
3. `ERROR`: Unhandled operational exception, I/O failure, or out-of-memory crash.

### 8.3 Enumeration Requirements
The baseline evaluation must explicitly enumerate and list all failure cases by `sample_id`:
1. **False Positives (FP) List**: Real samples misclassified as `FAKE` (with sample ID, true label, predicted fake probability, and logit).
2. **False Negatives (FN) List**: Fake samples misclassified as `REAL` (with sample ID, true label, predicted fake probability, and logit).
3. **Execution Failures List**: Any sample that resulted in an unexpected `ERROR` status (excluding expected rejection controls).

---

## 9. Deliverables Specifications

The Phase 4 baseline evaluation must generate exactly three primary deliverables without overwriting previous diagnostic results (`all_detector_predictions.*` and `detector_comparison.*`):

### 9.1 Deliverable 1: `benchmark/results/phase4_baseline_predictions.csv`
A standardized, long-format CSV recording every prediction across all 105 held-out test samples and all 3 detectors:
- **Total Rows**: Exactly 315 rows (105 samples $\times$ 3 detectors) + 1 header row.
- **Sorting**: Deterministic ordering by `detector_key` then `sample_id`.
- **Columns (18 Standardized Columns)**:

| # | Column Header | Data Type | Nullable? | Description & Example |
| :-: | :--- | :--- | :-: | :--- |
| 1 | `sample_id` | `string` | No | Manifest sample identifier (e.g. `test_face_real_001`) |
| 2 | `split` | `string` | No | Constant: `held_out_test` |
| 3 | `domain` | `string` | No | Visual domain (`facial`, `general_scene`, `control`) |
| 4 | `category` | `string` | No | Manifest category (`authentic_face`, `gan_face`, etc.) |
| 5 | `ground_truth` | `string` | No | Authoritative label (`REAL`, `FAKE`, `REJECTION_CONTROL`) |
| 6 | `file_path` | `string` | No | Absolute file path on disk |
| 7 | `detector_key` | `string` | No | Registry key (`chen2024_convnext`, `univfd`, `vit`) |
| 8 | `detector_name` | `string` | No | Full detector name (`UniversalFakeDetect`, etc.) |
| 9 | `model_name` | `string` | No | Specific model descriptor string |
| 10 | `prediction` | `string` | No | Discrete verdict (`REAL`, `FAKE`, `ERROR`) |
| 11 | `fake_probability`| `float` | Yes | Model probability of FAKE in $[0.0, 1.0]$ (empty on ERROR) |
| 12 | `real_probability`| `float` | Yes | Model probability of REAL in $[0.0, 1.0]$ (empty on ERROR) |
| 13 | `raw_logit` | `float` | Yes | Pre-activation logit score (empty on ERROR) |
| 14 | `confidence` | `float` | Yes | Model confidence score in $[0.0, 1.0]$ (empty on ERROR) |
| 15 | `threshold` | `float` | No | Decision threshold (constant: `0.50`) |
| 16 | `status` | `string` | No | Execution status (`SUCCESS`, `REJECTED`, `ERROR`) |
| 17 | `error_message` | `string` | Yes | Captured exception string if error; empty string otherwise |
| 18 | `latency_ms` | `float` | Yes | Wall-clock inference latency in ms (empty on ERROR) |

### 9.2 Deliverable 2: `benchmark/results/phase4_baseline_metrics.json`
A machine-readable JSON document structuring complete benchmark metrics by detector and evaluation domain.

#### Standard JSON Hierarchy:
```json
{
  "metadata": {
    "timestamp": "2026-10-30T...",
    "manifest_path": "benchmark/evaluation_manifest.csv",
    "evaluated_split": "held_out_test",
    "total_samples_evaluated": 105,
    "eligible_classification_samples": 103,
    "facial_samples": 100,
    "scene_diagnostic_samples": 3,
    "rejection_control_samples": 2,
    "environment": {
      "os": "Windows",
      "python_version": "3.13...",
      "torch_version": "2.13...",
      "device": "cpu"
    }
  },
  "detectors": {
    "<detector_key>": {
      "model_info": {
        "detector_key": "chen2024_convnext | univfd | vit",
        "detector_name": "...",
        "model_name": "...",
        "architecture": "...",
        "decision_threshold": 0.50,
        "input_dimensions": "3x224x224",
        "preprocessing": "..."
      },
      "overall_classification_metrics": {
        "sample_counts": {
          "total_eligible": 103,
          "ground_truth_real": 51,
          "ground_truth_fake": 52,
          "valid_predictions": 103,
          "errors": 0
        },
        "confusion_matrix": { "tp": 0, "fp": 0, "tn": 0, "fn": 0 },
        "accuracy": 0.0,
        "accuracy_fraction": "X/103",
        "precision": 0.0,
        "precision_fraction": "X/X",
        "recall": 0.0,
        "recall_fraction": "X/52",
        "specificity": 0.0,
        "specificity_fraction": "X/51",
        "f1_score": 0.0,
        "balanced_accuracy": 0.0,
        "false_positives": [ "sample_id", ... ],
        "false_negatives": [ "sample_id", ... ],
        "execution_errors": []
      },
      "facial_evaluation_metrics": {
        "sample_counts": {
          "total_facial": 100,
          "ground_truth_real": 50,
          "ground_truth_fake": 50
        },
        "confusion_matrix": { "tp": 0, "fp": 0, "tn": 0, "fn": 0 },
        "accuracy": 0.0,
        "accuracy_fraction": "X/100",
        "accuracy_95ci": [ 0.0, 0.0 ],
        "precision": 0.0,
        "precision_fraction": "X/X",
        "precision_95ci": [ 0.0, 0.0 ],
        "recall": 0.0,
        "recall_fraction": "X/50",
        "recall_95ci": [ 0.0, 0.0 ],
        "specificity": 0.0,
        "specificity_fraction": "X/50",
        "specificity_95ci": [ 0.0, 0.0 ],
        "f1_score": 0.0,
        "balanced_accuracy": 0.0,
        "false_positives": [],
        "false_negatives": [],
        "mean_fake_prob_real_faces": 0.0,
        "mean_fake_prob_fake_faces": 0.0
      },
      "scene_diagnostic_results": {
        "test_scene_real_001": {
          "ground_truth": "REAL",
          "prediction": "REAL",
          "fake_probability": 0.0,
          "raw_logit": 0.0,
          "latency_ms": 0.0,
          "correct": true
        },
        "test_scene_diffusion_001": { ... },
        "test_scene_manip_001": { ... }
      },
      "rejection_control_results": {
        "test_control_corr_001": {
          "status": "REJECTED",
          "error_message": "...",
          "rejection_successful": true
        },
        "test_control_unsupp_001": { ... }
      },
      "latency_summary_ms": {
        "mean": 0.0,
        "median": 0.0,
        "min": 0.0,
        "max": 0.0
      }
    }
  },
  "pairwise_agreement": {
    "chen2024_convnext_vs_univfd": {
      "valid_sample_count": 103,
      "concordant_samples": 0,
      "discordant_samples": 0,
      "concordance_rate": 0.0,
      "concordance_fraction": "X/103",
      "agreement_breakdown": {
        "both_fake": 0,
        "both_real": 0,
        "model1_fake_model2_real": 0,
        "model1_real_model2_fake": 0
      },
      "cohen_kappa": 0.0
    },
    "chen2024_convnext_vs_vit": { ... },
    "univfd_vs_vit": { ... }
  }
}
```

### 9.3 Deliverable 3: `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`
A publication-quality, exhaustive markdown evaluation report.

#### Mandatory Report Sections:
1. **Executive Summary**: Core findings, baseline performance overview, key takeaways before fusion.
2. **Evaluation Framework & Integrity Protection**: Manifest provenance, split decoupling confirmation (zero leakage from `development_val` or `operational_smoke`), verification tool results.
3. **System Environment & Execution Protocols**: Hardware specifications (RAM, VRAM, CPU/GPU), memory-safe sequential lifecycle details, run timestamps, total execution latency.
4. **Detector Architectural Profiles**: Detailed summary of ConvNeXt-Base, UnivFD, and ViT (checkpoints, sizes, parameters, preprocessors, label mappings, thresholds).
5. **Primary Classification Performance (103 Eligible Samples)**:
   - Confusion matrix table (TP, FP, TN, FN).
   - Core metrics table (Accuracy, Precision, Recall, Specificity, F1, Balanced Accuracy) with explicit sample counts.
   - Comparative cross-model analysis.
6. **Facial Domain Deep Dive (100 Samples)**:
   - Performance on 50 FFHQ Real vs 50 StyleGAN Fake.
   - 95% Confidence Intervals table (Wilson score / bootstrap).
   - Analysis of logit distributions and probability margins.
   - Domain limitations and generator specificity caveats.
7. **Diagnostic Case Studies: Scenes & Composites (3 Samples)**:
   - Case 1: Authentic Phone Photograph (`test_scene_real_001`).
   - Case 2: Google Imagen 3 Latent Diffusion Scene (`test_scene_diffusion_001`).
   - Case 3: Local Digital Splicing Composite (`test_scene_manip_001`).
   - Cross-model continuous probability comparison.
8. **Rejection Controls & Robustness Analysis (2 Samples)**:
   - Evaluation of malformed headers and plain text rejection.
   - Verification of graceful error reporting (`prediction == "ERROR"`).
9. **Pairwise Model Agreement & Concordance**:
   - 3x3 Pairwise concordance matrix.
   - Concordance breakdowns ($N_{FF}, N_{RR}, N_{FR}, N_{RF}$) and Cohen's Kappa.
   - Model independence disclaimer (Agreement $\ne$ Authenticity).
10. **Failure Analysis & Error Catalog**:
    - Enumeration of False Positives by sample ID and model.
    - Enumeration of False Negatives by sample ID and model.
    - Enumeration of any runtime errors.
11. **Strategic Recommendations for Phase 5 (Decision Fusion)**:
    - Domain disconnect implications.
    - Why naive linear averaging fails on cross-domain imagery.
    - Architecture for domain-gated routing (Facial path vs General Scene path).
12. **Reproducibility & Verification Log**: Exact commands and hash verification instructions.

---

## 10. Architectural Implementation Strategy for Evaluation Runner

### 10.1 Analysis of Existing Runner (`scripts/run_detector_benchmark.py`)
Requirement R1 instructs:
> *"If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner."*

Analysis of `scripts/run_detector_benchmark.py` reveals:
1. **Target File Hardcoded**: It explicitly loads `benchmark/metadata.json` (lines 49, 315) which only defines 5 diagnostic samples (`real_001` through `unsupp_001`).
2. **Relative Path Assumption**: It constructs file paths as `full_path = os.path.join(IMAGES_DIR, rel_path)`, assuming all images live inside `benchmark/images/`. In contrast, manifest entries in `benchmark/evaluation_manifest.csv` point to absolute paths in `Dataset/Test/Real` and `Dataset/Test/Fake`.
3. **Missing Split Logic**: It does not parse splits (`held_out_test`, `development_val`, `operational_smoke`).
4. **Output Collision**: It exports to `benchmark/results/all_detector_predictions.*` and `detector_comparison.*`, which would overwrite Phase 2/3 diagnostic benchmark results.

### 10.2 Recommendation: Dedicated Evaluation Script
Implement a clean, self-contained evaluation script:
`scripts/run_phase4_baseline_evaluation.py`

#### Key Architectural Requirements of the Script:
1. Reads `benchmark/evaluation_manifest.csv` and filters strictly for `split == "held_out_test"` (105 samples).
2. Uses sequential model execution with explicit memory deallocation:
   - Mode A (In-process):
     * `adapter = get_detector(det_name, lazy_load=False, singleton=False)`
     * Predict over all 105 samples
     * `adapter.unload_model()`, `del adapter`, `reset_instances()`, `gc.collect()`, `torch.cuda.empty_cache()`
   - Mode B (Subprocess, optional `--mode subprocess`): Evaluates each detector in an isolated Python subprocess for 100% OS-level memory reclamation.
3. Automatically computes all classification metrics, Wilson score confidence intervals, diagnostic dictionaries, pairwise concordance, and error catalogs.
4. Directly outputs the three required deliverables:
   - `benchmark/results/phase4_baseline_predictions.csv`
   - `benchmark/results/phase4_baseline_metrics.json`
   - `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` (or outputs metrics for report generation).
5. Exits with code 0 on success.

---

## 11. Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Manifest Partitioning | Held-Out Test Split | 105 pristine benchmark samples strictly isolated from all previous training and tuning runs | Manifest rows where `split == "held_out_test"` | 105 manifest row dictionaries | Fail if sample count $\ne 105$ | `benchmark/evaluation_manifest.csv`, `docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md` §4.1 |
| 2 | Manifest Partitioning | Development / Validation Split | 50 pristine facial samples (25 real, 25 StyleGAN) reserved for Phase 5 fusion tuning | Manifest rows where `split == "development_val"` | 50 manifest row dictionaries | Must be excluded from Phase 4 metrics | `benchmark/evaluation_manifest.csv`, `docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md` §4.1 |
| 3 | Manifest Partitioning | Operational Smoke Split | 4 test fixture samples reserved for CI contract testing | Manifest rows where `split == "operational_smoke"` | 4 manifest row dictionaries | Must be excluded from Phase 4 metrics | `benchmark/evaluation_manifest.csv`, `docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md` §4.1 |
| 4 | Detector Contract | Standardized Result Schema | Dataclass defining contract for all detector outputs | Class initialization kwargs | `DetectorResult` dataclass instance | Raises `ValueError` if prediction not in `VALID_PREDICTIONS` | `src/detector_schema.py` |
| 5 | Memory Management | Explicit Adapter Unload | Release weights and buffers without process termination | Call `adapter.unload_model()` | `model = None`, VRAM cleared | Graceful warning on exception | `src/convnext_adapter.py`, `src/univfd_adapter.py`, `src/vit_adapter.py`, `scripts/MEMORY_SAFE_BENCHMARK_REPORT.md` |
| 6 | Memory Management | Sequential Detector Loop | Evaluate detectors one by one to avoid concurrent memory exhaustion (`os error 1455`) | Detector registry list | Per-detector sequential execution | Intercepts errors per detector; continues loop | `scripts/MEMORY_SAFE_BENCHMARK_REPORT.md` §3 |
| 7 | Robustness | Rejection Controls Handling | Intercept non-image and corrupted files at ingress | Malformed PNG (`corrupted_image.png`) or ASCII TXT (`unsupported_doc.txt`) | `DetectorResult(prediction="ERROR")` | Gracefully returns ERROR, never crashes | `src/univfd_adapter.py`, `src/vit_adapter.py`, `src/convnext_adapter.py` |
| 8 | Evaluation Metrics | Exact Sample Counts Requirement | All percentages accompanied by exact integer counts | Integer TP, FP, TN, FN | Formatted string (e.g. `98.00% (49/50)`) | Avoids ambiguity in small-sample reporting | `ORIGINAL_REQUEST.md` R3, AC |
| 9 | Statistical Rigor | Wilson Score Confidence Interval | Asymmetric confidence interval for binomial proportions | $\hat{p}, n, z=1.96$ | Tuple $[L, U]$ bounded in $[0, 1]$ | Degenerate-free at $\hat{p} \in \{0, 1\}$ | `ORIGINAL_REQUEST.md` R3, statistical methodology |
| 10 | Model Comparison | Valid-Prediction Pairwise Agreement | Measure concordance strictly over samples where both models returned valid classifications | Model A and Model B prediction vectors | Concordance rate $P_o$, 2x2 matrix, Cohen's Kappa $\kappa$ | Excludes `ERROR` cases from denominator | `ORIGINAL_REQUEST.md` R3, AC |
| 11 | Diagnostic Reporting | Single-Example Scene Case Studies | In-depth qualitative case studies for $N=1$ scene/composite samples | 3 scene/composite samples | Structured case-study analysis table | Prohibits statistical extrapolation from $N=1$ | `ORIGINAL_REQUEST.md` R3, AC |
| 12 | Fault Isolation | Failure Retention Policy | Retain execution errors rather than silently dropping rows | Manifest sample with exception | Record with `status="ERROR"`, `error_message` | Logs error details, preserves row count | `ORIGINAL_REQUEST.md` R3 |
| 13 | Output Serialization | Phase 4 Predictions CSV | 18-column long-format CSV recording 315 predictions | 105 samples $\times$ 3 detectors | `benchmark/results/phase4_baseline_predictions.csv` | Fail if duplicate or missing rows | `ORIGINAL_REQUEST.md` R4 |
| 14 | Output Serialization | Phase 4 Metrics JSON | Structured JSON capturing metrics across all domains | Evaluated baseline outputs | `benchmark/results/phase4_baseline_metrics.json` | Valid schema with all required keys | `ORIGINAL_REQUEST.md` R4 |
| 15 | Documentation | Phase 4 Baseline Evaluation Report | Comprehensive markdown report documenting baseline | All metrics, tables, and analysis | `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` | Complete coverage of mandatory sections | `ORIGINAL_REQUEST.md` R4 |

---

## 12. Edge Cases Discovered

| # | Feature / Context | Input / Edge Condition | Observed Behavior / Required Handling |
|---|-------------------|------------------------|----------------------------------------|
| 1 | Rejection Controls | Corrupted file with truncated 20-byte header (`corrupted_image.png`) | PIL raises `UnidentifiedImageError`; adapter catches exception and returns `prediction: "ERROR"`. Handled as successful rejection (`status: "REJECTED"`). Must NOT count towards classification metrics. |
| 2 | Rejection Controls | Plain ASCII text document (`unsupported_doc.txt`) | PIL raises `UnidentifiedImageError`; adapter catches exception and returns `prediction: "ERROR"`. Handled as successful rejection (`status: "REJECTED"`). Must NOT count towards classification metrics. |
| 3 | Pairwise Agreement | One or both detectors return `prediction == "ERROR"` for a sample | The sample is excluded from pairwise agreement calculation. $N_{\text{valid\_both}}$ includes only samples where $\text{pred}_A, \text{pred}_B \in \{\text{"REAL"}, \text{"FAKE"}\}$. |
| 4 | Metric Calculation | Zero True Positives ($TP = 0$) or Zero Flagged Fakes ($TP + FP = 0$) | Precision formula denominator is 0; must be defined as $0.0$ to avoid `ZeroDivisionError`. |
| 5 | Metric Calculation | Perfect Accuracy on Sub-Benchmark ($\hat{p} = 1.0$, e.g. 50/50 fake recall) | Normal Wald interval collapses to $[1.0, 1.0]$ with zero width; Wilson score interval properly computes asymmetric bound $[0.9289, 1.0000]$. |
| 6 | General Scene Diffusion | Google Imagen 3 diffusion scene evaluated by UnivFD | UnivFD outputs $P(\text{fake}) = 0.0828\%$ (logit $-7.096$), misclassifying modern diffusion as `REAL`. Captured as diagnostic case study highlighting generator cutoff limitation. |
| 7 | General Scene Composite | Localized spatial splicing evaluated by whole-image backbones | All detectors predict `REAL` because global composition appears coherent. Captured as diagnostic case study highlighting architectural need for localization heads. |
| 8 | Windows Memory Heap | Sequential in-process execution across hundreds of samples | MSVCRT heap does not immediately release all memory to OS kernel. Sequential lifecycle deallocation (`unload_model()`, `gc.collect()`, `torch.cuda.empty_cache()`) maintains peak RSS well below the 2 GB threshold. Subprocess mode available as fallback if needed. |
| 9 | Ground-Truth Column Naming | Different column names across historical files (`ground_truth` vs `label` vs `explicit_label`) | The authoritative manifest uses `ground_truth`. The runner must prioritize `ground_truth` from the manifest. |
| 10 | Model Label Indexing | Different label mappings across adapters | `UnivFD`: Index 0=REAL, 1=FAKE. `ViT`: dynamic `id2label` (`0=real`, `1=fake`). `ConvNeXt`: Index 0=REAL, 1=FAKE. All three map threshold $\ge 0.50$ fake probability to `FAKE`. Normalized by adapter contract. |
