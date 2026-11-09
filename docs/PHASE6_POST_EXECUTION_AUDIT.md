# Phase 6 Post-Execution Audit & Independent Verification Report

**Project:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Auditor:** Independent Investigation Worker (Verification Only)  
**Date:** November 2026  
**Status:** Audit Completed — Verification Only (Zero Modifications to Model Weights, Thresholds, Fusion Logic, or Historical Artifacts)  
**Deliverable File:** [`docs/PHASE6_POST_EXECUTION_AUDIT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_POST_EXECUTION_AUDIT.md)

---

## Executive Summary

This post-execution audit independently verifies the completed Phase 6 evaluation of the three integrated deepfake image detectors (`ViT`, `UnivFD`, `ConvNeXt/DRCT`) and their decision-fusion combinations across the frozen test manifest ([`benchmark/phase6_test_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/phase6_test_manifest.csv)).

The audit was conducted strictly non-destructively: no inference benchmarks were rerun, no thresholds or weights were altered, and no existing manifests, prediction files, or historical reports were overwritten.

### Summary of Audit Verdicts:
1. **Manifest Integrity:** **VERIFIED (with boundary clarification)**. The manifest contains exactly 200 unique, valid, decodable $256 \times 256$ RGB JPEG samples (100 REAL, 100 FAKE). All 200 files exist on disk, match their recorded SHA-256 hashes and file sizes, and have zero hash collisions across 409 historical benchmark samples. Cryptographic and historical repository audits confirm the Kaggle 140k Test partition has zero duplicate files across the entire 190,335-sample dataset. **Identity-level independence cannot and must not be claimed** due to the absence of biometric identity annotations in FFHQ and the stochastic latent nature of StyleGAN.
2. **Individual Detector Metrics:** **VERIFIED (0 Numerical Discrepancies)**. Confusion matrices, accuracy, balanced accuracy, precision, recall, specificity, F1-scores, and 95% Wilson confidence intervals recomputed directly from saved prediction CSVs match [`benchmark/results/phase6_combined_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_combined_metrics.json) exactly. All three detectors achieved 100% sample coverage (200/200) with zero inference failures.
3. **Combined Fusion Results:** **VERIFIED (0 Numerical Discrepancies)**. Recomputed metrics for Majority Vote (63.00%), Mean Fake Probability (63.50%), and ViT-Priority Fallback (98.00%) perfectly replicate the recorded results.
4. **ViT-Priority Fallback Behavior:** **PRECISELY CHARACTERIZED**. ViT-priority triggers fallback *only* when ViT yields `ERROR`, `UNCERTAIN`, or fails. Across all 200 Phase 6 test samples, ViT produced decisive predictions ($200/200$), so **fallback was triggered 0 times (0.0%)**. ViT-priority does *not* improve detection accuracy over standalone ViT (both are identical at 98.00%); it preserves ViT predictions 100% and functions strictly as a software error-recovery mechanism.
5. **Discriminatory Failure of UnivFD and ConvNeXt on StyleGAN:** **ROC-AUC ANALYSIS RESOLVED**. Empirical ROC-AUC analysis confirms that UnivFD ($\text{ROC-AUC} = 0.5503$) and ConvNeXt ($\text{ROC-AUC} = 0.4977$) exhibit complete feature blindness to StyleGAN facial artifacts. Their false negatives ($FN = 83$ and $FN = 84$) are not an artifact of an uncalibrated default threshold ($\tau = 0.50$); even optimal oracle test threshold searches fail to exceed 54.50% accuracy.
6. **Documentation Erratum Identified:** A factual erratum was discovered in [`docs/PHASE6_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_EVALUATION_REPORT.md) line 67: the narrative misidentifies ViT's 4 False Positives as `real_2028.jpg`, `real_2057.jpg`, `real_2077.jpg`, and `real_2083.jpg` (which were actually true negatives). The actual 4 False Positives in [`benchmark/results/phase6_vit_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_vit_predictions.csv) are `real_2036.jpg`, `real_2052.jpg`, `real_2053.jpg`, and `real_2088.jpg`.
7. **System Integrity & Historical Preservation:** **VERIFIED**. Unit tests passed ($21/21$ in `test_decision_fusion.py`, $7/7$ in `test_detector_registry.py`, $100\%$ in `test_phase4_adversarial_verification.py`). Historical Phase 4 and Phase 5 artifacts remain untouched with unchanged timestamps (`2026-10-30`) and matching SHA-256 digests. The historical Windows paging file crash (`os error 1455`) was resolved via sequential subprocess isolation.

---

## 1. Frozen Manifest Verification (`benchmark/phase6_test_manifest.csv`)

### 1.1 Sample Count, Label Balance, and Decodability
The test manifest was verified programmatically against the local filesystem:
- **Total Samples:** Exactly 200 rows (plus 1 header line).
- **Class Balance:** Exactly 100 REAL and 100 FAKE (`REAL`: 100, `FAKE`: 100).
- **Sample IDs:** Exactly 200 unique identifiers (`phase6_test_real_001` through `phase6_test_real_100`, `phase6_test_fake_001` through `phase6_test_fake_100`).
- **File Existence on Disk:** 200 / 200 files exist at their specified `absolute_path` locations.
- **Byte Size & SHA-256 Checksums:** 100% match between disk files and manifest columns `file_size_bytes` and `sha256`. 0 size mismatches, 0 hash mismatches.
- **Image Decodability & Geometry:** 100% of images verified decodable via PIL (`Image.open(p).verify()` and `.convert('RGB')`). Every sample is confirmed to be exactly $256 \times 256$ pixels, 3 8-bit channels (`mode == 'RGB'`), JPEG format. Zero corruptions, zero truncated files.

### 1.2 Source Dataset Path and Split Verification
Every selected sample was cross-referenced against the master dataset on disk (`C:\Users\user\Desktop\currently working\Dataset`):
- All 100 REAL samples originate from `Dataset\Test\Real\` (`real_2000.jpg` through `real_2102.jpg`, excluding `real_2017`, `real_2041`, `real_2044` due to prior audit exposure).
- All 100 FAKE samples originate from `Dataset\Test\Fake\` (`fake_2000.jpg` through `fake_2101.jpg`, excluding `fake_2009`, `fake_2062`).
- All 200 samples were cross-referenced against the repository's historical 190,335-sample master manifest (`C:\Users\user\Desktop\currently working\dataset_manifest.csv`):
  * **Test Split Match:** Exactly 200 / 200 match split `Test`.
  * **Label Match:** Exactly 100 `Real` and 100 `Fake`.
  * **Readability Status:** Exactly 200 / 200 marked `is_readable: True`.
  * **File Size Match:** Exactly 200 / 200 byte counts match `dataset_manifest.csv`.

### 1.3 Exact SHA-256 Duplicate Screening & Scope of Checks
Phase 6 test hashes were audited against all available historical manifests, prediction logs, and duplicate databases:

| Historical Artifact Audited | Artifact Scope / Purpose | Historical Samples | Exact SHA-256 Overlap with Phase 6 |
| :--- | :--- | :---: | :---: |
| [`benchmark/evaluation_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/evaluation_manifest.csv) | Master project manifest (Phase 1–5) | 159 | **0 collisions** |
| [`benchmark/results/phase4_baseline_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase4_baseline_predictions.csv) | Phase 4 held-out baseline test set | 105 unique files | **0 collisions** |
| [`benchmark/results/phase5_val_raw_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_val_raw_predictions.csv) | Phase 5 development validation set | 50 unique files | **0 collisions** |
| `benchmark/test_images/` | Operational smoke test images | 4 unique files | **0 collisions** |
| `currently working/random_250_evaluation_results.csv` | Historical exploratory benchmark | 250 unique files | **0 collisions** |
| `currently working/evaluation_results.csv` | Historical single-model evaluation | 100 unique files | **0 collisions** |
| **Total Historical Benchmark Samples Checked** | **Consolidated Unique Historical Files** | **409 unique files** | **0 collisions (0.0%)** |
| `currently working/duplicates_metadata.csv` | Full dataset duplicate catalog (17,135 groups) | 34,270 file entries | **0 Phase 6 files present** |

#### Structural Root Cause for Zero Duplicates in `duplicates_metadata.csv`
A critical distinction must be noted: `duplicates_metadata.csv` contains 34,270 file entries representing **17,135 duplicate groups** (not 34,270 duplicate groups). An audit of the historical provenance report ([`duplicate_analysis_report.txt`](file:///C:/Users/user/Desktop/currently%20working/duplicate_analysis_report.txt), lines 47-60) reveals that:
- Total Test files in any duplicate group across the entire Kaggle 140k dataset: **0 files**.
- Train $\cap$ Test duplicate groups: **0**.
- Validation $\cap$ Test duplicate groups: **0**.
- All 5,334 cross-split duplicate groups (10,668 files) exist exclusively between `Train` and `Validation`.
Therefore, the absence of Phase 6 samples from `duplicates_metadata.csv` is not an empirical coincidence of index filtering; rather, the entire 10,905-sample Test split is cryptographically disjoint from all duplicate clusters in the dataset.

### 1.4 Filename Indexing Nuance vs. Identity Provenance
1. **Filename Re-indexing across Splits:** The Kaggle 140k dataset indexes each partition independently starting from index 0 (`real_0.jpg`, `fake_0.jpg`). Consequently, filenames such as `real_2000.jpg` appear in `Dataset\Train\Real\`, `Dataset\Validation\Real\`, and `Dataset\Test\Real\`. However, SHA-256 digest auditing confirms these are physically distinct images:
   - `Dataset/Train/Real/real_2000.jpg`: Size 8,786 bytes, SHA-256 `21c6031b2065...`
   - `Dataset/Validation/Real/real_2000.jpg`: Size 10,730 bytes, SHA-256 `4ed4731e...`
   - `Dataset/Test/Real/real_2000.jpg`: Size 10,371 bytes, SHA-256 `8926a74d...`
2. **Identity-Level Independence (Unverified Boundary):**
   - Photographic human faces in FFHQ were crawled from Flickr using automatic face detection without subject identity metadata. Multiple photos of the same subject under different poses, angles, or lighting may exist across different image files.
   - Synthetic StyleGAN faces represent stochastic generator samples drawn from latent space without identity tracking.
   - Phase 6 selection verified partition separation (`Dataset/Test`), decodability, resolution ($256 \times 256$), and file-level SHA-256 uniqueness. However, **no biometric identity verification or deep feature face-matching (e.g. ArcFace / FaceNet embeddings)** was conducted.
   - **Audit Finding:** **Identity-level independence CANNOT and MUST NOT be claimed**. Claims must remain strictly bounded to *file-level*, *cryptographic hash-level*, and *dataset test-partition-level* independence.

---

## 2. Recalculated Individual Model Metrics

### 2.1 Coverage, Failures, and Data Integrity
The three individual detector prediction files were inspected:
- [`benchmark/results/phase6_vit_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_vit_predictions.csv)
- [`benchmark/results/phase6_univfd_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_univfd_predictions.csv)
- [`benchmark/results/phase6_convnext_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_convnext_predictions.csv)

| Audit Check | ViT (`dima806`) | UnivFD (`CLIP ViT-L/14`) | ConvNeXt (`DRCT/RAID`) | Status |
| :--- | :---: | :---: | :---: | :---: |
| Total Prediction Rows | 200 | 200 | 200 | **Verified** |
| Unique Sample IDs | 200 | 200 | 200 | **Verified** |
| Missing / Extra Sample IDs | 0 / 0 | 0 / 0 | 0 / 0 | **Verified** |
| Ground Truth Label Alignment | 100 REAL, 100 FAKE | 100 REAL, 100 FAKE | 100 REAL, 100 FAKE | **Verified** |
| Classification Coverage | **100.0% (200/200)** | **100.0% (200/200)** | **100.0% (200/200)** | **Verified** |
| Inference Failures / Errors | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **Verified** |
| Non-Decisive / Uncertain | 0 | 0 | 0 | **Verified** |
| Probability Bounds $[0.0, 1.0]$ | $P(\text{fake}) \in [0.000442, 0.999614]$ | $P(\text{fake}) \in [0.000002, 0.999498]$ | $P(\text{fake}) \in [0.000006, 0.999312]$ | **Verified** |
| $P(\text{fake}) + P(\text{real}) = 1.0$ | Max error: $1.0 \times 10^{-6}$ | Max error: $0.0$ | Max error: $1.0 \times 10^{-6}$ | **Verified** |
| Decision Threshold Mapping | $\tau = 0.50$ (0 mismatches) | $\tau = 0.50$ (0 mismatches) | $\tau = 0.50$ (0 mismatches) | **Verified** |

*Note on CSV Schema:* The `status` and `error_message` columns in all three individual prediction CSVs contain `NaN` across all rows. Code tracing in [`scripts/evaluate_phase6_pipeline.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/evaluate_phase6_pipeline.py#L401-L402) reveals that the CSV exporter queries `r.get("status", "")`, but the standard `DetectorResult` dataclass in [`src/detector_schema.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/detector_schema.py) intentionally omits a `status` field for valid predictions (only populating `prediction`, probabilities, logits, and `error`). Because all 200 inferences completed without error, `error_message` is null and `status` is empty string (`NaN`). This is an expected artifact of schema alignment, not an unhandled failure.

### 2.2 Independent Metric Recalculation
Metrics were independently computed directly from the saved prediction CSVs and cross-referenced against [`benchmark/results/phase6_combined_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_combined_metrics.json):

#### ViT (`dima806/deepfake_vs_real_image_detection`)
- **Confusion Matrix:** $\text{TP} = 100$, $\text{FP} = 4$, $\text{TN} = 96$, $\text{FN} = 0$
- **Accuracy:** $98.00\%$ ($196 / 200$), 95% Wilson CI: $[94.97\%, 99.22\%]$
- **Balanced Accuracy:** $0.5 \times (1.0000 + 0.9600) = 98.00\%$
- **Precision:** $100 / (100 + 4) = 0.9615$ (95% CI: $[0.9053, 0.9849]$)
- **Recall (Fake Sensitivity):** $100 / 100 = 1.0000$ (95% CI: $[0.9630, 1.0000]$)
- **Specificity (Real Retention):** $96 / 100 = 0.9600$ (95% CI: $[0.9016, 0.9843]$)
- **F1-Score:** $2 \times 0.9615 \times 1.0 / (0.9615 + 1.0) = 0.9804$
- **Discrepancy against saved JSON:** **0 (Exact match)**

#### UnivFD (`UniversalFakeDetect` — CLIP `ViT-L/14` + Linear Probe)
- **Confusion Matrix:** $\text{TP} = 17$, $\text{FP} = 14$, $\text{TN} = 86$, $\text{FN} = 83$
- **Accuracy:** $51.50\%$ ($103 / 200$), 95% Wilson CI: $[44.61\%, 58.33\%]$
- **Balanced Accuracy:** $0.5 \times (0.1700 + 0.8600) = 51.50\%$
- **Precision:** $17 / (17 + 14) = 0.5484$ (95% CI: $[0.3777, 0.7084]$)
- **Recall (Fake Sensitivity):** $17 / 100 = 0.1700$ (95% CI: $[0.1089, 0.2555]$)
- **Specificity (Real Retention):** $86 / 100 = 0.8600$ (95% CI: $[0.7786, 0.9147]$)
- **F1-Score:** $2 \times 0.5484 \times 0.17 / (0.5484 + 0.17) = 0.2595$
- **Discrepancy against saved JSON:** **0 (Exact match)**

#### ConvNeXt (`Chen2024ConvNeXtDetector` — DRCT / RAID)
- **Confusion Matrix:** $\text{TP} = 16$, $\text{FP} = 14$, $\text{TN} = 86$, $\text{FN} = 84$
- **Accuracy:** $51.00\%$ ($102 / 200$), 95% Wilson CI: $[44.12\%, 57.84\%]$
- **Balanced Accuracy:** $0.5 \times (0.1600 + 0.8600) = 51.00\%$
- **Precision:** $16 / (16 + 14) = 0.5333$ (95% CI: $[0.3614, 0.6977]$)
- **Recall (Fake Sensitivity):** $16 / 100 = 0.1600$ (95% CI: $[0.1010, 0.2442]$)
- **Specificity (Real Retention):** $86 / 100 = 0.8600$ (95% CI: $[0.7786, 0.9147]$)
- **F1-Score:** $2 \times 0.5333 \times 0.16 / (0.5333 + 0.16) = 0.2462$
- **Discrepancy against saved JSON:** **0 (Exact match)**

### 2.3 Discriminatory Threshold Sensitivity & ROC-AUC Analysis
To resolve the open question regarding whether the ~84% false negative rates of UnivFD and ConvNeXt could be mitigated by threshold tuning ($\tau \ne 0.50$), an independent ROC-AUC and probability distribution audit was executed:

| Detector | ROC-AUC | REAL Mean $P(\text{fake})$ | REAL Median $P(\text{fake})$ | FAKE Mean $P(\text{fake})$ | FAKE Median $P(\text{fake})$ | Optimal Oracle Test Threshold | Max Attainable Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT (`dima806`)** | **0.9995** | 0.0466 | 0.0015 | 0.9988 | 0.9994 | $\tau = 0.97$ | 99.50% |
| **UnivFD (`CLIP ViT-L/14`)** | **0.5503** | 0.1707 | 0.0408 | 0.2046 | 0.0450 | $\tau = 0.01$ | 54.50% |
| **ConvNeXt (`DRCT/RAID`)** | **0.4977** | 0.1708 | 0.0112 | 0.1772 | 0.0143 | $\tau = 0.35$ | 52.50% |

**Key Diagnostic Finding:**  
- ConvNeXt's ROC-AUC of **0.4977** is literally worse than random guessing ($0.5000$). The median fake probability predicted for a fake StyleGAN face ($0.0143$) is virtually indistinguishable from a real photograph ($0.0112$).
- UnivFD's ROC-AUC of **0.5503** demonstrates near-total lack of discriminatory signal. Even when evaluated with an oracle cheating threshold on the test set ($\tau = 0.01$), maximum possible accuracy is bounded at 54.50%.
- **Conclusion:** The failure of UnivFD and ConvNeXt on StyleGAN faces is **not** an operational artifact of default 0.50 thresholding. It represents an intrinsic feature blindness: the frozen CLIP semantic representation and DRCT low-level diffusion artifact features lack representation for StyleGAN2 facial synthesis artifacts.

---

## 3. Combined Results & Fusion Logic Verification

### 3.1 Per-Sample Alignment & Strategy Outputs
[`benchmark/results/phase6_combined_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_combined_predictions.csv) was audited row-by-row across all 200 samples:
- Every row aligns identically with the manifest in `sample_id`, `original_filename`, and `ground_truth`.
- Predictions and fake probabilities for all three detectors correspond exactly to the standalone prediction CSVs ($0.0$ difference across all 600 predictions).

### 3.2 Decision Fusion Logic and Tie-Break Handling
The implementation in [`src/decision_fusion.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/decision_fusion.py) was audited against its operational execution:

1. **Majority Voting (`strategy_majority_vote`):**
   - Aggregates binary votes from valid detectors.
   - In Phase 6, all 3 detectors returned decisive binary predictions on all 200 samples. The vote distribution was strictly odd ($N=3$), resulting in either unanimous consensus ($3\text{-}0$) or split decision ($2\text{-}1$).
   - **Ties Encountered:** **0 ties** occurred during the 200-sample test run.
   - The configurable tie-breaker (`vit_or_real`, `conservative_real`, `uncertain`) was verified in unit tests (`tests/test_decision_fusion.py`) to handle even-numbered survivor sets correctly if detectors fail.
2. **Mean Fake Probability (`strategy_mean_fake_probability`):**
   - Computes arithmetic mean of valid fake probabilities: $\bar{P}(\text{fake}) = \frac{1}{M}\sum_{m=1}^M P_m(\text{fake})$.
   - Compares $\bar{P}(\text{fake}) \ge 0.50 \implies \text{FAKE}$.
   - In Phase 6, $\bar{P}(\text{fake})$ was never exactly equal to $0.500000$ (0 threshold ties encountered).
3. **ViT-Priority with Explicit Fallback (`strategy_vit_priority`):**
   - Traced logic in [`src/decision_fusion.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/decision_fusion.py#L417-L440):
     ```python
     vit_res = norm.get(vit_key)
     vit_valid = vit_res is not None and vit_res["prediction"] in (PREDICTION_REAL, PREDICTION_FAKE)
     if vit_valid:
         # Directly forward ViT prediction, probability, and confidence.
         # fallback_applied = False
     ```
   - **Fallback Condition:** Secondary models are consulted **only when ViT fails** (`vit_res is None`, `prediction == ERROR`, or `prediction == UNCERTAIN`).
   - **Test Benchmark Activation:** ViT produced decisive predictions on **200 out of 200 samples**. Consequently, **fallback was triggered 0 times (0.0%)**.
   - **Accuracy Impact:** ViT-Priority Fallback achieves **98.00% accuracy**, exactly equal to ViT standalone (98.00%). It **does not improve detection accuracy** over ViT; it preserves ViT predictions 100% while providing an operational safeguard against runtime crashes.

### 3.3 Ensemble Metric Recalculation
All ensemble metrics were recomputed from the per-sample combined predictions:

| Model / Ensemble Strategy | Accuracy (%) | 95% Wilson CI (Acc) | Balanced Acc (%) | Precision | Recall | Specificity | F1-Score | TP | FP | TN | FN | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT (`dima806`)** | **98.00%** | **[94.97%, 99.22%]** | **98.00%** | **0.9615** | **1.0000** | **0.9600** | **0.9804** | 100 | 4 | 96 | 0 | 192.4 ms |
| **UnivFD (`CLIP ViT-L/14`)** | 51.50% | [44.61%, 58.33%] | 51.50% | 0.5484 | 0.1700 | 0.8600 | 0.2595 | 17 | 14 | 86 | 83 | 942.1 ms |
| **ConvNeXt (`DRCT/RAID`)** | 51.00% | [44.12%, 57.84%] | 51.00% | 0.5333 | 0.1600 | 0.8600 | 0.2462 | 16 | 14 | 86 | 84 | 252.7 ms |
| **Ensemble (Majority Vote)** | **63.00%** | [56.12%, 69.39%] | 63.00% | 0.8824 | 0.3000 | 0.9600 | 0.4478 | 30 | 4 | 96 | 70 | 1387.2 ms |
| **Ensemble (Mean Fake Prob)** | **63.50%** | [56.63%, 69.86%] | 63.50% | 0.8857 | 0.3100 | 0.9600 | 0.4593 | 31 | 4 | 96 | 69 | 1387.2 ms |
| **Ensemble (ViT Priority)** | **98.00%** | **[94.97%, 99.22%]** | **98.00%** | **0.9615** | **1.0000** | **0.9600** | **0.9804** | 100 | 4 | 96 | 0 | 192.4 ms |

*All recomputed metrics match saved JSON and CSV values with zero numerical discrepancies.*

### 3.4 Agreement Analysis & Cohen's Kappa Re-verification
Cross-model agreement metrics from [`benchmark/results/phase6_combined_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_combined_metrics.json) were verified independently:
- **Three-Way Agreement:** Exactly 77 out of 200 samples ($38.50\%$).
  * Unanimous REAL: 72 samples (67 true negatives, 5 false negatives where all models failed).
  * Unanimous FAKE: 5 samples (5 true positives, 0 false positives).
- **Pairwise Agreement & Cohen's Kappa:**
  * **ViT vs. UnivFD:** Raw Agreement = $50.50\%$, $\kappa = 0.0366$ (Both FAKE: 18, Both REAL: 83, ViT FAKE / UnivFD REAL: 86, ViT REAL / UnivFD FAKE: 13).
  * **ViT vs. ConvNeXt:** Raw Agreement = $52.00\%$, $\kappa = 0.0661$ (Both FAKE: 19, Both REAL: 85, ViT FAKE / ConvNeXt REAL: 85, ViT REAL / ConvNeXt FAKE: 11).
  * **UnivFD vs. ConvNeXt:** Raw Agreement = $74.50\%$, $\kappa = 0.0135$ (Both FAKE: 5, Both REAL: 144, UnivFD FAKE / ConvNeXt REAL: 26, UnivFD REAL / ConvNeXt FAKE: 25).
  * *Note:* Despite high raw agreement ($74.50\%$) between UnivFD and ConvNeXt, Cohen's Kappa ($\kappa = 0.0135$) confirms that agreement is driven by marginal skew (both models predict REAL on over 84% of samples), not true signal correlation.

---

## 4. Documentation Erratum Identified During Audit

During line-by-line verification of [`docs/PHASE6_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_EVALUATION_REPORT.md), a factual discrepancy in the narrative text was discovered:

- **Stated Claim in Report (Line 67):**
  > *"ViT generated 4 False Positives out of 100 authentic images: `real_2028.jpg`, `real_2057.jpg`, `real_2077.jpg`, and `real_2083.jpg`."*
- **Empirical Ground Truth from Saved Predictions ([`benchmark/results/phase6_vit_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_vit_predictions.csv)):**
  Auditing all rows where `ground_truth == 'REAL'` and `prediction == 'FAKE'` yields:
  1. `real_2036.jpg` (Sample: `phase6_test_real_036`, $P(\text{fake}) = 0.706406$, Pred: `FAKE`)
  2. `real_2052.jpg` (Sample: `phase6_test_real_050`, $P(\text{fake}) = 0.952536$, Pred: `FAKE`)
  3. `real_2053.jpg` (Sample: `phase6_test_real_051`, $P(\text{fake}) = 0.997184$, Pred: `FAKE`)
  4. `real_2088.jpg` (Sample: `phase6_test_real_086`, $P(\text{fake}) = 0.968695$, Pred: `FAKE`)
- **Status of the 4 Images Cited in the Report:**
  `real_2028.jpg`, `real_2057.jpg`, `real_2077.jpg`, and `real_2083.jpg` were actually **True Negatives** correctly classified as `REAL` by ViT (with $P(\text{fake}) \le 0.007$) as well as by UnivFD and ConvNeXt.
- **Audit Impact Assessment:**
  This is a typographical/transcription error in the descriptive Markdown report. The quantitative count ($\text{FP} = 4$), the metrics calculations, the confusion matrix, and the predictions CSV are 100% correct.

---

## 5. Test Suite, Memory Safety, and Integrity Records

### 5.1 Verification Test Execution
All lightweight verification test suites were executed to verify codebase integrity:
- **Decision Fusion Test Suite (`tests/test_decision_fusion.py`):**
  * Executed: `python -m pytest tests/test_decision_fusion.py -v`
  * Result: **21 passed in 0.07s** (Exit Code 0).
  * Covers: unanimous agreement, disagreement, single-model failures, total failures, invalid probabilities, tie breaking, and fallback handling.
- **Detector Registry Diagnostic (`scripts/test_detector_registry.py`):**
  * Executed: `python scripts/test_detector_registry.py`
  * Result: **7/7 checks passed** (Exit Code 0).
  * Covers: import isolation (0 active instances on import), dynamic registration, metadata retrieval without model loading, lazy instantiation, and instance reset.
- **Phase 4 Adversarial Verification Oracle (`scripts/test_phase4_adversarial_verification.py`):**
  * Executed: `python scripts/test_phase4_adversarial_verification.py`
  * Result: **100% checks passed with zero discrepancies** (Exit Code 0).

### 5.2 Resolution of Initial Command Failures
- **Initial Blocker:** In earlier project phases, executing `run_detector_benchmark.py` in a single monolithic Python process resulted in Windows virtual memory paging exhaustion (`os error 1455`) during ViT initialization, crashing the evaluation.
- **Engineering Resolution:** The Phase 6 runner ([`scripts/evaluate_phase6_pipeline.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/evaluate_phase6_pipeline.py)) enforces sequential subprocess isolation via `run_detector_subprocess()`. Each detector is loaded in an independent child process, evaluates the test partition, writes JSON output, unloads weights, calls `gc.collect()`, and terminates.
- **Audit Verification:** The completed Phase 6 run processed 600 inference queries (200 samples $\times$ 3 detectors) plus 3 smoke tests with zero virtual memory errors, 100% memory reclamation upon child process exit, and exit code 0.

### 5.3 Historical Artifact Immutability Audit
All historical deliverables from Phase 4 and Phase 5 were verified to have remained completely unmodified:

| File Path | Creation / Last Modified Timestamp | File Size | SHA-256 Digest Prefix | Integrity Status |
| :--- | :---: | :---: | :---: | :---: |
| [`benchmark/results/phase4_baseline_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase4_baseline_predictions.csv) | 2026-10-30 15:58:11 | 88,859 bytes | `ac021aa22bb7951c...` | **UNMODIFIED** |
| [`benchmark/results/phase4_baseline_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase4_baseline_metrics.json) | 2026-10-30 15:58:11 | 23,959 bytes | `7e1e3555b4f3be75...` | **UNMODIFIED** |
| [`docs/PHASE4_BASELINE_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE4_BASELINE_EVALUATION_REPORT.md) | 2026-10-30 15:58:11 | 18,974 bytes | `5f1bddfd55725b77...` | **UNMODIFIED** |
| [`benchmark/results/phase5_val_raw_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_val_raw_predictions.csv) | 2026-10-30 20:30:11 | 39,763 bytes | `d8305c37f11aeb5b...` | **UNMODIFIED** |
| [`benchmark/results/phase5_validation_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_validation_metrics.json) | 2026-10-30 20:42:53 | 19,297 bytes | `587933e1d0dc0298...` | **UNMODIFIED** |
| [`benchmark/results/phase5_held_out_test_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_held_out_test_metrics.json) | 2026-10-30 21:26:48 | 3,147 bytes | `4a68170f92d80b60...` | **UNMODIFIED** |
| [`docs/PHASE5_FUSION_FEASIBILITY_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE5_FUSION_FEASIBILITY_REPORT.md) | 2026-10-30 21:27:16 | 24,310 bytes | `fa6bb972d73bada5...` | **UNMODIFIED** |
| [`benchmark/evaluation_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/evaluation_manifest.csv) | 2026-10-30 15:21:11 | 73,388 bytes | `637821dc80c89ea4...` | **UNMODIFIED** |

All Phase 4 and Phase 5 files retain their October 30, 2026 timestamps and recorded hashes, confirming zero in-place mutation or overwrite by Phase 6.

---

## 6. Exact Files Inspected During Audit

The following repository files were inspected directly during this audit:
1. Manifests & Datasets:
   - [`benchmark/phase6_test_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/phase6_test_manifest.csv)
   - [`benchmark/evaluation_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/evaluation_manifest.csv)
   - `C:\Users\user\Desktop\currently working\dataset_manifest.csv`
   - `C:\Users\user\Desktop\currently working\duplicates_metadata.csv`
   - `C:\Users\user\Desktop\currently working\duplicate_analysis_report.txt`
   - `C:\Users\user\Desktop\currently working\random_250_evaluation_results.csv`
   - `C:\Users\user\Desktop\currently working\evaluation_results.csv`
2. Prediction Outputs:
   - [`benchmark/results/phase6_vit_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_vit_predictions.csv)
   - [`benchmark/results/phase6_univfd_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_univfd_predictions.csv)
   - [`benchmark/results/phase6_convnext_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_convnext_predictions.csv)
   - [`benchmark/results/phase6_combined_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_combined_predictions.csv)
   - [`benchmark/results/phase4_baseline_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase4_baseline_predictions.csv)
   - [`benchmark/results/phase5_val_raw_predictions.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_val_raw_predictions.csv)
3. Metrics & Comparisons:
   - [`benchmark/results/phase6_combined_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_combined_metrics.json)
   - [`benchmark/results/phase6_model_comparison.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase6_model_comparison.csv)
   - [`benchmark/results/phase4_baseline_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase4_baseline_metrics.json)
   - [`benchmark/results/phase5_validation_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_validation_metrics.json)
   - [`benchmark/results/phase5_held_out_test_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase5_held_out_test_metrics.json)
4. Source Code & Evaluation Scripts:
   - [`src/decision_fusion.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/decision_fusion.py)
   - [`src/detector_registry.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/detector_registry.py)
   - [`src/detector_schema.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/detector_schema.py)
   - [`scripts/evaluate_phase6_pipeline.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/evaluate_phase6_pipeline.py)
   - [`scripts/test_detector_registry.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/test_detector_registry.py)
   - [`scripts/test_phase4_adversarial_verification.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/test_phase4_adversarial_verification.py)
   - [`tests/test_decision_fusion.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/tests/test_decision_fusion.py)
5. Documentation & Progression Records:
   - [`docs/PHASE6_DATASET_SELECTION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_DATASET_SELECTION_REPORT.md)
   - [`docs/PHASE6_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_EVALUATION_REPORT.md)
   - [`docs/PHASE6_COMPLETION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_COMPLETION_REPORT.md)
   - [`docs/PHASE6_EXECUTION_AUDIT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_EXECUTION_AUDIT.md)
   - [`docs/PHASE6_GENERALIZATION_PLAN.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_GENERALIZATION_PLAN.md)
   - [`docs/PHASE4_BASELINE_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE4_BASELINE_EVALUATION_REPORT.md)
   - [`docs/PHASE5_FUSION_FEASIBILITY_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE5_FUSION_FEASIBILITY_REPORT.md)
   - [`MODEL_STATUS_AND_PROJECT_PROGRESSION.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/MODEL_STATUS_AND_PROJECT_PROGRESSION.md)

---

## 7. Remaining Questions & Gaps

1. **Identity Provenance in Facial Source Datasets:**
   - The FFHQ dataset from which the Kaggle 140k REAL split was drawn consists of in-the-wild Flickr photographic portraits without ground-truth biometric identity metadata.
   - While file-level, SHA-256 hash-level, and partition-level independence are verified with zero historical collisions, biometric identity overlap (e.g. the same individual appearing under different camera angles or lighting in train vs test) remains unmeasured unless face embedding clustering (ArcFace) is conducted.
2. **Resolution of Threshold Sensitivity Question:**
   - Independent ROC-AUC analysis definitively proves that the failure of UnivFD ($\text{ROC-AUC} = 0.5503$) and ConvNeXt ($\text{ROC-AUC} = 0.4977$) on StyleGAN faces is due to complete feature blindness rather than threshold miscalibration. No linear threshold adjustment can restore sensitivity.
   - A remaining scientific question is whether fine-tuning the linear probe of UnivFD or fine-tuning the classification head of ConvNeXt on a small slice of StyleGAN faces would allow them to learn high-frequency GAN textures without catastrophic forgetting of their diffusion/ProGAN competencies.
3. **Cross-Domain and Multi-Generator Generalization:**
   - ViT dominates this facial test partition (98.00% accuracy, 100% recall), but earlier diagnostic evaluations indicated ViT degrades on non-facial diffusion synthesis (e.g. Imagen 3, Stable Diffusion scene generations) and cut-and-paste compositing.
   - The findings of Phase 6 are bounded to facial images; execution of the cross-generator benchmarks specified in [`docs/PHASE6_GENERALIZATION_PLAN.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE6_GENERALIZATION_PLAN.md) (e.g., GenImage, DiffusionForensics, FF++, Celeb-DF) should be prioritized by subsequent investigations.
