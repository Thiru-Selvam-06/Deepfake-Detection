# Forensic Audit Handoff Report: Milestone 1 Phase 4 Baseline Evaluation

## Forensic Audit Report

**Work Product**: Milestone 1: Phase 4 Baseline Evaluation Deliverables
**Target Deliverables**:
- `scripts/run_phase4_baseline_evaluation.py`
- `benchmark/results/phase4_baseline_predictions.csv`
- `benchmark/results/phase4_baseline_metrics.json`
- `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`
**Authoritative Request**: `c:\Users\user\Desktop\multi model connection\.agents\teamwork\ORIGINAL_REQUEST.md`
**Profile**: General Project (Integrity Mode: `development`)
**Verdict**: **CLEAN**

---

### Phase Verification Summary

| Check # | Forensic Verification Check | Status | Empirical Result Summary |
|:---:|:---|:---:|:---|
| 1 | **Git Status & File Integrity** | **PASS** | Model weights (Sept 2026), adapters (15:05), dataset files, and manifest untouched during Phase 4. |
| 2 | **Dataset & Manifest Verification** | **PASS** | `python scripts/verify_dataset_manifest.py` passed with 0 errors across all 159 images. |
| 3 | **Code Authenticity & Anti-Cheat Inspection** | **PASS** | Zero hardcoding, zero facade methods, zero mock lookup tables. Real model loading and inference. |
| 4 | **Data Leakage & Split Contamination** | **PASS** | Exactly 105 samples evaluated from `held_out_test`. Zero leakage from `development_val` or `operational_smoke`. |
| 5 | **Metric Derivation & Provenance Consistency** | **PASS** | JSON metrics and Markdown report match independent recalculations from raw CSV predictions 100%. |
| 6 | **Rejection Control Segregation** | **PASS** | 2 rejection controls caught gracefully as PIL errors, segregated completely from binary metrics. |
| 7 | **Live Empirical Reproducibility** | **PASS** | Live adapter inference on `real_1500.jpg` reproduced CSV predictions down to the 6th decimal place. |

---

## 1. Observation

Direct observations and raw tool outputs collected during forensic audit:

### 1.1 Model Weights and Adapter Integrity
1. Model weight inspection (`models/` and `UniversalFakeDetect/pretrained_weights/`):
   - `models/chen2024_convnext/16_acc0.9993.pth`: 354,605,453 bytes, `LastWriteTime: 18-09-2026 15:27:43`
   - `models/deepfake_vs_real_image_detection/model.safetensors`: 343,223,968 bytes, `LastWriteTime: 16-09-2026 20:47:20`
   - `UniversalFakeDetect/pretrained_weights/fc_weights.pth`: 4,083 bytes, `LastWriteTime: 17-09-2026 21:58:28`
   - All weights predate Phase 4 execution by over a month and remain unmodified.
2. Detector adapter files:
   - `src/convnext_adapter.py`: `LastWriteTime: 30-10-2026 15:05:38`
   - `src/vit_adapter.py`: `LastWriteTime: 30-10-2026 15:05:45`
   - `src/univfd_adapter.py`: `LastWriteTime: 30-10-2026 15:05:52`
   - `src/detector_schema.py`: `LastWriteTime: 17-09-2026 23:00:55`
   - Git diff confirms only `unload_model()` was added prior to Phase 4 (for memory deallocation) without altering inference logic, preprocessing, or thresholds. Zero modifications were made during Phase 4 execution (which began at 15:51).

### 1.2 Manifest Integrity
Executing `python scripts/verify_dataset_manifest.py`:
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

### 1.3 Baseline Evaluation Script Architecture (`scripts/run_phase4_baseline_evaluation.py`)
1. Model Loading (lines 109-112, 227-230):
   ```python
   adapter = get_detector(det_key, lazy_load=False, singleton=False)
   ```
2. Real-time Inference (lines 138-140, 253-255):
   ```python
   res = adapter.predict(file_path)
   res_dict = res.to_dict()
   ```
3. Split Isolation (lines 778-784):
   ```python
   eval_samples = [row for row in reader if row.get("split") == args.split]
   if len(eval_samples) != 105:
       raise ValueError(f"Integrity check failed: Expected exactly 105 samples in '{args.split}', found {len(eval_samples)}")
   ```
4. Control Segregation (lines 918-921):
   ```python
   overall_rows = [r for r in rows if r["ground_truth"] in ("REAL", "FAKE")]
   facial_rows = [r for r in rows if r["domain"] == "facial"]
   scene_rows = [r for r in rows if r["domain"] == "general_scene"]
   ctrl_rows = [r for r in rows if r["domain"] == "control"]
   ```
5. Dynamic Metric Derivation (lines 285-343): Computes `tp`, `fp`, `tn`, `fn`, `accuracy`, `precision`, `recall`, `f1_score`, `balanced_accuracy`, Wilson score 95% confidence intervals, and Cohen's kappa dynamically from sample prediction lists.

### 1.4 Data Leakage & Distribution Audit
Independent verification across `benchmark/results/phase4_baseline_predictions.csv` (315 total rows):
1. Split Distribution:
   - Exactly 1 split present: `{'held_out_test'}`
   - Contaminated rows from `development_val` (50 samples): **0**
   - Contaminated rows from `operational_smoke` (4 samples): **0**
2. Prediction Statistics:
   - `chen2024_convnext`: 105 rows, 103 valid, 2 errors. Unique logits: 103/103. Latency range: 361.6ms – 5431.5ms.
   - `univfd`: 105 rows, 103 valid, 2 errors. Unique logits: 103/103. Latency range: 1530.2ms – 24170.0ms.
   - `vit`: 105 rows, 103 valid, 2 errors. Unique logits: 103/103. Latency range: 335.2ms – 1341.6ms.
3. Rejection Controls:
   - Exactly 2 samples (`test_control_corr_001`, `test_control_unsupp_001`) evaluated per detector (6 total evaluations).
   - All 6 evaluations resulted in `status = ERROR`, `prediction = ERROR`, `rejection_successful = True`, capturing PIL decode/format errors gracefully without uncaught exceptions.

### 1.5 Live Inference Bit-Level Reproducibility
Live execution of all 3 detector adapters on held-out test sample `test_face_real_001` (`Dataset/Test/Real/real_1500.jpg`):
- `chen2024_convnext`:
  * Live inference: `pred=REAL, prob=0.00485, logit=-5.323988`
  * CSV record: `pred=REAL, prob=0.00485, logit=-5.323988` (EXACT MATCH)
- `univfd`:
  * Live inference: `pred=FAKE, prob=0.903896, logit=2.241279`
  * CSV record: `pred=FAKE, prob=0.903896, logit=2.241279` (EXACT MATCH)
- `vit`:
  * Live inference: `pred=REAL, prob=0.001544, logit=-6.472051`
  * CSV record: `pred=REAL, prob=0.001544, logit=-6.472051` (EXACT MATCH)

---

## 2. Logic Chain

1. **Premise 1 (Model and Dataset Stability)**: If model weight binaries, detector adapters, dataset files, and verification scripts are identical to their pre-evaluation state (Observation 1.1), the models were not modified, fine-tuned, or tampered with to bias evaluation results.
2. **Premise 2 (Manifest Integrity)**: The master evaluation manifest successfully passed independent verification of 159 files across SHA-256 hashes, PIL headers, and file sizes (Observation 1.2), establishing ground truth integrity.
3. **Premise 3 (Inference Authenticity)**: Because the runner explicitly loads model adapters and predicts on image filepaths via `adapter.predict()`, producing continuous, distinct logits and latencies (Observation 1.4), and because live model execution reproduces the exact floating-point probabilities and logits recorded in `phase4_baseline_predictions.csv` to 6 decimal places (Observation 1.5), the evaluation results represent genuine model inference rather than mocked, hardcoded, or pre-calculated data.
4. **Premise 4 (Zero Leakage and Segregation Compliance)**: Because all 315 rows in `phase4_baseline_predictions.csv` are strictly annotated with `split = held_out_test` and 0 rows originate from `development_val` or `operational_smoke` (Observation 1.4), and because `overall_rows` explicitly filters `ground_truth in ('REAL', 'FAKE')` to exclude `REJECTION_CONTROL` samples (Observation 1.3), the evaluation design complies strictly with requirements R2 and acceptance criteria.
5. **Premise 5 (Artifact Numerical Provenance)**: Re-computing all confusion matrices, accuracies, F1 scores, balanced accuracies, Wilson score 95% confidence intervals, and Cohen's kappa values independently from the raw CSV yields exact numerical concordance with `phase4_baseline_metrics.json` and `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` (Observation 1.4).
6. **Conclusion**: The deliverables satisfy all requirements with zero integrity violations.

---

## 3. Caveats

- **Device Runtime**: Evaluation was performed on host CPU, which is standard for reproducibility in this environment. Inference latencies reflect CPU execution.
- **Pre-existing Unload Method**: Git status indicates `unload_model()` was added to `univfd_adapter.py` and `vit_adapter.py` during prior memory-safety integration before Phase 4 commenced; verification confirmed this method only sets object references to `None` and clears caches, having zero impact on prediction math or model weights.
- No other caveats.

---

## 4. Conclusion

The Milestone 1: Phase 4 Baseline Evaluation deliverables (`scripts/run_phase4_baseline_evaluation.py`, `benchmark/results/phase4_baseline_predictions.csv`, `benchmark/results/phase4_baseline_metrics.json`, and `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`) are **CLEAN**. 

There is zero evidence of hardcoding, mocking, facade patterns, data leakage, split contamination, weight tampering, or fabricated outputs. The evaluation is 100% genuine, leakage-free, and reproducible. The work product is approved without reservations.

---

## 5. Verification Method

To independently verify this audit verdict:

1. **Verify Manifest Integrity**:
   ```powershell
   python scripts/verify_dataset_manifest.py
   ```
   *Expected*: Passes 100% with 0 errors across 159 samples.

2. **Verify Prediction Provenance and Metrics**:
   ```powershell
   python .agents/teamwork/auditor_1/verify_provenance.py
   ```
   *Expected*: Prints `ALL NUMERICAL CHECKS PASSED 100%!`, validating all 315 rows, zero leakage, exact confusion matrices, Wilson CIs, and pairwise concordances.

3. **Verify Git Immutability of Model Weights**:
   ```powershell
   git status models/ UniversalFakeDetect/pretrained_weights/
   ```
   *Expected*: No tracked modifications to model weight checkpoints.
