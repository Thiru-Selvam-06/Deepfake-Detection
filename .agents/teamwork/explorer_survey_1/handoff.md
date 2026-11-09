# Phase 4 Baseline Evaluation Survey: Handoff Report

**Sender:** Teamwork Explorer (`explorer_survey_1`)  
**Recipient:** Parent Orchestrator (`c50fe691-3bf8-4942-952c-4bc10ec2187c`)  
**Date:** 2026-10-30  
**Context:** Phase 4 Baseline Evaluation Infrastructure Survey  
**Working Directory:** `c:\Users\user\Desktop\multi model connection`

---

## 1. Observation

### 1.1 Dataset Manifest (`benchmark/evaluation_manifest.csv`)
1. **Schema & Dimensions**: The manifest contains 1 header line and 159 data rows (160 total lines), with 16 columns:
   `sample_id, split, domain, category, ground_truth, source_dataset, generation_or_manipulation_method, label_evidence, verification_status, file_path, relative_path, file_format, width, height, file_size_bytes, sha256`
2. **Split Distribution**:
   - `held_out_test`: 105 samples (66.04%)
   - `development_val`: 50 samples (31.45%)
   - `operational_smoke`: 4 samples (2.52%)
   - Total rows = 159
3. **`held_out_test` Composition**:
   - Facial domain: 100 samples
     - `authentic_face` (`ground_truth == "REAL"`): 50 samples (`test_face_real_001` .. `test_face_real_050`), sourced from `Dataset/Test/Real/real_1500.jpg` to `real_1549.jpg` (FFHQ).
     - `gan_face` (`ground_truth == "FAKE"`): 50 samples (`test_face_fake_001` .. `test_face_fake_050`), sourced from `Dataset/Test/Fake/fake_1500.jpg` to `fake_1549.jpg` (StyleGAN2).
   - General scene domain: 3 samples
     - `authentic_scene` (`ground_truth == "REAL"`): 1 sample (`test_scene_real_001`, PNG, 4000x3000, `benchmark/images/real/real_phone_photo.png`).
     - `diffusion_scene` (`ground_truth == "FAKE"`): 1 sample (`test_scene_diffusion_001`, JPEG, 1024x1024, `benchmark/images/ai_generated/ai_cyberpunk_city.jpg`).
     - `manipulated_scene` (`ground_truth == "FAKE"`): 1 sample (`test_scene_manip_001`, PNG, 800x800, `benchmark/images/manipulated/manipulated_image.png`).
   - Control domain: 2 samples
     - `corrupted_file` (`ground_truth == "REJECTION_CONTROL"`): 1 sample (`test_control_corr_001`, NON_IMAGE, 20 bytes, `benchmark/images/corrupted/corrupted_image.png`).
     - `unsupported_format` (`ground_truth == "REJECTION_CONTROL"`): 1 sample (`test_control_unsupp_001`, NON_IMAGE, 82 bytes ASCII, `benchmark/images/corrupted/unsupported_doc.txt`).
4. **Disk Verification**:
   Programmatic inspection via Python confirmed that all 159 file paths recorded in `file_path` exist on disk (`Missing files: 0`).

### 1.2 Dataset Verification Harness (`scripts/verify_dataset_manifest.py`)
Executing `python scripts/verify_dataset_manifest.py` passed with return code 0 and verbatim output:
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

### 1.3 Benchmark Runner (`scripts/run_detector_benchmark.py`)
1. **Manifest Ingestion**: Lines 49–50 and 315–318 hardcode:
   ```python
   METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")
   IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
   with open(METADATA_FILE, "r", encoding="utf-8") as f:
       manifest = json.load(f)
   entries = manifest.get("entries", [])
   ```
   The runner does not ingest CSV files, cannot parse `benchmark/evaluation_manifest.csv`, and assumes all target files are located relative to `benchmark/images/`.
2. **Subprocess Script Ingestion**: Lines 188–193 in `run_detector_subprocess` hardcode the same JSON file path:
   ```python
   metadata_file = r"{METADATA_FILE}"
   images_dir = r"{IMAGES_DIR}"
   with open(metadata_file, "r", encoding="utf-8") as f:
       entries = json.load(f)["entries"]
   ```
3. **CLI Arguments**: Lines 479–486 define only `--mode` with choices `["inprocess", "subprocess"]`. No `--manifest`, `--split`, or `--output-dir` arguments exist.
4. **Split Filtering**: There is zero filtering by split. It processes 100% of entries in `metadata.json`.
5. **Memory Management**: Lines 141–154 implement sequential unloading:
   ```python
   if adapter is not None:
       if hasattr(adapter, "unload_model"):
           adapter.unload_model()
       del adapter
       adapter = None
   reset_instances()
   gc.collect()
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```
   Subprocess mode (`--mode subprocess`) runs child processes that terminate, allowing the operating system to reclaim 100% of allocated virtual memory.
6. **Outputs**: Overwrites `benchmark/results/all_detector_predictions.*` and `detector_comparison.*`. Computes no classification confusion matrices, accuracy, precision, recall, F1, or confidence intervals.

---

## 2. Logic Chain

1. **Premise (R1 & R2)**: Phase 4 Baseline Evaluation requires evaluating ConvNeXt, UnivFD, and ViT sequentially across the verified `held_out_test` partition (105 samples) of `benchmark/evaluation_manifest.csv`, generating non-overwriting deliverables (`phase4_baseline_predictions.csv`, `phase4_baseline_metrics.json`, `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`), and computing domain-specific metrics without including development or smoke samples.
2. **From Observation 1.1 & 1.2**: `benchmark/evaluation_manifest.csv` is fully verified against disk and meets all integrity criteria. Exactly 105 samples are marked `split == "held_out_test"`, comprising 100 facial crops (50 real, 50 fake), 3 general scenes (1 real, 1 diffusion, 1 composite), and 2 rejection controls. 103 samples are eligible for classification metrics, and 2 are rejection controls.
3. **From Observation 1.3**: The existing benchmark runner (`scripts/run_detector_benchmark.py`) cannot read `benchmark/evaluation_manifest.csv`, does not filter by split, does not compute classification metrics, and overwrites previous diagnostic benchmark outputs in `benchmark/results/`.
4. **From Requirement R1**: *"If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner."*
5. **Inference**: Rewriting `scripts/run_detector_benchmark.py` would violate R1, risk breaking existing diagnostic tests for `metadata.json`, and overwrite existing benchmark artifacts. Therefore, creating a dedicated evaluation script—`scripts/run_phase4_baseline_evaluation.py`—is the architecturally safe, modular, and compliant path.
6. **Implementation Blueprint**: The new script should:
   - Read `benchmark/evaluation_manifest.csv` using Python's standard `csv.DictReader`.
   - Filter strictly for `row['split'] == 'held_out_test'` (105 samples).
   - Evaluate detectors sequentially (`chen2024_convnext`, `univfd`, `vit`) using subprocess isolation (`--mode subprocess` default, `--mode inprocess` option).
   - Compute domain-partitioned metrics (103 eligible REAL/FAKE, separate 100 facial with Wilson 95% CIs, 3 single-example diagnostic cases, 2 rejection controls, and pairwise agreement on non-error samples).
   - Save deliverables to:
     - `benchmark/results/phase4_baseline_predictions.csv`
     - `benchmark/results/phase4_baseline_metrics.json`
     - `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`

---

## 3. Caveats

1. **GPU vs CPU Latency**: Inference on CPU for 3 detectors $\times$ 105 samples (315 total inferences) may take 2–5 minutes depending on CPU thread contention. ConvNeXt and ViT inference takes ~100–150 ms per sample on CPU; UnivFD takes ~200–300 ms per sample. Subprocess startup adds ~5–10 seconds per detector for initial weight loading.
2. **Rejection Controls in Adapters**: Two samples in `held_out_test` are non-images (`corrupted_image.png` and `unsupported_doc.txt`). All three adapters have been verified in `test_detector_contract.py` to catch PIL exceptions and return `prediction == "ERROR"`. The runner must record these as successful rejections rather than unhandled script crashes.
3. **Scope Constraint**: As an explorer subagent in read-only mode, I have surveyed, analyzed, and designed the required architecture without modifying or adding files outside `.agents/teamwork/explorer_survey_1/`. The implementation of `scripts/run_phase4_baseline_evaluation.py` and execution of the evaluation should be assigned to the builder/implementer agent.

---

## 4. Conclusion

1. The dataset manifest `benchmark/evaluation_manifest.csv` is 100% structurally sound, verified against disk, and ready for baseline evaluation on its 105 `held_out_test` samples.
2. The existing runner `scripts/run_detector_benchmark.py` is specialized for the 5-sample diagnostic suite (`benchmark/metadata.json`) and must not be overwritten or refactored.
3. In accordance with Requirement R1, the implementer should construct a dedicated runner:
   `scripts/run_phase4_baseline_evaluation.py`.
4. The dedicated runner should execute `chen2024_convnext`, `univfd`, and `vit` sequentially using memory-safe subprocess isolation, compute all domain-specific classification and diagnostic metrics, and output the required Phase 4 deliverables cleanly without touching historical results.

---

## 5. Verification Method

To independently verify the observations and conclusions in this report:

1. **Verify Manifest Integrity**:
   Run the project's automated verification command:
   ```powershell
   python scripts/verify_dataset_manifest.py
   ```
   *Expected outcome*: Exits with code 0, confirms 159 total entries, 105 held_out_test, 77 REAL, 79 FAKE, 3 REJECTION_CONTROL, and prints `[SUCCESS] Manifest verification passed 100% with zero errors!`.

2. **Verify Detector Contracts & Memory Unload**:
   Run the registry diagnostic test:
   ```powershell
   python scripts/test_detector_registry.py
   ```
   *Expected outcome*: Exits with code 0, verifies import isolation, dynamic registration of `chen2024_convnext`, `univfd`, and `vit`, and zero instances on import.

3. **Inspect Existing Runner Architecture**:
   Inspect `scripts/run_detector_benchmark.py` lines 48–51 and lines 188–193 to confirm hardcoded `metadata.json` dependencies and absence of CSV ingestion or split filtering.

4. **Review Detailed Technical Analysis**:
   Inspect `c:\Users\user\Desktop\multi model connection\.agents\teamwork\explorer_survey_1\analysis.md` for full column schemas, mathematical formulas for Wilson score confidence intervals, and the complete functional specification for `scripts/run_phase4_baseline_evaluation.py`.
