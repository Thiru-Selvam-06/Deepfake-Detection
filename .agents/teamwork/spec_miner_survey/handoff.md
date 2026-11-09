# Handoff Report: Phase 4 Baseline Evaluation Technical Specification

**Author**: Specification Miner (`spec_miner_survey`)  
**Target Recipient**: Orchestrator (`c50fe691-3bf8-4942-952c-4bc10ec2187c`)  
**Date**: 2026-10-30  
**Handoff Type**: Hard (Task Complete)  
**Analysis File**: `c:\Users\user\Desktop\multi model connection\.agents\teamwork\spec_miner_survey\analysis.md`  

---

## 1. Observation

Direct observations and quotes from authoritative files and tools:

1. **Manifest Partitioning & Integrity**:
   - `benchmark/evaluation_manifest.csv`: 161 lines (1 header + 159 rows).
   - Rows where `split == "held_out_test"`: Exactly 105 samples (50 `authentic_face`, 50 `gan_face`, 1 `authentic_scene`, 1 `diffusion_scene`, 1 `manipulated_scene`, 1 `corrupted_file`, 1 `unsupported_format`).
   - Rows where `split == "development_val"`: Exactly 50 samples (`val_face_real_001`–`025`, `val_face_fake_001`–`025`).
   - Rows where `split == "operational_smoke"`: Exactly 4 samples (`smoke_scene_real_001`, `smoke_scene_diffusion_001`, `smoke_scene_manip_001`, `smoke_control_corr_001`).
   - Ground truth breakdown: 77 REAL, 79 FAKE, 3 REJECTION_CONTROL.
   - Command `python scripts/verify_dataset_manifest.py` output:
     ```text
     [SUCCESS] Manifest verification passed 100% with zero errors!
     ```

2. **Existing Benchmark Runner Limitations**:
   - `scripts/run_detector_benchmark.py`:
     * Line 49: `METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")`
     * Line 100: `full_path = os.path.join(IMAGES_DIR, rel_path)`
     * Lines 433, 451: Exports to `all_detector_predictions.csv` and `detector_comparison.csv`.
     * Does NOT parse CSV splits or manifest columns; cannot directly evaluate `evaluation_manifest.csv` without modification or collisions.
   - `ORIGINAL_REQUEST.md` R1:
     > *"If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner."*

3. **Detector Adapter Contracts & Memory Management**:
   - `src/detector_schema.py`: `DetectorResult` dataclass requires `prediction` in `{"REAL", "FAKE", "UNCERTAIN", "ERROR"}` and bounded probabilities `[0.0, 1.0]`.
   - `src/convnext_adapter.py`, `src/univfd_adapter.py`, `src/vit_adapter.py`: All three implement `unload_model()` resetting weights to `None` and clearing CUDA cache.
   - Command `python scripts/test_detector_contract.py` output:
     ```text
     CONTRACT VERIFICATION SUMMARY: ALL ADAPTERS FULLY COMPLIANT
     Testing UnivFDAdapter... [PASS]
     Testing ViTAdapter... [PASS]
     Testing ConvNeXtDetectorAdapter... [PASS]
     ```

4. **Rejection Controls Markers**:
   - Manifest lines 105–106:
     * `test_control_corr_001`: `split=held_out_test`, `domain=control`, `category=corrupted_file`, `ground_truth=REJECTION_CONTROL`, `file_format=NON_IMAGE`
     * `test_control_unsupp_001`: `split=held_out_test`, `domain=control`, `category=unsupported_format`, `ground_truth=REJECTION_CONTROL`, `file_format=NON_IMAGE`
   - Both return `prediction: "ERROR"` with informative error string.

5. **Deliverables Requirements**:
   - `ORIGINAL_REQUEST.md` R4:
     * `benchmark/results/phase4_baseline_predictions.csv`
     * `benchmark/results/phase4_baseline_metrics.json`
     * `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`

---

## 2. Logic Chain

1. **Step 1 (Scope & Decoupling)**:
   From Observation 1, the dataset contains three splits. To prevent data leakage and preserve audit integrity, the evaluation runner must filter exclusively for `split == "held_out_test"` (105 samples). Samples from `development_val` (50 samples) and `operational_smoke` (4 samples) must never be loaded or counted in primary metrics.

2. **Step 2 (Metric Segregation)**:
   From Observation 1 and Observation 4, `held_out_test` has 103 binary samples (`REAL` / `FAKE`) and 2 rejection controls (`REJECTION_CONTROL`). Rejection controls are format verification probes, not binary classification instances. If included in classification accuracy, they would distort TP/TN/FP/FN. Therefore, they must be evaluated solely for graceful error handling (`prediction == "ERROR"`) and reported in a separate table/dictionary.

3. **Step 3 (Metric Formulation)**:
   For the 103 eligible classification samples, standard deepfake detection defines `Positive = FAKE` and `Negative = REAL`. Precision, recall, specificity, accuracy, balanced accuracy, and F1 must be computed with exact sample counts alongside any percentages (Observation 5).

4. **Step 4 (Facial Sub-Benchmark & Uncertainty Quantification)**:
   The 100 facial samples (50 Real FFHQ, 50 StyleGAN2 Fake) form a balanced sub-benchmark. Because $N=100$ is finite, Wilson score confidence intervals (Observation 3 and Analysis §5) are required to avoid degenerate zero-width intervals at perfect recall/accuracy. Documented limitations regarding $256 \times 256$ crop bias and generator specificity must accompany these metrics.

5. **Step 5 (Diagnostic Case Studies)**:
   The remaining 3 samples in `held_out_test` (`test_scene_real_001`, `test_scene_diffusion_001`, `test_scene_manip_001`) have $N=1$ per category. Computing statistical rates on $N=1$ is scientifically invalid; they must be presented in a dedicated qualitative case-study table.

6. **Step 6 (Pairwise Concordance)**:
   Pairwise model concordance measures consistency across model pairs (`ConvNeXt vs UnivFD`, `ConvNeXt vs ViT`, `UnivFD vs ViT`). To avoid undefined or skewed comparisons, concordance must be evaluated strictly on the intersection of samples where both models produced valid predictions (`REAL` or `FAKE`), excluding errors and controls.

7. **Step 7 (Execution Runner Design)**:
   From Observation 2, `scripts/run_detector_benchmark.py` targets `metadata.json` (5 diagnostic samples) and writes to Phase 2/3 output files. Modifying it risks breaking legacy regression tests. Thus, creating a dedicated `scripts/run_phase4_baseline_evaluation.py` strictly satisfies R1 and R4.

---

## 3. Caveats

1. **GPU vs CPU Latency**:
   On the target system (RTX 3050 Laptop GPU / CPU fallback), running all 105 samples sequentially across 3 detectors will take approximately 1–3 minutes depending on device placement. Sequential execution is required to avoid memory paging errors (`os error 1455`).
2. **Deterministic Confidence Intervals**:
   Wilson score intervals are analytical and closed-form. If bootstrap percentile intervals are chosen instead, random seeds must be fixed (`seed=42`) for reproducibility.
3. **No Code Written Outside Working Directory**:
   As a read-only specification miner, no evaluation runner or project code was written. The implementation of `scripts/run_phase4_baseline_evaluation.py` and generation of Phase 4 deliverables is delegated to the implementation agent.

---

## 4. Conclusion

The specification mining investigation is complete and fully documented in `analysis.md`. The technical boundaries, exact split definitions, metric equations, rejection control handling, deliverable schemas, and implementation architecture are unambiguous, verified against the filesystem, and ready for immediate execution by the implementation agent.

---

## 5. Verification Method

To verify the findings in this report:

1. **Verify Manifest Integrity**:
   ```powershell
   python scripts/verify_dataset_manifest.py
   ```
   *Expected*: Passes with exit code 0, confirming 159 total entries (105 `held_out_test`, 50 `development_val`, 4 `operational_smoke`).

2. **Verify Detector Contracts & Memory Management**:
   ```powershell
   python scripts/test_detector_contract.py
   ```
   *Expected*: Passes with exit code 0, confirming all 3 adapters support valid inference, corrupted file rejection, and `unload_model()`.

3. **Inspect Specification Analysis**:
   Inspect `c:\Users\user\Desktop\multi model connection\.agents\teamwork\spec_miner_survey\analysis.md` for full schemas, formulas, tables, and edge cases.
