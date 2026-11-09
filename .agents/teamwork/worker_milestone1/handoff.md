# Milestone 1 Handoff Report: Phase 4 Baseline Evaluation Runner, Execution, and Deliverables Generation

**Author:** `worker_milestone1` (Implementer / QA / Specialist)  
**Date:** 2026-10-30T10:31:00Z  
**Parent Orchestrator:** `c50fe691-3bf8-4942-952c-4bc10ec2187c`  
**Working Directory:** `c:\Users\user\Desktop\multi model connection\.agents\teamwork\worker_milestone1\`  

---

## 1. Observation

1. **Manifest Integrity & Split Composition:**
   - Manifest path: `benchmark/evaluation_manifest.csv` contains 159 rows.
   - Running `python scripts/verify_dataset_manifest.py` passed with 0 errors:
     ```text
     Manifest Path: C:\Users\user\Desktop\multi model connection\benchmark\evaluation_manifest.csv
     Total manifest entries: 159
     [Split Breakdown]
       - development_val: 50 samples
       - held_out_test: 105 samples
       - operational_smoke: 4 samples
     [SUCCESS] Manifest verification passed 100% with zero errors!
     ```
   - Filtering for `split == 'held_out_test'` yielded exactly 105 samples:
     * Facial Domain: 100 samples (50 REAL FFHQ portraits `real_1500.jpg`–`real_1549.jpg`, 50 FAKE StyleGAN2 portraits `fake_1500.jpg`–`fake_1549.jpg`).
     * General Scene Domain: 3 samples (`test_scene_real_001`, `test_scene_diffusion_001`, `test_scene_manip_001`).
     * Control Domain: 2 samples (`test_control_corr_001`, `test_control_unsupp_001`).
     * Total eligible binary classification samples: 103 (51 REAL, 52 FAKE).
     * Total rejection controls: 2 (`REJECTION_CONTROL`).

2. **Runner Implementation & Execution:**
   - Script created: `scripts/run_phase4_baseline_evaluation.py`.
   - Command executed: `python scripts/run_phase4_baseline_evaluation.py`.
   - Execution duration: 345.91 seconds (~5.7 minutes) on CPU with Windows Subprocess isolation per detector.
   - Result: Exit code 0, 315 total predictions evaluated (105 samples $\times$ 3 detectors).
   - Rejection Controls: 2/2 rejected gracefully across all 3 detectors (6/6 rejections returning `prediction = ERROR`, 100.0% success rate).

3. **Performance Observations by Detector:**
   - **`chen2024_convnext` (ConvNeXt-Base / RAID):**
     * Overall (103 samples): Accuracy = 41.75% (43/103), Precision = 30.00% (6/20), Recall = 11.54% (6/52), Specificity = 72.55% (37/51), F1 = 0.1667, Balanced Accuracy = 42.04%.
     * Facial (100 samples): Accuracy = 42.00% (42/100) [95% Wilson CI: 32.8%–51.8%], Recall = 12.00% (6/50), Precision = 30.00% (6/20), Specificity = 72.00% (36/50).
     * False Positives: 14; False Negatives: 46.
   - **`univfd` (UniversalFakeDetect / CLIP ViT-L/14):**
     * Overall (103 samples): Accuracy = 59.22% (61/103), Precision = 91.67% (11/12), Recall = 21.15% (11/52), Specificity = 98.04% (50/51), F1 = 0.3437, Balanced Accuracy = 59.60%.
     * Facial (100 samples): Accuracy = 60.00% (60/100) [95% Wilson CI: 50.2%–69.1%], Recall = 22.00% (11/50), Precision = 91.67% (11/12), Specificity = 98.00% (49/50).
     * False Positives: 1 (`test_face_real_001`); False Negatives: 41.
   - **`vit` (ViTDeepfakeDetector / dima806):**
     * Overall (103 samples): Accuracy = 98.06% (101/103), Precision = 100.00% (50/50), Recall = 96.15% (50/52), Specificity = 100.00% (51/51), F1 = 0.9804, Balanced Accuracy = 98.08%.
     * Facial (100 samples): Accuracy = 100.00% (100/100) [95% Wilson CI: 96.3%–100.0%], Recall = 100.00% (50/50), Precision = 100.00% (50/50), Specificity = 100.00% (50/50).
     * False Positives: 0; False Negatives: 2 (`test_scene_diffusion_001`, `test_scene_manip_001`).

4. **Qualitative Diagnostic Case Studies (3 Scene Samples):**
   - `test_scene_real_001` (Authentic Indoor Smartphone Photo, PNG 4000x3000):
     * ConvNeXt: `REAL` ($P_{\text{fake}} = 2.12\%$)
     * UnivFD: `REAL` ($P_{\text{fake}} = 0.00\%$)
     * ViT: `REAL` ($P_{\text{fake}} = 19.78\%$)
     * Verdict: Unanimous correct classification.
   - `test_scene_diffusion_001` (Google Imagen 3 Latent Diffusion Scene, JPEG 1024x1024):
     * ConvNeXt: `REAL` ($P_{\text{fake}} = 0.03\%$) — Missed
     * UnivFD: `REAL` ($P_{\text{fake}} = 0.08\%$) — Missed (demonstrates generator cutoff)
     * ViT: `REAL` ($P_{\text{fake}} = 34.24\%$) — Elevated fake probability but misses 0.50 threshold.
     * Verdict: Unanimous false negative at 0.50 threshold.
   - `test_scene_manip_001` (Digital Splicing Composite, PNG 800x800):
     * ConvNeXt: `REAL` ($P_{\text{fake}} = 3.22\%$) — Missed
     * UnivFD: `REAL` ($P_{\text{fake}} = 0.60\%$) — Missed
     * ViT: `REAL` ($P_{\text{fake}} = 4.91\%$) — Missed
     * Verdict: Unanimous false negative; confirms whole-image classification pooling fails on localized spatial splices.

5. **Pairwise Model Agreement ($N_{\text{valid}} = 103$):**
   - `chen2024_convnext` vs `univfd`: Concordance = 72.82% (75/103), Cohen's $\kappa = -0.0241$.
   - `chen2024_convnext` vs `vit`: Concordance = 43.69% (45/103), Cohen's $\kappa = -0.1466$.
   - `univfd` vs `vit`: Concordance = 61.17% (63/103), Cohen's $\kappa = 0.2056$.

6. **Deliverables Verified on Disk:**
   - `benchmark/results/phase4_baseline_predictions.csv`: Exactly 316 lines (1 header + 315 data rows). Exactly 105 rows per detector, 0 duplicates, valid values for all 20 columns.
   - `benchmark/results/phase4_baseline_metrics.json`: Fully structured hierarchical document with `metadata`, `detectors`, and `pairwise_agreement`.
   - `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`: 12-section formal report with complete statistical tables, Wilson CIs, cross-model comparisons, and strategic recommendations for Phase 5 fusion.

---

## 2. Logic Chain

1. **Split Decoupling Guarantee:** By enforcing `split == 'held_out_test'` at the manifest ingestion point, all 50 `development_val` and 4 `operational_smoke` samples were completely excluded from baseline evaluations, preserving `development_val` for Phase 5 fusion calibration with zero data leakage.
2. **Memory Safety Mechanism:** Host environment constraints (8 GB RAM, 0.85 GB available, CPU execution) trigger Windows OS error 1455 if multiple models reside in memory concurrently. By deploying child subprocess execution (`run_detector_subprocess`), memory pages were reclaimed at the OS kernel level after each detector's 105 samples, preventing heap fragmentation and memory exhaustion.
3. **Statistical Validity:** Small sample benchmarks ($N=100$) suffer from unstated variance if only point accuracy is reported. Computing Wilson score 95% confidence intervals bounded within $[0, 1]$ avoids degeneracy at 100% recall/precision (e.g., $[0.929, 1.000]$) and provides valid confidence bounds.
4. **Domain Disconnect Confirmation:** ViT achieved 100% on StyleGAN2 faces ($P_{\text{fake}} = 99.84\%$), yet scored $P_{\text{fake}} = 34.24\%$ on latent diffusion and $4.91\%$ on composites. UnivFD scored $P_{\text{fake}} = 0.08\%$ on Imagen 3 diffusion. This empirical proof demonstrates that naive linear averaging in Phase 5 would fail, necessitating domain-gated routing.

---

## 3. Caveats

- **Host Device:** Evaluation was executed using CPU inference due to CUDA being unavailable on the host machine. Latencies reflect single-thread/multi-core CPU processing.
- **Generator Representation:** The synthetic facial benchmark is homogenous to StyleGAN/StyleGAN2. Generalization to diffusion faces (Flux, SD3) or deepfake face swaps was not directly evaluated in the 100 facial samples.
- **Diagnostic Sample Size:** Diagnostic scene cases are single-instance qualitative probes ($N=1$). They demonstrate artifact sensitivity trends but cannot be used for statistical generalization.

---

## 4. Conclusion

Milestone 1 is complete. All three detectors (`chen2024_convnext`, `univfd`, `vit`) have been rigorously evaluated against the verified master manifest without modifying weights, adapters, thresholds, or dataset files. All three required deliverables (`phase4_baseline_predictions.csv`, `phase4_baseline_metrics.json`, and `PHASE4_BASELINE_EVALUATION_REPORT.md`) have been generated, validated, and confirmed on disk.

---

## 5. Verification Method

1. **Manifest Integrity:**
   ```powershell
   python scripts/verify_dataset_manifest.py
   ```
   *Expected Result:* Exit code 0, 100% verified, 159 samples, 0 errors.

2. **Re-run Baseline Evaluation Runner:**
   ```powershell
   python scripts/run_phase4_baseline_evaluation.py
   ```
   *Expected Result:* Exit code 0, evaluates 105 samples $\times$ 3 detectors, writes all 3 deliverables.

3. **Programmatic Deliverable Verification:**
   ```powershell
   python -c "
   import csv, json
   with open('benchmark/results/phase4_baseline_predictions.csv', 'r') as f:
       rows = list(csv.DictReader(f))
   assert len(rows) == 315
   with open('benchmark/results/phase4_baseline_metrics.json', 'r') as f:
       data = json.load(f)
   assert len(data['detectors']) == 3
   print('Verification passed!')
   "
   ```
   *Expected Result:* Prints `Verification passed!`.
