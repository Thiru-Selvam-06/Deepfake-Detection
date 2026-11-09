# Original User Request

## Initial Request — 2026-10-30T10:01:01Z

Evaluate the three currently integrated image detectors (ConvNeXt/DRCT, UnivFD, ViT) individually against the verified manifest to establish a reproducible baseline before implementing decision fusion or changing any model.

Working directory: C:\Users\user\Desktop\multi model connection
Integrity mode: development

## Reference Material
- `docs/DATASET_AUDIT_AND_EVALUATION_PLAN.md`
- `benchmark/evaluation_manifest.csv`
- `scripts/verify_dataset_manifest.py`
- `scripts/run_detector_benchmark.py`
- `scripts/MEMORY_SAFE_BENCHMARK_REPORT.md`
- Integrated detector adapters: `src/convnext_adapter.py`, `src/univfd_adapter.py`, `src/vit_adapter.py`

## Requirements

### R1. Pre-Execution Inspection
Read the evaluation plan, manifest, benchmark runner, memory report, and detector adapters to confirm the manifest schema, ground-truth conventions, supported CLI arguments, label mappings, and prediction formats. If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner.

### R2. Baseline Execution and Evaluation Design Protection
Evaluate ConvNeXt/DRCT, UnivFD, and ViT (dima806) sequentially using memory-safe execution (load, predict, unload) on the held-out test split of `benchmark/evaluation_manifest.csv` (105 samples: 50 authentic facial, 50 StyleGAN facial, 1 authentic scene, 1 diffusion scene, 1 composite, 2 rejection controls). Preserve all original files, weights, adapters, class mappings, thresholds, and preprocessing. Do not download data, install packages, alter the dataset, use `development_val` for final metrics or threshold tuning, include `operational_smoke` samples in accuracy metrics, or implement decision fusion.

### R3. Domain-Specific Metrics and Reporting
Record prediction, predicted class, fake probability when available, ground truth, status, and error details for every sample, retaining failures rather than dropping them. Produce confusion matrix, accuracy, precision, recall, F1 score, balanced accuracy, and sample counts for eligible REAL/FAKE samples; separate metrics (with confidence intervals if possible, stating limitations) for the 100 facial samples; single-example diagnostic reporting for the 3 individual scene/composite samples; separate rejection-control reporting; pairwise agreement on samples where both detectors succeeded (excluding all-error cases); and lists of false positives, false negatives, and failures with sample IDs.

### R4. Verification and Deliverables Generation
Run `scripts/verify_dataset_manifest.py` to confirm manifest integrity, verify that no development or smoke samples entered primary metrics, and ensure original model weights and datasets remain untouched. Generate deliverables without overwriting previous results:
- `benchmark/results/phase4_baseline_predictions.csv`
- `benchmark/results/phase4_baseline_metrics.json`
- `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`

## Verification Resources
- `scripts/verify_dataset_manifest.py` for verifying manifest entries against disk.
- Programmatic output verification checking for missing rows, duplicate sample IDs, and unhandled errors in output files.

## Acceptance Criteria

### Execution & Integrity
- [ ] Evaluation uses strictly the `held_out_test` split from `benchmark/evaluation_manifest.csv` (105 samples) for primary evaluation.
- [ ] No `development_val` or `operational_smoke` samples are included in accuracy metrics.
- [ ] `REJECTION_CONTROL` samples are evaluated and reported separately from REAL/FAKE classification metrics.
- [ ] All three detectors (ConvNeXt/DRCT, UnivFD, ViT) are evaluated sequentially with memory-safe unload between runs.
- [ ] Original detector weights, adapters, thresholds, preprocessing, and dataset files remain unmodified.

### Metrics & Reporting
- [ ] Overall metrics (confusion matrix, accuracy, precision, recall, F1 score, balanced accuracy) are reported with exact sample counts alongside any percentages.
- [ ] Facial evaluation (50 real, 50 fake) metrics are reported separately, including confidence intervals where feasible.
- [ ] Single-example cases (authentic scene, diffusion scene, composite) are explicitly flagged as diagnostic only.
- [ ] Pairwise agreement is computed strictly over samples where both models returned valid predictions (excluding all-error cases).
- [ ] False positives, false negatives, and execution errors are enumerated with sample IDs.

### Deliverables & Verification
- [ ] `scripts/verify_dataset_manifest.py` passes successfully with zero errors.
- [ ] `benchmark/results/phase4_baseline_predictions.csv` contains predictions and error logs for all eligible samples without duplicate or missing rows.
- [ ] `benchmark/results/phase4_baseline_metrics.json` records complete evaluation metrics by domain.
- [ ] `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` documents environment details, exact commands, timestamps, model identifiers, sample counts, and limitations.
