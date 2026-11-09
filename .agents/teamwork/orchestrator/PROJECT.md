# Project: Phase 4 Baseline Evaluation

## Architecture
- Standalone execution runner `scripts/run_phase4_baseline_evaluation.py` targeting `benchmark/evaluation_manifest.csv`
- Subprocess-isolated memory-safe execution across 3 detectors: ConvNeXt/DRCT (`src/convnext_adapter.py`), UnivFD (`src/univfd_adapter.py`), and ViT (`src/vit_adapter.py`)
- Domain-partitioned metrics calculation (overall 103 real/fake, 100 facial with Wilson 95% CIs, 3 single-example diagnostic cases, 2 rejection controls, pairwise concordance)
- Three deliverables:
  * `benchmark/results/phase4_baseline_predictions.csv`
  * `benchmark/results/phase4_baseline_metrics.json`
  * `docs/PHASE4_BASELINE_EVALUATION_REPORT.md`

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Manifest Filtering & Ingestion | Parse `benchmark/evaluation_manifest.csv` and strictly filter for `held_out_test` (105 samples) | M1 | Survey |
| 2 | Memory-Safe Sequential Detector Execution | Sequentially load, infer, and unload ConvNeXt, UnivFD, and ViT with OS-level memory reclamation via subprocess isolation | M1 | Survey |
| 3 | Raw Predictions Export | Generate `benchmark/results/phase4_baseline_predictions.csv` with exactly 315 rows and full error tracking | M1 | Survey |
| 4 | Classification Metrics Computation | Compute Confusion Matrix, Accuracy, Precision, Recall, F1, Balanced Accuracy with exact sample counts for 103 REAL/FAKE samples | M1 | Survey |
| 5 | Facial Sub-Benchmark & Wilson CIs | Compute facial metrics on 100 samples (50 real, 50 fake) with Wilson 95% confidence intervals and documented limitations | M1 | Survey |
| 6 | Diagnostic Case Studies & Controls | Report 3 single-example scene/composite cases and 2 rejection control samples separately | M1 | Survey |
| 7 | Pairwise Concordance | Compute pairwise agreement strictly over samples where both models returned valid predictions | M1 | Survey |
| 8 | Baseline Evaluation Report | Compile comprehensive documentation in `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` | M1 | Survey |
| 9 | Manifest & Weight Integrity Verification | Validate manifest integrity via `scripts/verify_dataset_manifest.py` and confirm weights/adapters remain untouched | M1 | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Baseline Evaluation & Deliverables | Implement `scripts/run_phase4_baseline_evaluation.py`, execute evaluation, generate all deliverables, verify integrity | Survey complete | DONE |

## Code Layout
- `scripts/run_phase4_baseline_evaluation.py` (New dedicated baseline evaluation runner)
- `benchmark/results/phase4_baseline_predictions.csv` (New deliverable: 315 prediction rows)
- `benchmark/results/phase4_baseline_metrics.json` (New deliverable: metrics dictionary)
- `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` (New deliverable: comprehensive evaluation report)
- `benchmark/evaluation_manifest.csv` (Read-only, unmodified)
- `scripts/verify_dataset_manifest.py` (Read-only, unmodified verification tool)
- `src/convnext_adapter.py`, `src/univfd_adapter.py`, `src/vit_adapter.py` (Read-only, unmodified adapters)
- `models/`, `UniversalFakeDetect/pretrained_weights/`, `Dataset/` (Read-only, unmodified weights and data)
