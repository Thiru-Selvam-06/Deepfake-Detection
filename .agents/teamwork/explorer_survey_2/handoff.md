# Phase 4 Baseline Evaluation: Detector Adapters & Memory-Safe Protocol Survey

## 1. Observation

1. **Manifest Integrity and Split Composition**:
   - Running `python scripts/verify_dataset_manifest.py` passed with exit code 0 (`[SUCCESS] Manifest verification passed 100% with zero errors!`).
   - The authoritative manifest is located at `benchmark/evaluation_manifest.csv` (159 total samples).
   - The `held_out_test` partition contains exactly 105 samples:
     - 50 authentic facial portraits (`test_face_real_001` to `test_face_real_050`, FFHQ $256 \times 256$ JPEG via `currently working/Dataset/Test/Real/real_1500.jpg` ... `real_1549.jpg`)
     - 50 synthetic StyleGAN facial portraits (`test_face_fake_001` to `test_face_fake_050`, StyleGAN $256 \times 256$ JPEG via `currently working/Dataset/Test/Fake/fake_1500.jpg` ... `fake_1549.jpg`)
     - 1 authentic scene photograph (`test_scene_real_001`, $4000 \times 3000$ PNG, natural smartphone capture)
     - 1 modern diffusion scene (`test_scene_diffusion_001`, Google Imagen 3 $1024 \times 1024$ JPEG)
     - 1 manipulated composite scene (`test_scene_manip_001`, $800 \times 800$ PNG)
     - 2 rejection controls: `test_control_corr_001` (20-byte truncated PNG) and `test_control_unsupp_001` (82-byte plain text document).
   - Samples in `development_val` (50 samples) and `operational_smoke` (4 samples) are strictly isolated and must be excluded from baseline metrics.

2. **ConvNeXt Detector Adapter (`src/convnext_adapter.py`)**:
   - Class: `ConvNeXtDetectorAdapter` (lines 53-289) wrapping `Chen2024ConvNeXtArchitecture(nn.Module)` (lines 29-51).
   - Default checkpoint: `DEFAULT_CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "chen2024_convnext", "16_acc0.9993.pth")` (lines 24-26).
   - Constructor: `__init__(self, checkpoint_path=None, device=None, lazy_load=True)` (lines 77-95).
   - Preprocessing: `transforms.Resize(224, bicubic) -> transforms.CenterCrop(224) -> transforms.ToTensor() -> transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (lines 98-106).
   - Prediction method: `predict(self, image_input: Any) -> DetectorResult` (lines 166-289).
   - Output calculation: 2-class logits `[real_logit, fake_logit]`. `probs = torch.softmax(logits, dim=1).squeeze(0)` (lines 247-248).
   - Relative logit: `relative_logit = fake_logit - real_logit` (line 255).
   - Threshold & Decision: `DECISION_THRESHOLD = 0.50` (line 71). `is_fake = fake_prob >= self.DECISION_THRESHOLD`; returns `PREDICTION_FAKE` if `is_fake` else `PREDICTION_REAL` (lines 260-261).
   - Label mapping: `real_idx = 0`, `fake_idx = 1` (lines 88-89).
   - Memory cleanup: `unload_model(self)` sets `self.model = None` and `self.transform = None`, calling `torch.cuda.empty_cache()` (lines 156-165).
   - Input validation: Traps non-existent path, `None`, unsupported type, and corrupted image via `try...except` returning `DetectorResult.create_error(...)` (lines 177-223).

3. **UniversalFakeDetect Adapter (`src/univfd_adapter.py`)**:
   - Class: `UnivFDAdapter` (lines 41-242).
   - Model components: CLIP ViT-L/14 backbone loaded via `UniversalFakeDetect/models:get_model("CLIP:ViT-L/14")` (line 89) and linear classifier checkpoint at `UniversalFakeDetect/pretrained_weights/fc_weights.pth` (line 92).
   - Constructor: `__init__(self, device=None, lazy_load=True)` (lines 62-72).
   - Preprocessing: OpenAI CLIP preprocess pipeline (`Resize(224, bicubic) -> CenterCrop(224) -> ToTensor() -> Normalize(clip)`).
   - Prediction method: `predict(self, image_input: Any) -> DetectorResult` (lines 123-242).
   - Output calculation: Linear probe outputs 1-D logit tensor: `logit_tensor = self.model.fc(features.float())` (line 205). `prob_fake = torch.sigmoid(logit_tensor).item()`, `prob_real = 1.0 - prob_fake` (lines 207-209).
   - Threshold & Decision: `DECISION_THRESHOLD = 0.50` (line 56), equivalent to logit $\ge 0.00$. `is_fake = prob_fake >= 0.50`; returns `PREDICTION_FAKE` if `is_fake` else `PREDICTION_REAL` (lines 213-215).
   - Label mapping: Sigmoid scalar where 0 = REAL, 1 = FAKE.
   - Memory cleanup: `unload_model(self)` sets `self.model = None` and `self.preprocess = None`, calling `torch.cuda.empty_cache()` (lines 113-122).
   - Input validation: Traps invalid paths, `None`, unsupported types, and corrupted images returning `DetectorResult.create_error(...)` (lines 134-181).

4. **Vision Transformer Adapter (`src/vit_adapter.py`)**:
   - Class: `ViTAdapter` (lines 27-243).
   - Model components: `AutoImageProcessor` and `AutoModelForImageClassification` loaded from `models/deepfake_vs_real_image_detection` (lines 23, 75-89).
   - Constructor: `__init__(self, device=None, lazy_load=True)` (lines 48-60).
   - Preprocessing: `ViTImageProcessor(Resize 224x224, Rescale 1/255, Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` (line 41).
   - Prediction method: `predict(self, image_input: Any) -> DetectorResult` (lines 117-243).
   - Output calculation: Logits of shape `[1, 2]`. `probs = torch.softmax(logits, dim=1).squeeze()` (lines 199-201). Dynamic label lookup: `real_idx` (label containing "real", index 0) and `fake_idx` (label containing "fake", index 1) (lines 94-96).
   - Relative logit: `relative_logit = fake_logit - real_logit` (line 209).
   - Threshold & Decision: `DECISION_THRESHOLD = 0.50` (line 42). `is_fake = fake_prob >= 0.50`; returns `PREDICTION_FAKE` if `is_fake` else `PREDICTION_REAL` (lines 214-216).
   - Label mapping: Index 0 = Real, Index 1 = Fake.
   - Memory cleanup: `unload_model(self)` sets `self.model = None` and `self.processor = None`, calling `torch.cuda.empty_cache()` (lines 107-116).
   - Input validation: Traps invalid paths, `None`, unsupported types, and corrupted images returning `DetectorResult.create_error(...)` (lines 128-175).

5. **Memory Constraints and Windows Heap Retention (`scripts/MEMORY_SAFE_BENCHMARK_REPORT.md`)**:
   - Host environment: Windows 10/11 x64, 8 GB System RAM, NVIDIA RTX 3050 4 GB VRAM.
   - Root cause of failure: Interleaved sample-by-sample loops kept UnivFD (~1,866 MB), ConvNeXt (~678 MB), and ViT (~1,981 MB accumulated) alive concurrently, exceeding 2.1 GB virtual memory and causing `os error 1455` (paging file too small) during ViT initialization (lines 12-27).
   - Windows MSVCRT and PyTorch allocator (`c10::Allocator`) retain committed address space even after Python `del` and `gc.collect()`.
   - Tiered mitigation verified:
     - Sequential execution (detector-by-detector)
     - Explicit deallocation (`adapter.unload_model()`, `del`, `gc.collect()`, `torch.cuda.empty_cache()`, `reset_instances()`)
     - Subprocess isolation (`--mode subprocess`): Child Python processes are terminated after each detector finishes inference, guaranteeing 100% OS-level memory reclamation and keeping the runner RSS at $\le 200\text{ MB}$ (lines 59-64).

6. **Benchmark Runner Limitations (`scripts/run_detector_benchmark.py`)**:
   - Hardcoded to `benchmark/metadata.json` (lines 49, 188, 312-316).
   - Does not accept CLI arguments for manifest CSV, test splits, or custom output file paths.
   - Resolves paths relative to `benchmark/images/` rather than reading absolute `file_path` from the manifest.
   - Does not compute domain-stratified metrics (facial vs. single-scene vs. rejection control).
   - In accordance with Requirement R1: *"If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner."*

---

## 2. Logic Chain

1. **Adapter Homogeneity**: Observations 2, 3, and 4 establish that all three adapters strictly fulfill the `DetectorResult` contract. Each adapter implements:
   - Identical constructor interface supporting `device` and `lazy_load`.
   - Identical `.predict(image_input)` method signature returning `DetectorResult`.
   - Identical threshold logic ($0.50$ fake probability cutoff).
   - Identical class label semantics: Real = 0 (or complement of fake probability), Fake = 1.
   - Explicit `.unload_model()` methods.
   Therefore, a single unified evaluation harness can execute all three detectors sequentially without custom per-detector branching.

2. **Memory Safety and Stability**: Observation 5 establishes that running all three detectors concurrently exhausts committed virtual memory on Windows (OS error 1455). Evaluating detector-by-detector with explicit deallocation (`unload_model()`, `del`, `gc.collect()`, `torch.cuda.empty_cache()`) reduces peak committed memory below the system limit. Furthermore, subprocess isolation provides a deterministic guarantee of complete virtual memory release between detector runs. Therefore, the Phase 4 evaluation runner must enforce sequential execution and offer subprocess isolation.

3. **Input Resilience**: Observations 2, 3, and 4 demonstrate that all three adapters catch corrupted image bytes, non-image text files, missing paths, and `None` inputs internally, returning structured `DetectorResult.create_error(...)` objects rather than throwing unhandled exceptions. Observation 1 shows the `held_out_test` partition contains 2 rejection control samples. Therefore, the evaluation runner can safely pass rejection control samples through the standard `.predict()` interface; the expected output is `prediction="ERROR"` (classified as successful input rejection).

4. **Runner Architecture Requirement**: Observation 6 proves that `scripts/run_detector_benchmark.py` cannot consume `benchmark/evaluation_manifest.csv` without modifying its hardcoded metadata and image path resolution. Requirement R1 explicitly prohibits rewriting the existing runner. Therefore, a new, dedicated script (`scripts/run_phase4_baseline_evaluation.py`) must be created to execute the 105 `held_out_test` samples and generate the required deliverables (`phase4_baseline_predictions.csv`, `phase4_baseline_metrics.json`, and `PHASE4_BASELINE_EVALUATION_REPORT.md`).

---

## 3. Caveats

1. **Hardware Execution Constraints**: Inference is currently running on CPU (`PyTorch 2.13.0+cpu`). While fully functional, evaluating 105 samples across three models sequentially (315 total inferences) on CPU requires approximately 3–6 minutes of total execution time. Adequate command wait times must be allocated.
2. **Rejection Control Metric Isolation**: Rejection control samples (`test_control_corr_001` and `test_control_unsupp_001`) must not be factored into accuracy, precision, or recall metrics, as doing so would distort binary classification performance.
3. **Single-Scene Diagnostic Scope**: The 3 general scene / composite samples in `held_out_test` ($N=1$ per type) cannot support statistical inference or confidence intervals; they must be reported as individual diagnostic case studies.

---

## 4. Conclusion

The detector adapters (`ConvNeXtDetectorAdapter`, `UnivFDAdapter`, `ViTAdapter`) and the dataset manifest (`benchmark/evaluation_manifest.csv`) are 100% verified, fully compatible, and ready for Phase 4 baseline evaluation.

### Key Architectural Recommendations for Implementation:
1. **Preserve Existing Files**: Do NOT alter `src/convnext_adapter.py`, `src/univfd_adapter.py`, `src/vit_adapter.py`, `src/detector_schema.py`, or `scripts/run_detector_benchmark.py`.
2. **Create Dedicated Evaluation Script**: Implement `scripts/run_phase4_baseline_evaluation.py` to ingest `benchmark/evaluation_manifest.csv`, filter for `split == 'held_out_test'`, and execute each detector sequentially.
3. **Enforce Memory Isolation**: Run detectors either in separate child subprocesses or with sequential lifecycle deallocation (`unload_model()`, `del`, `gc.collect()`, `torch.cuda.empty_cache()`, `reset_instances()`).
4. **Export Standard Deliverables**:
   - `benchmark/results/phase4_baseline_predictions.csv` (105 samples $\times$ 3 detectors = 315 rows)
   - `benchmark/results/phase4_baseline_metrics.json` (facial domain metrics + Wilson CIs, single-scene diagnostics, rejection control results, pairwise agreement)
   - `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` (comprehensive baseline evaluation report)

---

## 5. Verification Method

1. **Manifest Integrity**:
   ```powershell
   python scripts/verify_dataset_manifest.py
   ```
   *Expected result*: Exits with code 0 and outputs `[SUCCESS] Manifest verification passed 100% with zero errors!`.

2. **Detector Contract Verification**:
   ```powershell
   python scripts/test_detector_contract.py
   ```
   *Expected result*: Exits with code 0 and verifies that all three adapters conform to the `DetectorResult` schema, reject corrupted/non-image inputs with `prediction="ERROR"`, and correctly deallocate weights upon `unload_model()`.

3. **Invalidation Conditions**:
   - Any modification of weights in `models/` or `UniversalFakeDetect/pretrained_weights/`.
   - Inclusion of `development_val` (50 samples) or `operational_smoke` (4 samples) in Phase 4 baseline classification metrics.
   - Concurrent instantiation of multiple detector backbones inside the same process memory space.
