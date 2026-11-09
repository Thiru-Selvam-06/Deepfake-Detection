# Comprehensive Investigation Report: Detector Adapters & Memory-Safe Execution Protocols for Phase 4 Baseline Evaluation

**Investigation Target:** Integrated Detector Adapters, Memory Management, Input Handling, and Evaluation Protocol  
**Investigator:** Teamwork Explorer (`explorer_survey_2`)  
**Project:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Date:** 2026-10-30  
**Status:** Complete & Empirically Verified against Local Codebase  

---

## 1. Executive Summary

This report provides a rigorous, exhaustive technical analysis of the three integrated deepfake detector adapters (`ConvNeXtDetectorAdapter`, `UnivFDAdapter`, and `ViTAdapter`), their memory allocation and deallocation lifecycles, input validation resilience, and evaluation requirements for Phase 4 Baseline Evaluation on the verified manifest `benchmark/evaluation_manifest.csv` (`held_out_test` split: 105 samples).

### Core Findings:
1. **Three Fully Integrated Adapters Adhering to Standard Schema**: All three detectors wrap distinct deepfake detection backbones into a uniform contract returning `DetectorResult` (`src/detector_schema.py`). Every adapter provides a standardized `.predict(image_input)` method, records fake and real probabilities bounded in $[0.0, 1.0]$, relative logits, confidence aligned with prediction, threshold ($0.50$), inference latency, compute device, and pre-processing pipeline metadata.
2. **Explicit Memory Deallocation Methods Already Present**: Crucially, each of the three adapters already implements an explicit `.unload_model()` method that releases network weights, deletes preprocessing transforms/processors, and calls `torch.cuda.empty_cache()`.
3. **Windows Virtual Memory Heap Retention Requires Sequential Process Isolation**: In Windows 10/11 x64, PyTorch's underlying C++ allocator (`c10::Allocator`) and standard MSVCRT heap retain committed virtual memory pages. Loading all three models simultaneously causes Windows OS Error 1455 (`The paging file is too small for this operation to complete`). The benchmark runner must evaluate models **detector-by-detector sequentially** (outer loop detectors, inner loop samples). Furthermore, for absolute memory safety over 105 samples, **subprocess isolation mode** guarantees 100% OS-level reclamation of virtual memory address space upon child process exit.
4. **Resilient Structured Error Handling**: All three adapters safely intercept `None` inputs, nonexistent paths, corrupted image bytes, and unsupported non-image text files, returning well-formed `DetectorResult` instances with `prediction="ERROR"` and detailed error strings without raising uncaught exceptions.
5. **Existing Runner Inadequacy for Phase 4 Manifest**: `scripts/run_detector_benchmark.py` is hardcoded to `benchmark/metadata.json` (5 diagnostic samples) and does not support CLI arguments for dataset manifests, test splits, or domain-stratified metric computation. As mandated by Requirement R1, a separate, narrowly scoped evaluation runner (e.g. `scripts/run_phase4_baseline_evaluation.py`) should be created to evaluate the 105 `held_out_test` samples cleanly.

---

## 2. Detailed Technical Breakdown of Detector Adapters

### 2.1 ConvNeXt / DRCT (`src/convnext_adapter.py`)

- **Class Name**: `ConvNeXtDetectorAdapter` (accompanying architecture: `Chen2024ConvNeXtArchitecture(nn.Module)`)
- **Origin & Reference**: Chen et al. 2024 / constituent detector from the RAID benchmark (`aimagelab/RAID_ckpt`).
- **Constructor Parameters**:
  ```python
  def __init__(
      self,
      checkpoint_path: Optional[str] = None,
      device: Optional[Union[str, torch.device]] = None,
      lazy_load: bool = True,
  )
  ```
- **Default Weight Paths**:
  `os.path.join(BASE_DIR, "models", "chen2024_convnext", "16_acc0.9993.pth")`
- **Device Selection**:
  Defaults to `torch.device("cuda" if torch.cuda.is_available() else "cpu")` if `device` is `None`. Includes dynamic runtime device placement inside `predict()` to adjust to ZeroGPU or CPU fallback contexts.
- **Predict Method Signature & Return Format**:
  `predict(self, image_input: Any) -> DetectorResult`
  Accepts PIL `Image.Image`, file path `str`, or numpy/array-like objects convertible to PIL RGB.
- **Preprocessing Pipeline**:
  `transforms.Compose([Resize(224, bicubic), CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])`
- **Model Output & Score/Probability Interpretation**:
  - The model outputs a 2-dimensional logit tensor of shape `[1, 2]`.
  - Softmax is applied across dimension 1: `probs = torch.softmax(logits, dim=1).squeeze(0)`.
  - `real_prob = float(probs[0].item())`, `fake_prob = float(probs[1].item())`.
  - Relative logit: `relative_logit = fake_logit - real_logit` (such that relative logit $\ge 0.0$ corresponds directly to $P(\text{fake}) \ge 0.50$).
  - Confidence: `fake_prob if is_fake else real_prob`.
- **Threshold Logic**:
  `DECISION_THRESHOLD = 0.50`.
  `is_fake = fake_prob >= 0.50`.
- **Label Mapping**:
  Index 0 = `PREDICTION_REAL` ("REAL").
  Index 1 = `PREDICTION_FAKE` ("FAKE").
  If `is_fake` is True, returns `prediction = PREDICTION_FAKE`, else `PREDICTION_REAL`.

---

### 2.2 UniversalFakeDetect (`src/univfd_adapter.py`)

- **Class Name**: `UnivFDAdapter`
- **Origin & Reference**: CVPR 2023 UniversalFakeDetect (Ojeda et al.). Uses frozen CLIP ViT-L/14 image encoder combined with a trained binary linear classification probe (`fc_weights.pth`).
- **Constructor Parameters**:
  ```python
  def __init__(
      self,
      device: Optional[Union[str, torch.device]] = None,
      lazy_load: bool = True,
  )
  ```
- **Default Weight Paths**:
  - CLIP ViT-L/14 backbone loaded via `UniversalFakeDetect/models/__init__.py:get_model("CLIP:ViT-L/14")`.
  - Linear classifier checkpoint: `os.path.join(BASE_DIR, "UniversalFakeDetect", "pretrained_weights", "fc_weights.pth")`.
- **Device Selection**:
  Defaults to CUDA if available, otherwise CPU. Runtime verification ensures model parameters match current torch device before inference.
- **Predict Method Signature & Return Format**:
  `predict(self, image_input: Any) -> DetectorResult`
- **Preprocessing Pipeline**:
  OpenAI CLIP preprocessing: `Resize(224, bicubic) -> CenterCrop(224) -> ToTensor() -> Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])`.
- **Model Output & Score/Probability Interpretation**:
  - The frozen CLIP encoder generates a 768-dimensional feature embedding: `features = self.model.model.encode_image(input_tensor)`.
  - The linear probe layer outputs a single scalar logit: `logit_tensor = self.model.fc(features.float())`.
  - `raw_logit = logit_tensor.item()`.
  - Fake probability is computed via the standard sigmoid activation: `prob_fake = torch.sigmoid(logit_tensor).item()`.
  - Real probability is the exact complement: `prob_real = 1.0 - prob_fake`.
  - Confidence: `prob_fake if is_fake else prob_real`.
- **Threshold Logic**:
  `DECISION_THRESHOLD = 0.50` (corresponds exactly to `raw_logit >= 0.00`).
  `is_fake = prob_fake >= 0.50`.
- **Label Mapping**:
  Scalar sigmoid output: 0 corresponds to Real, 1 corresponds to Fake.
  If `is_fake` is True, returns `prediction = PREDICTION_FAKE`, else `PREDICTION_REAL`.

---

### 2.3 Vision Transformer (`src/vit_adapter.py`)

- **Class Name**: `ViTAdapter`
- **Origin & Reference**: Hugging Face checkpoint `dima806/deepfake_vs_real_image_detection` (fine-tuned `google/vit-base-patch16-224`).
- **Constructor Parameters**:
  ```python
  def __init__(
      self,
      device: Optional[Union[str, torch.device]] = None,
      lazy_load: bool = True,
  )
  ```
- **Default Weight Paths**:
  - Preferred local checkpoint: `os.path.join(BASE_DIR, "models", "deepfake_vs_real_image_detection")` (containing `config.json` and `model.safetensors`).
  - Remote fallback ID: `"dima806/deepfake_vs_real_image_detection"`.
- **Device Selection**:
  Defaults to CUDA if available, otherwise CPU. Runtime check verifies parameter device placement.
- **Predict Method Signature & Return Format**:
  `predict(self, image_input: Any) -> DetectorResult`
- **Preprocessing Pipeline**:
  `ViTImageProcessor(Resize 224x224, Rescale 1/255, Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`.
- **Model Output & Score/Probability Interpretation**:
  - Model outputs classification logits across 2 classes: `logits = outputs.logits` (shape `[1, 2]`).
  - Softmax across classes: `probs = torch.softmax(logits, dim=1).squeeze()`.
  - Dynamic label index mapping extracted directly from `model.config.id2label`:
    - `real_idx`: index matching `"real"` (Index 0).
    - `fake_idx`: index matching `"fake"` (Index 1).
  - `fake_prob = float(probs[self.fake_idx].item())`, `real_prob = float(probs[self.real_idx].item())`.
  - Relative logit: `relative_logit = fake_logit - real_logit` (aligned so that relative logit $\ge 0.0 \iff \text{fake\_prob} \ge 0.50$).
  - Confidence: `fake_prob if is_fake else real_prob`.
- **Threshold Logic**:
  `DECISION_THRESHOLD = 0.50`.
  `is_fake = fake_prob >= 0.50`.
- **Label Mapping**:
  Index 0 = Real, Index 1 = Fake (dynamically verified from `id2label`).
  If `is_fake` is True, returns `prediction = PREDICTION_FAKE`, else `PREDICTION_REAL`.

---

### 2.4 Comparative Adapter Architecture Matrix

| Specification | ConvNeXt (`chen2024_convnext`) | UnivFD (`univfd`) | ViT (`vit`) |
| :--- | :--- | :--- | :--- |
| **Adapter Class** | `ConvNeXtDetectorAdapter` | `UnivFDAdapter` | `ViTAdapter` |
| **Model Registry Key** | `chen2024_convnext` | `univfd` | `vit` |
| **Backbone Architecture** | ConvNeXt-Base (timm) | CLIP ViT-L/14 (OpenAI) | ViT-Base/16 (Google / HF) |
| **Head Architecture** | `nn.Linear(1024, 2)` | `nn.Linear(768, 1)` | `ViTForImageClassification` (2-class) |
| **Checkpoint Path** | `models/chen2024_convnext/16_acc0.9993.pth` | `UniversalFakeDetect/pretrained_weights/fc_weights.pth` | `models/deepfake_vs_real_image_detection` |
| **Primary Domain Focus** | Frequency / synthetic generator artifacts | General semantic / visual artifacts | Facial deepfake artifacts |
| **Input Resolution** | $224 \times 224$ (Bicubic + CenterCrop) | $224 \times 224$ (Bicubic + CenterCrop) | $224 \times 224$ (Bicubic resize) |
| **Input Color Channels** | RGB ($0 - 1$ normalized) | RGB (CLIP normalized) | RGB ([-1, 1] normalized) |
| **Raw Output Type** | 2-logit tensor | 1-logit scalar | 2-logit tensor |
| **Probability Function** | `torch.softmax(logits, dim=1)` | `torch.sigmoid(logits)` | `torch.softmax(logits, dim=1)` |
| **Threshold** | $P(\text{fake}) \ge 0.50$ | $P(\text{fake}) \ge 0.50$ ($\text{logit} \ge 0$) | $P(\text{fake}) \ge 0.50$ |
| **Explicit Cleanup** | `adapter.unload_model()` | `adapter.unload_model()` | `adapter.unload_model()` |

---

## 3. Memory-Safe Lifecycle Protocol & Windows OS Constraints

### 3.1 Memory Footprints per Model
Empirical measurement during the memory benchmarking study (`scripts/MEMORY_SAFE_BENCHMARK_REPORT.md`) recorded the following process memory footprints:
- **`ConvNeXtDetectorAdapter`**: ~355 MB weights + activation graphs $\implies$ Peak Resident Set Size (RSS): **~678 MB**.
- **`UnivFDAdapter`**: ~933 MB CLIP ViT-L/14 backbone + ~500 MB graph buffers $\implies$ Peak RSS: **~1,866 MB**.
- **`ViTAdapter`**: ~343 MB weights + Hugging Face model state $\implies$ Peak RSS: **~1,981 MB** (when evaluated in accumulated process memory).

### 3.2 The Root Cause of Windows OS Error 1455
When all three models were loaded into memory concurrently inside a single Python process:
1. Committed virtual memory exceeded $2.1\text{ GB}$.
2. On the host environment (Windows 10/11 x64, 8 GB System RAM, NVIDIA RTX 3050 4 GB VRAM), the Windows virtual memory allocation limit / paging file ceiling was exceeded.
3. Windows MSVCRT runtime failed with:
   ```text
   Model loading failed: The paging file is too small for this operation to complete. (os error 1455)
   ```
4. Even after Python object deletion, PyTorch's underlying C++ allocator (`c10::Allocator`) and the Windows C-Runtime heap retain committed virtual memory pages rather than immediately returning them to the OS.

### 3.3 The Multi-Tiered Memory Mitigation Strategy

To guarantee reproducible, crash-free execution over the 105 samples in `benchmark/evaluation_manifest.csv`, the following four tiers of protection must be observed:

#### Tier 1: Inverted Sequential Loop Execution
- **Never iterate sample-by-sample with multiple models alive**:
  $$\text{Prohibited:} \quad \forall \text{image} \in \text{Dataset} \implies \forall \text{model} \in \text{Detectors}$$
- **Always iterate detector-by-detector sequentially**:
  $$\text{Required:} \quad \forall \text{model} \in \text{Detectors} \implies [\text{Load} \implies \forall \text{image} \in \text{Dataset} \implies \text{Unload}]$$
  Only ONE model is resident in memory at any point in time.

#### Tier 2: Explicit In-Process Lifecycle Deallocation
After a detector finishes evaluating all samples:
1. `adapter.unload_model()`: Drops internal `model`, `transform`, `processor`, and `preprocess` references to `None`.
2. `del adapter`: Clears the local Python variable reference.
3. `src.detector_registry.reset_instances()`: Clears any cached singleton references in the registry.
4. `gc.collect()`: Forces immediate generational garbage collection across cyclic references.
5. `torch.cuda.empty_cache()`: Flushes cached CUDA allocations back to the GPU allocator if GPU acceleration is present.

#### Tier 3: Subprocess Process-Level Isolation (`--mode subprocess`)
While in-process sequential execution frees Python-level objects, Windows MSVCRT memory fragmentation can still accumulate over hundreds of inferences. Subprocess isolation executes each detector in an independent Python child process:
- The parent coordinator process stays at $<200\text{ MB}$ RSS.
- The child process spawns, loads a single detector, evaluates the 105 samples, emits JSON results to stdout/file, and terminates.
- **Operating System Guarantee**: Upon child process exit, Windows unconditionally reclaims 100% of the process's committed virtual address space, heap pages, and file descriptors.
- Subprocess execution eliminates any possibility of memory leaks or `os error 1455`.

#### Tier 4: Detector-Level Fault Isolation
If an unforeseen model initialization or memory error occurs, the runner catches the failure and generates a valid `DetectorResult.create_error(...)` record for all remaining samples. The evaluation of subsequent detectors continues without crashing the overall benchmark run.

---

## 4. Input Handling, Validation, and Error Safety

### 4.1 Input Format & Dimension Handling
Each adapter implements identical multi-modal input ingestion logic:
- **Input Types Accepted**:
  - File path string (`str`): validated for existence via `os.path.exists()`, loaded via `PIL.Image.open(path)` inside a context manager, and converted via `.convert("RGB")`.
  - PIL Image (`Image.Image`): converted via `.convert("RGB")`.
  - Numpy Array (`__array__`): wrapped via `Image.fromarray(arr).convert("RGB")`.
- **Arbitrary Resolutions & Aspect Ratios**:
  - High-resolution photographs (e.g., $4000 \times 3000$ `test_scene_real_001`), medium-resolution composites ($800 \times 800$), and standard facial crops ($256 \times 256$) are automatically scaled and cropped to the model's required $224 \times 224$ input dimensions by the torchvision or Hugging Face preprocessing transform.
  - No dimension-related exceptions escape the pipeline for valid image files.

### 4.2 Handling of Corrupted Files & Unsupported Formats
The manifest contains non-image rejection controls:
- `test_control_corr_001`: 20-byte truncated file with malformed PNG headers.
- `test_control_unsupp_001`: 82-byte plain text file (`unsupported_doc.txt`).

When passed to any of the three adapters:
1. `Image.open` raises `PIL.UnidentifiedImageError` or `OSError`.
2. The error is intercepted by the adapter's `try...except Exception as img_err:` block.
3. The adapter immediately returns:
   ```python
   DetectorResult.create_error(
       detector_name=self.DETECTOR_NAME,
       model_name=self.MODEL_NAME,
       error_message=f"Invalid or corrupted image: {str(img_err)}",
       warning=self.LIMITATION_WARNING,
       device=str(self.device) if self.device else None,
       preprocessing=self.PREPROCESSING_NAME,
   )
   ```
4. **No uncaught exceptions escape** to the runner. The result has `prediction="ERROR"`, `fake_probability=None`, and a descriptive error string.

### 4.3 Safe Runner Error Capture Protocol
Although adapters handle errors internally, the evaluation runner must follow defensive coding practices:
1. **Outer Try-Catch**:
   ```python
   try:
       res = adapter.predict(full_path)
       res_dict = res.to_dict()
   except Exception as exc:
       err_res = DetectorResult.create_error(
           detector_name=det_name,
           model_name=model_name,
           error_message=f"Runner caught unhandled inference exception: {str(exc)}",
       )
       res_dict = err_res.to_dict()
   ```
2. **Never Drop Failed Rows**:
   Every manifest row must produce an output record in `phase4_baseline_predictions.csv`. If an error occurs, the row is recorded with `prediction="ERROR"`, retaining full traceability.
3. **Rejection Control Separation**:
   In metrics computation:
   - For `REJECTION_CONTROL` rows, a prediction of `ERROR` is classified as **SUCCESSFUL_REJECTION** (pass). A prediction of `REAL` or `FAKE` is classified as **FAILED_REJECTION** (fail).
   - `REJECTION_CONTROL` samples must **NEVER** be mixed into accuracy, precision, recall, or confusion matrices for binary REAL/FAKE classification.

---

## 5. Manifest Ingestion & Evaluation Runner Architecture

### 5.1 Verification of the Evaluation Manifest
The authoritative dataset manifest is:
`C:\Users\user\Desktop\multi model connection\benchmark\evaluation_manifest.csv`

Running `scripts/verify_dataset_manifest.py` passed with **100% integrity across all 159 rows**:
- Total entries: 159
- `held_out_test`: **105 samples**
  - 50 Authentic Real Faces (FFHQ via `Dataset/Test/Real`, $256 \times 256$ JPEG)
  - 50 Synthetic GAN Faces (StyleGAN via `Dataset/Test/Fake`, $256 \times 256$ JPEG)
  - 1 Authentic Real Scene (`benchmark/images/real/real_phone_photo.png`, $4000 \times 3000$ PNG)
  - 1 Diffusion Scene (`benchmark/images/ai_generated/ai_cyberpunk_city.jpg`, $1024 \times 1024$ JPEG)
  - 1 Composite Scene (`benchmark/images/manipulated/manipulated_image.png`, $800 \times 800$ PNG)
  - 2 Rejection Controls (corrupted PNG and plain text TXT)
- `development_val`: 50 samples (25 real, 25 fake) — **strictly excluded from Phase 4 metrics**
- `operational_smoke`: 4 samples — **strictly excluded from Phase 4 metrics**

### 5.2 Existing Runner Limitations (Requirement R1 Analysis)
Inspection of `scripts/run_detector_benchmark.py` reveals that it cannot be used as-is for Phase 4:
1. **Hardcoded Metadata File**: Line 49 hardcodes `METADATA_FILE = os.path.join(BENCHMARK_DIR, "metadata.json")`, which contains only the 5 diagnostic samples.
2. **No Split Filtering**: Does not support filtering by `split == "held_out_test"`.
3. **Hardcoded Path Resolution**: Assumes all images are under `benchmark/images/` using relative filename paths, whereas the manifest contains absolute paths (`file_path`) pointing to both `currently working/Dataset/Test` and `benchmark/images`.
4. **No Domain Stratification**: Does not compute separate metrics for the 100 facial samples vs. single-scene diagnostic samples vs. rejection controls.
5. **No CLI Arguments for Manifest**: CLI only accepts `--mode {inprocess,subprocess}`.

**Conclusion under R1**: As explicitly instructed in Requirement R1:
> *"If the current benchmark runner cannot consume the manifest safely, create a separate, narrowly scoped evaluation script rather than rewriting the existing runner."*

A new evaluation script should be developed: `scripts/run_phase4_baseline_evaluation.py`.

### 5.3 Architectural Design for `run_phase4_baseline_evaluation.py`

The new evaluation script should adhere to the following architecture:
1. **Input Arguments**:
   - `--manifest`: defaults to `benchmark/evaluation_manifest.csv`.
   - `--split`: defaults to `held_out_test` (105 samples).
   - `--mode`: choices `["subprocess", "inprocess"]`, defaulting to `subprocess` (or `inprocess` with explicit deallocation).
   - `--output-predictions`: defaults to `benchmark/results/phase4_baseline_predictions.csv`.
   - `--output-metrics`: defaults to `benchmark/results/phase4_baseline_metrics.json`.
2. **Execution Order**:
   - Sequential loop over detectors (`chen2024_convnext`, `univfd`, `vit`).
   - For each detector, evaluate all 105 samples in `held_out_test`.
   - Full memory deallocation between detectors (`unload_model()`, `del`, `gc.collect()`, `empty_cache()`, or subprocess boundary).
3. **Predictions CSV Schema**:
   Columns must match the requirements in R3/R4:
   `sample_id,split,domain,category,ground_truth,file_path,detector,detector_name,model_name,prediction,fake_probability,real_probability,raw_logit,confidence,threshold,device,latency_ms,status,error,warning`
4. **Domain-Stratified Metrics JSON**:
   Metrics computed separately for:
   - **Facial Domain (100 samples: 50 Real, 50 Fake)**:
     - Confusion Matrix (TP, TN, FP, FN where FAKE = positive, REAL = negative).
     - Accuracy, Precision, Recall, Specificity, F1 Score, Balanced Accuracy.
     - 95% Wilson score confidence intervals for Accuracy, Recall, and Specificity.
     - Enumeration of False Positives, False Negatives, and Errors with `sample_id`s.
   - **Diagnostic General Scene Domain (3 individual samples)**:
     - Single-sample diagnostic reporting (sample ID, true label, predicted label, fake probability, status).
     - Explicit documentation that $N=1$ samples are diagnostic only and cannot establish statistical accuracy.
   - **Rejection Control Domain (2 samples)**:
     - Operational status: `INPUT_REJECTION_VERIFIED` (predicted `ERROR`) vs `FAILURE_DID_NOT_REJECT`.
   - **Pairwise Detector Agreement**:
     - Calculated strictly across samples where both detectors produced valid non-error predictions (excluding error cases).
5. **Report Deliverable**:
   `docs/PHASE4_BASELINE_EVALUATION_REPORT.md` documenting environment details, exact execution commands, memory profiles, metrics tables, cross-model comparison, and limitations.

---

## 6. Summary of Actionable Implementation Recommendations

1. **Keep Original Code Intact**: Do NOT modify `src/convnext_adapter.py`, `src/univfd_adapter.py`, `src/vit_adapter.py`, `src/detector_schema.py`, or `scripts/run_detector_benchmark.py`.
2. **Implement Dedicated Runner**: Create `scripts/run_phase4_baseline_evaluation.py` implementing the sequential memory-safe protocol over `held_out_test`.
3. **Prefer Subprocess Isolation**: Use subprocess execution (`python -c ...` or dedicated per-detector entrypoint) to ensure zero virtual memory leakage on Windows.
4. **Strict Ground-Truth Conventions**:
   - `REAL` = Authentic photographic portrait or scene (Negative).
   - `FAKE` = Synthetic GAN/Diffusion or composite manipulation (Positive).
   - `REJECTION_CONTROL` = Corrupted or non-image input (evaluated for error generation only).
5. **Verify Against Disk**: Run `scripts/verify_dataset_manifest.py` before and after evaluation to guarantee that no manifest entries, weights, or dataset files were altered.
