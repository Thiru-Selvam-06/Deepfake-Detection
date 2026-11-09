# Project Audit: Multi-Model Deepfake Image Detection

**Audit date:** 2026-10-30  
**Project directory reported by user:** `C:\Users\user\Desktop\multi model connection`  
**Audit basis:** Project tree listing and pasted contents of detector adapters, registry, benchmark/design documents. This is a source-and-document audit, not a fresh execution of the code or an independent reproduction of reported metrics.

---

## 1. Executive summary

The project has evolved from a single-model deepfake detector into a **modular, multi-detector evaluation framework**. Its purpose is to run different image detectors through a consistent interface, compare their outputs, and preserve each model's own preprocessing and decision logic.

The current project evidence shows three registered detector identifiers:

1. `univfd` — UniversalFakeDetect, based on CLIP ViT-L/14 plus a trained linear probe.
2. `vit` — a fine-tuned ViT-Base/16 image classifier from `dima806/deepfake_vs_real_image_detection`.
3. `chen2024_convnext` — a ConvNeXt-Base detector checkpoint associated with the RAID checkpoint collection.

Each detector is wrapped in an adapter that returns a shared `DetectorResult` object. The registry supports lazy construction and caches adapter instances.

**Important architecture distinction:** registering several detectors does not itself create an ensemble. The available evaluation/design document explicitly states that the detectors were evaluated independently and that no prediction-fusion or agreement layer had been implemented in that documented phase. A combined final verdict must not be claimed unless additional current code proves it.

The older `df_model.pt` baseline is part of the project's reported history, but the pasted current directory listing does not establish its exact path, code, checkpoint architecture, or whether it is still integrated into this repository. Treat its relationship to the current detector registry as **not yet verified**.

## 2. Audit scope and evidence quality

### Inspected evidence

- Project file listing, including `src/`, `models/`, `scripts/`, `benchmark/results/`, `docs/`, and `UniversalFakeDetect/`.
- `src/detector_schema.py`
- `src/detector_registry.py`
- `src/vit_adapter.py`
- `src/convnext_adapter.py`
- Contents of project documentation including:
  - `DETECTOR_3_CANDIDATE_ANALYSIS.md`
  - `DETECTOR_BENCHMARK_RESULTS.md`
  - `MULTI_MODEL_EVALUATION_PLAN.md`
  - `UNIVFD_LIMITATIONS.md`
- Benchmark/evaluation snippets included in the pasted terminal output.

### Not independently verified in this audit

- The project was not run locally as part of this audit.
- Test scripts were not executed again.
- The actual model-weight files were not opened or hashed.
- Accuracy figures were not independently recomputed from a complete labeled dataset.
- The original `df_model.pt` implementation was not present in the pasted source excerpts.
- The current deployed Hugging Face Space may differ from the local working directory.

Where a claim comes from a project document rather than a fresh run, this audit calls it a **reported result** or **documented design**.

## 3. Project evolution

### Stage 1 — Original single-model baseline

The project history identifies `df_model.pt` as the original working model. The reported experience was that it performed well on a particular dataset/type of data but did not generalize sufficiently to all target image types.

The exact architecture, class mapping, training dataset, preprocessing, and evaluation metrics of this checkpoint must be recovered from the original inference/training code and checkpoint metadata. Earlier project notes associated a video detector with ResNeXt-50 + LSTM, but that association should not automatically be treated as the architecture of `df_model.pt` in the current image-only multi-model project.

**Baseline preservation principle:** retain the original working detector unchanged while experiments with alternative models are isolated. Compare models on the same labeled test set before deciding whether to replace or combine them.

### Stage 2 — Separate model experiments

The current repository has a dedicated `src/` integration layer and benchmark scripts/results. The goal is to make each model independently loadable and measurable, instead of entangling model-specific code.

### Stage 3 — ViT candidate

The repository includes a fine-tuned ViT image classifier:

- Hugging Face model ID: `dima806/deepfake_vs_real_image_detection`
- Architecture: `ViTForImageClassification`, based on ViT-Base/16 (`google/vit-base-patch16-224`)
- Local model directory: `models/deepfake_vs_real_image_detection`
- Expected local checkpoint: `model.safetensors`
- Input: RGB image processed to 224 × 224
- Label mapping in adapter: real index 0, fake index 1
- Decision threshold: fake probability ≥ 0.50 means `FAKE`

The adapter loads a local model directory if its `config.json` exists; otherwise it falls back to the Hugging Face model ID. It reads label indices from the model config by searching for labels containing “real” and “fake”.

### Stage 4 — ConvNeXt candidate

The repository includes a Chen2024/RAID-associated ConvNeXt detector:

- Registry key: `chen2024_convnext`
- Architecture: ConvNeXt-Base backbone with a two-class linear head (1024 → 2)
- Expected checkpoint: `models/chen2024_convnext/16_acc0.9993.pth`
- Source described in code: `aimagelab/RAID_ckpt`
- Input: RGB, resize/crop to 224 × 224, ImageNet normalization
- Label mapping: index 0 = real, index 1 = fake
- Decision threshold: fake probability ≥ 0.50 means `FAKE`

The filename contains `acc0.9993`, but this is not proof that the model achieves 99.93% accuracy on the project's target data. It is a checkpoint filename / source-associated metric and must not be presented as the current application's measured accuracy.

### Stage 5 — UniversalFakeDetect

The project also integrates UniversalFakeDetect:

- Registry key: `univfd`
- Backbone: CLIP ViT-L/14
- Classifier: linear probe (`nn.Linear(768, 1)`)
- Probe checkpoint: `UniversalFakeDetect/pretrained_weights/fc_weights.pth`
- The probe checkpoint contains the linear head, not the entire CLIP backbone; the CLIP backbone weights are separate.
- Output is a single logit converted to fake probability using sigmoid.
- Documented label mapping: 0 = real, 1 = fake.
- Its historical training/evaluation domain emphasizes ProGAN-era artifacts and earlier generators, so modern diffusion images and localized manipulations can be out-of-distribution.

The UniversalFakeDetect limitation report records examples where AI-generated or manipulated images were confidently labeled `REAL`. This is a significant reminder that confidence is not equivalent to correctness.

## 4. Current architecture

```text
Image input
   |
   v
Input validation / RGB conversion (adapter-level handling)
   |
   +----------------------+-------------------------+
   |                      |                         |
   v                      v                         v
UniversalFakeDetect       ViT adapter               ConvNeXt adapter
CLIP ViT-L/14             ViT-Base/16               ConvNeXt-Base
+ linear probe            2-class head              + 2-class head
   |                      |                         |
   v                      v                         v
DetectorResult            DetectorResult             DetectorResult
   |                      |                         |
   +----------------------+-------------------------+
                          |
                          v
                Benchmark / comparison harness
```

This diagram represents the **independent detector architecture** supported by the code/documentation. It does not imply that a production API always calls all three detectors or fuses their scores. That behavior must be verified in the application entry point and benchmark runner.

### Main integration components

#### `src/detector_schema.py`

Defines a dataclass named `DetectorResult`, the common return structure for detector predictions.

Fields include:

- `detector_name`
- `model_name`
- `prediction`
- `fake_probability`
- `real_probability`
- `raw_logit`
- `confidence`
- `threshold`
- `device`
- `preprocessing`
- `inference_time_ms`
- `error`
- `warning`

Allowed prediction strings are `REAL`, `FAKE`, `UNCERTAIN`, and `ERROR`. The dataclass rejects any other prediction label. `to_dict()` converts a result to a serializable dictionary. `create_error()` constructs a consistent error result.

Note: the schema permits `UNCERTAIN`, but the visible ViT and ConvNeXt adapters make binary REAL/FAKE decisions at a 0.50 threshold. A schema value being available does not prove that a composite uncertainty policy exists.

#### `src/detector_registry.py`

Provides centralized registration and lookup of detector adapters.

Observed behavior:

- Normalizes detector names to lowercase.
- Rejects duplicate registrations unless `overwrite=True`.
- Supports registration metadata.
- Can instantiate an adapter by name.
- Can cache a singleton adapter instance.
- Supports lazy loading by passing `lazy_load` when the factory accepts it.
- Verifies that the created adapter has a callable `predict()` method.
- Provides `get_all_detectors()` and `reset_instances()` helpers.
- Registers `univfd`, `vit`, and `chen2024_convnext`.

Lazy loading reduces unnecessary model initialization during imports. It does not guarantee that all models fit in available memory if the application loads several models simultaneously.

#### `src/vit_adapter.py`

Responsibilities:

1. Accepts a PIL image, file path, or array-like image.
2. Rejects empty, missing, unsupported, or corrupted input with an `ERROR` result.
3. Converts valid images to RGB.
4. Loads the processor and model only when required.
5. Selects CPU or CUDA based on availability/requested device.
6. Applies the Hugging Face image processor.
7. Runs inference under `torch.inference_mode()`.
8. Applies softmax to the two logits.
9. Extracts fake and real probabilities using model-config label indices.
10. Returns `DetectorResult` with prediction, confidence, timing, preprocessing description, and limitation warning.

Potential code-review item: the adapter dynamically moves the model to the currently available CUDA/CPU device during inference. This may be appropriate in a hosted GPU runtime, but device switching should be tested under the actual deployment environment.

#### `src/convnext_adapter.py`

Responsibilities:

1. Builds a ConvNeXt-Base architecture with a 1024-dimensional output followed by a 2-class linear head.
2. Loads the local checkpoint with `torch.load(..., weights_only=True)`.
3. Loads weights with `strict=True` so architecture/checkpoint mismatches fail rather than silently passing.
4. Applies resize, center crop, tensor conversion, and ImageNet normalization.
5. Runs inference under `torch.inference_mode()`.
6. Uses softmax to compute real/fake probabilities.
7. Returns a standardized `DetectorResult` or an `ERROR` result.

Potential code-review item: strict loading verifies state-dictionary compatibility, not whether the checkpoint's preprocessing, label mapping, and architecture match the original research implementation. Those details should be cross-checked against the source repository/checkpoint documentation.

## 5. Model preprocessing and output comparison

| Detector | Preprocessing described in code/docs | Output conversion | Decision |
|---|---|---|---|
| UniversalFakeDetect | Resize/crop 224 × 224; CLIP normalization | Single logit → sigmoid | Fake probability ≥ 0.50 |
| ViT | Hugging Face processor; 224 × 224, rescale, normalize with mean/std 0.5 | Two logits → softmax | Fake probability ≥ 0.50 |
| Chen2024 ConvNeXt | Resize 224, center crop 224, tensor conversion, ImageNet normalization | Two logits → softmax | Fake probability ≥ 0.50 |

These detectors do not necessarily measure the same learned cues. Their probability values should not be assumed to be calibrated or directly comparable simply because all are in the range [0, 1].

The raw logit in the ViT/ConvNeXt adapters is defined as `fake_logit - real_logit`. UniversalFakeDetect uses a single output logit. These are related decision scores, but their scales and calibration differ.

## 6. What the available benchmark says

The documented five-sample diagnostic comparison for UniversalFakeDetect and ViT reports:

| Sample | Ground truth | UniversalFakeDetect | ViT |
|---|---|---|---|
| `real_001` | REAL | REAL | REAL |
| `ai_001` | FAKE | REAL | REAL |
| `manip_001` | FAKE | REAL | REAL |
| `corr_001` | Invalid/corrupt | ERROR | ERROR |
| `unsupp_001` | Unsupported/non-image | ERROR | ERROR |

Reported diagnostic summary:

- Same discrete result on 5/5 cases.
- Both models rejected the two invalid inputs.
- Both missed the two known fake/manipulated examples in this tiny sample.

**Interpretation:** 5/5 agreement is not 100% detection accuracy. The three valid images include one real and two fake examples; both detectors misclassified both fake examples as real. If these three valid examples are treated as the full labeled set, their accuracy on valid inputs is 1/3 (33.3%). This is only a tiny diagnostic set and is not a reliable estimate of real-world generalization.

The documentation also reports an example UniversalFakeDetect output of about 99.92% confidence for REAL on a modern AI-generated image. That result illustrates overconfident failure under distribution shift.

### ConvNeXt benchmark status

The visible excerpts confirm the ConvNeXt adapter and benchmark result file names, but do not provide a complete, auditable confusion matrix or full independent evaluation for ConvNeXt in the text supplied. Do not claim that ConvNeXt outperformed ViT or the original `df_model.pt` until its actual result file is inspected and the same labeled test set is used.

### What is needed for a valid performance claim

Use a held-out, labeled dataset with enough examples from each relevant category:

- Authentic camera photographs.
- Face swaps and facial reenactments.
- GAN-generated images.
- Diffusion-generated images from multiple generators.
- Local edits/inpainting/splicing.
- Compressed, resized, or otherwise transformed media.

Report, at minimum:

- Sample count and class balance.
- Accuracy.
- Precision, recall, and F1 for the FAKE class.
- False-real rate (fake inputs predicted REAL).
- False-fake rate (real inputs predicted FAKE).
- Confusion matrix.
- Per-category results.
- Inference latency and device.
- Threshold and calibration method.

Avoid data leakage between training, validation, and test sets. A fair comparison requires the same held-out images and ground-truth labels for every detector.

## 7. Why model diversity was pursued

The motivation is sound: detectors trained on different datasets or using different architectures may learn different artifacts. A single detector can perform well on the distribution it learned while failing on new generators or manipulation methods.

- **ViT:** uses patch-based self-attention to model global and local relationships across image patches.
- **ConvNeXt:** uses a modern convolutional architecture with hierarchical feature extraction and strong spatial inductive bias.
- **CLIP ViT-L/14 + linear probe:** reuses a large pretrained visual representation and applies a small classifier trained to separate real/fake examples.

Different architecture families may provide useful diversity, but architecture diversity alone does not guarantee complementary errors. Their predictions must be tested on the same examples to determine whether combining them actually helps.

## 8. Ensemble design: what exists and what remains planned

### Existing

- Common result schema.
- Independent detector adapters.
- Registry for independent model discovery and retrieval.
- Benchmark/evaluation documentation.
- Individual model scores and diagnostic comparison in the documented evaluation plan.

### Not established as implemented by the inspected evidence

- A production ensemble endpoint that always runs all detectors.
- A calibrated weighted average of model probabilities.
- A learned meta-classifier or feature-level fusion model.
- A final agreement/voting layer.
- A robust composite `UNCERTAIN` decision rule.
- A verified accuracy improvement over `df_model.pt`.

### Recommended safe next step

Before implementing fusion, create one evaluation table where every row is one labeled image and each detector contributes its prediction and fake probability. Examine:

1. Which errors each model makes.
2. Whether their errors are complementary.
3. Whether probabilities are calibrated on a held-out validation set.
4. How a proposed combination performs on a separate untouched test set.

Only then choose a fusion rule. Do not average uncalibrated scores merely because the values are probabilities. A conservative initial policy can report individual detector results and mark disagreement as `UNCERTAIN`, but it should be validated before production use.

## 9. Reliability, security, and deployment review

### Positive design choices visible in the code

- Shared result schema simplifies logging and comparisons.
- Lazy loading avoids unnecessary initialization at import time.
- Adapter boundaries isolate model-specific preprocessing.
- `torch.inference_mode()` avoids gradient-tracking overhead.
- Explicit RGB conversion reduces channel-format inconsistencies.
- Invalid inputs are represented as errors rather than silently treated as real/fake.
- ConvNeXt checkpoint loading uses `weights_only=True` and strict state-dictionary loading.
- Per-detector error results can help isolate a model failure from other evaluation work.

### Risks / follow-up checks

1. **Model loading failures:** Confirm the application surfaces `load_error` clearly and does not silently return a default class.
2. **Memory pressure:** ViT, CLIP ViT-L/14, and ConvNeXt models can consume substantial RAM/VRAM. Test sequential versus concurrent loading on the actual hosting hardware.
3. **Label mapping:** Validate each checkpoint's class indices against its authoritative model config or source code. A reversed mapping can invalidate all results.
4. **Preprocessing fidelity:** Reproduce each model's official transforms exactly. Small differences can alter predictions.
5. **Probability calibration:** A score of 0.95 from one model is not automatically equivalent to 0.95 from another.
6. **Out-of-distribution content:** The models may fail on new generators, heavy compression, screenshots, crops, and localized manipulations.
7. **Checkpoint provenance:** Record source URL, version/revision, license, file hash, and download date for each weight file.
8. **Licensing:** Review model/code/checkpoint licenses before public or commercial deployment, especially for research benchmark checkpoints and non-commercial datasets.
9. **Input security:** Enforce upload size/type limits, safe decoding, timeouts, and resource caps at the web/API boundary.
10. **Evaluation integrity:** Keep test images separate from any model selection or threshold tuning.

## 10. Recommended repository map

The following map is based on names surfaced in the project listing. Not every file's full contents were inspected.

```text
multi model connection/
├── benchmark/
│   └── results/
│       ├── all_detectors...
│       ├── chen2024...
│       ├── detector...
│       └── vit_results...
├── docs/
│   ├── DETECTOR_3_CANDIDATE_ANALYSIS.md
│   ├── DETECTOR_BENCHMARK_RESULTS.md
│   ├── FUTURE_DETECTOR_REQUIREMENTS...
│   ├── MULTI_MODEL_EVALUATION_PLAN.md
│   └── UNIVFD_LIMITATIONS.md
├── models/
│   ├── deepfake_vs_real_image_detection/
│   └── chen2024_convnext/
├── scripts/
│   ├── run_detector_benchmark...
│   ├── test_convnext_adapter...
│   ├── test_detector_contract...
│   ├── test_detector_registry...
│   └── test_vit_adapter.py
├── src/
│   ├── detector_schema.py
│   ├── detector_registry.py
│   ├── vit_adapter.py
│   ├── convnext_adapter.py
│   └── univfd_adapter.py
└── UniversalFakeDetect/
    ├── train.py
    ├── models/
    └── networks/
```

Confirm exact names and current contents locally before treating this as a complete inventory. The original `df_model.pt` path and its connection to this repository remain to be traced.

## 11. Suggested audit commands

Run these in PowerShell from the project root to verify current state.

### A. Show all relevant files with full names

```powershell
Get-ChildItem -Recurse -File |
  Where-Object {
    $_.FullName -notmatch '\\(\.git|venv|\.venv|node_modules)\\'
  } |
  Select-Object FullName, Length
```

### B. Inspect the benchmark result files

```powershell
Get-ChildItem .\benchmark\results -File |
  ForEach-Object {
    "`n===== $($_.FullName) ====="
    if ($_.Extension -in ".json", ".csv", ".md", ".txt", ".log") {
      Get-Content $_.FullName -Raw
    } else {
      "Binary or unsupported text format; inspect separately."
    }
  }
```

### C. Locate the original checkpoint and its references

```powershell
Get-ChildItem -Recurse -File |
  Where-Object { $_.Name -match 'df_model\.pt|\.pth$|\.safetensors$' } |
  Select-Object FullName, Length

Get-ChildItem -Recurse -File -Include *.py,*.md,*.json,*.yml,*.yaml |
  Select-String -Pattern 'df_model\.pt|ResNeXt|LSTM|ensemble|accuracy|confusion_matrix' |
  Select-Object Path, LineNumber, Line
```

### D. Run tests in the project's intended Python environment

First inspect `requirements.txt`, `pyproject.toml`, or the environment documentation. Then run the relevant tests using that environment, for example:

```powershell
python -m pytest .\scripts -v
```

This command is only appropriate if the repository's tests are compatible with pytest and dependencies are installed. Save the full output as evidence. A passing adapter/contract test establishes integration behavior, not detection accuracy.

## 12. Prioritized action plan

### Priority 1 — Establish the baseline
- Find the original `df_model.pt` file and inference code.
- Record architecture, class mapping, transforms, and dataset.
- Re-run it on the same fixed, labeled test set as the candidate models.

### Priority 2 — Verify all candidate detectors
- Confirm the ViT and ConvNeXt checkpoints exist and load.
- Run adapter contract tests.
- Verify class-index mapping and preprocessing.
- Record per-image outputs and errors.

### Priority 3 — Produce a reproducible benchmark
- Use a balanced, held-out test set.
- Calculate confusion matrices and class-specific metrics.
- Include modern diffusion and manipulation examples.
- Save dataset manifest, predictions, metrics, environment, and model hashes.

### Priority 4 — Decide whether to ensemble
- Check model error overlap.
- Calibrate scores only using a separate validation set.
- Compare each single model against proposed fusion on a final test set.
- Keep a disagreement/uncertainty path.

### Priority 5 — Document production integration
- Trace the web application entry point and API routes.
- Establish which detector(s) are actually invoked by the UI.
- Document deployment memory, timeout behavior, and failure responses.

## 13. Interview-ready explanation

> “The first version used a single trained checkpoint, `df_model.pt`. It worked reasonably well on the data distribution it had been tested on, but we wanted to reduce dependence on one model's learned artifacts. We preserved the original baseline and developed a modular evaluation framework. Each detector has its own adapter and preprocessing pipeline, but all return the same `DetectorResult` schema. We integrated a fine-tuned Vision Transformer and a ConvNeXt-based detector, alongside UniversalFakeDetect, so we could evaluate their behavior independently. The next engineering step is to compare them on the same labeled test set and validate whether their errors are complementary. We do not claim that simply registering multiple models improves accuracy; an ensemble needs a separately tested and calibrated fusion strategy.”

Do not state a percentage improvement unless a reproducible benchmark against the baseline supports it.

## 14. Final audit verdict

**Current status: Modular multi-detector integration and evaluation framework; ensemble improvement not proven by the supplied evidence.**

The project has useful engineering structure: model adapters, a shared result contract, a lazy registry, benchmark scripts, and limitation documentation. The key unresolved question is not whether multiple model files exist, but whether the candidate detectors reliably improve detection on a representative, labeled test set compared with the original `df_model.pt` baseline.

The most important next deliverable is a reproducible comparison of the baseline, ViT, ConvNeXt, and UniversalFakeDetect using identical test examples, followed by a separate evaluation of any proposed fusion layer.
