# Multi-Model Evaluation and Comparison Plan

**Target Space**: [`https://huggingface.co/spaces/Thiru1122/Multimodel_setup`](https://huggingface.co/spaces/Thiru1122/Multimodel_setup)  
**Status**: Architectural Planning Document (Preparation Phase — No Agreement Layer Implemented Yet)

---

## 1. Overview & Objective

No individual deepfake detection architecture generalizes universally across all generative algorithms (GANs, autoregressive transformers, latent diffusion, flow-matching) and manipulation techniques (inpainting, face-swapping, splicing).

The objective of this framework is to establish a rigorous, standardized, multi-model evaluation harness that allows multiple independent detectors to analyze identical input assets under consistent protocols, producing standardized diagnostic outputs and agreement metrics without modifying individual model behaviors.

---

## 2. Common Input Format

To ensure parity across all candidate models, all inputs entering the evaluation harness must conform to the following pipeline:

1. **Primary Input Types**:
   - `PIL.Image.Image` object, OR
   - Absolute file path string pointing to a local image file.
2. **Standard Ingress Transformation**:
   - Validation against corruption and truncated headers using `PIL.Image.open()`.
   - Explicit conversion to 3-channel RGB mode (`.convert("RGB")`).
   - Preservation of original image aspect ratio and native resolution metadata prior to model-specific preprocessing.
3. **Model-Specific Downstream Adapters**:
   - Each detector adapter is responsible for applying its own exact author-defined preprocessing (e.g., specific resizing interpolation, cropping dimensions, normalization means and standard deviations).

---

## 3. Common Result Schema

All candidate detectors must implement an adapter adhering strictly to the `DetectorResult` schema defined in [`src/detector_schema.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/detector_schema.py):

```python
@dataclass
class DetectorResult:
    detector_name: str           # Unique identifier of detector (e.g. "UniversalFakeDetect")
    model_name: str              # Backbone and checkpoint descriptor
    prediction: str              # Strictly one of: "REAL", "FAKE", "UNCERTAIN", "ERROR"
    fake_probability: float      # Normalized probability in [0.0, 1.0] that input is synthetic
    real_probability: float      # Normalized probability in [0.0, 1.0] that input is real
    raw_logit: float             # Unbounded pre-activation output logit
    confidence: float            # Confidence score in [0.0, 1.0]
    threshold: float             # Operational decision boundary (e.g., 0.50)
    device: str                  # Execution hardware ("cuda", "cpu")
    preprocessing: str           # Summary description of transform pipeline
    inference_time_ms: float     # Wall-clock execution time in milliseconds
    error: Optional[str]         # Descriptive error trace if prediction is "ERROR"
    warning: Optional[str]       # Mandatory authenticity limitation notice
```

Allowed prediction values are strictly bounded to:
- `REAL`: Detector classified input as natural photograph.
- `FAKE`: Detector classified input as synthetic / manipulated.
- `UNCERTAIN`: Prediction cannot be reliably assigned (e.g., logit falls in dead-band or model disagreement).
- `ERROR`: Input was invalid, corrupted, or pipeline encountered an unrecoverable exception.

---

## 4. Common Error Handling Protocol

1. **Rejection at Ingress**:
   - Corrupted files, non-image formats, and truncated byte streams must produce `prediction: "ERROR"` with a non-null `error` field.
2. **No Silent Error Fallback**:
   - Under no circumstances may an unhandled exception or decode failure default silently to `REAL` or `FAKE`.
3. **Hardware & OOM Isolation**:
   - If an individual detector exhausts GPU memory or times out, its adapter must return an `ERROR` result without crashing the enclosing process or aborting sibling detector evaluations.

---

## 5. Per-Model Reporting & Aggregation

When multiple detectors evaluate a single input, the evaluation harness must report:

1. **Individual Model Records**:
   - Tabular display of each detector's raw logit, calibrated fake probability, assigned binary label, and latency.
2. **Consensus & Disagreement**:
   - Quantitative concordance count (e.g., `2/3 detectors predict FAKE`).
   - Mean and variance of predicted fake probabilities across models.
3. **The `UNCERTAIN` State**:
   - When constituent models produce conflicting predictions (e.g., Detector A outputs `REAL` with 99% confidence, while Detector B outputs `FAKE` with 95% confidence), the composite verdict must be explicitly designated as **`UNCERTAIN`**.
   - The system must not arbitrate conflicting predictions by simple average if the underlying feature distributions are uncalibrated.

---

## 6. Rigorous Evaluation Protocol & Benchmark Requirements

Future evaluations across multiple models must adhere to the following experimental standards:

1. **Curated Ground-Truth Benchmark**:
   - Evaluation must be conducted on a balanced, labeled benchmark dataset containing verified real photographs, known GAN images (ProGAN, StyleGAN), modern diffusion images (Midjourney, Stable Diffusion, Imagen, Flux), and localized image manipulations.
2. **Separate Reporting of Error Types**:
   - Metrics must report False-Real (False Negative / Missed AI) and False-Fake (False Positive / Flagged Real) error rates independently.
   - Aggregate accuracy alone is misleading on imbalanced or out-of-distribution test sets.
3. **Calibration Requirements**:
   - Raw logits from distinct architectures cannot be compared directly without Platt scaling or temperature calibration on a shared validation set.

---

## 7. Fundamental Disclaimer on Model Agreement

> [!IMPORTANT]
> **Consensus $\neq$ Authenticity Verification:**  
> If all detectors in a multi-model ensemble classify an image as `REAL`, this indicates only that none of the models detected the specific synthetic patterns for which they were trained. It **does not verify** that the image is an authentic photograph. Generative models developed after a detector's training cutoff may bypass all detectors simultaneously.

---

## 8. Implementation Boundary Notice

**Current State**: Only UniversalFakeDetect and its standardized adapter (`UnivFDAdapter`) are active.  
**Strict Boundary**: The multi-model agreement layer, additional detector repositories (e.g., RAID), and ensemble fusion logic are **not implemented** in this phase.
