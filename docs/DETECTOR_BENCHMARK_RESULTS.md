# Multi-Model Detector Benchmark Results & Comparative Analysis

**Target Space**: [`https://huggingface.co/spaces/Thiru1122/Multimodel_setup`](https://huggingface.co/spaces/Thiru1122/Multimodel_setup)  
**Evaluated Detectors**: 
- UniversalFakeDetect (`UnivFDAdapter`, CLIP ViT-L/14 + Linear Probe)
- Vision Transformer (`ViTAdapter`, dima806/deepfake_vs_real_image_detection, ViT-Base/16)

> [!IMPORTANT]
> **Independence and Authenticity Notice:**  
> Both detectors were evaluated in complete isolation. Predictions are not combined into an ensemble, and no agreement layer is implemented. High confidence from either detector does NOT guarantee authenticity.

---

## 1. Verified Model Specifications

| Parameter | UniversalFakeDetect (`UnivFDAdapter`) | Vision Transformer (`ViTAdapter`) |
| :--- | :--- | :--- |
| **Model ID / Source** | `UniversalFakeDetect` (WisconsinAIVision) | `dima806/deepfake_vs_real_image_detection` |
| **Architecture** | CLIP `ViT-L/14` Backbone + `nn.Linear(768, 1)` | `ViTForImageClassification` (`google/vit-base-patch16-224`) |
| **Checkpoint Path** | `UniversalFakeDetect/pretrained_weights/fc_weights.pth` | `models/deepfake_vs_real_image_detection/model.safetensors` |
| **Checkpoint Size** | 4,083 bytes (probe head) | 343,223,968 bytes (~327.32 MB) |
| **Checkpoint Format**| PyTorch `.pth` | SafeTensors (`.safetensors`), float32 |
| **Parameters** | ~304M (CLIP) + 769 (linear probe) | 85,800,194 parameters |
| **Input Dimensions** | $3 \times 224 \times 224$ (RGB) | $3 \times 224 \times 224$ (RGB) |
| **Preprocessing** | `Resize(224, bicubic) -> CenterCrop(224) -> Normalize(clip_mean, clip_std)` | `Resize(224) -> Rescale(1/255) -> Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` |
| **Output Format** | Single raw logit $\in (-\infty, +\infty)$, Sigmoid to $P(\text{fake})$ | 2 logits $\in \mathbb{R}^2$, Softmax to $[P(\text{real}), P(\text{fake})]$ |
| **Label Mapping** | $0 = \text{REAL}, 1 = \text{FAKE}$ | $0 = \text{Real}, 1 = \text{Fake}$ |
| **Threshold** | $0.50$ (logit $0.00$) | $0.50$ (relative logit $0.00$) |
| **Confidence** | $P(\text{pred})$: $P(\text{fake})$ if Fake else $(1 - P(\text{fake}))$ | $P(\text{pred})$: $P(\text{fake})$ if Fake else $P(\text{real})$ |
| **Memory Footprint** | ~1.7 GB (CLIP weights + forward pass) | ~400–500 MB RAM/VRAM |
| **Schema Compliant** | `DetectorResult` in `src/detector_schema.py` | `DetectorResult` in `src/detector_schema.py` |

---

## 2. Independent Benchmark Evaluation Results

All test samples were evaluated independently without cross-model dependency or score blending:

| Sample ID | Category | Explicit Label | UnivFD Prediction | UnivFD Fake Prob | ViT Prediction | ViT Fake Prob | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `real_001` | Real Image | **REAL** | **REAL** (100.0%) | `0.000018` | **REAL** (80.2%) | `0.197797` | **AGREE** |
| `ai_001` | Diffusion AI | **FAKE** | **REAL** (99.9%) | `0.000828` | **REAL** (65.8%) | `0.342434` | **AGREE** |
| `manip_001`| Manipulated | **FAKE** | **REAL** (99.4%) | `0.006011` | **REAL** (95.1%) | `0.049064` | **AGREE** |
| `corr_001` | Corrupted | **CORRUPTED** | **ERROR** (N/A) | `None` | **ERROR** (N/A) | `None` | **AGREE** |
| `unsupp_001`| Non-Image | **UNSUPPORTED**| **ERROR** (N/A) | `None` | **ERROR** (N/A) | `None` | **AGREE** |

### Summary Statistics
- **Total Diagnostic Samples**: 5
- **Identical Discrete Classifications**: 5 / 5 (100.0%)
- **Disagreements**: 0 / 5 (0.0%)
- **Input Rejection Compliance**: 2 / 2 (100.0% rejected malformed/invalid inputs with `ERROR`)

---

## 3. Comparative Observations & Detailed Nuances

While both models agreed on the discrete binary classifications at the $0.50$ threshold, significant differences in continuous probability estimates were observed:

### 1. Known Real Photograph (`real_001`)
- **UnivFD**: Emits an extremely confident negative logit ($-10.9329$), calculating $P(\text{fake}) \approx 0.0018\%$ and $P(\text{real}) \approx 99.9982\%$.
- **ViT**: Emits logit $-1.4001$, calculating $P(\text{fake}) = 19.78\%$ and $P(\text{real}) = 80.22\%$.
- **Observation**: Both correctly classify the natural smartphone photograph as `REAL`. UnivFD's CLIP linear probe produces a more extreme logit saturation compared to ViT's balanced cross-entropy head.

### 2. Modern AI-Generated Diffusion Image (`ai_001` - Imagen 3)
- **UnivFD**: Predicts `REAL` with $99.9\%$ confidence ($P(\text{fake}) = 0.0828\%$, logit $-7.0961$).
- **ViT**: Predicts `REAL` with $65.8\%$ confidence ($P(\text{fake}) = 34.24\%$, logit $-0.6525$).
- **Observation**: 
  - Neither detector classified this modern diffusion image as `FAKE` at the default $0.50$ threshold.
  - However, **ViT assigned a substantially higher fake probability ($34.24\%$ vs $0.08\%$)**, indicating that ViT detected far more unnatural frequency/texture artifacts than UnivFD's CLIP probe.
  - UnivFD's training on 2018 ProGAN artifacts renders it largely blind to modern diffusion generation, projecting it as a natural image manifold.

### 3. Digitally Manipulated Spliced Composite (`manip_001`)
- **UnivFD**: Predicts `REAL` with $99.4\%$ confidence ($P(\text{fake}) = 0.6011\%$, logit $-5.1081$).
- **ViT**: Predicts `REAL` with $95.1\%$ confidence ($P(\text{fake}) = 4.9064\%$, logit $-2.9643$).
- **Observation**: Both detectors misclassify local digital composite splicing as `REAL` when global image composition appears coherent. Neither model was specifically trained with localized spatial localization heads.

### 4. Rejection of Corrupted and Unsupported Inputs (`corr_001`, `unsupp_001`)
- Both adapters intercepted invalid files before inference and returned `prediction: "ERROR"`.
- This confirms that invalid inputs will never silently produce arbitrary confidence scores or misleading `REAL`/`FAKE` classifications.

---

## 4. Contract and Independence Verification

- **Contract Tests**: Verified by [`scripts/test_detector_contract.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/test_detector_contract.py). Both adapters strictly implement the `DetectorResult` schema with bounded probabilities in $[0.0, 1.0]$ summing to $1.0$.
- **No Agreement Layer**: No score averaging, voting, or agreement arbitration exists. Both models execute and log independently.
- **UI and Space Integrity**: `app.py` remains completely untouched, serving the single-detector Gradio UI as originally verified.
