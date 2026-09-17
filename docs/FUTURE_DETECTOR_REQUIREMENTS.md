# Requirements Specification for Future Secondary Detector Integration

**Target Space**: [`https://huggingface.co/spaces/Thiru1122/Multimodel_setup`](https://huggingface.co/spaces/Thiru1122/Multimodel_setup)  
**Purpose**: Defines the mandatory technical specifications and artifacts required before any secondary detector (e.g. ViT, RAID, or other architecture) can be integrated into the multi-model benchmark framework.

---

## Required Specification Checklist

To prevent integration failures, unverified assumptions, or runtime crashes, any candidate detector proposed for integration must provide complete, verified answers to the following 14 requirements:

### 1. Repository & Code Source
- Exact public GitHub or Hugging Face repository URL.
- Specific commit SHA or release tag to pin for reproducibility.

### 2. Software & Model License
- Formal license type (e.g., MIT, Apache 2.0, Non-Commercial CC-BY-NC).
- Confirmation of compatibility with Hugging Face Spaces hosting.

### 3. Model Architecture & Backbone
- Base neural network architecture (e.g., `google/vit-base-patch16-224`, `timm/vit_large_patch14_224`, `ResNet50`).
- Any custom classification neck, pooling layer, or multi-scale projection heads.

### 4. Checkpoint Location & Integrity
- Accessible public download URL or Hugging Face model hub identifier (`repo_id`).
- SHA256 checksum of the weight file.
- Exact filename and format (e.g., `pytorch_model.bin`, `model.safetensors`, `model.pth`).
- Confirmation whether the checkpoint contains the full model or classification head only.

### 5. Input Modality
- Explicit confirmation that the model evaluates single static images (or frame sequences).

### 6. Input Dimensions & Aspect Ratio
- Target tensor resolution (e.g., $224 \times 224$, $384 \times 384$, or variable).
- Aspect ratio handling (e.g., isotropic scaling with padding vs. anisotropic bicubic resize vs. center cropping).

### 7. Preprocessing Transform Pipeline
- Exact color space (RGB, BGR, or Grayscale).
- Resizing interpolation mode (Bilinear, Bicubic, Lanczos).
- Cropping strategy (Center crop, random crop, or no crop).
- Tensor scaling (range `[0.0, 1.0]` vs. `[-1.0, 1.0]` vs. `[0, 255]`).
- Exact channel mean and standard deviation vectors for normalization.

### 8. Output Format & Tensor Shape
- Number of output logits:
  - Binary single scalar logit (`[batch, 1]`), OR
  - Two-class logits (`[batch, 2]` for `[real_logit, fake_logit]`), OR
  - Softmax probability distribution.

### 9. Label Mapping
- Explicit class indexing:
  - Which index/sign corresponds to **REAL** (e.g., index `0` or negative logit).
  - Which index/sign corresponds to **FAKE** (e.g., index `1` or positive logit).

### 10. Operational Decision Threshold
- Author-defined threshold value (e.g., $0.50$, $0.65$, or logit $0.00$).
- Calibration details (whether Platt scaling or temperature scaling is required).

### 11. Confidence Calculation Formula
- Mathematical formula used to compute user-facing confidence percentage from logits/probabilities.

### 12. Device & Memory Footprint
- Peak GPU VRAM required during batch=1 inference.
- Host system RAM required to load weights into memory.
- Compatibility with Hugging Face ZeroGPU (`zero-a10g`) and PyTorch CPU execution.

### 13. Inference Function & Entry Point
- Python function signature for single-image execution (e.g., `predict(image: PIL.Image) -> float`).
- Whether model loading can be decoupled and executed lazily once at startup.

### 14. Dependency Requirements & Conflicts
- Required Python library versions (`transformers`, `timm`, `torchvision`, etc.).
- Confirmation of no conflicting dependencies with existing UniversalFakeDetect requirements.

---

## 15. Labeled Benchmark Evaluation Data

Before performing comparative benchmarking, a curated set of test images with documentary proof of ground-truth origin must be supplied:
- Verified natural photographs (diverse camera sensors, lighting, and subjects).
- Verified legacy GAN synthetic images (ProGAN, StyleGAN2/3).
- Verified modern diffusion synthetic images (Google Imagen, Stable Diffusion XL, Midjourney v6, Flux).
- Localized image manipulations (splicing, inpainting).
