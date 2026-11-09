# Detector #3 Candidate Discovery & Architectural Feasibility Analysis

**Target Space**: [`https://huggingface.co/spaces/Thiru1122/Multimodel_setup`](https://huggingface.co/spaces/Thiru1122/Multimodel_setup)  
**Status**: Pre-Implementation Candidate Research (No Code or Weights Modified)

> [!IMPORTANT]
> **Candidate Discovery Notice:**  
> This report evaluates potential candidate models for Detector #3 based exclusively on documented technical characteristics from official sources. No model has been selected, downloaded, or implemented. Implementation remains blocked pending explicit user approval.

---

## 1. Candidate Model Profiles

### Candidate 1: `prithivMLmods/Deep-Fake-Detector-Model` (SigLIP Image Classifier)

- **Name**: Deep-Fake-Detector-Model (SigLIP-Base/16)
- **Repository**: [`https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model)
- **Architecture**: `SiglipForImageClassification` (`model_type: "siglip"`). Fine-tuned from Google SigLIP (`google/siglip-base-patch16-512` / `google/siglip2-base-patch16-224`) with a dedicated binary classification head.
- **Checkpoint**: `model.safetensors` (Exact size: `371,568,640` bytes ~354.35 MB).
- **Checkpoint Format**: SafeTensors (`.safetensors`), `float32` precision.
- **Input Resolution**: $3 \times 224 \times 224$ (RGB format).
- **Preprocessing**: `SiglipImageProcessor` (via Hugging Face `transformers`):
  - Resize: $224 \times 224$ (`resample=2`, bicubic interpolation).
  - Rescale: $1/255 \approx 0.00392157$.
  - Normalization: Mean `[0.5, 0.5, 0.5]`, Std `[0.5, 0.5, 0.5]` (maps $[0, 1] \to [-1, 1]$).
- **Output Format**: 2 raw logits (`torch.Size([1, 2])`), normalized via Softmax to $[P(\text{Fake}), P(\text{Real})]$.
- **Label Mapping**:
  - Index `0`: `"Fake"`
  - Index `1`: `"Real"`
- **Decision Threshold**: `0.50` fake probability ($P(\text{Fake}) \ge 0.50 \implies \text{FAKE}$, else $\text{REAL}$).
- **Inference Entry Point**: `AutoModelForImageClassification` + `AutoImageProcessor` under `torch.inference_mode()`.
- **License**: **Apache 2.0** (Open commercial and research use).
- **Dependencies**: `torch`, `transformers`, `safetensors`, `pillow`. *(All 4 dependencies are already installed and verified in the current Space environment).*
- **Resource Requirements**:
  - Checkpoint size: 354.35 MB.
  - Runtime RAM / VRAM: ~354 MB static weights + ~50–100 MB forward activation tensors.
- **Domain**: Modern synthetic media and facial deepfakes (trained on the `OpenDeepfake-Preview` dataset, encompassing recent generative media).
- **Known Limitations**: Optimized for synthetic face and portrait artifacts; generalization to open-domain non-facial landscape/macro imagery is unverified.
- **Verification Sources**: Hugging Face Model Card, `config.json`, `preprocessor_config.json`, direct HTTP HEAD headers.
- **Integration Risks**: Low. Uses standard Hugging Face pipeline matching the architecture pattern already proven in `src/vit_adapter.py`.

---

### Candidate 2: `umm-maybe/AI-image-detector` (Swin Transformer Image Classifier)

- **Name**: Swin AI Image Detector
- **Repository**: [`https://huggingface.co/umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector)
- **Architecture**: `SwinForImageClassification` (`model_type: "swin"`). Hierarchical Vision Transformer using shifted window self-attention (Swin-Base/Tiny configuration).
- **Checkpoint**: `pytorch_model.bin` (Exact size: `347,591,615` bytes ~331.49 MB).
- **Checkpoint Format**: PyTorch binary (`.bin`), `float32` precision.
- **Input Resolution**: $3 \times 224 \times 224$ (RGB format).
- **Preprocessing**: `AutoImageProcessor` (`ViTFeatureExtractor`):
  - Resize: $224 \times 224$ (`resample=3`, bicubic interpolation).
  - Rescale: $1/255$.
  - Normalization: ImageNet standard mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- **Output Format**: 2 raw logits (`torch.Size([1, 2])`), normalized via Softmax to $[P(\text{artificial}), P(\text{human})]$.
- **Label Mapping**:
  - Index `0`: `"artificial"` (Synthetic / AI-generated)
  - Index `1`: `"human"` (Authentic / Real)
- **Decision Threshold**: `0.50` probability on `"artificial"`.
- **Inference Entry Point**: `AutoModelForImageClassification` + `AutoImageProcessor` under `torch.inference_mode()`.
- **License**: **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
- **Dependencies**: `torch`, `transformers`, `pillow`. *(All dependencies already installed in Space).*
- **Resource Requirements**:
  - Checkpoint size: 331.49 MB.
  - Runtime RAM / VRAM: ~331 MB static weights + ~60 MB forward activation tensors.
- **Domain**: General AI-generated images vs. human photographs (trained on synthetic images from mid-2023 generative models).
- **Known Limitations**: Trained via AutoTrain on 2023-era synthetic data; sensitivity to newer late-2024 to 2026 diffusion architectures (e.g. Imagen 3, Flux) is limited. Checkpoint is in legacy `.bin` format rather than `.safetensors`.
- **Verification Sources**: Hugging Face Model Card, `config.json`, `preprocessor_config.json`, HTTP HEAD headers.
- **Integration Risks**: Low to Medium. Legacy `.bin` format triggers Hugging Face security warnings if loaded across untrusted environments, though loadable via standard PyTorch weights loader.

---

### Candidate 3: `CNNDetection` (Wang et al., CVPR 2020 — ResNet-50)

- **Name**: CNNDetection (Wang2020 Baseline)
- **Repository**: [`https://github.com/PeterWang512/CNNDetection`](https://github.com/PeterWang512/CNNDetection)
- **Architecture**: `ResNet-50` convolutional neural network backbone + custom binary linear head (`models/resnet.py`).
- **Checkpoint**: `blur_jpg_prob0.5.pth` (Exact size: ~97.8 MB).
- **Checkpoint Format**: PyTorch `.pth` state dictionary.
- **Input Resolution**: $3 \times 224 \times 224$ (cropped from $256 \times 256$).
- **Preprocessing**: `transforms.Resize(256)` followed by `transforms.CenterCrop(224)` and ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- **Output Format**: Single scalar logit ($\in \mathbb{R}$), mapped via Sigmoid to $P(\text{Fake})$.
- **Label Mapping**: `0 = Real`, `1 = Fake`.
- **Decision Threshold**: `0.50` probability (logit `0.00`).
- **Inference Entry Point**: Custom ResNet forward pass in `PeterWang512/CNNDetection`.
- **License**: **Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)** / Research Only.
- **Dependencies**: `torch`, `torchvision`, `pillow`.
- **Resource Requirements**: ~98 MB RAM/VRAM weights. Very low computational overhead.
- **Domain**: ProGAN (2018) with aggressive Gaussian blur and JPEG compression augmentation.
- **Known Limitations**:
  - Severe generalization failure on modern diffusion models (documented by the original authors and confirmed across multiple benchmarks).
  - Checkpoint hosting links on UC Berkeley servers are notoriously unstable and frequently experience 404/throttling errors.
  - **Restrictive Non-Commercial License**: Prohibits commercial deployment.
- **Verification Sources**: CVPR 2020 paper, official GitHub repository, `weights/download_weights.sh`.
- **Integration Risks**: High. Fragile weight download links, non-commercial licensing constraints, and known failure on modern diffusion images.

---

### Candidate 4: `RAID` Adversarial Evaluation Benchmark Suite (`pralab/RAID`)

- **Name**: RAID Adversarial Evaluation Benchmark Suite
- **Repository**: [`https://github.com/pralab/RAID`](https://github.com/pralab/RAID) & [`https://huggingface.co/aimagelab/RAID_ckpt`](https://huggingface.co/aimagelab/RAID_ckpt)
- **Architecture**: **NOT A MONOLITHIC DETECTOR MODEL**. RAID is an evaluation benchmark and dataset consisting of 72,000 adversarial examples attacking 7 distinct literature detectors. The checkpoint repository `aimagelab/RAID_ckpt` contains individual retrained checkpoints of disparate models:
  - `wang2020/model_best.pth.tar` (ResNet-50)
  - `ojha2023/model_best.pth.tar` (UniversalFakeDetect CLIP ViT-L/14)
  - `corvi2023/model_best.pth.tar` (DMimageDetection)
  - `chen2024_clip/last_acc0.9112.pth` (CLIP-based detector)
  - `chen2024_convnext/16_acc0.9993.pth` (ConvNeXt-based detector)
  - `pretrained_linear/siglip_retrain_d3/model_best.pth.tar`
- **Checkpoint**: Disparate individual `.pth.tar` files ranging from 97 MB to 1.7 GB.
- **Checkpoint Format**: Varied (`.pth.tar`, `.pth`).
- **Input Resolution**: `UNVERIFIED` as a single standard; varies by constituent model.
- **Preprocessing**: `UNVERIFIED` as a single standard; depends on selected underlying model.
- **Output Format**: Varied across constituent models.
- **Label Mapping**: Varied across constituent models.
- **Decision Threshold**: `UNVERIFIED` without per-model calibration curves.
- **Inference Entry Point**: Requires cloning the `pralab/RAID` repository framework and its custom submodule wrappers (`raid/models/`).
- **License**: Dataset on HF under `CC-BY-NC-4.0`. Code on GitHub.
- **Dependencies**: Requires `pralab/RAID` repository files and multiple third-party libraries.
- **Resource Requirements**: Highly variable depending on which constituent model is chosen.
- **Domain**: Adversarial robustness evaluation against text-to-image generators (SD 1.4, SD 2.1, SDXL, DeepFloyd IF).
- **Known Limitations**: **Cannot be integrated as a single "RAID detector"**. An engineer attempting to "install RAID" must choose which specific constituent model (e.g., Corvi2023 ConvNeXt, Chen2024 CLIP, or Wang2020 ResNet) to deploy.
- **Verification Sources**: arXiv:2506.03988, `aimagelab/RAID_ckpt` repository index, `pralab/RAID` repository.
- **Integration Risks**: Critical. "RAID" is a benchmark suite, not a standalone model. Attempting to integrate "RAID" directly without picking a specific underlying model would violate the requirement not to guess or invent architectures.

---

## 2. Factual Comparison Table

| Feature / Criteria | Candidate 1: SigLIP (`prithivMLmods`) | Candidate 2: Swin (`umm-maybe`) | Candidate 3: CNNDetection (`Wang2020`) | Candidate 4: RAID Suite (`pralab`) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type** | Standalone Classifier | Standalone Classifier | Standalone Classifier | Benchmark Suite of 7+ Models |
| **Architecture** | `SiglipForImageClassification` | `SwinForImageClassification` | `ResNet-50` | `UNVERIFIED` (Varies per sub-model) |
| **Checkpoint Path** | Hugging Face Hub (Public) | Hugging Face Hub (Public) | External Berkeley Server / Mirror | `aimagelab/RAID_ckpt` (Multiple) |
| **Checkpoint Format** | `.safetensors` | `.bin` | `.pth` | `.pth.tar` |
| **Checkpoint Size** | **354.35 MB** | **331.49 MB** | **~97.8 MB** | Varies (97 MB – 1.7 GB) |
| **Input Dimensions** | $3 \times 224 \times 224$ | $3 \times 224 \times 224$ | $3 \times 224 \times 224$ | `UNVERIFIED` (Varies) |
| **Preprocessing** | Resize $224 \times 224$, Norm $[-1, 1]$ | Resize $224 \times 224$, ImageNet Norm | Resize $256 \to$ Crop $224$, ImageNet Norm | `UNVERIFIED` (Varies) |
| **Output Head** | 2-class Softmax | 2-class Softmax | 1-class Sigmoid | Varies |
| **Label Mapping** | `0 = Fake`, `1 = Real` | `0 = artificial`, `1 = human` | `0 = Real`, `1 = Fake` | `UNVERIFIED` (Varies) |
| **Threshold** | `0.50` | `0.50` | `0.50` | `UNVERIFIED` |
| **License** | **Apache 2.0** | **CC-BY-4.0** | **CC BY-NC 4.0** (Non-Commercial) | **CC-BY-NC-4.0** (Dataset) |
| **Space Dependency Compatibility** | **100% Compatible** (`transformers`) | **100% Compatible** (`transformers`) | Requires custom `models/` code | Requires `pralab/RAID` repository clone |
| **Memory Footprint** | ~354 MB RAM/VRAM | ~332 MB RAM/VRAM | ~98 MB RAM/VRAM | Varies |

---

## 3. Integration Readiness Checklist

| Readiness Question | Candidate 1: SigLIP | Candidate 2: Swin | Candidate 3: CNNDetection | Candidate 4: RAID Suite |
| :--- | :---: | :---: | :---: | :---: |
| 1. Can it implement `DetectorResult` contract? | **YES** | **YES** | **YES** | Depends on selected sub-model |
| 2. Can it be loaded lazily in `detector_registry.py`? | **YES** | **YES** | **YES** | Requires complex submodule loading |
| 3. Operates independently from UnivFD? | **YES** | **YES** | **YES** | **YES** |
| 4. Operates independently from ViT? | **YES** | **YES** | **YES** | **YES** |
| 5. Requires changes to `app.py`? | **NO** | **NO** | **NO** | **NO** |
| 6. Requires changes to existing checkpoints? | **NO** | **NO** | **NO** | **NO** |
| 7. Requires incompatible dependency versions? | **NO** | **NO** | **NO** | Potential conflicts with submodules |
| 8. Requires additional large support files? | **NO** (Standalone) | **NO** (Standalone) | **NO** (Standalone) | **YES** (Full repo structure needed) |
| 9. Uses image input rather than video? | **YES** (Still image) | **YES** (Still image) | **YES** (Still image) | **YES** (Still image) |
| 10. License permits intended deployment? | **YES** (Apache 2.0) | **YES** (CC-BY-4.0) | **RESTRICTED** (Non-commercial) | **RESTRICTED** (Non-commercial) |

---

## 4. Key Analytical Insights & Clarifications

### Why RAID Cannot Be Selected As-Is
In academic and commercial literature, "RAID" refers to:
1. A benchmark dataset of 72,000 adversarial synthetic images (`aimagelab/RAID`).
2. An adversarial evaluation attack harness (`pralab/RAID`).
3. A collection of re-evaluated models from other authors (`aimagelab/RAID_ckpt`).

There is no single authoritative "RAID neural network". Attempting to integrate "RAID" without selecting a specific constituent model (e.g., `corvi2023/model_best.pth.tar` or `chen2024_convnext/16_acc0.9993.pth`) is technically undefined.

### Strongest Technical Candidate: SigLIP (`prithivMLmods/Deep-Fake-Detector-Model`)
- **Modern Backbone**: Leverages Google's SigLIP (Sigmoid Loss for Language Image Pre-training), providing an entirely different visual feature representation than CLIP ViT-L/14 (UnivFD) and standard ViT-Base (Detector #2).
- **Clean License**: Permissive Apache 2.0 license.
- **Native SafeTensors**: Distributed natively as `model.safetensors`, matching the zero-code-execution security posture of the Space.
- **Dependency Coexistence**: Runs directly within the existing `transformers` version already installed in the Space.

### Viable Alternative Candidate: Swin Transformer (`umm-maybe/AI-image-detector`)
- **Hierarchical Representation**: Uses shifted window attention (Swin), offering a multi-scale inductive bias distinct from both standard ViT and CNN architectures.
- **License**: CC-BY-4.0.
- **Caveat**: Uses legacy PyTorch `.bin` weights and older training distribution.

---

## 5. Candidates Requiring User Approval

The following candidates are presented for your explicit review and decision:

1. **Option A: `prithivMLmods/Deep-Fake-Detector-Model` (SigLIP Image Classifier)**  
   - *Pros*: Modern architecture (SigLIP), Apache 2.0 license, native SafeTensors (354 MB), full Transformers compatibility.
   - *Cons*: Primary training focus on facial deepfakes.
2. **Option B: `umm-maybe/AI-image-detector` (Swin Transformer)**  
   - *Pros*: Distinct hierarchical vision transformer architecture (Swin), CC-BY-4.0 license, 331 MB.
   - *Cons*: Older 2023 training distribution, PyTorch `.bin` checkpoint format.
3. **Option C: Specific constituent model from the RAID evaluation checkpoint repository** (e.g. `corvi2023` ConvNeXt or `chen2024_clip`).  
   - *Pros*: Directly benchmarked against adversarial attacks in the RAID paper.
   - *Cons*: Requires custom model loading harness from `pralab/RAID`, `.pth.tar` format, non-commercial restrictions.
4. **Option D: User-specified alternative model.**

---

## 6. Stop Condition & Explicit Approval Required

In accordance with strict instructions:
- **No model has been downloaded.**
- **No adapter code has been created.**
- **No configuration or checkpoint has been modified.**
- **No Git commit or push has been performed.**

### Exact Approval Required Before Proceeding:
Please confirm:
1. **Which candidate model** (e.g., Option A: SigLIP, Option B: Swin, Option C: Specific RAID constituent, or Option D: custom model) you approve for integration as Detector #3.
2. Once you provide explicit approval, the model files will be audited, an isolated `src/siglip_adapter.py` (or corresponding adapter) will be implemented to conform to `DetectorResult`, registered into `detector_registry.py`, tested under the contract suite, and evaluated independently on the benchmark without combining predictions.
