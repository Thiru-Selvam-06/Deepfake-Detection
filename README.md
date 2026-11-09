---
title: Multimodel Setup
emoji: 🦀
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.20.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# Multi-Model Deepfake Detection & Cross-Generator Evaluation

A multi-model deepfake detection framework that combines specialized computer vision architectures to detect AI-generated and manipulated images across diverse generator families (GANs, Diffusion, Midjourney, StyleGAN, ADM).

---

## 💡 The Core Idea

Single deepfake detectors suffer from **generator overfitting**:
- A model trained on facial GANs (like StyleGAN) often fails on diffusion models (like Midjourney).
- A model trained on general diffusion artifacts might miss fine-grained facial manipulations.

**Our Solution:** 
Instead of relying on a single fallible detector, we run multiple complementary vision models with distinct inductive biases and combine their outputs using **deterministic decision fusion**.

```mermaid
flowchart TD
    IMG["Input Image"] --> P1["ViT Preprocessor (224x224 Resize)"]
    IMG --> P2["UnivFD Preprocessor (Native CenterCrop + CLIP Norm)"]
    IMG --> P3["ConvNeXt Preprocessor (Native CenterCrop + ImageNet Norm)"]

    P1 --> M1["ViT Detector<br/>(Facial & GAN Artifacts)"]
    P2 --> M2["UnivFD<br/>(CLIP ViT-L/14 + Linear Probe)"]
    P3 --> M3["ConvNeXt DRCT<br/>(Diffusion Contrastive Base)"]

    M1 -->|Fake Prob / Vote| FUSION{"Decision Fusion Engine<br/>(src/decision_fusion.py)"}
    M2 -->|Fake Prob / Vote| FUSION
    M3 -->|Fake Prob / Vote| FUSION

    FUSION --> DECISION["Final Output:<br/>• Prediction: REAL / FAKE<br/>• Fused Confidence Score<br/>• Per-Model Explanation"]
```

---

## 🔍 The Base Detectors

| Model | Architecture | Specialization / Training | Key Strength |
|---|---|---|---|
| **ViT** | Vision Transformer (`dima806`) | Facial deepfakes & GAN faces | High sensitivity on facial boundaries and warping |
| **UnivFD** | CLIP ViT-L/14 + Linear Classifier | Multi-generator frequency & semantic cues | Universal zero-shot artifact sensitivity |
| **ConvNeXt DRCT** | ConvNeXt-Base (CVPR 2024) | Diffusion Reconstruction Contrastive Training | Outstanding generalization to diffusion (100% on Midjourney) |

---

## ⚙️ How It Works: Inference & Fusion Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as Web UI / API (app.py)
    participant Reg as Detector Registry
    participant Models as Base Detectors (ViT, UnivFD, ConvNeXt)
    participant Fusion as Decision Fusion

    User->>App: Upload Image
    App->>Reg: Request Predictions
    par Run Detectors in Parallel / Safe Sequential
        Reg->>Models: ViT Inference
        Reg->>Models: UnivFD Inference
        Reg->>Models: ConvNeXt DRCT Inference
    end
    Models-->>Reg: Individual Probabilities & Votes
    Reg->>Fusion: Submit {ViT: p1, UnivFD: p2, ConvNeXt: p3}
    Note over Fusion: Apply Strategy (Majority Vote / Mean Prob / Priority)
    Fusion-->>App: Unified Fused Verdict + Confidence Breakdown
    App-->>User: Display Results, Charts & Confidence
```

### Fusion Strategies
1. **Majority Voting:** Collects discrete votes (`REAL` vs `FAKE`). If a tie occurs, falls back to probability thresholding.
2. **Mean Fake Probability:** Computes the unweighted average of calibrated fake probabilities across all active detectors.
3. **Domain Priority:** Selects primary detector based on input context (e.g. facial vs non-facial diffusion synthesis).
4. **Graceful Degradation:** If any detector fails or checkpoint is absent, the engine falls back gracefully without crashing.

---

## 📊 Evaluation & Parity Verification

Our framework was evaluated against both facial GAN benchmarks and external diffusion datasets (Midjourney v5, ADM, ImageNet nature photos):

```mermaid
flowchart LR
    A["Test Datasets"] --> B["Phase 6: FFHQ vs StyleGAN (Facial)"]
    A --> C["Phase 7: Midjourney + ADM vs ImageNet (Diffusion)"]
    
    B --> D["ViT: 99.5% Acc<br/>ConvNeXt: 82.5% Acc"]
    C --> E["ConvNeXt DRCT: 77.0% Acc (100% Midjourney Recall)<br/>UnivFD: 57.5% Acc<br/>ViT: 52.0% Acc"]
```

- **Reference Parity:** All adapters match official paper implementations (CVPR 2023 UniversalFakeDetect and CVPR 2024 DRCT) with numerical differences $< 10^{-6}$.
- **Comprehensive Regression Tests:** Automated test suite with 25 unit and contract tests verifying preprocessing shapes, normalization, and fusion edge cases.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Thiru-Selvam-06/Deepfake-Detection.git
cd Deepfake-Detection
pip install -r requirements.txt
```

### 2. Run the Interactive Web UI
```bash
python app.py
```
Open your browser at `http://localhost:7860`.

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```text
├── app.py                   # Gradio Web Interface
├── src/
│   ├── convnext_adapter.py  # DRCT Native Crop Preprocessing & Inference
│   ├── univfd_adapter.py    # UniversalFakeDetect CLIP Preprocessing & Inference
│   ├── vit_adapter.py       # Vision Transformer Adapter
│   ├── decision_fusion.py   # Multi-Model Voting & Probability Fusion Logic
│   └── detector_registry.py # Unified Lifecycle & Model Registry
├── benchmark/               # Labeled test manifests and evaluation results
├── tests/                   # Regression and unit tests for adapters and fusion
├── docs/                    # Detailed scientific evaluation reports
└── models/                  # Pretrained model weights and checkpoints (Git LFS)
```
