# Phase 7 — External AI-Generated Image Generalization Evaluation
## Stage 1 Audit & Proposed Dataset Plan

**Document Path:** [`docs/PHASE7_DATASET_AUDIT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_DATASET_AUDIT.md)  
**Date:** November 2026  
**Status:** Stage 1 Audit Completed — Pending User Review & Approval  
**Operating Mode:** Read-Only Audit (Zero model weights modified, zero inferences initiated, zero downloads started)

---

## 1. Executive Summary

This audit establishes the empirical foundation for **Phase 7: External AI-Generated Image Generalization Evaluation**. 

In Phase 6, our evaluation benchmark focused on **facial portraits** (FFHQ real faces vs. StyleGAN synthetic faces), where the facial ViT detector achieved 98.00% accuracy while ConvNeXt and UnivFD achieved ~51% accuracy (due to feature blindness to StyleGAN manifold structures). 

Phase 7 investigates whether our pretrained detectors can distinguish authentic photographs from AI-generated images produced by modern generator architectures (e.g., latent diffusion, pixel diffusion, and large-scale GANs) across **general object and scene categories**.

In strict accordance with the **Mandatory Stage 1** directive, no files have been downloaded and no inference runs have been triggered. This report audits model training provenance, assesses authoritative dataset sources and host storage constraints, verifies class distributions, and outlines a rigorous plan for Stage 2.

---

## 2. Model Training Provenance & Generator Family Overlap

We audited the training lineages of all three integrated detectors against source checkpoints, configuration files, and published literature:

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                    DETECTOR TRAINING PROVENANCE & DOMAIN MAP                                            |
+--------------------------+------------------------------+------------------------------+--------------------------------+
| Attribute                | ViT (dima806)                | UnivFD (Ojha et al., 2023)   | ConvNeXt / DRCT (Chen et al.)  |
+--------------------------+------------------------------+------------------------------+--------------------------------+
| Architecture             | ViT-Base/16 (85.8M params)   | CLIP ViT-L/14 + Linear Probe | ConvNeXt-Base + Linear Head    |
| Foundation Pretraining   | ImageNet-21k (Google, 2020)  | WebImageText 400M (CLIP 2020)| ImageNet-22k (timm convnext_b) |
| Fine-Tuning Task         | Facial Real vs Fake          | Cross-domain Synthetic Det.  | Contrastive Diffusion Rec.     |
| Real Images in Training  | FFHQ (Flickr human faces)    | LSUN (20 object categories)  | MS-COCO train2017 (objects)    |
| Fake Images in Training  | StyleGAN / StyleGAN2 faces   | ProGAN (20 LSUN categories)  | SD v1.4, SD-Inpaint, SD v2 rec |
| Checkpoint Path          | models/.../model.safetensors | UniversalFakeDetect/...      | models/.../16_acc0.9993.pth    |
| Input Dimensions         | 3 x 224 x 224                | 3 x 224 x 224                | 3 x 224 x 224                  |
| Seen: StyleGAN?          | YES (In-distribution)        | NO                           | NO                             |
| Seen: ProGAN?            | NO                           | YES (In-distribution)        | NO                             |
| Seen: Stable Diffusion?  | NO                           | NO                           | YES (v1.4 & v2 reconstruction) |
| Seen: Midjourney?        | NO (Unseen)                  | NO (Unseen)                  | NO (Unseen)                    |
| Seen: ADM?               | NO (Unseen)                  | NO (Unseen)                  | NO (Unseen)                    |
| Seen: BigGAN?            | NO (Unseen)                  | NO (Unseen in probe)         | NO (Unseen in training)        |
+--------------------------+------------------------------+------------------------------+--------------------------------+
```

### Detailed Lineage Findings:
1. **ViT (`dima806/deepfake_vs_real_image_detection`):**
   - Fine-tuned exclusively on Kaggle 140k faces (FFHQ vs. StyleGAN2).
   - **Has never seen diffusion models or general-scene AI generators.**
   - In general-scene domains, ViT is operating completely out-of-distribution.
2. **UnivFD (`CLIP ViT-L/14` + Linear Probe):**
   - Linear probe trained exclusively on ProGAN across 20 LSUN scene categories.
   - Backbone visual encoder (OpenAI CLIP ViT-L/14) was trained on 400M web pairs collected prior to mid-2020.
   - **Has never seen diffusion models (Midjourney, ADM) or StyleGAN during linear probe training.**
3. **ConvNeXt / DRCT (`16_acc0.9993.pth`):**
   - Trained contrastively on MS-COCO train2017 real photographs vs. **Stable Diffusion v1.4, SD Inpainting, and SD v2 reconstruction**.
   - **Stable Diffusion is in-distribution for ConvNeXt.**
   - **Has never seen Midjourney, ADM, or BigGAN in its training data.**
   - In published DRCT benchmarks ([`temp_DRCT_reference/README.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/temp_DRCT_reference/README.md)), Chen et al. demonstrated that ConvNeXt achieved high accuracy on unseen diffusion models but dropped to 59.81% on BigGAN.

---

## 3. Host Storage, Dependencies & External Dataset Sources

### 3.1 Host Environment
- **Operating System:** Windows 10/11 Enterprise (64-bit AMD64).
- **Available Storage:** **92.46 GB free** on drive `C:`.
- **Installed Dependencies:** `torch` (2.13.0+cpu), `torchvision`, `transformers` (5.17.0), `timm` (1.0.29), `pyarrow` (24.0.0), `huggingface_hub` (1.31.0), `PIL` (12.2.0), `pandas` (3.0.2), `scikit-learn` (1.9.0).
- *No additional package installations are required.*

### 3.2 Evaluation of Authoritative Dataset Sources
1. **GenImage Official Google Drive (`1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS`):**
   - Packaged as multi-part split zip archives: Midjourney spans `imagenet_midjourney.z01` to `.z50` (>100 GB); ADM spans `imagenet_ai_0508_adm.z01` to `.z12` (>50 GB).
   - Downloading multi-part archives would exhaust local disk storage (92.46 GB free) and is **technically unviable**.
2. **DiffusionDB (`poloclub/diffusiondb`):**
   - Contains millions of SD-generated images with text prompts, but **zero authentic photographs**. Pairing with external photo collections introduces uncontrollable confounding variables (camera sensor bias, compression artifacts, semantic mismatch). Unsuitable as a balanced benchmark.
3. **Curated GenImage Validation Split (`TheKernel01/Tiny-GenImage` on Hugging Face):**
   - Preserves official NeurIPS 2023 GenImage validation partition under **CC BY-NC-SA 4.0**.
   - Packaged as 4 modular Parquet shards of ~413 MB each.
   - Remote Parquet columnar queries confirmed that **Shard 0 (`validation-00000-of-00004.parquet`, 433.18 MB)** contains:
     * **875 Real photographs** (ImageNet-1k `val/nature`)
     * **125 Midjourney** samples
     * **125 ADM** samples
     * **125 BigGAN** samples
     * **125 GLIDE** samples
     * **125 SD v1.5** samples (Note: SD v1.4 is absent in validation)
     * **125 VQDM** samples
     * **125 Wukong** samples
   - **Verdict:** Downloading Shard 0 alone requires only **~413 MB** of temporary download space and **~20 MB** of extracted image space (<0.5 GB total), which is completely safe for our 92.46 GB free storage.

---

## 4. Proposed Stage 2 Dataset Plan ($N = 200$)

### 4.1 Target Composition
- **Total Images:** Exactly 200 images.
- **Authentic Class (`REAL`):** **100 authentic photographs** from ImageNet-1k validation set (`val/nature`).
- **AI-Generated Class (`FAKE`):** **100 AI-generated images**, balanced across two candidate generator families:
  - **Generator A (50 samples):** **Midjourney** (commercial closed-source latent diffusion).
  - **Generator B (50 samples):** **ADM (Ablated Diffusion Model)** (academic pixel-space guided diffusion).
  *(Alternative Generator B: **BigGAN** if GAN vs. Diffusion comparison is preferred).*

### 4.2 Semantic Matching & Data Hygiene
- **Category Alignment:** Real images are ImageNet-1k validation natural objects; fake images were generated using ImageNet-1k synset text prompts (`"a photo of {class_name}"`).
- **Pristine Preservation:** Source image bytes will be extracted directly from Parquet records without resizing, cropping, or re-encoding. Each detector's adapter will apply its native preprocessing in-memory.
- **Collision Screening:** Every sample's SHA-256 hash will be verified against all historical benchmark manifests (Phase 4, Phase 5, Phase 6, random 250, and master Kaggle duplicate tables) to guarantee zero overlap.
- **Frozen Manifest:** `benchmark/external_generalization_manifest.csv` with 15 schema columns including `sample_id`, `relative_path`, `label`, `generator_family`, `sha256`, dimensions, and licensing.

---

## 5. Domain Inversion Hypothesis & Decision Fusion Forecast

In Phase 6 (facial portraits), ViT achieved 98.00% while ConvNeXt and UnivFD achieved ~51%, causing democratic voting to degrade accuracy by -35.00%.

In Phase 7 (general ImageNet objects/scenes), empirical conditions are **completely inverted**:
- **ViT (`dima806`):** Expected to struggle or become uncertain because it lacks general-scene detection capabilities and was fine-tuned purely on facial features.
- **ConvNeXt (`chen2024`):** Expected to lead detection because it was trained contrastively on diffusion reconstruction artifacts.
- **UnivFD (`CLIP ViT-L/14`):** Expected to demonstrate improved cross-generator transfer compared to facial StyleGAN.
- **Ensemble Dynamics:** Unlike Phase 6, democratic voting (Majority Vote, Mean Fake Probability) is expected to **outperform ViT-Priority**, because ConvNeXt and UnivFD should form a correct consensus to outvote an erroneous ViT.

---

## 6. Key Decisions Submitted for User Approval

Before proceeding to Stage 2 downloads and extraction, user confirmation is requested on the following design choices:

1. **Generator Family Pair Selection (100 fake images = $2 \times 50$):**
   - **Recommended (Option 1):** **Midjourney (50) + ADM (50)**.
     * Evaluates true cross-generator zero-shot generalization on two generator families **completely unseen in training to all three detectors** (commercial latent diffusion vs. academic pixel diffusion).
   - **Alternative (Option 2):** **Midjourney (50) + BigGAN (50)**.
     * Evaluates commercial diffusion against modern large-scale GAN (both unseen in detector training).
   - **Alternative (Option 3):** **Midjourney (50) + Stable Diffusion v1.5 (50)**.
     * Evaluates one unseen generator (Midjourney) and one near-in-distribution generator for ConvNeXt (SD v1.5).
2. **Dataset Ingestion Source:**
   - **Recommended:** Ingest from Shard 0 (`validation-00000-of-00004.parquet`, ~413 MB) of `TheKernel01/Tiny-GenImage` on Hugging Face Hub. (Consumes <0.5 GB total space; avoids >100 GB multi-part Google Drive download).
3. **Scientific Claim Boundary:**
   - Option 1/2 scientifically evaluates **unseen cross-generator generalization across architectures absent from all three training sets**.
   - Option 3 tests **generalization beyond our existing facial benchmark**, noting that SD v1.5 is within ConvNeXt's training family.

