# Phase 7 Version 2: Corrected External Generalization Evaluation Report
## Official Reference Preprocessing Alignment & Empirical Re-evaluation on GenImage

**Project:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Evaluation Scope:** Phase 7 Corrective Action — Official Preprocessing Alignment across ConvNeXt, UnivFD, and ViT on GenImage Benchmark ($N = 200$)  
**Benchmark Partition:** [`benchmark/external_generalization_manifest.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/external_generalization_manifest.csv) (100 Authentic ImageNet-1k, 50 Synthetic Midjourney, 50 Synthetic ADM)  
**Execution Timestamp:** November 2026  
**Status:** **FULLY VERIFIED & INDEPENDENTLY REPRODUCED**  
**Preceding Investigation:** [`docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md)  
**Version 1 Baseline:** [`docs/PHASE7_EXTERNAL_GENERALIZATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_EXTERNAL_GENERALIZATION_REPORT.md)  

---

## 1. Reason for Correcting Version 1

During Phase 7 Version 1 evaluation, the three integrated detectors exhibited unexpectedly poor generalization across external diffusion-generated images:
- ViT: 52.00% Accuracy, 18.00% Fake Recall, ROC-AUC 0.4673
- UnivFD: 57.50% Accuracy, 15.00% Fake Recall, ROC-AUC 0.6662
- ConvNeXt: 56.50% Accuracy, 13.00% Fake Recall, ROC-AUC 0.7926

A forensic root-cause parity audit ([`docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/ROOT_CAUSE_MODEL_INFERENCE_INVESTIGATION.md)) was conducted comparing project adapters directly against the authors' official upstream repositories:
1. **ConvNeXt DRCT Downsampling Defect:** The Version 1 adapter applied `transforms.Resize(224, bicubic) -> transforms.CenterCrop(224)`. Downsampling $1024 \times 1024$ diffusion images down to 224 smoothed and attenuated the subtle high-frequency spatial and spectral artifacts that DRCT's contrastive reconstruction objective targets. In contrast, the author's official GenImage evaluation script ([`temp_DRCT_reference/scripts/test_GenImage.sh`](file:///c:/Users/user/Desktop/multi%20model%20connection/temp_DRCT_reference/scripts/test_GenImage.sh)) explicitly runs with `--is_crop`, performing native-resolution center cropping without prior resizing.
2. **UnivFD Transform Divergence:** The Version 1 adapter used standard OpenAI CLIP preprocessing (which downsamples the shortest edge before center cropping). The official CVPR 2023 evaluation script ([`UniversalFakeDetect/validate.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/UniversalFakeDetect/validate.py)) uses unscaled `CenterCrop(224) -> ToTensor() -> Normalize(clip_stats)`.
3. **ViT Baseline Integrity:** ViT was verified to have 100% bit-for-bit parity with Hugging Face upstream inference; its poor performance stems entirely from facial training domain collapse on non-facial scene photography.

To eliminate integration defects as a confounding variable and establish authentic out-of-domain generalization limits, Phase 7 Version 2 was executed with official reference-exact preprocessing.

---

## 2. Original Preprocessing versus Official Preprocessing

| Detector | Phase 7 Version 1 Preprocessing | Phase 7 Version 2 (Official Reference) Preprocessing | Upstream Reference File & Setting |
| :--- | :--- | :--- | :--- |
| **ConvNeXt (DRCT)** | `Resize(224, bicubic) -> CenterCrop(224) -> ToTensor() -> Normalize(imagenet)` | `PadIfNeeded(224, value=0) -> CenterCrop(224, native resolution) -> ToTensor() -> Normalize(imagenet)` | [`temp_DRCT_reference/data/transform.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/temp_DRCT_reference/data/transform.py#L69-L77), `is_crop=True` in `test_GenImage.sh` |
| **UnivFD (CLIP)** | `Resize(224, bicubic) -> CenterCrop(224) -> ToTensor() -> Normalize(clip)` | `CenterCrop(224, native resolution) -> ToTensor() -> Normalize(clip_mean, clip_std)` | [`UniversalFakeDetect/validate.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/UniversalFakeDetect/validate.py#L198-L202) |
| **ViT (dima806)** | `AutoImageProcessor` (`Resize(224) -> CenterCrop(224) -> Rescale(1/255) -> Normalize(0.5)`) | `AutoImageProcessor` (Unchanged, 100% official match) | `dima806/deepfake_vs_real_image_detection` |

### Code Implementation Details

#### ConvNeXt: `DRCTNativeCropTransform` in [`src/convnext_adapter.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/convnext_adapter.py#L55-L114)
```python
class DRCTNativeCropTransform:
    """Official DRCT evaluation transform (--is_crop, Chen et al. ICML 2024)."""
    def __init__(self, size: int = 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        # Pad symmetrically if height or width < 224
        h, w = img_np.shape[:2]
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2; bottom = pad_h - top
            left = pad_w // 2; right = pad_w - left
            img_np = cv2.copyMakeBorder(img_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            h, w = img_np.shape[:2]
        # Direct CenterCrop(224) at 1:1 native resolution
        start_y = (h - self.size) // 2
        start_x = (w - self.size) // 2
        cropped = img_np[start_y:start_y + self.size, start_x:start_x + self.size]
        norm = (cropped.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(norm.transpose(2, 0, 1)).float()
```

#### UnivFD: `transforms.Compose` in [`src/univfd_adapter.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/univfd_adapter.py#L106-L114)
```python
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]
self.preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std),
])
```

---

## 3. Reference Parity Test Results

Prior to running full-benchmark inference, parity was verified across deterministic test batches side-by-side against upstream reference execution code using [`scripts/verify_v2_reference_preprocessing.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/scripts/verify_v2_reference_parity.py).

Summary from [`benchmark/results/phase7_v2_reference_preprocessing/parity_verification_audit.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_v2_reference_preprocessing/parity_verification_audit.json):

| Detector Tested | Sample Size ($N$) | Max Pixel Diff | Max Input Tensor Diff | Max Logit Diff | Max Prob Diff | Binary Prediction Matches | Parity Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt (DRCT)** | 30 | 0 | $0.000000$ | $9.16 \times 10^{-7}$ | $4.71 \times 10^{-7}$ | 30 / 30 (100.0%) | **PASSED** |
| **UnivFD (CLIP)** | 30 | 0 | $0.000000$ | $4.88 \times 10^{-7}$ | $4.76 \times 10^{-7}$ | 30 / 30 (100.0%) | **PASSED** |
| **ViT (dima806)** | 10 | 0 | $0.000000$ | $4.80 \times 10^{-7}$ | $4.94 \times 10^{-7}$ | 10 / 10 (100.0%) | **PASSED** |

### Regression Test Suite Results
The regression test suite [`tests/test_adapter_preprocessing_regression.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/tests/test_adapter_preprocessing_regression.py) was executed with `pytest`:
- `test_convnext_no_resize_on_1024x1024`: **PASSED** (confirms $1024 \times 1024$ central feature pattern is preserved with zero border blending).
- `test_convnext_small_image_padding`: **PASSED** (confirms symmetric zero-padding for images $<224$).
- `test_univfd_no_resize_on_1024x1024`: **PASSED** (confirms `CenterCrop(224)` without prior resizing).
- `test_vit_preprocessing_remains_unchanged`: **PASSED** (confirms Hugging Face AutoImageProcessor integrity).
- **Full Test Suite:** 25/25 tests passed across unit and regression suites.

---

## 4. Corrected Model Metrics (Phase 7 Version 2)

Evaluated across the frozen 200 manifest samples (100 Authentic ImageNet, 50 Midjourney, 50 ADM) under default threshold $\tau = 0.50$:

| Model / Strategy | Aggregate Acc (%) | 95% Wilson CI | Balanced Acc (%) | Precision | Recall (Fake) | Specificity (Real) | F1 Score | ROC-AUC | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT (`dima806`)** | 52.00% | [45.10%, 58.82%] | 52.00% | 0.5625 | 18.00% | 86.00% | 0.2727 | 0.4673 | 225.1 ms |
| **UnivFD (CLIP ViT-L/14)** | 56.50% | [49.57%, 63.18%] | 56.50% | 0.8824 | 15.00% | 98.00% | 0.2564 | 0.8037 | 1028.0 ms |
| **ConvNeXt (DRCT / RAID)** | **77.00%** | **[70.69%, 82.29%]** | **77.00%** | **0.9821** | **55.00%** | **99.00%** | **0.7051** | **0.9271** | 273.3 ms |
| **Ensemble (Majority Vote)** | 58.50% | [51.57%, 65.11%] | 58.50% | 0.9474 | 18.00% | 99.00% | 0.3025 | 0.7662 | 1526.5 ms |
| **Ensemble (Mean Fake Prob)** | 61.50% | [54.60%, 67.97%] | 61.50% | **1.0000** | 23.00% | **100.00%** | 0.3740 | 0.8309 | 1526.5 ms |
| **Ensemble (ViT Priority)** | 52.00% | [45.10%, 58.82%] | 52.00% | 0.5625 | 18.00% | 86.00% | 0.2727 | 0.4673 | 225.1 ms |

### Confusion Matrices ($N = 200$)
- **ViT:** $\text{TP}=18, \text{FP}=14, \text{TN}=86, \text{FN}=82$
- **UnivFD:** $\text{TP}=15, \text{FP}=2, \text{TN}=98, \text{FN}=85$
- **ConvNeXt:** $\mathbf{\text{TP}=55, \text{FP}=1, \text{TN}=99, \text{FN}=45}$
- **Majority Vote:** $\text{TP}=18, \text{FP}=1, \text{TN}=99, \text{FN}=82$
- **Mean Fake Probability:** $\text{TP}=23, \text{FP}=0, \text{TN}=100, \text{FN}=77$
- **ViT Priority:** $\text{TP}=18, \text{FP}=14, \text{TN}=86, \text{FN}=82$

---

## 5. Version 1 versus Version 2 Comparison

| Detector / Strategy | Metric | Phase 7 Version 1 (Downsampled) | Phase 7 Version 2 (Official Crop) | Absolute Delta ($\Delta$) | Attribution / Effect |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **ConvNeXt (DRCT)** | **Accuracy** | 56.50% | **77.00%** | $\mathbf{+20.50\%}$ | Resolves Midjourney downsampling defect |
| | **Fake Recall** | 13.00% | **55.00%** | $\mathbf{+42.00\%}$ | Midjourney recall surges from 22% to 100% |
| | **Specificity** | 100.00% | 99.00% | $-1.00\%$ | 99/100 authentic images correctly classified |
| | **ROC-AUC** | 0.7926 | **0.9271** | $\mathbf{+0.1345}$ | Near-perfect separation on Midjourney |
| **UnivFD (CLIP)** | **Accuracy** | 57.50% | 56.50% | $-1.00\%$ | Net-zero change in binary recall |
| | **Fake Recall** | 15.00% | 15.00% | $\pm 0.00\%$ | Bounded by ProGAN pretraining |
| | **Specificity** | 100.00% | 98.00% | $-2.00\%$ | 2 false positives on natural photos |
| | **ROC-AUC** | 0.6662 | **0.8037** | $\mathbf{+0.1375}$ | Continuous ranking significantly improved |
| **ViT (dima806)** | **Accuracy** | 52.00% | 52.00% | $\pm 0.00\%$ | Control model; zero modifications made |
| | **Fake Recall** | 18.00% | 18.00% | $\pm 0.00\%$ | Facial domain collapse confirmed |
| | **ROC-AUC** | 0.4673 | 0.4673 | $\pm 0.00\%$ | Remains worse than random coin toss |
| **Majority Vote** | **Accuracy** | 52.50% | 58.50% | $\mathbf{+6.00\%}$ | Lifted by ConvNeXt's 55 correct fake votes |
| | **Fake Recall** | 5.00% | 18.00% | $\mathbf{+13.00\%}$ | Improved from 5 to 18 detected fakes |
| | **ROC-AUC** | 0.7181 | 0.7662 | $+0.0481$ | Continuous vote margin ranking |
| **Mean Probability**| **Accuracy** | 52.50% | 61.50% | $\mathbf{+9.00\%}$ | ConvNeXt pulls ensemble mean over $\tau=0.5$ |
| | **Fake Recall** | 5.00% | 23.00% | $\mathbf{+18.00\%}$ | Improved from 5 to 23 detected fakes |
| | **ROC-AUC** | 0.7181 | 0.8309 | $\mathbf{+0.1128}$ | Stronger probability calibration |

---

## 6. Generator-Specific Performance

> [!IMPORTANT]
> The authentic ImageNet control partition ($N=100$) is shared between Midjourney ($N=50$) and ADM ($N=50$) cohorts (150 total evaluations per cohort). Cohort accuracies must not be averaged to obtain aggregate benchmark accuracy.

### Cohort Breakdown Table

| Detector / Strategy | Metric | Authentic ImageNet ($N=100$) | Midjourney Cohort ($N=150$) | ADM Cohort ($N=150$) |
| :--- | :--- | :---: | :---: | :---: |
| **ConvNeXt (DRCT / RAID)** | **Accuracy** | **99.00% (99/100)** | **99.33% (149/150)** | 69.33% (104/150) |
| | **Cohort Fake Recall** | — | **100.00% (50/50)** | **10.00% (5/50)** |
| | **Cohort Specificity** | 99.00% | 99.00% | 99.00% |
| | **Cohort Precision** | — | 0.9804 | 0.8333 |
| | **Cohort F1-Score** | — | 0.9901 | 0.1786 |
| | **Cohort ROC-AUC** | — | **0.9992** | 0.8550 |
| | **Score Mean $\pm$ Std** | $0.0146 \pm 0.1021$ | $\mathbf{0.9904 \pm 0.0349}$ | $0.1608 \pm 0.2518$ |
| **UnivFD (CLIP ViT-L/14)** | **Accuracy** | **98.00% (98/100)** | 70.67% (106/150) | 70.00% (105/150) |
| | **Cohort Fake Recall** | — | 16.00% (8/50) | 14.00% (7/50) |
| | **Cohort Specificity** | 98.00% | 98.00% | 98.00% |
| | **Cohort ROC-AUC** | — | 0.7952 | 0.8122 |
| | **Score Mean $\pm$ Std** | $0.0368 \pm 0.1249$ | $0.2084 \pm 0.3078$ | $0.1913 \pm 0.2555$ |
| **ViT (`dima806`)** | **Accuracy** | 86.00% (86/100) | 64.00% (96/150) | 62.67% (94/150) |
| | **Cohort Fake Recall** | — | 20.00% (10/50) | 16.00% (8/50) |
| | **Cohort Specificity** | 86.00% | 86.00% | 86.00% |
| | **Cohort ROC-AUC** | — | 0.5336 | 0.4010 |
| | **Score Mean $\pm$ Std** | $0.3009 \pm 0.1813$ | $0.3316 \pm 0.1873$ | $0.2531 \pm 0.2405$ |
| **Ensemble (Majority Vote)** | **Accuracy** | 99.00% (99/100) | 76.00% (114/150) | 68.00% (102/150) |
| | **Cohort Fake Recall** | — | 30.00% (15/50) | 6.00% (3/50) |
| | **Cohort ROC-AUC** | — | 0.9393 | 0.5931 |
| **Ensemble (Mean Prob)** | **Accuracy** | **100.00% (100/100)**| 80.00% (120/150) | 68.67% (103/150) |
| | **Cohort Fake Recall** | — | 40.00% (20/50) | 6.00% (3/50) |
| | **Cohort ROC-AUC** | — | 0.9870 | 0.6748 |

---

## 7. Updated Fusion Performance

### Does Decision Fusion Beat the Strongest Single Detector?
**NO.** On the Phase 7 general-scene external generalization benchmark:
- **Strongest Individual Detector:** **ConvNeXt DRCT / RAID**
  * Accuracy: **77.00%**
  * Fake Recall: **55.00%**
  * Midjourney Recall: **100.00%**
  * ROC-AUC: **0.9271**
- **Best Fusion Strategy:** **Ensemble (Mean Fake Probability)**
  * Accuracy: **61.50%** ($\mathbf{-15.50\%}$ compared to ConvNeXt)
  * Fake Recall: **23.00%** ($\mathbf{-32.00\%}$ compared to ConvNeXt)
  * Midjourney Recall: **40.00%** ($\mathbf{-60.00\%}$ compared to ConvNeXt)
  * ROC-AUC: **0.8309** ($\mathbf{-0.0962}$ compared to ConvNeXt)
- **Democratic Majority Vote:**
  * Accuracy: **58.50%** ($\mathbf{-18.50\%}$ compared to ConvNeXt)
  * Fake Recall: **18.00%** ($\mathbf{-37.00\%}$ compared to ConvNeXt)
  * ROC-AUC: **0.7662**

### Why Fusion Degrades Performance on Out-of-Domain Media
In democratic voting or arithmetic averaging, each detector has equal authority regardless of input domain:
1. On Midjourney images, ConvNeXt outputs near-ceiling fake probabilities ($\mu = 0.9904$, 50/50 detected).
2. However, both ViT ($\mu = 0.3316$) and UnivFD ($\mu = 0.2084$) fail to recognize Midjourney artifacts and vote `REAL` on 40 and 42 of the 50 images, respectively.
3. In majority voting ($2 \text{ vs } 1$), **ViT and UnivFD combine to outvote ConvNeXt on 35 out of 50 Midjourney fakes**, causing the ensemble to incorrectly classify them as authentic photographs.
4. In mean probability fusion, averaging $0.999 + 0.20 + 0.30 \approx 0.499 < 0.50$, pulling 30 of the 50 Midjourney samples below the decision boundary.
5. In ViT-priority fusion, the ensemble blindly forwards ViT's decisions, inheriting ViT's complete domain collapse (52.00% accuracy).

---

## 8. Interpretation of Findings

1. **Resolution of the Preprocessing Mismatch:** Restoring official native-resolution cropping (`--is_crop`) proved conclusively that the Version 1 failure on Midjourney was an **integration defect, not a model limitation**. ConvNeXt achieved **100.00% recall (50/50)** on Midjourney with a cohort ROC-AUC of **0.9992**.
2. **Asymmetric Forensic Generalization across Diffusion Architectures:**
   - **Latent Diffusion (Midjourney):** High detection rate. Latent diffusion decoders (e.g., VAE upsampling) leave distinct high-frequency checkerboard and phase artifacts that are preserved by native cropping and readily captured by DRCT's residual feature extractor.
   - **Pixel-Space Diffusion (ADM):** Low detection rate. Guided Diffusion / ADM operates directly in pixel space without a latent VAE decoder. It exhibits fundamentally different spatial and frequency profiles that evade DRCT's contrastive reconstruction representations (only 10.00% recall). This confirms the published findings of Chen et al. (ICML 2024 Spotlight, Table 7), where DRCT reported only 60.00% accuracy on GenImage ADM.
3. **UnivFD Probe Limitations:** While official `CenterCrop(224)` improved UnivFD's continuous ranking separation (ROC-AUC surged from 0.6662 to 0.8037), binary recall at default threshold $\tau = 0.50$ remained unchanged at 15.00%. A linear probe trained exclusively on 2018 ProGAN artifacts lacks the representation scale to cross over to modern diffusion generators at default calibration.
4. **Catastrophic Domain Collapse of ViT:** ViT achieved 98.00% accuracy on facial deepfakes (Phase 6), but collapsed to 52.00% on natural scene objects (Phase 7). Its self-attention heads learned facial geometric priors (eye alignment, mouth synthesis) rather than generalizable generative artifacts.

---

## 9. Limitations and Unresolved Questions

1. **Single-Crop vs. Test-Time Augmentation (TTA):** The official DRCT training code provides a 9-view TTA routine (`TTA()` averaging flips and crops). Phase 7 V2 evaluated deterministic single center crops. Whether multi-crop TTA can marginally lift ADM recall above 10% remains an open experimental question.
2. **ADM Detection Failure across Forensic Literature:** Table 7 of the DRCT paper demonstrates that virtually all published detectors (CNNSpot: 53.07%, CLIP/RN50: 52.06%, GramNet: 51.83%, UnivFD: 55.95%, DIRE: 49.93%) fail on ADM. Pixel-space diffusion remains an open challenge for deepfake detection.
3. **Threshold Calibration ($\tau$):** Both ConvNeXt and UnivFD achieved near-perfect specificity on ImageNet photos ($99\%$ and $98\%$). However, fixed $\tau = 0.50$ leaves UnivFD operating far from its optimal operating point ($\tau^* \approx 0.05$). Threshold tuning must be performed on dedicated calibration splits, not test benchmarks.

---

## 10. Final Recommendation

1. **Adopt Corrected Adapters as Project Standard:** Permanently retain `DRCTNativeCropTransform` in [`src/convnext_adapter.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/convnext_adapter.py) and native `CenterCrop(224)` in [`src/univfd_adapter.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/src/univfd_adapter.py).
2. **Deprecate Unweighted Democratic Fusion on Unconstrained Media:** Democratic voting across heterogeneous detectors is counter-productive when detector domains differ. A weak or out-of-domain detector actively degrades stronger detectors.
3. **Implement Domain-Gated Hierarchical Routing for Phase 8:**
   - Deploy an upstream domain detector / face-detector router.
   - If human faces are detected $\to$ Route to **ViT (`dima806`)** (98% accuracy on faces).
   - If general scenes / non-facial media are detected $\to$ Route exclusively to **ConvNeXt (DRCT / RAID)** (77% accuracy, 99.3% on Midjourney).

---

## Master Artifacts Produced

- **Full Corrected Evaluation Report:** [`docs/PHASE7_V2_CORRECTED_EVALUATION_REPORT.md`](file:///c:/Users/user/Desktop/multi%20model%20connection/docs/PHASE7_V2_CORRECTED_EVALUATION_REPORT.md)
- **Reference Parity Audit Record:** [`benchmark/results/phase7_v2_reference_preprocessing/parity_verification_audit.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_v2_reference_preprocessing/parity_verification_audit.json)
- **Model Comparison Table:** [`benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_model_comparison.csv`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_model_comparison.csv)
- **Full Metrics JSON:** [`benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_combined_metrics.json`](file:///c:/Users/user/Desktop/multi%20model%20connection/benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_combined_metrics.json)
- **Individual Prediction Datasets:**
  * `benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_convnext_predictions.csv`
  * `benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_univfd_predictions.csv`
  * `benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_vit_predictions.csv`
  * `benchmark/results/phase7_v2_reference_preprocessing/phase7_v2_combined_predictions.csv`
- **Regression Test Suite:** [`tests/test_adapter_preprocessing_regression.py`](file:///c:/Users/user/Desktop/multi%20model%20connection/tests/test_adapter_preprocessing_regression.py)
