# Phase 6 Dataset Selection & Integrity Report
**Project:** Cross-Model Evaluation of Deepfake Image Detection Under Diverse Image Generation and Manipulation Conditions  
**Date:** November 2026  
**Status:** Frozen & Verified  
**Manifest Path:** [`benchmark/phase6_test_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/phase6_test_manifest.csv)

---

## 1. Executive Summary

As part of Phase 6 autonomous master goal execution, a rigorous, leak-free, balanced test set of 200 images (100 REAL, 100 FAKE) was selected from the primary test partition located at:
`C:\Users\user\Desktop\currently working\Dataset\Test`

To ensure absolute scientific reproducibility and prevent optimistic evaluation bias from data leakage or repeated testing, the selection protocol strictly isolated these samples from all training data, validation splits, and prior benchmark runs (Phases 1 through 5, `random_250`, `extended_predictions`, and CommunityForensics).

---

## 2. Selection Methodology & Contamination Screening

### 2.1 Candidate Pool
- **Source Root:** `C:\Users\user\Desktop\currently working\Dataset\Test`
  - Real subfolder: 5,413 candidate JPEG files (`real_0.jpg` to `real_5412.jpg`)
  - Fake subfolder: 5,492 candidate JPEG files (`fake_0.jpg` to `fake_5491.jpg`)
- **Dataset Lineage:** Derived from the *Kaggle 140k Real and Fake Faces* dataset.
  - REAL images represent photographic captures of real human faces sourced from Flickr-Faces-HQ (FFHQ).
  - FAKE images represent synthetic facial images generated via StyleGAN / StyleGAN2.

### 2.2 Historical Leakage Screening & Exclusion Sets
Prior project audits, development phases, and historical scripts utilized several test subsets:
1. **Index Range 0 to 699:** Screened and used in early project explorations and `evaluation_manifest.csv` (development val & held-out test).
2. **Prior Manifest Overlaps:** `benchmark/evaluation_manifest.csv` utilized index ranges up to 1574 for held-out tests and smoke testing.
3. **Random 250 Evaluation:** `C:\Users\user\Desktop\currently working\random_250_evaluation_results.csv` evaluated 250 samples spanning various index ranges.
4. **Extended Predictions:** `C:\Users\user\Desktop\currently working\extended_predictions.csv` evaluated 300 samples across the dataset.
5. **CommunityForensics Evaluation:** `C:\Users\user\Desktop\currently working\CommunityForensics\community_predictions.csv`.

To ensure complete statistical independence:
- An **exclusion database** was compiled containing every filename referenced across all historical prediction logs, manifests, and scripts (`touched_names`).
- A minimum index boundary of **$\ge 2000$** was enforced for both Real and Fake candidate pools.
- When traversing candidates starting at index 2000, each candidate was cross-referenced against `touched_names`.
- Candidates identified in historical runs were disqualified:
  * `real_2017.jpg`, `real_2041.jpg`, `real_2044.jpg` (disqualified due to prior touch)
  * `fake_2009.jpg`, `fake_2062.jpg` (disqualified due to prior touch)
- 100 consecutive pristine samples passing all filters were selected for each class:
  * REAL: `real_2000.jpg` through `real_2102.jpg` (100 selected, 3 excluded)
  * FAKE: `fake_2000.jpg` through `fake_2101.jpg` (100 selected, 2 excluded)

---

## 3. Data Integrity & Verification Audit

Every selected image underwent programmatic inspection verifying:
1. **File Readability & Decodability:** Verified via PIL (`Image.open(path).verify()` and `.convert('RGB')`). Zero corruptions, zero truncated JPEGs, zero decode exceptions.
2. **Dimensions & Spatial Uniformity:** Every image is exactly $256 \times 256$ pixels (width: 256, height: 256).
3. **Color Space:** All images contain 3 8-bit RGB color channels (`mode == 'RGB'`).
4. **Cryptographic Uniqueness:** SHA-256 digests were computed for all 200 selected files. All 200 hashes are strictly unique (0 collision rate).
5. **Train / Validation Independence:** Cross-verification against `C:\Users\user\Desktop\currently working\Dataset\Train` and `Dataset\Validation` confirmed zero filename collisions and zero hash collisions.

---

## 4. Manifest Schema

The manifest is stored at [`benchmark/phase6_test_manifest.csv`](file:///C:/Users/user/Desktop/multi%20model%20connection/benchmark/phase6_test_manifest.csv). It contains 201 lines (1 header + 200 data rows) conforming to the following schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | String | Unique identifier: `phase6_test_real_001`..`100`, `phase6_test_fake_001`..`100` |
| `original_filename` | String | Base filename, e.g., `real_2000.jpg` |
| `relative_path` | String | Project relative path: `currently_working/Dataset/Test/...` |
| `absolute_path` | String | Forward-slashed absolute file path on disk |
| `ground_truth` | String | Enum: `REAL` or `FAKE` |
| `domain` | String | `facial` |
| `category` | String | `authentic_face` or `gan_face` |
| `source_dataset` | String | Kaggle 140k Real and Fake Faces (FFHQ / StyleGAN) |
| `generation_or_manipulation_method` | String | Photographic Capture vs. StyleGAN / StyleGAN2 Generative Synthesis |
| `format` | String | `JPEG` |
| `width` | Integer | `256` |
| `height` | Integer | `256` |
| `file_size_bytes` | Integer | File size on disk in bytes |
| `sha256` | Hex String | 64-character SHA-256 checksum |
| `selection_reason` | String | Pristine, decodable test sample with zero historical leakage |

---

## 5. Verification Checklist

- [x] Exactly 100 REAL samples and 100 FAKE samples.
- [x] Balanced 50/50 test distribution ($N=200$).
- [x] Zero overlap with Phase 4 or Phase 5 evaluation sets.
- [x] Zero overlap with historical validation, random 250, or extended prediction logs.
- [x] 100% of images verified decodable, $256 \times 256$ RGB, non-corrupt.
- [x] Manifest frozen at `benchmark/phase6_test_manifest.csv`.

