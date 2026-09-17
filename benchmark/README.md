# Benchmark Dataset Directory

This directory provides an isolated, standardized benchmark harness for testing deepfake detectors.

## Directory Structure

```text
benchmark/
├── README.md
├── metadata.json
├── images/
│   ├── real/            # Ground-truth photographic images
│   ├── ai_generated/    # Ground-truth synthetically generated images
│   ├── manipulated/     # Ground-truth edited/spliced images
│   └── corrupted/       # Corrupted or malformed files for rejection testing
└── results/             # Benchmark execution outputs (JSON and CSV)
```

## Input Availability & Addition Protocol

In accordance with strict repository isolation rules:
- No files were copied from local workspaces or external paths outside this Space repository.
- No ground-truth labels were fabricated.

### How to Manually Add Benchmark Images
1. Place natural photographs into `benchmark/images/real/`.
2. Place AI-generated images (e.g., from Diffusion, GAN, or Flow-Matching models) into `benchmark/images/ai_generated/`.
3. Place edited or composite images into `benchmark/images/manipulated/`.
4. Place truncated or corrupted test files into `benchmark/images/corrupted/`.
5. Update `benchmark/metadata.json` with the filename, category, and verified ground-truth origin. If ground truth cannot be verified with documentary proof, mark `ground_truth: "UNKNOWN"`.

## Non-Authenticity Disclaimer

This benchmark framework is an operational diagnostic tool. Successful execution does not imply detector accuracy, and consensus across models does not verify photographic authenticity.
