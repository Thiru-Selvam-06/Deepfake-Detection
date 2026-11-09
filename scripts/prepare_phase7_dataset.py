"""
Phase 7 Dataset Preparation Script
Downloads GenImage validation shard 0 (TheKernel01/Tiny-GenImage),
extracts 200 balanced, pristine images:
  - 100 REAL photographs (ImageNet-1k val/nature)
  - 50 FAKE Midjourney images
  - 50 FAKE ADM images
Screens against historical benchmark hashes, validates RGB decodability,
and generates benchmark/external_generalization_manifest.csv.
"""

import os
import sys
import io
import csv
import hashlib
from typing import Dict, Any, List

import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
REAL_DIR = os.path.join(IMAGES_DIR, "real")
MIDJOURNEY_DIR = os.path.join(IMAGES_DIR, "fake_midjourney")
ADM_DIR = os.path.join(IMAGES_DIR, "fake_adm")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")

REPO_ID = "TheKernel01/Tiny-GenImage"
SHARD_FILENAME = "data/validation-00000-of-00004.parquet"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_historical_hashes() -> set:
    """Collects all SHA-256 hashes from prior project manifests and logs to guarantee zero leakage."""
    historical_hashes = set()

    # Phase 6 test manifest
    p6_manifest = os.path.join(PROJECT_ROOT, "benchmark", "phase6_test_manifest.csv")
    if os.path.exists(p6_manifest):
        with open(p6_manifest, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = row.get("sha256")
                if h:
                    historical_hashes.add(h.lower())

    # Phase 4/5 evaluation manifest
    eval_manifest = os.path.join(PROJECT_ROOT, "benchmark", "evaluation_manifest.csv")
    if os.path.exists(eval_manifest):
        with open(eval_manifest, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = row.get("sha256")
                if h:
                    historical_hashes.add(h.lower())

    print(f"Loaded {len(historical_hashes)} historical hashes for anti-leakage screening.")
    return historical_hashes


def main():
    print("=" * 80)
    print("PHASE 7: EXTRACTING & FREEZING EXTERNAL GENERALIZATION DATASET (GENIMAGE)")
    print("=" * 80)

    for d in [REAL_DIR, MIDJOURNEY_DIR, ADM_DIR]:
        os.makedirs(d, exist_ok=True)

    historical_hashes = load_historical_hashes()

    # Step 1: Download Shard 0 Parquet file
    print(f"\n[Step 1] Downloading GenImage validation shard 0 from Hugging Face Hub ({REPO_ID})...")
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=SHARD_FILENAME,
        repo_type="dataset",
    )
    print(f"  Downloaded to local cache: {parquet_path}")

    # Step 2: Read Parquet table
    print("\n[Step 2] Reading Parquet records from shard 0...")
    table = pq.read_table(parquet_path)
    print(f"  Table loaded: {table.num_rows} rows, {table.num_columns} columns.")

    p_image = table.column("image")
    p_label = table.column("label").to_pylist()
    p_gen = table.column("generator").to_pylist()

    # Step 3: Extract balanced subsets
    print("\n[Step 3] Selecting 100 Real, 50 Midjourney, 50 ADM samples...")
    selected_real = []
    selected_midjourney = []
    selected_adm = []

    for idx in range(table.num_rows):
        gen_id = p_gen[idx]
        lbl = p_label[idx]

        if gen_id == 0 and len(selected_real) < 100:
            selected_real.append(idx)
        elif gen_id == 4 and len(selected_midjourney) < 50:
            selected_midjourney.append(idx)
        elif gen_id == 1 and len(selected_adm) < 50:
            selected_adm.append(idx)

        if len(selected_real) == 100 and len(selected_midjourney) == 50 and len(selected_adm) == 50:
            break

    print(f"  Selected counts: REAL={len(selected_real)}, Midjourney={len(selected_midjourney)}, ADM={len(selected_adm)}")
    assert len(selected_real) == 100, f"Expected 100 Real, got {len(selected_real)}"
    assert len(selected_midjourney) == 50, f"Expected 50 Midjourney, got {len(selected_midjourney)}"
    assert len(selected_adm) == 50, f"Expected 50 ADM, got {len(selected_adm)}"

    # Step 4: Extract and save image files to disk with integrity validation
    print("\n[Step 4] Writing pristine images to disk and validating integrity...")
    manifest_rows = []
    seen_hashes = set()

    extraction_targets = [
        ("real", selected_real, REAL_DIR, "REAL", "none"),
        ("midjourney", selected_midjourney, MIDJOURNEY_DIR, "FAKE", "midjourney"),
        ("adm", selected_adm, ADM_DIR, "FAKE", "adm"),
    ]

    sample_counter = 1
    for name_prefix, indices, target_dir, ground_truth, gen_fam in extraction_targets:
        for idx in indices:
            row_dict = p_image[idx].as_py()
            img_bytes = row_dict["bytes"]
            orig_path = row_dict.get("path") or f"{name_prefix}_{idx}.jpg"
            orig_filename = os.path.basename(orig_path)

            sha256 = compute_sha256(img_bytes).lower()
            file_size = len(img_bytes)

            # Assert no duplicates
            if sha256 in seen_hashes:
                raise ValueError(f"Duplicate hash within selection: {sha256}")
            seen_hashes.add(sha256)

            if sha256 in historical_hashes:
                raise ValueError(f"Hash collision with historical benchmark dataset: {sha256}")

            # Verify image decode
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                width, height = pil_img.size
                img_format = pil_img.format or "JPEG"
                # Ensure RGB decodable
                _ = pil_img.convert("RGB")

            # Determine clean filename and relative/absolute paths
            if name_prefix == "real":
                sample_id = f"phase7_ext_real_{len([r for r in manifest_rows if r['label'] == 'REAL']) + 1:03d}"
                out_filename = f"genimage_real_{len([r for r in manifest_rows if r['label'] == 'REAL']) + 1:03d}.jpg"
            elif name_prefix == "midjourney":
                sample_id = f"phase7_ext_fake_midjourney_{len([r for r in manifest_rows if r['generator_family'] == 'midjourney']) + 1:03d}"
                out_filename = f"genimage_midjourney_{len([r for r in manifest_rows if r['generator_family'] == 'midjourney']) + 1:03d}.jpg"
            else:
                sample_id = f"phase7_ext_fake_adm_{len([r for r in manifest_rows if r['generator_family'] == 'adm']) + 1:03d}"
                out_filename = f"genimage_adm_{len([r for r in manifest_rows if r['generator_family'] == 'adm']) + 1:03d}.jpg"

            out_path = os.path.join(target_dir, out_filename)
            with open(out_path, "wb") as f_out:
                f_out.write(img_bytes)

            rel_path = os.path.relpath(out_path, PROJECT_ROOT).replace("\\", "/")
            abs_path = os.path.abspath(out_path).replace("\\", "/")

            manifest_rows.append({
                "sample_id": sample_id,
                "relative_path": rel_path,
                "absolute_path": abs_path,
                "label": ground_truth,
                "generator_family": gen_fam,
                "source_dataset": "GenImage (TheKernel01/Tiny-GenImage)",
                "source_split": "validation",
                "original_filename": orig_filename,
                "file_size_bytes": file_size,
                "sha256": sha256,
                "width": width,
                "height": height,
                "format": img_format,
                "category": "ImageNet-1k object/scene",
                "license_or_usage_reference": "CC BY-NC-SA 4.0 / ImageNet research non-commercial",
            })

    print(f"  Successfully extracted and verified {len(manifest_rows)} pristine images.")

    # Step 5: Save manifest CSV
    print(f"\n[Step 5] Writing frozen manifest to {MANIFEST_PATH}...")
    fieldnames = list(manifest_rows[0].keys())
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  Frozen manifest generated successfully: {MANIFEST_PATH}")
    print(f"  Rows: {len(manifest_rows)} (100 REAL, 50 Midjourney, 50 ADM).")
    print("=" * 80)
    print("STAGE 2 DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

