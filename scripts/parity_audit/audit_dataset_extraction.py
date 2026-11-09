"""
Audit Phase 3: Dataset Extraction and Semantic Composition Audit
Cross-references the Parquet source shard with the extracted Phase 7 image files and manifest.
"""

import os
import sys
import io
import csv
import hashlib
import json
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "benchmark", "external_generalization_manifest.csv")
PARQUET_PATH = r"C:\Users\user\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\validation-00000-of-00004.parquet"

def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    print("=" * 80)
    print("AUDITING PHASE 7 DATASET EXTRACTION AND COMPOSITION")
    print("=" * 80)
    
    # Check parquet existence
    if not os.path.exists(PARQUET_PATH):
        print(f"ERROR: Parquet file not found at {PARQUET_PATH}")
        sys.exit(1)
        
    print(f"Reading Parquet file: {PARQUET_PATH}")
    table = pq.read_table(PARQUET_PATH)
    print(f"Parquet schema:\n{table.schema}")
    print(f"Total rows in shard 0: {table.num_rows}")
    
    p_image = table.column("image")
    p_label = table.column("label").to_pylist()
    p_gen = table.column("generator").to_pylist()
    
    # Check distinct generators and labels in shard 0
    gen_counter = Counter(p_gen)
    lbl_counter = Counter(p_label)
    print(f"Generator distribution in shard 0: {gen_counter}")
    print(f"Label distribution in shard 0: {lbl_counter}")
    
    # Replicate selection
    selected_real = []
    selected_midjourney = []
    selected_adm = []
    
    for idx in range(table.num_rows):
        gen_id = p_gen[idx]
        if gen_id == 0 and len(selected_real) < 100:
            selected_real.append(idx)
        elif gen_id == 4 and len(selected_midjourney) < 50:
            selected_midjourney.append(idx)
        elif gen_id == 1 and len(selected_adm) < 50:
            selected_adm.append(idx)
        if len(selected_real) == 100 and len(selected_midjourney) == 50 and len(selected_adm) == 50:
            break
            
    print(f"Selection counts: REAL={len(selected_real)}, Midjourney={len(selected_midjourney)}, ADM={len(selected_adm)}")
    
    # Load manifest
    manifest_rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            manifest_rows.append(r)
    print(f"Manifest rows loaded: {len(manifest_rows)}")
    
    # Map by sample_id
    manifest_by_id = {r["sample_id"]: r for r in manifest_rows}
    
    # Perform 1-to-1 byte and metadata comparison
    targets = [
        ("real", selected_real, "REAL", "none"),
        ("midjourney", selected_midjourney, "FAKE", "midjourney"),
        ("adm", selected_adm, "FAKE", "adm"),
    ]
    
    byte_mismatches = 0
    size_mismatches = 0
    hash_mismatches = 0
    total_audited = 0
    
    extraction_details = []
    
    counter = {"real": 0, "midjourney": 0, "adm": 0}
    for prefix, indices, expected_label, expected_gen in targets:
        for idx in indices:
            counter[prefix] += 1
            if prefix == "real":
                sample_id = f"phase7_ext_real_{counter[prefix]:03d}"
            elif prefix == "midjourney":
                sample_id = f"phase7_ext_fake_midjourney_{counter[prefix]:03d}"
            else:
                sample_id = f"phase7_ext_fake_adm_{counter[prefix]:03d}"
                
            m_row = manifest_by_id[sample_id]
            file_on_disk = os.path.join(PROJECT_ROOT, m_row["relative_path"])
            
            # Read parquet record
            p_rec = p_image[idx].as_py()
            p_bytes = p_rec["bytes"]
            p_path = p_rec.get("path", "")
            p_sha = compute_sha256(p_bytes).lower()
            p_size = len(p_bytes)
            
            # Read disk file
            with open(file_on_disk, "rb") as f_disk:
                disk_bytes = f_disk.read()
            disk_sha = compute_sha256(disk_bytes).lower()
            disk_size = len(disk_bytes)
            
            # Compare bytes
            if p_bytes != disk_bytes:
                byte_mismatches += 1
            if p_size != disk_size:
                size_mismatches += 1
            if p_sha != disk_sha:
                hash_mismatches += 1
                
            # Decode images to inspect format, mode, channels, color profiles
            with Image.open(io.BytesIO(p_bytes)) as pil_p:
                p_format = pil_p.format
                p_mode = pil_p.mode
                p_w, p_h = pil_p.size
                p_icc = pil_p.info.get("icc_profile") is not None
                
            with Image.open(file_on_disk) as pil_d:
                d_format = pil_d.format
                d_mode = pil_d.mode
                d_w, d_h = pil_d.size
                d_icc = pil_d.info.get("icc_profile") is not None
                
            extraction_details.append({
                "sample_id": sample_id,
                "prefix": prefix,
                "label": expected_label,
                "gen": expected_gen,
                "parquet_orig_filename": os.path.basename(p_path),
                "disk_filename": os.path.basename(file_on_disk),
                "parquet_size": p_size,
                "disk_size": disk_size,
                "parquet_sha": p_sha,
                "disk_sha": disk_sha,
                "format": p_format,
                "disk_format": d_format,
                "mode": p_mode,
                "width": p_w,
                "height": p_h,
                "has_icc": p_icc,
                "exact_byte_match": p_bytes == disk_bytes,
            })
            total_audited += 1
            
    print(f"\nAUDIT RESULTS ({total_audited} samples):")
    print(f"  Exact byte matches: {total_audited - byte_mismatches} / {total_audited}")
    print(f"  Byte mismatches:    {byte_mismatches}")
    print(f"  Size mismatches:    {size_mismatches}")
    print(f"  SHA-256 mismatches: {hash_mismatches}")
    
    # Save extraction audit json
    audit_out_path = os.path.join(PROJECT_ROOT, "benchmark", "results", "parity_audit_dataset_extraction.json")
    with open(audit_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_audited": total_audited,
            "byte_mismatches": byte_mismatches,
            "size_mismatches": size_mismatches,
            "hash_mismatches": hash_mismatches,
            "details": extraction_details,
        }, f, indent=2)
    print(f"Saved extraction details to: {audit_out_path}")
    
    # Composition and dimension analysis
    print("\n" + "=" * 80)
    print("DATASET COMPOSITION & DIMENSION ANALYSIS")
    print("=" * 80)
    for group_name in ["real", "midjourney", "adm"]:
        group_items = [d for d in extraction_details if d["prefix"] == group_name]
        formats = Counter(d["format"] for d in group_items)
        disk_formats = Counter(d["disk_format"] for d in group_items)
        modes = Counter(d["mode"] for d in group_items)
        has_icc = sum(1 for d in group_items if d["has_icc"])
        widths = [d["width"] for d in group_items]
        heights = [d["height"] for d in group_items]
        aspect_ratios = [round(d["width"] / d["height"], 3) for d in group_items]
        file_sizes_kb = [round(d["disk_size"] / 1024, 1) for d in group_items]
        
        print(f"\n--- Group: {group_name.upper()} (N = {len(group_items)}) ---")
        print(f"  Source Formats in Parquet: {dict(formats)}")
        print(f"  Decoded Formats on Disk:   {dict(disk_formats)}")
        print(f"  Color Modes:               {dict(modes)}")
        print(f"  Has ICC profile:           {has_icc} / {len(group_items)}")
        print(f"  Widths:  min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
        print(f"  Heights: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")
        print(f"  Aspect Ratios: min={min(aspect_ratios)}, max={max(aspect_ratios)}, square_count={sum(1 for r in aspect_ratios if r == 1.0)}")
        print(f"  File Size (KB): min={min(file_sizes_kb)}, max={max(file_sizes_kb)}, mean={sum(file_sizes_kb)/len(file_sizes_kb):.1f}")
        sample_filenames = [d["parquet_orig_filename"] for d in group_items[:5]]
        print(f"  Sample Source Filenames:   {sample_filenames}")

if __name__ == "__main__":
    main()
