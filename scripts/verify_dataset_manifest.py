"""
Verification script for the benchmark evaluation manifest.
Validates file existence, file sizes, SHA-256 hashes, image formats,
resolutions, ground-truth label consistency, and split partitioning.
"""

import csv
import hashlib
import os
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "benchmark" / "evaluation_manifest.csv"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_manifest() -> bool:
    print("=" * 80)
    print("DATASET MANIFEST INTEGRITY VERIFICATION")
    print("=" * 80)
    print(f"Manifest Path: {MANIFEST_PATH}")
    
    if not MANIFEST_PATH.exists():
        print(f"[FAIL] Manifest file does not exist: {MANIFEST_PATH}")
        return False
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        
    total_entries = len(reader)
    print(f"Total manifest entries: {total_entries}")
    
    expected_total = 159
    if total_entries != expected_total:
        print(f"[FAIL] Expected {expected_total} rows, found {total_entries}")
        return False
        
    seen_ids = set()
    seen_hashes = set()
    split_counts = {}
    gt_counts = {}
    domain_counts = {}
    errors = []
    
    for idx, row in enumerate(reader):
        sample_id = row.get("sample_id", f"row_{idx}")
        if sample_id in seen_ids:
            errors.append(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        
        file_path_str = row.get("file_path", "")
        file_path = Path(file_path_str)
        if not file_path.exists():
            errors.append(f"File missing on disk: {file_path} (ID: {sample_id})")
            continue
            
        actual_size = os.path.getsize(file_path)
        expected_size = int(row.get("file_size_bytes", 0))
        if actual_size != expected_size:
            errors.append(f"Size mismatch for {sample_id}: actual {actual_size} != recorded {expected_size}")
            
        actual_hash = compute_sha256(file_path)
        expected_hash = row.get("sha256", "")
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch for {sample_id}: actual {actual_hash} != recorded {expected_hash}")
        if actual_hash in seen_hashes:
            errors.append(f"Duplicate hash across dataset for {sample_id}: {actual_hash}")
        seen_hashes.add(actual_hash)
        
        fmt = row.get("file_format", "")
        if fmt in ["JPEG", "PNG"]:
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    exp_w = int(row.get("width", 0))
                    exp_h = int(row.get("height", 0))
                    if w != exp_w or h != exp_h:
                        errors.append(f"Resolution mismatch for {sample_id}: actual {w}x{h} != recorded {exp_w}x{exp_h}")
            except Exception as e:
                errors.append(f"Failed to open image {file_path} for {sample_id}: {e}")
        elif row.get("category") not in ["corrupted_file", "unsupported_format"]:
            errors.append(f"Unrecognized file format without control category: {sample_id}")
            
        split = row.get("split", "unknown")
        gt = row.get("ground_truth", "unknown")
        dom = row.get("domain", "unknown")
        
        split_counts[split] = split_counts.get(split, 0) + 1
        gt_counts[gt] = gt_counts.get(gt, 0) + 1
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    if errors:
        print(f"\n[FAIL] Found {len(errors)} integrity errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        return False
        
    print("\n[Split Breakdown]")
    for s, c in sorted(split_counts.items()):
        print(f"  - {s}: {c} samples")
        
    print("\n[Ground Truth Breakdown]")
    for g, c in sorted(gt_counts.items()):
        print(f"  - {g}: {c} samples")
        
    print("\n[Domain Breakdown]")
    for d, c in sorted(domain_counts.items()):
        print(f"  - {d}: {c} samples")
        
    # Check expected partition sizes
    assert split_counts.get("held_out_test") == 105, "held_out_test count mismatch"
    assert split_counts.get("development_val") == 50, "development_val count mismatch"
    assert split_counts.get("operational_smoke") == 4, "operational_smoke count mismatch"
    assert gt_counts.get("REAL") == 77, "REAL count mismatch"
    assert gt_counts.get("FAKE") == 79, "FAKE count mismatch"
    assert gt_counts.get("REJECTION_CONTROL") == 3, "REJECTION_CONTROL count mismatch"
    assert domain_counts.get("facial") == 150, "facial domain count mismatch"
    assert domain_counts.get("general_scene") == 6, "general_scene domain count mismatch"
    assert domain_counts.get("control") == 3, "control domain count mismatch"

    print("\n[SUCCESS] Manifest verification passed 100% with zero errors!")
    return True

if __name__ == "__main__":
    success = verify_manifest()
    sys.exit(0 if success else 1)

