"""
Package and Upload Full Dataset and Legacy Project to Private Hugging Face Dataset Repo
Repo: https://huggingface.co/datasets/Thiru1122/deepfake-dataset-full
"""

import os
import sys
import time
import zipfile
import hashlib
import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Thiru1122/deepfake-dataset-full"
REPO_TYPE = "dataset"
SOURCE_ROOT = Path(r"C:\Users\user\Desktop\multi model connection\currently working")
STAGING_DIR = Path(r"C:\Users\user\Desktop\hf_dataset_staging")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_zip_archive(source_dir: Path, output_zip: Path, arc_prefix: str = ""):
    print(f"[{time.strftime('%X')}] Archiving {source_dir} -> {output_zip.name}...")
    start_time = time.time()
    
    file_count = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(source_dir)
                archive_name = f"{arc_prefix}/{rel_path}" if arc_prefix else str(rel_path)
                zf.write(full_path, archive_name)
                file_count += 1
                if file_count % 20000 == 0:
                    print(f"  ... archived {file_count:,} files ({time.time() - start_time:.1f}s)")
                    
    duration = time.time() - start_time
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[{time.strftime('%X')}] Done: {output_zip.name} ({file_count:,} files, {size_mb:.2f} MB in {duration:.1f}s)")
    return file_count, size_mb


def create_project_zip(source_dir: Path, output_zip: Path):
    print(f"[{time.strftime('%X')}] Archiving project source files -> {output_zip.name}...")
    start_time = time.time()
    file_count = 0
    
    # Exclude Dataset directory since it is archived separately
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for root, dirs, files in os.walk(source_dir):
            rel_root = Path(root).relative_to(source_dir)
            # Skip Dataset and .git directories
            if rel_root.parts and rel_root.parts[0] in ("Dataset", ".git"):
                continue
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(source_dir)
                zf.write(full_path, str(rel_path))
                file_count += 1
                
    duration = time.time() - start_time
    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"[{time.strftime('%X')}] Done: {output_zip.name} ({file_count:,} files, {size_mb:.2f} MB in {duration:.1f}s)")
    return file_count, size_mb


def main():
    print(f"=== Starting Hugging Face Dataset Backup to {REPO_ID} ===")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    
    api = HfApi()
    
    # 1. Ensure private repository exists
    repo_url = api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=True,
        exist_ok=True,
    )
    print(f"Repository verified: {repo_url}")
    
    checksums = {}
    
    # 2. Archive Dataset Splits
    dataset_dir = SOURCE_ROOT / "Dataset"
    splits = [
        ("Train", "Dataset_Train.zip", "Train"),
        ("Validation", "Dataset_Validation.zip", "Validation"),
        ("Test", "Dataset_Test.zip", "Test"),
    ]
    
    for folder_name, zip_name, arc_prefix in splits:
        split_path = dataset_dir / folder_name
        zip_path = STAGING_DIR / zip_name
        if not zip_path.exists():
            create_zip_archive(split_path, zip_path, arc_prefix)
        else:
            print(f"Archive {zip_name} already exists. Skipping packaging.")
            
        sha = compute_sha256(zip_path)
        checksums[zip_name] = (sha, zip_path.stat().st_size)
        print(f"  SHA-256: {sha}")
        
    # 3. Archive Full Project (Excluding Dataset)
    project_zip = STAGING_DIR / "currently_working_project.zip"
    if not project_zip.exists():
        create_project_zip(SOURCE_ROOT, project_zip)
    else:
        print(f"Archive {project_zip.name} already exists. Skipping packaging.")
    sha_proj = compute_sha256(project_zip)
    checksums[project_zip.name] = (sha_proj, project_zip.stat().st_size)
    print(f"  SHA-256: {sha_proj}")
    
    # 4. Copy standalone manifest CSVs directly
    manifest_csv = SOURCE_ROOT / "dataset_manifest.csv"
    duplicates_csv = SOURCE_ROOT / "duplicates_metadata.csv"
    
    staged_manifest = STAGING_DIR / "dataset_manifest.csv"
    staged_duplicates = STAGING_DIR / "duplicates_metadata.csv"
    
    if manifest_csv.exists() and not staged_manifest.exists():
        shutil.copy2(manifest_csv, staged_manifest)
    if duplicates_csv.exists() and not staged_duplicates.exists():
        shutil.copy2(duplicates_csv, staged_duplicates)
        
    for staged_file in [staged_manifest, staged_duplicates]:
        if staged_file.exists():
            sha = compute_sha256(staged_file)
            checksums[staged_file.name] = (sha, staged_file.stat().st_size)
            print(f"Copied {staged_file.name} | SHA-256: {sha}")
            
    # 5. Generate Checksums File
    checksum_file = STAGING_DIR / "checksums.sha256"
    with open(checksum_file, "w") as f:
        for fname, (sha, size) in checksums.items():
            f.write(f"{sha}  {fname}  ({size:,} bytes)\n")
            
    # 6. Generate Dataset README.md
    readme_path = STAGING_DIR / "README.md"
    readme_content = f"""---
license: mit
task_categories:
- image-classification
size_categories:
- 100K<n<1M
tags:
- deepfake
- face-manipulation
- dataset-backup
---

# Deepfake Detection Clean Dataset & Project Backup

Private lossless archive of the deepfake image dataset and associated project assets.

## Dataset Summary
- **Total Images:** 190,335
- **Train Split:** 140,002 images (`Dataset_Train.zip`)
- **Validation Split:** 39,428 images (`Dataset_Validation.zip`)
- **Test Split:** 10,905 images (`Dataset_Test.zip`)
- **Manifest:** `dataset_manifest.csv` (19.8 MB)
- **Duplicates Metadata:** `duplicates_metadata.csv` (5.2 MB)
- **Legacy Project Code & Models:** `currently_working_project.zip` (Flask app, LSTM/ResNet `df_model.pt`, notebooks, test videos)

## File Integrity & Checksums (SHA-256)
```text
"""
    for fname, (sha, size) in checksums.items():
        readme_content += f"{sha}  {fname} ({size / (1024*1024):.2f} MB)\n"
    readme_content += """```

## How to Restore
```python
from huggingface_hub import hf_hub_download
import zipfile

for zip_name in ["Dataset_Train.zip", "Dataset_Validation.zip", "Dataset_Test.zip"]:
    local_path = hf_hub_download(
        repo_id="Thiru1122/deepfake-dataset-full",
        repo_type="dataset",
        filename=zip_name,
    )
    with zipfile.ZipFile(local_path, "r") as zf:
        zf.extractall("Dataset")
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print(f"Generated {readme_path.name}")
    
    # 7. Upload All Files to Hugging Face
    files_to_upload = [
        "README.md",
        "checksums.sha256",
        "dataset_manifest.csv",
        "duplicates_metadata.csv",
        "Dataset_Test.zip",
        "Dataset_Validation.zip",
        "Dataset_Train.zip",
        "currently_working_project.zip",
    ]
    
    print("\n=== Commencing Uploads to Hugging Face ===")
    for fname in files_to_upload:
        fpath = STAGING_DIR / fname
        if not fpath.exists():
            print(f"Warning: {fpath} does not exist. Skipping.")
            continue
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"[{time.strftime('%X')}] Uploading {fname} ({size_mb:.2f} MB)...")
        upload_start = time.time()
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Upload {fname} (SHA-256: {checksums.get(fname, ('', ''))[0][:8]})",
        )
        upload_duration = time.time() - upload_start
        print(f"[{time.strftime('%X')}] Uploaded {fname} in {upload_duration:.1f}s")
        
    # 8. Verify Files on Hugging Face
    print("\n=== Verifying Files on Remote Repository ===")
    remote_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    for rf in remote_files:
        print(f"  [Remote Verified] {rf}")
        
    print(f"\nAll files successfully uploaded to: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
