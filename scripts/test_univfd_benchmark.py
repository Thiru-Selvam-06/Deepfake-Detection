"""
Test Suite for UnivFD Benchmark Framework
Validates metadata loading, resilience to missing files, corrupted file rejection,
unsupported format handling, DetectorResult contract adherence, and probability bounds.
"""

import os
import sys
import json
import tempfile
from PIL import Image
import torch

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.univfd_adapter import UnivFDAdapter
from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
    VALID_PREDICTIONS,
)

METADATA_PATH = os.path.join(BASE_DIR, "benchmark", "metadata.json")


def test_metadata_loading():
    print("[TEST 1] Verifying benchmark/metadata.json structure...")
    assert os.path.exists(METADATA_PATH), f"Metadata file missing at {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "entries" in data, "Missing 'entries' key in metadata.json"
    entries = data["entries"]
    assert len(entries) > 0, "Metadata contains zero entries"

    for e in entries:
        assert "id" in e, f"Entry missing 'id': {e}"
        assert "filename" in e, f"Entry missing 'filename': {e}"
        assert "category" in e, f"Entry missing 'category': {e}"
        assert "ground_truth" in e, f"Entry missing 'ground_truth': {e}"
    print(f"  PASS: Successfully loaded {len(entries)} valid metadata entries.\n")


def test_missing_files_handling():
    print("[TEST 2] Verifying resilience to missing files...")
    adapter = UnivFDAdapter(lazy_load=True)
    non_existent_path = os.path.join(BASE_DIR, "benchmark", "images", "non_existent_image.png")
    
    result = adapter.predict(non_existent_path)
    assert isinstance(result, DetectorResult), "Return value must be DetectorResult"
    assert result.prediction == PREDICTION_ERROR, f"Expected ERROR for missing file, got: {result.prediction}"
    assert result.error is not None, "Error message must be present for missing file"
    assert "does not exist" in result.error or "Invalid" in result.error
    print("  PASS: Missing file correctly reported as ERROR without crashing.\n")


def test_valid_image_inference():
    print("[TEST 3] Verifying valid image inference & DetectorResult schema...")
    adapter = UnivFDAdapter(lazy_load=True)
    
    # Create a small valid test PIL image
    test_img = Image.new("RGB", (256, 256), color=(120, 150, 180))
    result = adapter.predict(test_img)

    assert isinstance(result, DetectorResult), "Return value must be DetectorResult"
    assert result.prediction in {PREDICTION_REAL, PREDICTION_FAKE}, f"Unexpected prediction: {result.prediction}"
    assert result.error is None, f"Valid image produced unexpected error: {result.error}"

    # Verify probability bounds [0.0, 1.0]
    assert result.fake_probability is not None, "fake_probability must not be None"
    assert 0.0 <= result.fake_probability <= 1.0, f"fake_probability out of bounds: {result.fake_probability}"

    assert result.real_probability is not None, "real_probability must not be None"
    assert 0.0 <= result.real_probability <= 1.0, f"real_probability out of bounds: {result.real_probability}"

    prob_sum = result.fake_probability + result.real_probability
    assert abs(prob_sum - 1.0) < 1e-4, f"Real and Fake probabilities must sum to 1.0 (got {prob_sum})"

    assert result.confidence is not None, "confidence must not be None"
    assert 0.0 <= result.confidence <= 1.0, f"confidence out of bounds: {result.confidence}"
    assert result.threshold == 0.50, f"threshold must be 0.50, got {result.threshold}"
    assert result.raw_logit is not None, "raw_logit must not be None"
    assert result.inference_time_ms is not None and result.inference_time_ms >= 0, "Invalid latency"

    print(f"  PASS: Valid image successfully inferred (Prediction: {result.prediction}, Latency: {result.inference_time_ms:.2f} ms).\n")


def test_corrupted_file_rejection():
    print("[TEST 4] Verifying corrupted file rejection...")
    adapter = UnivFDAdapter(lazy_load=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"CORRUPTED_TRUNCATED_HEADER_TEST")
        tmp_path = tmp.name

    try:
        result = adapter.predict(tmp_path)
        assert isinstance(result, DetectorResult), "Return value must be DetectorResult"
        assert result.prediction == PREDICTION_ERROR, f"Corrupted file must return ERROR, got: {result.prediction}"
        assert result.error is not None, "Error message must be present for corrupted file"
        print(f"  PASS: Corrupted file cleanly rejected with ERROR ('{result.error}').\n")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_unsupported_file_rejection():
    print("[TEST 5] Verifying unsupported format rejection...")
    adapter = UnivFDAdapter(lazy_load=True)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"This is a plain text document, not an image format.")
        tmp_path = tmp.name

    try:
        result = adapter.predict(tmp_path)
        assert isinstance(result, DetectorResult), "Return value must be DetectorResult"
        assert result.prediction == PREDICTION_ERROR, f"Unsupported file must return ERROR, got: {result.prediction}"
        assert result.error is not None, "Error message must be present for unsupported file"
        print(f"  PASS: Unsupported text format cleanly rejected with ERROR.\n")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_all_tests():
    print("=" * 75)
    print("RUNNING UNIVFD BENCHMARK TEST SUITE")
    print("=" * 75 + "\n")

    test_metadata_loading()
    test_missing_files_handling()
    test_valid_image_inference()
    test_corrupted_file_rejection()
    test_unsupported_file_rejection()

    print("=" * 75)
    print("ALL 5 BENCHMARK CONTRACT TESTS PASSED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()
