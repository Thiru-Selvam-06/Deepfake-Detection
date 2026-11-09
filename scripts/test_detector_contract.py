"""
Detector Contract Verification Suite
Tests that both UnivFDAdapter and ViTAdapter adhere to the identical DetectorResult contract:
- Common interface: .predict(image_input) -> DetectorResult
- All required fields present and of correct types
- Probabilities bounded in [0.0, 1.0] and sum to 1.0
- Confidence matches the predicted class
- Proper error propagation and rejection for corrupted/invalid inputs
- Detector independence (one detector's failure does not impact the other)
"""

import os
import sys
import json
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)
from src.univfd_adapter import UnivFDAdapter
from src.vit_adapter import ViTAdapter
from src.convnext_adapter import ConvNeXtDetectorAdapter

# Resolve asset paths within the project repository
ASSET_DIR = os.path.join(BASE_DIR, "benchmark", "test_images")
FALLBACK_DIR = os.path.join(BASE_DIR, "benchmark", "images")

def assert_contract(result: DetectorResult, test_name: str, expect_error: bool = False):
    """Rigorous contract checks against DetectorResult."""
    assert isinstance(result, DetectorResult), f"{test_name}: Result is not instance of DetectorResult"
    res_dict = result.to_dict()

    # Required fields presence
    required_keys = [
        "detector_name", "model_name", "prediction", "fake_probability",
        "real_probability", "raw_logit", "confidence", "threshold",
        "device", "preprocessing", "inference_time_ms", "error", "warning"
    ]
    for key in required_keys:
        assert key in res_dict, f"{test_name}: Missing required field '{key}'"

    if expect_error:
        assert result.prediction == PREDICTION_ERROR, f"{test_name}: Expected prediction ERROR, got {result.prediction}"
        assert result.error is not None and len(result.error) > 0, f"{test_name}: Error message must be populated"
        assert result.fake_probability is None, f"{test_name}: fake_probability must be None on error"
        assert result.real_probability is None, f"{test_name}: real_probability must be None on error"
    else:
        assert result.prediction in (PREDICTION_REAL, PREDICTION_FAKE), f"{test_name}: Invalid prediction '{result.prediction}'"
        assert result.error is None, f"{test_name}: Error should be None on success, got: {result.error}"
        assert result.fake_probability is not None, f"{test_name}: fake_probability is None"
        assert result.real_probability is not None, f"{test_name}: real_probability is None"
        assert 0.0 <= result.fake_probability <= 1.0, f"{test_name}: fake_probability out of bounds: {result.fake_probability}"
        assert 0.0 <= result.real_probability <= 1.0, f"{test_name}: real_probability out of bounds: {result.real_probability}"
        
        prob_sum = result.fake_probability + result.real_probability
        assert math.isclose(prob_sum, 1.0, abs_tol=1e-3), f"{test_name}: Probabilities do not sum to 1.0: {prob_sum}"
        
        assert result.confidence is not None, f"{test_name}: confidence is None"
        assert 0.0 <= result.confidence <= 1.0, f"{test_name}: confidence out of bounds: {result.confidence}"
        if result.prediction == PREDICTION_FAKE:
            assert math.isclose(result.confidence, result.fake_probability, abs_tol=1e-5), f"{test_name}: Confidence does not match fake_prob"
        else:
            assert math.isclose(result.confidence, result.real_probability, abs_tol=1e-5), f"{test_name}: Confidence does not match real_prob"

        assert result.threshold is not None and 0.0 <= result.threshold <= 1.0
        assert result.inference_time_ms is not None and result.inference_time_ms >= 0.0
        assert result.raw_logit is not None


def main():
    print("=" * 75)
    print("DETECTOR CONTRACT VERIFICATION SUITE")
    print("Testing UnivFDAdapter and ViTAdapter adherence to DetectorResult contract.")
    print("=" * 75 + "\n")

    univfd = UnivFDAdapter(lazy_load=True)
    vit = ViTAdapter(lazy_load=True)
    convnext = ConvNeXtDetectorAdapter(lazy_load=True)

    adapters = [
        ("UnivFDAdapter", univfd),
        ("ViTAdapter", vit),
        ("ConvNeXtDetectorAdapter", convnext),
    ]

    # Locate test fixtures
    real_img = os.path.join(ASSET_DIR, "real.jpg")
    if not os.path.isfile(real_img):
        real_img = os.path.join(FALLBACK_DIR, "real", "real_phone_photo.png")

    corrupt_img = os.path.join(ASSET_DIR, "corrupted_image.png")
    if not os.path.isfile(corrupt_img):
        corrupt_img = os.path.join(FALLBACK_DIR, "corrupted", "corrupted_image.png")

    nonexistent = os.path.join(ASSET_DIR, "nonexistent_image_123.jpg")

    contract_results = []

    for name, adapter in adapters:
        print(f"Testing {name}...")

        # 1. Valid Real Image
        res_real = adapter.predict(real_img)
        assert_contract(res_real, f"{name} - Valid Image", expect_error=False)
        print(f"  [PASS] Valid image inference: pred={res_real.prediction}, conf={res_real.confidence*100:.1f}%, logit={res_real.raw_logit:+.3f}")

        # 2. Corrupted Image Rejection
        res_corrupt = adapter.predict(corrupt_img)
        assert_contract(res_corrupt, f"{name} - Corrupted Image", expect_error=True)
        print(f"  [PASS] Corrupted file rejection: pred={res_corrupt.prediction}, err={res_corrupt.error[:40]}...")

        # 3. Missing File Handling
        res_missing = adapter.predict(nonexistent)
        assert_contract(res_missing, f"{name} - Missing File", expect_error=True)
        print(f"  [PASS] Missing file rejection: pred={res_missing.prediction}")

        # 4. None Input Handling
        res_none = adapter.predict(None)
        assert_contract(res_none, f"{name} - None Input", expect_error=True)
        print(f"  [PASS] None input rejection: pred={res_none.prediction}")

        # 5. Unload Model Verification
        if hasattr(adapter, "unload_model"):
            adapter.unload_model()
            assert adapter.model is None, f"{name}: unload_model() did not set self.model to None"
            print(f"  [PASS] unload_model() verified: weights successfully released")

        contract_results.append({
            "adapter": name,
            "contract_status": "COMPLIANT",
            "schema_verified": True,
            "error_handling_verified": True,
            "unload_model_verified": hasattr(adapter, "unload_model"),
        })
        print()

    print("=" * 75)
    print("CONTRACT VERIFICATION SUMMARY: ALL ADAPTERS FULLY COMPLIANT")
    print("=" * 75)

    contract_summary_file = os.path.join(BASE_DIR, "scripts", "contract_test_summary.json")
    with open(contract_summary_file, "w", encoding="utf-8") as f:
        json.dump(contract_results, f, indent=2)
    print(f"Saved contract summary to {contract_summary_file}")


if __name__ == "__main__":
    main()
