
"""
Diagnostic Test Suite for ViTAdapter.

Validates adapter integration, schema adherence, inference execution,
and rejection of corrupted or unsupported inputs.

This is an operational diagnostic, NOT an accuracy evaluation.
"""

import os
import sys
import json
import traceback

from PIL import Image

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.vit_adapter import ViTAdapter
from src.detector_schema import (
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)

# All test images are stored inside the project.
ASSET_DIR = os.path.join(BASE_DIR, "benchmark", "test_images")
RESULTS_FILE = os.path.join(
    BASE_DIR, "scripts", "vit_adapter_test_results.json"
)


def prepare_test_files():
    """Create invalid-input samples and verify the required images exist."""
    os.makedirs(ASSET_DIR, exist_ok=True)

    # Intentionally invalid image bytes.
    corrupted_path = os.path.join(ASSET_DIR, "corrupted_image.png")
    with open(corrupted_path, "wb") as file:
        file.write(bytes([0, 1, 2, 3, 4, 5, 6, 7]))

    # Plain text submitted to the image pipeline.
    invalid_txt_path = os.path.join(ASSET_DIR, "_temp_vit_invalid.txt")
    with open(invalid_txt_path, "w", encoding="utf-8") as file:
        file.write("This is a plain text file, not an image.")

    required_images = [
        "real.jpg",
        "ai_generated.jpg",
        "manipulated.jpg",
    ]

    missing = [
        name for name in required_images
        if not os.path.isfile(os.path.join(ASSET_DIR, name))
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required test images in "
            f"{ASSET_DIR}: {', '.join(missing)}\n"
            "Copy real.jpg, ai_generated.jpg, and manipulated.jpg "
            "into benchmark/test_images before running this test."
        )

    return invalid_txt_path, corrupted_path


def build_test_cases():
    """Return test definitions using project-relative paths."""
    return [
        {
            "id": 1,
            "category": "VALID_IMAGE_PIPELINE",
            "name": "Real-image pipeline test",
            "input": os.path.join(ASSET_DIR, "real.jpg"),
            "ground_truth_label": "UNKNOWN",
            "expected_outcome": "SUCCESSFUL_INFERENCE",
            "nature": (
                "Synthetic test asset. Its actual origin is not verified "
                "as a real photograph."
            ),
        },
        {
            "id": 2,
            "category": "VALID_AI_IMAGE",
            "name": "AI-generated image",
            "input": os.path.join(ASSET_DIR, "ai_generated.jpg"),
            "ground_truth_label": "FAKE",
            "expected_outcome": "SUCCESSFUL_INFERENCE",
            "nature": "AI-generated test image.",
        },
        {
            "id": 3,
            "category": "VALID_MANIPULATED_IMAGE",
            "name": "Manipulated-style image",
            "input": os.path.join(ASSET_DIR, "manipulated.jpg"),
            "ground_truth_label": "MANIPULATED",
            "expected_outcome": "SUCCESSFUL_INFERENCE",
            "nature": "Synthetic composite-style test image.",
        },
        {
            "id": 4,
            "category": "CORRUPTED_FILE",
            "name": "Corrupted image file",
            "input": os.path.join(ASSET_DIR, "corrupted_image.png"),
            "ground_truth_label": "CORRUPTED",
            "expected_outcome": "INPUT_REJECTION",
            "nature": "Intentionally invalid image bytes.",
        },
        {
            "id": 5,
            "category": "UNSUPPORTED_INPUT",
            "name": "Plain text submitted as image",
            "input": os.path.join(ASSET_DIR, "_temp_vit_invalid.txt"),
            "ground_truth_label": "INVALID_TYPE",
            "expected_outcome": "INPUT_REJECTION",
            "nature": "Non-image text file.",
        },
    ]


def assess_result(test_case, result):
    """Assess operational behavior separately from prediction correctness."""
    is_error = result.prediction == PREDICTION_ERROR
    expects_rejection = (
        test_case["expected_outcome"] == "INPUT_REJECTION"
    )

    if expects_rejection:
        if is_error:
            operational_status = "INPUT_REJECTION_VERIFIED"
        else:
            operational_status = "FAILURE_DID_NOT_REJECT"
    else:
        if is_error:
            operational_status = "EXECUTION_FAILURE"
        else:
            operational_status = "SUCCESSFUL_INFERENCE"

    ground_truth = test_case["ground_truth_label"]

    # Unknown or non-binary labels must not be treated as model failures.
    if is_error:
        domain_assessment = "NOT_EVALUATED: inference/input error"
    elif ground_truth == "UNKNOWN":
        domain_assessment = "NOT_EVALUATED: ground truth unknown"
    elif ground_truth not in (PREDICTION_REAL, PREDICTION_FAKE):
        domain_assessment = (
            "NOT_EVALUATED: ground truth is not a supported "
            "binary REAL/FAKE label"
        )
    elif result.prediction == ground_truth:
        domain_assessment = "MATCH"
    else:
        domain_assessment = (
            f"MISMATCH: predicted {result.prediction}, "
            f"expected {ground_truth}"
        )

    return operational_status, domain_assessment


def print_result(test_case, result, operational_status, domain_assessment):
    """Print one test result."""
    print("-" * 75)
    print(f"TEST {test_case['id']}: {test_case['name']}")
    print(f"  Input Source:          {test_case['input']}")
    print(f"  Category:              {test_case['category']}")
    print(f"  Ground-Truth Label:    {test_case['ground_truth_label']}")
    print(f"  Expected Outcome:      {test_case['expected_outcome']}")
    print(f"  Test Nature:           {test_case['nature']}")
    print(f"  Prediction:            {result.prediction}")
    print(
        "  Raw Model Logit:       "
        + (
            f"{result.raw_logit:+.5f}"
            if result.raw_logit is not None else "N/A"
        )
    )
    print(
        "  Fake Probability:      "
        + (
            f"{result.fake_probability:.6f}"
            if result.fake_probability is not None else "N/A"
        )
    )
    print(
        "  Real Probability:      "
        + (
            f"{result.real_probability:.6f}"
            if result.real_probability is not None else "N/A"
        )
    )
    print(
        "  Confidence:            "
        + (
            f"{result.confidence * 100:.2f}%"
            if result.confidence is not None else "N/A"
        )
    )
    print(f"  Threshold:             {result.threshold}")
    print(f"  Compute Device:        {result.device}")
    print(
        "  Inference Latency:     "
        + (
            f"{result.inference_time_ms:.2f} ms"
            if result.inference_time_ms is not None else "N/A"
        )
    )
    print(f"  Error Field:           {result.error or 'None'}")
    print(f"  Operational Status:    {operational_status}")
    print(f"  Domain Assessment:     {domain_assessment}")
    print()


def main():
    print("=" * 75)
    print("VISION TRANSFORMER (ViT) ADAPTER DIAGNOSTIC TEST SUITE")
    print("Validates adapter integration and inference execution.")
    print("This is not an accuracy or authenticity evaluation.")
    print("=" * 75)

    try:
        invalid_txt_path, _ = prepare_test_files()
        test_cases = build_test_cases()
    except Exception as exc:
        print(f"\nTEST SETUP FAILED: {exc}")
        sys.exit(1)

    print(f"\nProject directory: {BASE_DIR}")
    print(f"Test image directory: {ASSET_DIR}\n")

    # Initialize once to test the actual registered adapter implementation.
    adapter = ViTAdapter(lazy_load=True)
    all_results = []

    try:
        for test_case in test_cases:
            try:
                result = adapter.predict(test_case["input"])
                result_dict = result.to_dict()

                operational_status, domain_assessment = assess_result(
                    test_case, result
                )

                print_result(
                    test_case,
                    result,
                    operational_status,
                    domain_assessment,
                )

            except Exception as exc:
                # Preserve the test suite if an unexpected exception occurs.
                result_dict = {
                    "prediction": PREDICTION_ERROR,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                operational_status = "UNHANDLED_EXCEPTION"
                domain_assessment = "NOT_EVALUATED: unhandled exception"

                print("-" * 75)
                print(f"TEST {test_case['id']}: {test_case['name']}")
                print(f"  Unexpected exception: {type(exc).__name__}: {exc}")
                print(f"  Operational Status:   {operational_status}")
                print()

            all_results.append({
                "test_id": test_case["id"],
                "name": test_case["name"],
                "category": test_case["category"],
                "input": test_case["input"],
                "ground_truth_label": test_case["ground_truth_label"],
                "expected_outcome": test_case["expected_outcome"],
                "operational_status": operational_status,
                "domain_assessment": domain_assessment,
                "result": result_dict,
            })

    finally:
        # Clean up the temporary text file.
        try:
            if os.path.exists(invalid_txt_path):
                os.remove(invalid_txt_path)
        except OSError as exc:
            print(f"Warning: could not remove temporary file: {exc}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print("=" * 75)
    print("TEST SUITE SUMMARY")
    print("=" * 75)

    for item in all_results:
        result = item["result"]
        print(
            f"Test {item['test_id']}: "
            f"{item['name'][:30]:<30} | "
            f"Pred: {str(result.get('prediction', 'ERROR')):<9} | "
            f"Status: {item['operational_status']}"
        )

    print(f"\nSaved structured results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
