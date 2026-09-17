"""
Diagnostic Test Suite for UnivFDAdapter
Validates adapter integration, schema adherence, error handling, and input rejection.
NOTE: This script verifies operational pipeline behavior. It is NOT an accuracy evaluation
and does NOT verify image authenticity.
"""

import os
import sys
import json
from PIL import Image

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.univfd_adapter import UnivFDAdapter
from src.detector_schema import PREDICTION_REAL, PREDICTION_FAKE, PREDICTION_ERROR

ASSET_DIR = r"C:\Users\user\.gemini\antigravity-ide\brain\3b54ff09-c09f-4fe6-9276-f4cca9fbc6d7"

def main():
    print("=" * 75)
    print("UNIVERSALFAKEDETECT ADAPTER DIAGNOSTIC TEST SUITE")
    print("Verifies adapter schema, error handling, and inference integrity.")
    print("DISCLAIMER: This test validates integration mechanics, not model accuracy.")
    print("=" * 75 + "\n")

    adapter = UnivFDAdapter(lazy_load=True)

    # Prepare non-image text file for test case 5
    invalid_txt_path = os.path.join(BASE_DIR, "scripts", "_temp_invalid.txt")
    with open(invalid_txt_path, "w", encoding="utf-8") as f:
        f.write("This is a plain text file, not an image format.")

    test_cases = [
        {
            "id": 1,
            "category": "VALID_REAL_IMAGE",
            "name": "Known real photograph (smartphone capture)",
            "input": os.path.join(ASSET_DIR, "scratch", "real_phone_photo.png"),
            "ground_truth_label": "REAL",
            "expected_outcome": "SUCCESSFUL_INFERENCE",
            "nature": "Natural photographic capture from smartphone camera.",
        },
        {
            "id": 2,
            "category": "VALID_AI_IMAGE",
            "name": "Known AI-generated image (Google Imagen 3)",
            "input": os.path.join(ASSET_DIR, "ai_cyberpunk_city_1789660863192.jpg"),
            "ground_truth_label": "FAKE",
            "expected_outcome": "SUCCESSFUL_INFERENCE",
            "nature": "Modern text-to-image diffusion generation (out-of-distribution for ProGAN-trained head).",
        },
        {
            "id": 3,
            "category": "VALID_MANIPULATED_IMAGE",
            "name": "Known manipulated image (digital composite)",
            "input": os.path.join(ASSET_DIR, "scratch", "manipulated_image.png"),
            "ground_truth_label": "MANIPULATED",
            "expected_outcome": "SUCCESSFUL_INFERENCE",
            "nature": "Digitally spliced composite of photographic and synthetic elements.",
        },
        {
            "id": 4,
            "category": "CORRUPTED_FILE",
            "name": "Corrupted image file (truncated header)",
            "input": os.path.join(ASSET_DIR, "scratch", "corrupted_image.png"),
            "ground_truth_label": "CORRUPTED",
            "expected_outcome": "INPUT_REJECTION",
            "nature": "Invalid 20-byte file that cannot be decoded by image decoders.",
        },
        {
            "id": 5,
            "category": "UNSUPPORTED_INPUT",
            "name": "Unsupported file format (plain text document)",
            "input": invalid_txt_path,
            "ground_truth_label": "INVALID_TYPE",
            "expected_outcome": "INPUT_REJECTION",
            "nature": "Non-image text file submitted to image inference pipeline.",
        },
    ]

    all_results = []

    for tc in test_cases:
        print("-" * 75)
        print(f"TEST {tc['id']}: {tc['name']}")
        print(f"  Input Source:          {tc['input']}")
        print(f"  Category:              {tc['category']}")
        print(f"  Ground-Truth Label:    {tc['ground_truth_label']}")
        print(f"  Expected Outcome:      {tc['expected_outcome']}")
        print(f"  Test Nature:           {tc['nature']}")

        result = adapter.predict(tc["input"])
        res_dict = result.to_dict()

        # Operational status assessment
        is_error = result.prediction == PREDICTION_ERROR
        if tc["expected_outcome"] == "INPUT_REJECTION":
            status_classification = "INPUT_REJECTION_VERIFIED" if is_error else "FAILURE (Did not reject)"
        else:
            status_classification = "SUCCESSFUL_INFERENCE" if not is_error else "EXECUTION_FAILURE"

        # Correctness vs Model Limitation assessment
        if not is_error:
            if result.prediction == tc["ground_truth_label"]:
                domain_assessment = "ACCORDANT (Prediction matches ground truth label)"
            else:
                domain_assessment = (
                    "MODEL_LIMITATION_OBSERVED (Known ProGAN-trained classifier boundary: "
                    f"output {result.prediction} on {tc['ground_truth_label']} input)"
                )
        else:
            domain_assessment = "N/A (Input correctly rejected before model evaluation)"

        print(f"  Prediction:            {result.prediction}")
        print(f"  Raw Model Logit:       {f'{result.raw_logit:+.5f}' if result.raw_logit is not None else 'N/A'}")
        print(f"  Fake Probability:      {f'{result.fake_probability:.6f}' if result.fake_probability is not None else 'N/A'}")
        print(f"  Confidence:            {f'{result.confidence*100:.2f}%' if result.confidence is not None else 'N/A'}")
        print(f"  Threshold:             {result.threshold}")
        print(f"  Compute Device:        {result.device}")
        print(f"  Inference Latency:     {f'{result.inference_time_ms:.2f} ms' if result.inference_time_ms is not None else 'N/A'}")
        print(f"  Error Field:           {result.error if result.error else 'None'}")
        print(f"  Operational Status:    {status_classification}")
        print(f"  Domain Assessment:     {domain_assessment}")

        all_results.append({
            "test_id": tc["id"],
            "name": tc["name"],
            "category": tc["category"],
            "ground_truth_label": tc["ground_truth_label"],
            "expected_outcome": tc["expected_outcome"],
            "operational_status": status_classification,
            "domain_assessment": domain_assessment,
            "result": res_dict
        })
        print()

    # Clean up temporary test file
    if os.path.exists(invalid_txt_path):
        try:
            os.remove(invalid_txt_path)
        except Exception:
            pass

    print("=" * 75)
    print("TEST SUITE SUMMARY")
    print("=" * 75)
    for r in all_results:
        res = r["result"]
        print(f"Test {r['test_id']}: {r['name'][:30]:<30} | Pred: {res['prediction']:<6} | Logit: {str(res['raw_logit']):<9} | Op: {r['operational_status']}")

    summary_file = os.path.join(BASE_DIR, "scripts", "adapter_test_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved structured test results to: {summary_file}")

if __name__ == "__main__":
    main()
