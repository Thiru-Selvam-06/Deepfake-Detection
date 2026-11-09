
"""
Diagnostic Test Suite for ConvNeXtDetectorAdapter.

Tests adapter integration, lazy loading, detector isolation, valid-image
inference, invalid-input handling, and probability consistency.

This is an operational test suite, not an accuracy evaluation.
"""

import os
import sys
import json
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.convnext_adapter import ConvNeXtDetectorAdapter
from src.detector_schema import (
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)

ASSET_DIR = os.path.join(BASE_DIR, "benchmark", "test_images")
RESULTS_FILE = os.path.join(
    BASE_DIR, "scripts", "convnext_adapter_test_results.json"
)


def safe_prediction(adapter, image_path):
    """Run prediction and return a structured error if an exception escapes."""
    try:
        result = adapter.predict(image_path)
        return result, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def record_check(results, name, passed, detail=""):
    """Record a diagnostic check without hiding failures."""
    status = "PASS" if passed else "FAIL"
    results[name] = {
        "status": status,
        "detail": detail,
    }
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def result_to_dict(result, exception=None):
    if exception:
        return {
            "prediction": PREDICTION_ERROR,
            "error": exception,
        }
    return result.to_dict()


def main():
    print("=" * 80)
    print("CHEN2024 CONVNEXT DETECTOR ADAPTER DIAGNOSTIC TEST")
    print("Operational checks only; predictions do not establish accuracy.")
    print(f"Project directory: {BASE_DIR}")
    print(f"Test images:       {ASSET_DIR}")
    print("=" * 80)

    os.makedirs(ASSET_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

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
        print("\nSETUP FAILED. Missing test images:")
        for name in missing:
            print(f"  - {os.path.join(ASSET_DIR, name)}")
        print("\nCopy the three images into benchmark\\test_images and rerun.")
        sys.exit(1)

    corrupt_path = os.path.join(ASSET_DIR, "corrupted_image.png")
    unsupported_path = os.path.join(ASSET_DIR, "_temp_convnext_unsupported.txt")
    missing_path = os.path.join(
        BASE_DIR, "models", "non_existent_image_12345.png"
    )

    # Create invalid-input fixtures.
    with open(corrupt_path, "wb") as file:
        file.write(bytes([0, 1, 2, 3, 4, 5, 6, 7]))

    with open(unsupported_path, "w", encoding="utf-8") as file:
        file.write("Plain text file, not an image.")

    image_paths = {
        "real": os.path.join(ASSET_DIR, "real.jpg"),
        "ai_generated": os.path.join(ASSET_DIR, "ai_generated.jpg"),
        "manipulated": os.path.join(ASSET_DIR, "manipulated.jpg"),
    }

    results = {
        "metadata": {
            "test_type": "operational_diagnostics",
            "model": "chen2024_convnext",
            "accuracy_evaluation": False,
            "test_images": image_paths,
        },
        "checks": {},
        "inferences": {},
    }

    adapter = None
    try:
        # Test 11: lazy loading.
        adapter = ConvNeXtDetectorAdapter(lazy_load=True)
        lazy_ok = getattr(adapter, "model", None) is None
        record_check(
            results["checks"],
            "test_11_lazy_loading",
            lazy_ok,
            "Model is not initialized before first prediction."
            if lazy_ok else
            "Model was already initialized in the constructor.",
        )

        # Test 12: check imports/instances without assuming absent variables
        # are proof of isolation.
        univfd_module = sys.modules.get("src.univfd_adapter")
        vit_module = sys.modules.get("src.vit_adapter")

        univfd_instances = (
            getattr(univfd_module, "_INSTANCES", None)
            if univfd_module is not None else None
        )
        vit_instances = (
            getattr(vit_module, "_INSTANCES", None)
            if vit_module is not None else None
        )

        isolation_ok = (
            not univfd_instances and not vit_instances
        )
        record_check(
            results["checks"],
            "test_12_detector_isolation",
            isolation_ok,
            "No UnivFD/ViT instances found at this point."
            if isolation_ok else
            "Other detector instances appear to exist; inspect module state.",
        )

        # Tests 1-3: valid image inference.
        valid_results = {}
        valid_cases = [
            ("test_01_valid_real_image", "real"),
            ("test_02_valid_ai_image", "ai_generated"),
            ("test_03_valid_manipulated_image", "manipulated"),
        ]

        for check_name, label in valid_cases:
            path = image_paths[label]
            print(f"\n{check_name}: {path}")
            result, exception = safe_prediction(adapter, path)
            results["inferences"][label] = result_to_dict(result, exception)

            valid = (
                result is not None
                and result.prediction in (PREDICTION_REAL, PREDICTION_FAKE)
                and result.error is None
            )

            record_check(
                results["checks"],
                check_name,
                valid,
                (
                    f"Prediction={result.prediction}, "
                    f"fake_probability={result.fake_probability}, "
                    f"real_probability={result.real_probability}, "
                    f"device={result.device}, "
                    f"latency_ms={result.inference_time_ms}"
                ) if result is not None else exception,
            )

            if result is not None:
                valid_results[label] = result

        # Verify on-demand loading after first prediction.
        model_loaded = getattr(adapter, "model", None) is not None
        record_check(
            results["checks"],
            "test_model_loaded_after_prediction",
            model_loaded,
            "Model is initialized." if model_loaded
            else "Model attribute is still None after predictions.",
        )

        # Test 4: corrupted file rejection.
        corrupt_result, corrupt_exception = safe_prediction(
            adapter, corrupt_path
        )
        results["inferences"]["corrupted"] = result_to_dict(
            corrupt_result, corrupt_exception
        )
        corrupt_ok = (
            corrupt_result is not None
            and corrupt_result.prediction == PREDICTION_ERROR
            and corrupt_result.error is not None
        )
        record_check(
            results["checks"],
            "test_04_corrupted_image",
            corrupt_ok,
            (
                f"Prediction={corrupt_result.prediction}, "
                f"error={corrupt_result.error}"
            ) if corrupt_result is not None else corrupt_exception,
        )

        # Test 5: unsupported file rejection.
        unsupported_result, unsupported_exception = safe_prediction(
            adapter, unsupported_path
        )
        results["inferences"]["unsupported_text"] = result_to_dict(
            unsupported_result, unsupported_exception
        )
        unsupported_ok = (
            unsupported_result is not None
            and unsupported_result.prediction == PREDICTION_ERROR
            and unsupported_result.error is not None
        )
        record_check(
            results["checks"],
            "test_05_unsupported_file",
            unsupported_ok,
            (
                f"Prediction={unsupported_result.prediction}, "
                f"error={unsupported_result.error}"
            ) if unsupported_result is not None else unsupported_exception,
        )

        # Test 6: missing path rejection.
        missing_result, missing_exception = safe_prediction(
            adapter, missing_path
        )
        results["inferences"]["missing_path"] = result_to_dict(
            missing_result, missing_exception
        )
        missing_ok = (
            missing_result is not None
            and missing_result.prediction == PREDICTION_ERROR
            and missing_result.error is not None
            and "does not exist" in missing_result.error.lower()
        )
        record_check(
            results["checks"],
            "test_06_missing_path",
            missing_ok,
            (
                f"Prediction={missing_result.prediction}, "
                f"error={missing_result.error}"
            ) if missing_result is not None else missing_exception,
        )

        # Test 7: None input rejection.
        none_result, none_exception = safe_prediction(adapter, None)
        results["inferences"]["none_input"] = result_to_dict(
            none_result, none_exception
        )
        none_ok = (
            none_result is not None
            and none_result.prediction == PREDICTION_ERROR
            and none_result.error is not None
        )
        record_check(
            results["checks"],
            "test_07_none_input",
            none_ok,
            (
                f"Prediction={none_result.prediction}, "
                f"error={none_result.error}"
            ) if none_result is not None else none_exception,
        )

        # Test 8: probability bounds.
        bounds_ok = True
        bounds_details = []
        for label, result in valid_results.items():
            values = {
                "fake_probability": result.fake_probability,
                "real_probability": result.real_probability,
                "confidence": result.confidence,
            }
            for field, value in values.items():
                if (
                    value is None
                    or not isinstance(value, (int, float))
                    or not 0.0 <= value <= 1.0
                ):
                    bounds_ok = False
                    bounds_details.append(f"{label}.{field}={value}")

        if len(valid_results) != 3:
            bounds_ok = False
            bounds_details.append("Not all three images produced results.")

        record_check(
            results["checks"],
            "test_08_probability_bounds",
            bounds_ok,
            "; ".join(bounds_details) if bounds_details
            else "All valid-result probabilities are within [0, 1].",
        )

        # Test 9: probability sum close to 1.
        sum_ok = len(valid_results) == 3
        sum_details = []
        for label, result in valid_results.items():
            if (
                result.fake_probability is None
                or result.real_probability is None
            ):
                sum_ok = False
                sum_details.append(f"{label}: missing probability")
                continue

            probability_sum = (
                result.fake_probability + result.real_probability
            )
            if abs(probability_sum - 1.0) >= 1e-4:
                sum_ok = False
                sum_details.append(f"{label}: sum={probability_sum}")

        record_check(
            results["checks"],
            "test_09_probability_sum",
            sum_ok,
            "; ".join(sum_details) if sum_details
            else "Each valid result's probabilities sum to approximately 1.",
        )

        # Test 10: confidence equals the probability of the predicted class.
        confidence_ok = len(valid_results) == 3
        confidence_details = []
        for label, result in valid_results.items():
            if result.confidence is None:
                confidence_ok = False
                confidence_details.append(f"{label}: confidence missing")
                continue

            if result.prediction == PREDICTION_FAKE:
                expected_confidence = result.fake_probability
            elif result.prediction == PREDICTION_REAL:
                expected_confidence = result.real_probability
            else:
                confidence_ok = False
                confidence_details.append(
                    f"{label}: unexpected prediction {result.prediction}"
                )
                continue

            if (
                expected_confidence is None
                or abs(result.confidence - expected_confidence) >= 1e-6
                or result.confidence < 0.5
            ):
                confidence_ok = False
                confidence_details.append(
                    f"{label}: confidence={result.confidence}, "
                    f"expected={expected_confidence}"
                )

        record_check(
            results["checks"],
            "test_10_confidence_alignment",
            confidence_ok,
            "; ".join(confidence_details) if confidence_details
            else "Confidence matches the predicted class probability.",
        )

    except Exception as exc:
        results["fatal_error"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"\nUNEXPECTED TEST ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    finally:
        for path in (unsupported_path,):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"Warning: could not remove {path}: {exc}")

        with open(RESULTS_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

    checks = results["checks"]
    passed = sum(
        1 for item in checks.values() if item.get("status") == "PASS"
    )
    failed = sum(
        1 for item in checks.values() if item.get("status") == "FAIL"
    )

    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC SUMMARY: {passed} passed, {failed} failed")
    print(f"Results saved to: {RESULTS_FILE}")
    print("Passing contract checks does not establish detector accuracy.")
    print("=" * 80)

    if failed or "fatal_error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
