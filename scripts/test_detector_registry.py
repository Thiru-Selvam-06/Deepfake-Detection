"""
Diagnostic Test Suite for Detector Registry
Validates:
- Lazy registration without model instantiation
- Registration and unregistration APIs
- Isolated retrieval (selecting one detector does NOT instantiate the other)
- Inspection without weight loading
- Dynamic multi-detector discovery
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detector_schema import DetectorResult, PREDICTION_REAL


class DummyTestDetector:
    """Mock detector to test registration without heavy ML dependencies."""
    DETECTOR_NAME = "DummyTestDetector"
    MODEL_NAME = "Mock/Model-v1"
    DECISION_THRESHOLD = 0.50

    def __init__(self, lazy_load: bool = True):
        self.lazy_load = lazy_load
        self.weights_loaded = False

    def load_model(self):
        self.weights_loaded = True
        return True

    def predict(self, image_input):
        return DetectorResult(
            detector_name=self.DETECTOR_NAME,
            model_name=self.MODEL_NAME,
            prediction=PREDICTION_REAL,
            fake_probability=0.1,
            real_probability=0.9,
            raw_logit=-2.0,
            confidence=0.9,
            threshold=self.DECISION_THRESHOLD,
        )


def test_registry():
    print("=" * 75)
    print("DETECTOR REGISTRY VERIFICATION SUITE")
    print("=" * 75 + "\n")

    # 1. Test Import isolation
    print("1. Testing registry import isolation...")
    import src.detector_registry as registry
    assert len(registry._INSTANCES) == 0, "No detectors should be instantiated on import!"
    print("   [PASS] Registry imported cleanly with 0 active instances.")

    # 2. Test List detectors
    print("\n2. Testing list_detectors()...")
    detectors = registry.list_detectors()
    print(f"   Registered detectors: {detectors}")
    assert "univfd" in detectors, "univfd should be registered by default"
    assert "vit" in detectors, "vit should be registered by default"
    assert "chen2024_convnext" in detectors, "chen2024_convnext should be registered by default"
    assert len(detectors) == 3, f"Expected 3 default detectors, found {len(detectors)}"
    print("   [PASS] All 3 default detectors correctly registered.")

    # 3. Test Metadata retrieval without instantiation
    print("\n3. Testing get_detector_info() without instantiation...")
    u_info = registry.get_detector_info("univfd")
    v_info = registry.get_detector_info("vit")
    c_info = registry.get_detector_info("chen2024_convnext")
    assert u_info["name"] == "univfd"
    assert v_info["name"] == "vit"
    assert c_info["name"] == "chen2024_convnext"
    assert not u_info["instantiated"]
    assert not v_info["instantiated"]
    assert not c_info["instantiated"]
    assert len(registry._INSTANCES) == 0, "Info check must not instantiate models!"
    print("   [PASS] Metadata retrieved with 0 model instantiations.")

    # 4. Test Single Detector Retrieval (Isolation)
    print("\n4. Testing isolated detector instantiation...")
    univfd_adapter = registry.get_detector("univfd", lazy_load=True)
    assert univfd_adapter is not None
    assert "univfd" in registry._INSTANCES
    assert "vit" not in registry._INSTANCES, "Getting univfd must NOT instantiate vit!"
    assert "chen2024_convnext" not in registry._INSTANCES, "Getting univfd must NOT instantiate convnext!"
    assert univfd_adapter.model is None, "Lazy loading must not load model weights into RAM!"
    print("   [PASS] UnivFD instantiated in lazy mode without touching ViT or ConvNeXt.")

    # 5. Test ViT Detector Retrieval
    print("\n5. Testing ViT detector instantiation...")
    vit_adapter = registry.get_detector("vit", lazy_load=True)
    assert vit_adapter is not None
    assert "vit" in registry._INSTANCES
    assert "chen2024_convnext" not in registry._INSTANCES
    assert vit_adapter.model is None, "Lazy loading must not load model weights into RAM!"
    print("   [PASS] ViT instantiated in lazy mode without touching ConvNeXt.")

    # 5b. Test ConvNeXt Detector Retrieval
    print("\n5b. Testing ConvNeXt detector instantiation...")
    convnext_adapter = registry.get_detector("chen2024_convnext", lazy_load=True)
    assert convnext_adapter is not None
    assert "chen2024_convnext" in registry._INSTANCES
    assert convnext_adapter.model is None, "Lazy loading must not load model weights into RAM!"
    print("   [PASS] ConvNeXt instantiated in lazy mode.")

    # 6. Test Dynamic Registration of New / Third Detector
    print("\n6. Testing dynamic registration of mock detector...")
    registry.register_detector(
        name="dummy",
        adapter_cls_or_factory=DummyTestDetector,
        metadata={"description": "Mock detector for testing third-slot capability"},
    )
    assert "dummy" in registry.list_detectors()
    dummy_info = registry.get_detector_info("dummy")
    assert dummy_info["metadata"]["description"] == "Mock detector for testing third-slot capability"

    dummy_adapter = registry.get_detector("dummy")
    assert isinstance(dummy_adapter, DummyTestDetector)
    dummy_res = dummy_adapter.predict("test")
    assert dummy_res.prediction == PREDICTION_REAL
    print("   [PASS] Dynamic registration and execution of candidate detector verified.")

    # 7. Test Unregistration
    print("\n7. Testing unregistration...")
    assert registry.unregister_detector("dummy") is True
    assert "dummy" not in registry.list_detectors()
    assert registry.unregister_detector("dummy") is False
    print("   [PASS] Unregistration successfully cleaned up mock detector.")

    # 8. Reset instances
    registry.reset_instances()
    assert len(registry._INSTANCES) == 0
    print("\n   [PASS] Reset instances cleanly wiped active cache.")

    print("\n" + "=" * 75)
    print("ALL DETECTOR REGISTRY CONTRACT TESTS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    test_registry()
