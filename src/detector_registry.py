"""
Detector Registry Module
Provides centralized, decoupled discovery, registration, and lazy retrieval of deepfake detector adapters.
Ensures strict independence between detector implementations and prevents unnecessary model weight loading.
"""

import inspect
import logging
from typing import Dict, Any, List, Optional, Callable, Union

logger = logging.getLogger(__name__)

# Registry mapping: detector_name -> {"factory": Callable, "metadata": Dict[str, Any]}
_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Active adapter instance cache: detector_name -> adapter_instance
_INSTANCES: Dict[str, Any] = {}


def register_detector(
    name: str,
    adapter_cls_or_factory: Union[type, Callable[..., Any]],
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """
    Registers a detector adapter class or factory function under a unique name.

    Args:
        name: Unique identifier string (case-insensitive).
        adapter_cls_or_factory: Adapter class or factory callable returning a DetectorResult-compliant adapter.
        metadata: Optional dictionary of informational metadata (architecture, threshold, etc.).
        overwrite: If True, replaces any existing registration under the same name.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("Detector name cannot be empty.")

    if key in _REGISTRY and not overwrite:
        raise ValueError(
            f"Detector '{name}' is already registered. Set overwrite=True to replace it."
        )

    meta = metadata.copy() if metadata else {}
    _REGISTRY[key] = {
        "factory": adapter_cls_or_factory,
        "metadata": meta,
    }
    # Invalidate cached instance if replacing
    if key in _INSTANCES:
        del _INSTANCES[key]

    logger.debug(f"Registered detector '{key}'.")


def unregister_detector(name: str) -> bool:
    """
    Unregisters a detector by name.

    Returns:
        True if removed, False if not found.
    """
    key = name.strip().lower()
    if key in _REGISTRY:
        del _REGISTRY[key]
        _INSTANCES.pop(key, None)
        return True
    return False


def is_detector_registered(name: str) -> bool:
    """Check if a detector name is currently registered."""
    return name.strip().lower() in _REGISTRY


def list_detectors() -> List[str]:
    """Returns a sorted list of all registered detector names."""
    return sorted(list(_REGISTRY.keys()))


def get_detector_info(name: str) -> Dict[str, Any]:
    """
    Retrieves informational metadata for a registered detector without instantiating it.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Detector '{name}' is not registered. Registered detectors: {list_detectors()}"
        )

    entry = _REGISTRY[key]
    factory = entry["factory"]
    meta = entry["metadata"].copy()

    # Enrich metadata with class attributes if available without instantiating
    if isinstance(factory, type):
        meta.setdefault("detector_name", getattr(factory, "DETECTOR_NAME", factory.__name__))
        meta.setdefault("model_name", getattr(factory, "MODEL_NAME", "Unknown"))
        meta.setdefault("threshold", getattr(factory, "DECISION_THRESHOLD", None))
        meta.setdefault("doc", inspect.getdoc(factory))

    return {
        "name": key,
        "factory_name": getattr(factory, "__name__", str(factory)),
        "metadata": meta,
        "instantiated": key in _INSTANCES,
    }


def get_detector(
    name: str,
    lazy_load: bool = True,
    singleton: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Retrieves or instantiates a detector adapter by name.

    Args:
        name: Name of the registered detector (case-insensitive).
        lazy_load: If True, instructs the adapter not to load heavyweight weights until first inference.
        singleton: If True, caches and returns the shared instance. If False, returns a new instance.
        **kwargs: Additional parameters forwarded to the adapter constructor/factory.

    Returns:
        An instantiated adapter adhering to the DetectorResult contract.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Detector '{name}' is not registered. Available detectors: {list_detectors()}"
        )

    if singleton and key in _INSTANCES:
        return _INSTANCES[key]

    entry = _REGISTRY[key]
    factory = entry["factory"]

    # Inspect factory/cls arguments to pass lazy_load if accepted
    pass_kwargs = kwargs.copy()
    try:
        sig = inspect.signature(factory)
        if "lazy_load" in sig.parameters and "lazy_load" not in pass_kwargs:
            pass_kwargs["lazy_load"] = lazy_load
    except (ValueError, TypeError):
        pass

    try:
        adapter = factory(**pass_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate detector '{name}': {e}") from e

    # Verify adapter has predict method
    if not hasattr(adapter, "predict") or not callable(getattr(adapter, "predict")):
        raise TypeError(
            f"Instantiated detector '{name}' does not implement required 'predict' method."
        )

    if singleton:
        _INSTANCES[key] = adapter

    return adapter


def get_all_detectors(
    lazy_load: bool = True,
    singleton: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Instantiates and returns a dictionary of all registered detector adapters {name: adapter}.
    """
    return {
        name: get_detector(name, lazy_load=lazy_load, singleton=singleton, **kwargs)
        for name in list_detectors()
    }


def reset_instances() -> None:
    """Clears all cached detector instances."""
    _INSTANCES.clear()


# ==============================================================================
# Built-in Lazy Factories for Default Detectors
# These factories ensure that importing this module does NOT import heavy ML
# packages or instantiate detector weights until get_detector is actually called.
# ==============================================================================

def _univfd_factory(**kwargs: Any) -> Any:
    from src.univfd_adapter import UnivFDAdapter
    return UnivFDAdapter(**kwargs)


def _vit_factory(**kwargs: Any) -> Any:
    from src.vit_adapter import ViTAdapter
    return ViTAdapter(**kwargs)


# Register UniversalFakeDetect
register_detector(
    name="univfd",
    adapter_cls_or_factory=_univfd_factory,
    metadata={
        "detector_name": "UniversalFakeDetect",
        "model_name": "CLIP:ViT-L/14 + fc_weights.pth",
        "architecture": "CLIP ViT-L/14 + Linear Probe",
        "threshold": 0.50,
        "input_dimensions": "3x224x224",
        "label_mapping": {0: "REAL", 1: "FAKE"},
        "description": "CVPR 2023 UniversalFakeDetect model using CLIP ViT-L/14 representation.",
    },
)

# Register Vision Transformer (dima806)
register_detector(
    name="vit",
    adapter_cls_or_factory=_vit_factory,
    metadata={
        "detector_name": "ViTDeepfakeDetector",
        "model_name": "dima806/deepfake_vs_real_image_detection (ViT-Base/16)",
        "architecture": "ViTForImageClassification (google/vit-base-patch16-224)",
        "threshold": 0.50,
        "input_dimensions": "3x224x224",
        "label_mapping": {0: "Real", 1: "Fake"},
        "description": "Fine-tuned Vision Transformer for deepfake detection.",
    },
)


def _convnext_factory(**kwargs: Any) -> Any:
    from src.convnext_adapter import ConvNeXtDetectorAdapter
    return ConvNeXtDetectorAdapter(**kwargs)


# Register RAID Chen2024 ConvNeXt
register_detector(
    name="chen2024_convnext",
    adapter_cls_or_factory=_convnext_factory,
    metadata={
        "detector_name": "Chen2024ConvNeXtDetector",
        "model_name": "chen2024_convnext (ConvNeXt-Base)",
        "architecture": "ConvNeXt-Base + Linear Head (1024 -> 2)",
        "threshold": 0.50,
        "input_dimensions": "3x224x224",
        "label_mapping": {0: "REAL", 1: "FAKE"},
        "description": "RAID benchmark constituent detector (Chen et al. 2024 ConvNeXt-Base binary classifier).",
    },
)
