"""
Standardized Detector Result Schema
Defines a consistent output schema for deepfake detectors to enable standardized logging,
diagnostics, and future multi-model comparison without modifying underlying detector behaviors.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Set

PREDICTION_REAL = "REAL"
PREDICTION_FAKE = "FAKE"
PREDICTION_UNCERTAIN = "UNCERTAIN"
PREDICTION_ERROR = "ERROR"

VALID_PREDICTIONS: Set[str] = {
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_UNCERTAIN,
    PREDICTION_ERROR,
}


@dataclass
class DetectorResult:
    """
    Standardized result data structure for detector outputs.

    Allowed prediction values:
      - 'REAL'
      - 'FAKE'
      - 'UNCERTAIN'
      - 'ERROR'
    """
    detector_name: str
    model_name: str
    prediction: str
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    raw_logit: Optional[float] = None
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    device: Optional[str] = None
    preprocessing: Optional[str] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def __post_init__(self):
        if self.prediction not in VALID_PREDICTIONS:
            raise ValueError(
                f"Invalid prediction value '{self.prediction}'. "
                f"Must be strictly one of: {sorted(list(VALID_PREDICTIONS))}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a standard serializable dictionary."""
        return asdict(self)

    @classmethod
    def create_error(
        cls,
        detector_name: str,
        model_name: str,
        error_message: str,
        warning: Optional[str] = None,
        device: Optional[str] = None,
        preprocessing: Optional[str] = None,
    ) -> "DetectorResult":
        """Factory helper to construct a well-formed error result."""
        return cls(
            detector_name=detector_name,
            model_name=model_name,
            prediction=PREDICTION_ERROR,
            error=str(error_message),
            warning=warning,
            device=device,
            preprocessing=preprocessing,
        )
