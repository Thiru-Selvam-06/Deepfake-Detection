"""
Decision Fusion Module
Combines predictions from multiple deepfake detectors (ConvNeXt, UnivFD, ViT)
using configurable, deterministic decision-fusion strategies.

Designed for robust software contracts, explicit fallback handling,
and leakage-free empirical evaluation.
"""

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_UNCERTAIN,
    PREDICTION_ERROR,
    VALID_PREDICTIONS,
)

logger = logging.getLogger(__name__)

# Canonical Strategy Identifiers
STRATEGY_VIT_ONLY = "vit_only"
STRATEGY_MAJORITY_VOTE = "majority_vote"
STRATEGY_MEAN_PROBABILITY = "mean_fake_probability"
STRATEGY_VIT_PRIORITY = "vit_priority"

VALID_STRATEGIES = {
    STRATEGY_VIT_ONLY,
    STRATEGY_MAJORITY_VOTE,
    STRATEGY_MEAN_PROBABILITY,
    STRATEGY_VIT_PRIORITY,
}


@dataclass
class FusionResult:
    """
    Standardized result structure for multi-model decision fusion.
    Compatible with DetectorResult while exposing fusion-specific audit fields.
    """
    strategy_name: str
    prediction: str
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    confidence: Optional[float] = None
    status: str = "SUCCESS"  # "SUCCESS", "FALLBACK", "ERROR", "REJECTED"
    detector_votes: Dict[str, str] = field(default_factory=dict)
    detector_probabilities: Dict[str, Optional[float]] = field(default_factory=dict)
    valid_detectors: List[str] = field(default_factory=list)
    failed_detectors: List[str] = field(default_factory=list)
    tie_broken: bool = False
    fallback_applied: bool = False
    explanation: str = ""
    error: Optional[str] = None
    warning: Optional[str] = None

    def __post_init__(self):
        if self.prediction not in VALID_PREDICTIONS:
            raise ValueError(
                f"Invalid fusion prediction '{self.prediction}'. "
                f"Must be strictly one of: {sorted(list(VALID_PREDICTIONS))}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)

    def to_detector_result(self) -> DetectorResult:
        """Convert to standard DetectorResult instance."""
        return DetectorResult(
            detector_name=f"Fusion_{self.strategy_name}",
            model_name=f"Ensemble_{self.strategy_name}",
            prediction=self.prediction,
            fake_probability=self.fake_probability,
            real_probability=self.real_probability,
            confidence=self.confidence,
            error=self.error,
            warning=self.warning,
        )


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert value to a valid probability float in [0.0, 1.0], or None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if 0.0 <= f <= 1.0:
            return round(f, 6)
        return None
    except (ValueError, TypeError):
        return None


def normalize_detector_record(
    det_key: str,
    result: Union[DetectorResult, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Normalizes a DetectorResult or raw prediction dictionary into a clean dictionary.
    """
    if isinstance(result, DetectorResult):
        raw = result.to_dict()
    elif isinstance(result, dict):
        raw = result.copy()
    else:
        raise TypeError(f"Expected DetectorResult or dict, got {type(result).__name__}")

    pred = raw.get("prediction", PREDICTION_ERROR)
    if pred not in VALID_PREDICTIONS:
        pred = PREDICTION_ERROR

    fake_p = _safe_float(raw.get("fake_probability"))
    real_p = _safe_float(raw.get("real_probability"))

    # Reconstruct complementary probability if only one is present
    if fake_p is not None and real_p is None:
        real_p = round(1.0 - fake_p, 6)
    elif real_p is not None and fake_p is None:
        fake_p = round(1.0 - real_p, 6)

    err = raw.get("error") or raw.get("error_message")
    if err is not None:
        err = str(err).strip() or None

    return {
        "detector_key": det_key,
        "detector_name": raw.get("detector_name", det_key),
        "prediction": pred,
        "fake_probability": fake_p,
        "real_probability": real_p,
        "raw_logit": raw.get("raw_logit"),
        "confidence": _safe_float(raw.get("confidence")),
        "error": err,
    }


def strategy_vit_only(
    results: Dict[str, Union[DetectorResult, Dict[str, Any]]],
    vit_key: str = "vit",
) -> FusionResult:
    """
    Strategy 1: ViT Alone.
    Directly forwards the prediction of the ViT detector.
    """
    norm = {}
    for k, v in results.items():
        norm[k] = normalize_detector_record(k, v)

    vit_res = norm.get(vit_key)
    votes = {k: v["prediction"] for k, v in norm.items()}
    probs = {k: v["fake_probability"] for k, v in norm.items()}
    failed_keys = [k for k, v in norm.items() if v["prediction"] == PREDICTION_ERROR or v.get("error")]

    if vit_res is None or vit_res["prediction"] == PREDICTION_ERROR:
        err_msg = vit_res["error"] if (vit_res and vit_res.get("error")) else f"Detector '{vit_key}' failed or not present"
        return FusionResult(
            strategy_name=STRATEGY_VIT_ONLY,
            prediction=PREDICTION_ERROR,
            status="ERROR",
            detector_votes=votes,
            detector_probabilities=probs,
            valid_detectors=[k for k, v in norm.items() if v["prediction"] in (PREDICTION_REAL, PREDICTION_FAKE, PREDICTION_UNCERTAIN)],
            failed_detectors=failed_keys or ([vit_key] if vit_res is None else []),
            explanation=f"ViT detector failed: {err_msg}",
            error=err_msg,
        )

    pred = vit_res["prediction"]
    fake_p = vit_res["fake_probability"]
    real_p = vit_res["real_probability"]
    if pred == PREDICTION_UNCERTAIN:
        conf = vit_res["confidence"] if vit_res["confidence"] is not None else 0.0
        status = "UNCERTAIN"
    else:
        conf = vit_res["confidence"] if vit_res["confidence"] is not None else (fake_p if pred == PREDICTION_FAKE else real_p)
        status = "SUCCESS"

    return FusionResult(
        strategy_name=STRATEGY_VIT_ONLY,
        prediction=pred,
        fake_probability=fake_p,
        real_probability=real_p,
        confidence=conf,
        status=status,
        detector_votes=votes,
        detector_probabilities=probs,
        valid_detectors=[vit_key],
        failed_detectors=[k for k in failed_keys if k != vit_key],
        explanation=f"ViT standalone prediction: {pred} (P(fake)={fake_p})",
    )


def strategy_majority_vote(
    results: Dict[str, Union[DetectorResult, Dict[str, Any]]],
    tie_breaker: str = "vit_or_real",
    vit_key: str = "vit",
) -> FusionResult:
    """
    Strategy 2: Majority Vote.
    Aggregates binary predictions (REAL vs FAKE) from all valid detectors.

    Tie-breaking behavior (when real_votes == fake_votes):
      - 'vit_or_real': Follows ViT if ViT is among valid votes, otherwise REAL.
      - 'vit': Follows ViT if valid, otherwise UNCERTAIN.
      - 'conservative_real': Deterministically chooses REAL.
      - 'conservative_fake': Deterministically chooses FAKE.
      - 'uncertain': Returns UNCERTAIN.
    """
    norm = {}
    for k, v in results.items():
        norm[k] = normalize_detector_record(k, v)

    votes = {k: v["prediction"] for k, v in norm.items()}
    probs = {k: v["fake_probability"] for k, v in norm.items()}

    valid_keys = [k for k, v in norm.items() if v["prediction"] in (PREDICTION_REAL, PREDICTION_FAKE)]
    uncertain_keys = [k for k, v in norm.items() if v["prediction"] == PREDICTION_UNCERTAIN]
    failed_keys = [k for k, v in norm.items() if v["prediction"] == PREDICTION_ERROR or v.get("error")]

    if not valid_keys:
        if uncertain_keys:
            valid_probs = [norm[k]["fake_probability"] for k in uncertain_keys if norm[k]["fake_probability"] is not None]
            mean_fake_p = round(sum(valid_probs) / len(valid_probs), 6) if valid_probs else None
            mean_real_p = round(1.0 - mean_fake_p, 6) if mean_fake_p is not None else None
            return FusionResult(
                strategy_name=STRATEGY_MAJORITY_VOTE,
                prediction=PREDICTION_UNCERTAIN,
                fake_probability=mean_fake_p,
                real_probability=mean_real_p,
                confidence=0.0,
                status="UNCERTAIN",
                detector_votes=votes,
                detector_probabilities=probs,
                valid_detectors=uncertain_keys,
                failed_detectors=failed_keys,
                explanation=f"Zero decisive votes; {len(uncertain_keys)} detector(s) reported UNCERTAIN.",
            )
        return FusionResult(
            strategy_name=STRATEGY_MAJORITY_VOTE,
            prediction=PREDICTION_ERROR,
            status="ERROR",
            detector_votes=votes,
            detector_probabilities=probs,
            valid_detectors=[],
            failed_detectors=failed_keys or list(norm.keys()),
            explanation="All detectors failed or produced invalid predictions.",
            error="No valid detector predictions available for majority vote.",
        )

    real_votes = sum(1 for k in valid_keys if norm[k]["prediction"] == PREDICTION_REAL)
    fake_votes = sum(1 for k in valid_keys if norm[k]["prediction"] == PREDICTION_FAKE)

    tie_broken = False
    if fake_votes > real_votes:
        prediction = PREDICTION_FAKE
        explanation = f"Majority vote FAKE ({fake_votes} FAKE vs {real_votes} REAL)."
    elif real_votes > fake_votes:
        prediction = PREDICTION_REAL
        explanation = f"Majority vote REAL ({real_votes} REAL vs {fake_votes} FAKE)."
    else:
        # Tie condition
        tie_broken = True
        if tie_breaker == "vit_or_real":
            if vit_key in valid_keys:
                prediction = norm[vit_key]["prediction"]
                explanation = f"Tie ({real_votes} vs {fake_votes}) broken by ViT preference: {prediction}."
            else:
                prediction = PREDICTION_REAL
                explanation = f"Tie ({real_votes} vs {fake_votes}) broken by default REAL."
        elif tie_breaker == "vit":
            if vit_key in valid_keys:
                prediction = norm[vit_key]["prediction"]
                explanation = f"Tie broken by ViT preference: {prediction}."
            else:
                prediction = PREDICTION_UNCERTAIN
                explanation = f"Tie ({real_votes} vs {fake_votes}) and ViT unavailable -> UNCERTAIN."
        elif tie_breaker in ("conservative_real", "real"):
            prediction = PREDICTION_REAL
            explanation = f"Tie broken by conservative REAL rule."
        elif tie_breaker in ("conservative_fake", "fake"):
            prediction = PREDICTION_FAKE
            explanation = f"Tie broken by conservative FAKE rule."
        elif tie_breaker == "uncertain":
            prediction = PREDICTION_UNCERTAIN
            explanation = f"Tie resulting in UNCERTAIN verdict."
        else:
            prediction = PREDICTION_REAL
            explanation = f"Tie broken by fallback REAL."

    # Compute mean probability across valid votes with probabilities
    valid_probs = [norm[k]["fake_probability"] for k in valid_keys if norm[k]["fake_probability"] is not None]
    mean_fake_p = round(sum(valid_probs) / len(valid_probs), 6) if valid_probs else None
    mean_real_p = round(1.0 - mean_fake_p, 6) if mean_fake_p is not None else None
    confidence = round(max(real_votes, fake_votes) / len(valid_keys), 4)

    return FusionResult(
        strategy_name=STRATEGY_MAJORITY_VOTE,
        prediction=prediction,
        fake_probability=mean_fake_p,
        real_probability=mean_real_p,
        confidence=confidence,
        status="SUCCESS" if prediction in (PREDICTION_REAL, PREDICTION_FAKE) else "UNCERTAIN",
        detector_votes=votes,
        detector_probabilities=probs,
        valid_detectors=valid_keys,
        failed_detectors=failed_keys,
        tie_broken=tie_broken,
        fallback_applied=False,
        explanation=explanation,
    )


def strategy_mean_fake_probability(
    results: Dict[str, Union[DetectorResult, Dict[str, Any]]],
    threshold: float = 0.50,
    tie_breaker: str = "real",
) -> FusionResult:
    """
    Strategy 3: Mean Fake Probability.
    Averages the fake probability across detectors that produced valid probabilities.
    Compares against a decision threshold (default 0.50).
    """
    norm = {}
    for k, v in results.items():
        norm[k] = normalize_detector_record(k, v)

    votes = {k: v["prediction"] for k, v in norm.items()}
    probs = {k: v["fake_probability"] for k, v in norm.items()}

    valid_keys = [k for k, v in norm.items() if v["fake_probability"] is not None and v["prediction"] != PREDICTION_ERROR]
    failed_keys = [k for k in norm.keys() if k not in valid_keys]

    if not valid_keys:
        return FusionResult(
            strategy_name=STRATEGY_MEAN_PROBABILITY,
            prediction=PREDICTION_ERROR,
            status="ERROR",
            detector_votes=votes,
            detector_probabilities=probs,
            valid_detectors=[],
            failed_detectors=failed_keys,
            explanation="No valid detector probabilities available for mean calculation.",
            error="Zero valid probabilities across all detectors.",
        )

    valid_probs = [norm[k]["fake_probability"] for k in valid_keys]
    mean_fake_p = round(sum(valid_probs) / len(valid_probs), 6)
    mean_real_p = round(1.0 - mean_fake_p, 6)

    tie_broken = False
    if math.isclose(mean_fake_p, threshold, abs_tol=1e-6):
        tie_broken = True
        if tie_breaker in ("conservative_real", "real"):
            prediction = PREDICTION_REAL
        elif tie_breaker in ("conservative_fake", "fake"):
            prediction = PREDICTION_FAKE
        elif tie_breaker == "uncertain":
            prediction = PREDICTION_UNCERTAIN
        else:
            prediction = PREDICTION_REAL
        explanation = f"Mean fake probability exactly equals threshold {threshold:.4f}; tie resolved to {prediction}."
    elif mean_fake_p >= threshold:
        prediction = PREDICTION_FAKE
        explanation = f"Mean fake probability {mean_fake_p:.4f} >= threshold {threshold:.4f} -> FAKE."
    else:
        prediction = PREDICTION_REAL
        explanation = f"Mean fake probability {mean_fake_p:.4f} < threshold {threshold:.4f} -> REAL."

    status = "SUCCESS" if prediction in (PREDICTION_REAL, PREDICTION_FAKE) else "UNCERTAIN"
    conf = round(abs(mean_fake_p - 0.5) * 2.0, 4) if prediction != PREDICTION_UNCERTAIN else 0.0

    return FusionResult(
        strategy_name=STRATEGY_MEAN_PROBABILITY,
        prediction=prediction,
        fake_probability=mean_fake_p,
        real_probability=mean_real_p,
        confidence=conf,
        status=status,
        detector_votes=votes,
        detector_probabilities=probs,
        valid_detectors=valid_keys,
        failed_detectors=failed_keys,
        tie_broken=tie_broken,
        fallback_applied=False,
        explanation=explanation,
    )


def strategy_vit_priority(
    results: Dict[str, Union[DetectorResult, Dict[str, Any]]],
    vit_key: str = "vit",
    fallback_strategy: str = "majority_vote",
) -> FusionResult:
    """
    Strategy 4: ViT Priority with Explicit Fallback.
    Prioritizes ViT because of its empirical domain dominance on facial media.
    If ViT succeeds with a decisive prediction (REAL or FAKE), its verdict is used.
    If ViT fails (ERROR, UNCERTAIN, or missing), dynamically falls back to
    aggregating the secondary detectors (UnivFD, ConvNeXt).
    """
    norm = {}
    for k, v in results.items():
        norm[k] = normalize_detector_record(k, v)

    votes = {k: v["prediction"] for k, v in norm.items()}
    probs = {k: v["fake_probability"] for k, v in norm.items()}
    failed_keys_all = [k for k, v in norm.items() if v["prediction"] == PREDICTION_ERROR or v.get("error")]

    vit_res = norm.get(vit_key)
    vit_valid = vit_res is not None and vit_res["prediction"] in (PREDICTION_REAL, PREDICTION_FAKE)

    if vit_valid:
        pred = vit_res["prediction"]
        fake_p = vit_res["fake_probability"]
        real_p = vit_res["real_probability"]
        conf = vit_res["confidence"] if vit_res["confidence"] is not None else (fake_p if pred == PREDICTION_FAKE else real_p)

        return FusionResult(
            strategy_name=STRATEGY_VIT_PRIORITY,
            prediction=pred,
            fake_probability=fake_p,
            real_probability=real_p,
            confidence=conf,
            status="SUCCESS",
            detector_votes=votes,
            detector_probabilities=probs,
            valid_detectors=[k for k, v in norm.items() if v["prediction"] in (PREDICTION_REAL, PREDICTION_FAKE, PREDICTION_UNCERTAIN)],
            failed_detectors=[k for k in failed_keys_all if k != vit_key],
            tie_broken=False,
            fallback_applied=False,
            explanation=f"ViT decisive prediction: {pred} (P(fake)={fake_p}). No fallback needed.",
        )

    # ViT is invalid, failed, or uncertain -> execute fallback
    secondary_results = {k: v for k, v in norm.items() if k != vit_key}
    valid_secondary = [k for k, v in secondary_results.items() if v["prediction"] in (PREDICTION_REAL, PREDICTION_FAKE)]
    uncertain_secondary = [k for k, v in secondary_results.items() if v["prediction"] == PREDICTION_UNCERTAIN]
    failed_secondary = [k for k, v in secondary_results.items() if v["prediction"] == PREDICTION_ERROR or v.get("error")]

    vit_failed = vit_res is None or vit_res["prediction"] == PREDICTION_ERROR or bool(vit_res.get("error"))
    failed_keys = ([vit_key] if vit_failed else []) + failed_secondary

    if not valid_secondary:
        if uncertain_secondary or (vit_res and vit_res["prediction"] == PREDICTION_UNCERTAIN):
            valid_probs = [norm[k]["fake_probability"] for k in uncertain_secondary if norm[k]["fake_probability"] is not None]
            mean_fake_p = round(sum(valid_probs) / len(valid_probs), 6) if valid_probs else None
            mean_real_p = round(1.0 - mean_fake_p, 6) if mean_fake_p is not None else None
            return FusionResult(
                strategy_name=STRATEGY_VIT_PRIORITY,
                prediction=PREDICTION_UNCERTAIN,
                fake_probability=mean_fake_p,
                real_probability=mean_real_p,
                confidence=0.0,
                status="UNCERTAIN",
                detector_votes=votes,
                detector_probabilities=probs,
                valid_detectors=uncertain_secondary or ([vit_key] if not vit_failed else []),
                failed_detectors=failed_keys,
                tie_broken=False,
                fallback_applied=True,
                explanation="ViT yielded UNCERTAIN and zero decisive secondary detectors remained.",
                warning="ViT was uncertain; secondary detectors were also non-decisive.",
            )

        vit_err = vit_res["error"] if (vit_res and vit_res.get("error")) else (f"Detector '{vit_key}' not found" if not vit_res else "Failed")
        return FusionResult(
            strategy_name=STRATEGY_VIT_PRIORITY,
            prediction=PREDICTION_ERROR,
            status="ERROR",
            detector_votes=votes,
            detector_probabilities=probs,
            valid_detectors=[],
            failed_detectors=failed_keys or list(norm.keys()),
            tie_broken=False,
            fallback_applied=True,
            explanation=f"ViT failed ({vit_err}) and zero valid secondary detectors remained.",
            error=f"ViT priority failed and fallback exhausted: {vit_err}",
        )

    # Apply fallback strategy over remaining decisive detectors
    if fallback_strategy == "majority_vote":
        fallback_res = strategy_majority_vote(secondary_results, tie_breaker="conservative_real")
    elif fallback_strategy == "mean_probability":
        fallback_res = strategy_mean_fake_probability(secondary_results)
    else:
        fallback_res = strategy_majority_vote(secondary_results)

    vit_status = f"failed ({vit_res['error']})" if vit_failed else (f"yielded {vit_res['prediction']}" if vit_res else "missing")
    explanation = (
        f"ViT {vit_status}. "
        f"Triggered secondary fallback ({fallback_strategy}) -> {fallback_res.prediction}."
    )
    status = "ERROR" if fallback_res.prediction == PREDICTION_ERROR else "FALLBACK"

    return FusionResult(
        strategy_name=STRATEGY_VIT_PRIORITY,
        prediction=fallback_res.prediction,
        fake_probability=fallback_res.fake_probability,
        real_probability=fallback_res.real_probability,
        confidence=fallback_res.confidence,
        status=status,
        detector_votes=votes,
        detector_probabilities=probs,
        valid_detectors=valid_secondary,
        failed_detectors=failed_keys,
        tie_broken=fallback_res.tie_broken,
        fallback_applied=True,
        explanation=explanation,
        error=fallback_res.error,
        warning="ViT was unavailable or indecisive; prediction derived via fallback.",
    )


class DecisionFusionPipeline:
    """
    Centralized pipeline for managing and executing decision fusion strategies.
    """

    def __init__(self, vit_key: str = "vit"):
        self.vit_key = vit_key
        self.strategies = {
            STRATEGY_VIT_ONLY: lambda res, **kw: strategy_vit_only(res, vit_key=self.vit_key),
            STRATEGY_MAJORITY_VOTE: lambda res, **kw: strategy_majority_vote(res, vit_key=self.vit_key, **kw),
            STRATEGY_MEAN_PROBABILITY: lambda res, **kw: strategy_mean_fake_probability(res, **kw),
            STRATEGY_VIT_PRIORITY: lambda res, **kw: strategy_vit_priority(res, vit_key=self.vit_key, **kw),
        }

    def fuse(
        self,
        results: Dict[str, Union[DetectorResult, Dict[str, Any]]],
        strategy: str = STRATEGY_VIT_PRIORITY,
        **kwargs: Any,
    ) -> FusionResult:
        """
        Executes a single specified fusion strategy on detector outputs.
        """
        strategy_func = self.strategies.get(strategy)
        if strategy_func is None:
            raise ValueError(
                f"Unknown fusion strategy '{strategy}'. Valid strategies: {sorted(list(VALID_STRATEGIES))}"
            )
        return strategy_func(results, **kwargs)

    def fuse_all_strategies(
        self,
        results: Dict[str, Union[DetectorResult, Dict[str, Any]]],
        **kwargs: Any,
    ) -> Dict[str, FusionResult]:
        """
        Runs all 4 canonical strategies concurrently on the provided detector results.
        Returns a dictionary mapping strategy_name -> FusionResult.
        """
        return {
            name: func(results, **kwargs)
            for name, func in self.strategies.items()
        }
