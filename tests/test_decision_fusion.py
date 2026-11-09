"""
Unit Test Suite for Decision Fusion Logic
Tests all required behavioral specifications:
- All models agree (REAL and FAKE)
- Models disagree (divergence, majority vs priority)
- One model fails (ViT fails vs ConvNeXt fails)
- Multiple models fail (single survivor)
- No valid predictions remain (all fail)
- Invalid or missing probabilities (None, NaN, out of bounds)
- Deterministic tie handling (configurable policies)
- Rejection controls / non-image error handling
- Conversion to DetectorResult
"""

import os
import sys
import math
import pytest
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_UNCERTAIN,
    PREDICTION_ERROR,
)
from src.decision_fusion import (
    FusionResult,
    DecisionFusionPipeline,
    strategy_vit_only,
    strategy_majority_vote,
    strategy_mean_fake_probability,
    strategy_vit_priority,
    normalize_detector_record,
    STRATEGY_VIT_ONLY,
    STRATEGY_MAJORITY_VOTE,
    STRATEGY_MEAN_PROBABILITY,
    STRATEGY_VIT_PRIORITY,
)


def make_detector_result(
    detector_name: str,
    prediction: str,
    fake_prob: float = None,
    real_prob: float = None,
    error: str = None,
) -> DetectorResult:
    """Helper to construct well-formed DetectorResult fixtures."""
    return DetectorResult(
        detector_name=detector_name,
        model_name=f"{detector_name}_model",
        prediction=prediction,
        fake_probability=fake_prob,
        real_probability=real_prob,
        error=error,
    )


class TestDecisionFusion:

    # 1. All models agree
    def test_all_models_agree_real(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL, fake_prob=0.05, real_prob=0.95),
            "univfd": make_detector_result("univfd", PREDICTION_REAL, fake_prob=0.01, real_prob=0.99),
            "vit": make_detector_result("vit", PREDICTION_REAL, fake_prob=0.02, real_prob=0.98),
        }
        pipeline = DecisionFusionPipeline()
        all_res = pipeline.fuse_all_strategies(ensemble)

        for strat_name, res in all_res.items():
            assert res.prediction == PREDICTION_REAL, f"{strat_name} failed on unanimous REAL"
            assert res.status == "SUCCESS"
            assert not res.tie_broken
            assert not res.fallback_applied
            assert res.fake_probability is not None
            assert res.fake_probability < 0.10

    def test_all_models_agree_fake(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_FAKE, fake_prob=0.92, real_prob=0.08),
            "univfd": make_detector_result("univfd", PREDICTION_FAKE, fake_prob=0.88, real_prob=0.12),
            "vit": make_detector_result("vit", PREDICTION_FAKE, fake_prob=0.99, real_prob=0.01),
        }
        pipeline = DecisionFusionPipeline()
        all_res = pipeline.fuse_all_strategies(ensemble)

        for strat_name, res in all_res.items():
            assert res.prediction == PREDICTION_FAKE, f"{strat_name} failed on unanimous FAKE"
            assert res.status == "SUCCESS"
            assert not res.tie_broken
            assert not res.fallback_applied
            assert res.fake_probability is not None
            assert res.fake_probability > 0.85

    # 2. Models disagree
    def test_models_disagree_majority_vs_vit(self):
        # ConvNeXt and UnivFD predict REAL; ViT predicts FAKE
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL, fake_prob=0.10, real_prob=0.90),
            "univfd": make_detector_result("univfd", PREDICTION_REAL, fake_prob=0.20, real_prob=0.80),
            "vit": make_detector_result("vit", PREDICTION_FAKE, fake_prob=0.95, real_prob=0.05),
        }
        pipeline = DecisionFusionPipeline()

        # Majority vote: 2 REAL vs 1 FAKE -> REAL
        res_maj = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)
        assert res_maj.prediction == PREDICTION_REAL
        assert not res_maj.tie_broken

        # ViT alone: FAKE
        res_vit = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_ONLY)
        assert res_vit.prediction == PREDICTION_FAKE

        # ViT priority: FAKE (ViT succeeds decisively)
        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_FAKE
        assert not res_prio.fallback_applied

        # Mean fake probability: (0.10 + 0.20 + 0.95) / 3 = 0.4167 < 0.50 -> REAL
        res_mean = pipeline.fuse(ensemble, strategy=STRATEGY_MEAN_PROBABILITY)
        assert res_mean.prediction == PREDICTION_REAL
        assert math.isclose(res_mean.fake_probability, 0.416667, abs_tol=1e-4)

    # 3. One model fails
    def test_one_model_fails_convnext(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_ERROR, error="CUDA out of memory"),
            "univfd": make_detector_result("univfd", PREDICTION_FAKE, fake_prob=0.85, real_prob=0.15),
            "vit": make_detector_result("vit", PREDICTION_FAKE, fake_prob=0.99, real_prob=0.01),
        }
        pipeline = DecisionFusionPipeline()

        res_maj = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)
        assert res_maj.prediction == PREDICTION_FAKE
        assert res_maj.valid_detectors == ["univfd", "vit"]
        assert res_maj.failed_detectors == ["chen2024_convnext"]

        res_mean = pipeline.fuse(ensemble, strategy=STRATEGY_MEAN_PROBABILITY)
        assert res_mean.prediction == PREDICTION_FAKE
        assert math.isclose(res_mean.fake_probability, 0.92, abs_tol=1e-4)

        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_FAKE
        assert not res_prio.fallback_applied

    def test_one_model_fails_vit_triggers_fallback(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL, fake_prob=0.05, real_prob=0.95),
            "univfd": make_detector_result("univfd", PREDICTION_REAL, fake_prob=0.08, real_prob=0.92),
            "vit": make_detector_result("vit", PREDICTION_ERROR, error="Paging file too small"),
        }
        pipeline = DecisionFusionPipeline()

        # ViT alone must return ERROR
        res_vit = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_ONLY)
        assert res_vit.prediction == PREDICTION_ERROR
        assert res_vit.status == "ERROR"

        # Majority vote proceeds on remaining 2 detectors
        res_maj = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)
        assert res_maj.prediction == PREDICTION_REAL

        # ViT priority must detect ViT failure and fall back gracefully
        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_REAL
        assert res_prio.status == "FALLBACK"
        assert res_prio.fallback_applied is True
        assert "ViT was unavailable" in res_prio.warning

    # 4. Multiple models fail
    def test_multiple_models_fail_single_survivor(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_ERROR, error="Model load error"),
            "univfd": make_detector_result("univfd", PREDICTION_ERROR, error="Clip weights corrupt"),
            "vit": make_detector_result("vit", PREDICTION_FAKE, fake_prob=0.98, real_prob=0.02),
        }
        pipeline = DecisionFusionPipeline()

        res_maj = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)
        assert res_maj.prediction == PREDICTION_FAKE
        assert res_maj.valid_detectors == ["vit"]

        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_FAKE
        assert not res_prio.fallback_applied

    def test_multiple_models_fail_including_vit_single_survivor(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL, fake_prob=0.10, real_prob=0.90),
            "univfd": make_detector_result("univfd", PREDICTION_ERROR, error="Failed"),
            "vit": make_detector_result("vit", PREDICTION_ERROR, error="Failed"),
        }
        pipeline = DecisionFusionPipeline()

        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_REAL
        assert res_prio.status == "FALLBACK"
        assert res_prio.fallback_applied is True

    # 5. Invalid or missing probabilities
    def test_invalid_or_missing_probabilities(self):
        ensemble = {
            "chen2024_convnext": {
                "prediction": PREDICTION_REAL,
                "fake_probability": None,  # Missing probability
            },
            "univfd": {
                "prediction": PREDICTION_REAL,
                "fake_probability": float("nan"),  # NaN probability
            },
            "vit": {
                "prediction": PREDICTION_REAL,
                "fake_probability": 0.05,
                "real_probability": 0.95,
            },
        }
        pipeline = DecisionFusionPipeline()

        # Majority vote does not need probabilities
        res_maj = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)
        assert res_maj.prediction == PREDICTION_REAL

        # Mean probability should skip None and NaN, using only valid ViT probability
        res_mean = pipeline.fuse(ensemble, strategy=STRATEGY_MEAN_PROBABILITY)
        assert res_mean.prediction == PREDICTION_REAL
        assert res_mean.fake_probability == 0.05
        assert res_mean.valid_detectors == ["vit"]

    def test_all_missing_probabilities(self):
        ensemble = {
            "chen2024_convnext": {"prediction": PREDICTION_REAL, "fake_probability": None},
            "univfd": {"prediction": PREDICTION_REAL, "fake_probability": None},
        }
        res_mean = strategy_mean_fake_probability(ensemble)
        assert res_mean.prediction == PREDICTION_ERROR
        assert res_mean.status == "ERROR"
        assert "Zero valid probabilities" in res_mean.error

    # 6. Deterministic tie handling
    def test_deterministic_tie_handling_majority_vote(self):
        # 1 REAL vs 1 FAKE, 1 ERROR
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL),
            "univfd": make_detector_result("univfd", PREDICTION_FAKE),
            "vit": make_detector_result("vit", PREDICTION_ERROR),
        }

        # tie_breaker='conservative_real'
        res_real = strategy_majority_vote(ensemble, tie_breaker="conservative_real")
        assert res_real.prediction == PREDICTION_REAL
        assert res_real.tie_broken is True

        # tie_breaker='conservative_fake'
        res_fake = strategy_majority_vote(ensemble, tie_breaker="conservative_fake")
        assert res_fake.prediction == PREDICTION_FAKE
        assert res_fake.tie_broken is True

        # tie_breaker='uncertain'
        res_unc = strategy_majority_vote(ensemble, tie_breaker="uncertain")
        assert res_unc.prediction == PREDICTION_UNCERTAIN
        assert res_unc.tie_broken is True

    def test_tie_handling_with_vit_preference(self):
        # ConvNeXt=REAL, ViT=FAKE -> 1 vs 1 tie. ViT preference should pick ViT's vote
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL),
            "vit": make_detector_result("vit", PREDICTION_FAKE),
        }
        res = strategy_majority_vote(ensemble, tie_breaker="vit_or_real")
        assert res.prediction == PREDICTION_FAKE
        assert res.tie_broken is True

    def test_mean_probability_exact_threshold_tie(self):
        # Mean fake probability = exactly 0.500000
        ensemble = {
            "det1": {"prediction": PREDICTION_REAL, "fake_probability": 0.40},
            "det2": {"prediction": PREDICTION_FAKE, "fake_probability": 0.60},
        }
        res_tie = strategy_mean_fake_probability(ensemble, threshold=0.50, tie_breaker="real")
        assert res_tie.prediction == PREDICTION_REAL
        assert res_tie.tie_broken is True

        res_tie_fake = strategy_mean_fake_probability(ensemble, threshold=0.50, tie_breaker="fake")
        assert res_tie_fake.prediction == PREDICTION_FAKE
        assert res_tie_fake.tie_broken is True

    # 7. No valid predictions remain (all models fail)
    def test_no_valid_predictions_remain(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_ERROR, error="Failed"),
            "univfd": make_detector_result("univfd", PREDICTION_ERROR, error="Failed"),
            "vit": make_detector_result("vit", PREDICTION_ERROR, error="Failed"),
        }
        pipeline = DecisionFusionPipeline()
        all_res = pipeline.fuse_all_strategies(ensemble)

        for strat_name, res in all_res.items():
            assert res.prediction == PREDICTION_ERROR, f"{strat_name} did not return ERROR on total failure"
            assert res.status == "ERROR"
            assert res.valid_detectors == []
            assert len(res.failed_detectors) == 3

    # 8. Rejection controls separate from REAL/FAKE
    def test_rejection_controls(self):
        # When all detectors reject an invalid file (corrupted or unsupported format)
        rejection_ensemble = {
            "chen2024_convnext": DetectorResult.create_error("convnext", "model", "Invalid image header"),
            "univfd": DetectorResult.create_error("univfd", "model", "Cannot identify image file"),
            "vit": DetectorResult.create_error("vit", "model", "Unrecognized image format"),
        }
        pipeline = DecisionFusionPipeline()
        res = pipeline.fuse(rejection_ensemble, strategy=STRATEGY_VIT_PRIORITY)

        assert res.prediction == PREDICTION_ERROR
        assert res.status == "ERROR"
        assert res.prediction not in (PREDICTION_REAL, PREDICTION_FAKE)

    # 9. Schema compliance and DetectorResult conversion
    def test_to_detector_result(self):
        res = FusionResult(
            strategy_name=STRATEGY_VIT_PRIORITY,
            prediction=PREDICTION_FAKE,
            fake_probability=0.985,
            real_probability=0.015,
            confidence=0.985,
        )
        det_res = res.to_detector_result()
        assert isinstance(det_res, DetectorResult)
        assert det_res.prediction == PREDICTION_FAKE
        assert det_res.fake_probability == 0.985
        assert det_res.detector_name == "Fusion_vit_priority"

    # 10. Boolean rejection in probability sanitization
    def test_boolean_probabilities_rejected(self):
        from src.decision_fusion import _safe_float
        assert _safe_float(True) is None
        assert _safe_float(False) is None
        assert _safe_float(1.0) == 1.0
        assert _safe_float(0.0) == 0.0

    # 11. UNCERTAIN handling across all models
    def test_all_models_uncertain(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_UNCERTAIN, fake_prob=0.50),
            "univfd": make_detector_result("univfd", PREDICTION_UNCERTAIN, fake_prob=0.50),
            "vit": make_detector_result("vit", PREDICTION_UNCERTAIN, fake_prob=0.50),
        }
        pipeline = DecisionFusionPipeline()

        # Majority vote should yield UNCERTAIN, not ERROR
        res_maj = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)
        assert res_maj.prediction == PREDICTION_UNCERTAIN
        assert res_maj.status == "UNCERTAIN"
        assert res_maj.error is None
        assert len(res_maj.failed_detectors) == 0

        # ViT priority should yield UNCERTAIN, not ERROR
        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_UNCERTAIN
        assert res_prio.status == "UNCERTAIN"
        assert res_prio.error is None
        assert len(res_prio.failed_detectors) == 0

        # ViT alone should yield UNCERTAIN with status UNCERTAIN
        res_vit = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_ONLY)
        assert res_vit.prediction == PREDICTION_UNCERTAIN
        assert res_vit.status == "UNCERTAIN"
        assert res_vit.error is None

    # 12. ViT UNCERTAIN triggers decisive fallback
    def test_vit_uncertain_fallback_to_decisive(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_REAL, fake_prob=0.08),
            "univfd": make_detector_result("univfd", PREDICTION_REAL, fake_prob=0.12),
            "vit": make_detector_result("vit", PREDICTION_UNCERTAIN, fake_prob=0.50),
        }
        pipeline = DecisionFusionPipeline()

        res_prio = pipeline.fuse(ensemble, strategy=STRATEGY_VIT_PRIORITY)
        assert res_prio.prediction == PREDICTION_REAL
        assert res_prio.status == "FALLBACK"
        assert res_prio.fallback_applied is True
        assert "ViT yielded UNCERTAIN" in res_prio.explanation
        assert "vit" not in res_prio.failed_detectors

    # 13. Mean probability tie-breaker with uncertain
    def test_mean_probability_tie_breaker_uncertain(self):
        ensemble = {
            "det1": {"prediction": PREDICTION_REAL, "fake_probability": 0.40},
            "det2": {"prediction": PREDICTION_FAKE, "fake_probability": 0.60},
        }
        res_unc = strategy_mean_fake_probability(ensemble, threshold=0.50, tie_breaker="uncertain")
        assert res_unc.prediction == PREDICTION_UNCERTAIN
        assert res_unc.status == "UNCERTAIN"
        assert res_unc.tie_broken is True

    # 14. Majority vote with 1 error and 2 uncertain
    def test_majority_vote_with_abstained_and_error(self):
        ensemble = {
            "chen2024_convnext": make_detector_result("convnext", PREDICTION_ERROR, error="Crash"),
            "univfd": make_detector_result("univfd", PREDICTION_UNCERTAIN, fake_prob=0.50),
            "vit": make_detector_result("vit", PREDICTION_UNCERTAIN, fake_prob=0.50),
        }
        pipeline = DecisionFusionPipeline()
        res = pipeline.fuse(ensemble, strategy=STRATEGY_MAJORITY_VOTE)

        assert res.prediction == PREDICTION_UNCERTAIN
        assert res.status == "UNCERTAIN"
        assert res.failed_detectors == ["chen2024_convnext"]
        assert "univfd" in res.valid_detectors
        assert "vit" in res.valid_detectors

    # 15. Invalid strategy name raises ValueError
    def test_invalid_strategy_raises_value_error(self):
        pipeline = DecisionFusionPipeline()
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            pipeline.fuse({}, strategy="non_existent_strategy")

