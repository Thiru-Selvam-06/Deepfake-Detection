"""
Vision Transformer (ViT) Deepfake Detector Adapter
Provides a clean, standardized wrapper around the dima806/deepfake_vs_real_image_detection ViT model.
Adheres strictly to the DetectorResult schema, supporting lazy loading and robust error handling.
"""

import os
import sys
import time
from typing import Optional, Union, Any, Dict
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "deepfake_vs_real_image_detection")
HUB_MODEL_ID = "dima806/deepfake_vs_real_image_detection"


class ViTAdapter:
    """
    Adapter for dima806 ViT Deepfake Detection Model.
    
    Model Details:
      - Architecture: ViTForImageClassification (google/vit-base-patch16-224 fine-tuned)
      - Checkpoint: models/deepfake_vs_real_image_detection (model.safetensors)
      - Image Preprocessing: ViTImageProcessor (Resize 224x224, Rescale 1/255, Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      - Decision Threshold: 0.50 Fake Probability
      - Label Mapping: 0 = Real, 1 = Fake
    """

    DETECTOR_NAME = "ViTDeepfakeDetector"
    MODEL_NAME = "dima806/deepfake_vs_real_image_detection (ViT-Base/16)"
    PREPROCESSING_NAME = "ViTImageProcessor(Resize 224x224, Rescale 1/255, Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])"
    DECISION_THRESHOLD = 0.50
    LIMITATION_WARNING = (
        "Vision Transformer trained for facial deepfake detection. "
        "Evaluates learned visual artifact representations; does not guarantee authenticity."
    )

    def __init__(self, device: Optional[Union[str, torch.device]] = None, lazy_load: bool = True):
        self.device_requested = device
        self.processor = None
        self.model = None
        self.device = None
        self.real_idx = 0
        self.fake_idx = 1
        self.load_error = None
        self.load_time_sec = None

        if not lazy_load:
            self.load_model()

    def load_model(self) -> bool:
        """Loads the ViT processor and classifier weights once if not already initialized."""
        if self.model is not None:
            return True

        start_t = time.perf_counter()
        try:
            # Determine compute device
            if self.device_requested is not None:
                self.device = torch.device(self.device_requested)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Determine model path (local directory preferred)
            if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
                model_source = LOCAL_MODEL_DIR
                local_only = True
            else:
                model_source = HUB_MODEL_ID
                local_only = False

            self.processor = AutoImageProcessor.from_pretrained(
                model_source,
                local_files_only=local_only
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                model_source,
                local_files_only=local_only
            )
            self.model.to(self.device)
            self.model.eval()

            # Verify label indices dynamically from model config
            config_id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
            self.real_idx = next(k for k, v in config_id2label.items() if "real" in v)
            self.fake_idx = next(k for k, v in config_id2label.items() if "fake" in v)

            self.load_time_sec = time.perf_counter() - start_t
            return True

        except Exception as e:
            self.load_error = str(e)
            self.processor = None
            self.model = None
            return False

    def unload_model(self) -> None:
        """
        Explicitly releases the Vision Transformer model and image processor
        to free system and GPU memory.
        """
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image_input: Any) -> DetectorResult:
        """
        Execute single-image inference and return a standardized DetectorResult.

        Accepts:
          - PIL Image
          - File path string
          - Numpy image array (or any object convertible to PIL Image)
        """
        start_time = time.perf_counter()

        # 1. Validation for empty input
        if image_input is None:
            return DetectorResult.create_error(
                detector_name=self.DETECTOR_NAME,
                model_name=self.MODEL_NAME,
                error_message="No image provided (input was None).",
                warning=self.LIMITATION_WARNING,
                device=str(self.device) if self.device else None,
                preprocessing=self.PREPROCESSING_NAME,
            )

        # 2. Safe image loading and RGB conversion
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return DetectorResult.create_error(
                        detector_name=self.DETECTOR_NAME,
                        model_name=self.MODEL_NAME,
                        error_message=f"Image file does not exist: {image_input}",
                        warning=self.LIMITATION_WARNING,
                        device=str(self.device) if self.device else None,
                        preprocessing=self.PREPROCESSING_NAME,
                    )
                with Image.open(image_input) as opened_img:
                    img_rgb = opened_img.convert("RGB")
            elif isinstance(image_input, Image.Image):
                img_rgb = image_input.convert("RGB")
            elif hasattr(image_input, "__array__"):
                img_rgb = Image.fromarray(image_input).convert("RGB")
            else:
                return DetectorResult.create_error(
                    detector_name=self.DETECTOR_NAME,
                    model_name=self.MODEL_NAME,
                    error_message=f"Unsupported image input type: {type(image_input).__name__}",
                    warning=self.LIMITATION_WARNING,
                    device=str(self.device) if self.device else None,
                    preprocessing=self.PREPROCESSING_NAME,
                )
        except Exception as img_err:
            return DetectorResult.create_error(
                detector_name=self.DETECTOR_NAME,
                model_name=self.MODEL_NAME,
                error_message=f"Invalid or corrupted image: {str(img_err)}",
                warning=self.LIMITATION_WARNING,
                device=str(self.device) if self.device else None,
                preprocessing=self.PREPROCESSING_NAME,
            )

        # 3. Ensure model is loaded
        if not self.load_model():
            return DetectorResult.create_error(
                detector_name=self.DETECTOR_NAME,
                model_name=self.MODEL_NAME,
                error_message=f"Model loading failed: {self.load_error}",
                warning=self.LIMITATION_WARNING,
                device=str(self.device) if self.device else None,
                preprocessing=self.PREPROCESSING_NAME,
            )

        # 4. Inference execution under torch.inference_mode()
        try:
            # Dynamic device adjustment if running inside ZeroGPU context
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if next(self.model.parameters()).device != current_device:
                self.model.to(current_device)
                self.device = current_device

            inputs = self.processor(images=img_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape [1, 2]
                probs = torch.softmax(logits, dim=1).squeeze()

            fake_prob = float(probs[self.fake_idx].item())
            real_prob = float(probs[self.real_idx].item())

            # Relative logit (fake_logit - real_logit), such that logit >= 0 aligns with prob >= 0.50
            real_logit = float(logits[0, self.real_idx].item())
            fake_logit = float(logits[0, self.fake_idx].item())
            relative_logit = fake_logit - real_logit

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Output decision and confidence
            is_fake = fake_prob >= self.DECISION_THRESHOLD
            prediction = PREDICTION_FAKE if is_fake else PREDICTION_REAL
            confidence = fake_prob if is_fake else real_prob

            return DetectorResult(
                detector_name=self.DETECTOR_NAME,
                model_name=self.MODEL_NAME,
                prediction=prediction,
                fake_probability=round(fake_prob, 6),
                real_probability=round(real_prob, 6),
                raw_logit=round(relative_logit, 6),
                confidence=round(confidence, 6),
                threshold=self.DECISION_THRESHOLD,
                device=str(self.device),
                preprocessing=self.PREPROCESSING_NAME,
                inference_time_ms=round(latency_ms, 2),
                error=None,
                warning=self.LIMITATION_WARNING,
            )

        except Exception as inf_err:
            return DetectorResult.create_error(
                detector_name=self.DETECTOR_NAME,
                model_name=self.MODEL_NAME,
                error_message=f"Inference execution failed: {str(inf_err)}",
                warning=self.LIMITATION_WARNING,
                device=str(self.device) if self.device else None,
                preprocessing=self.PREPROCESSING_NAME,
            )
