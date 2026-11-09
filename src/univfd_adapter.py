"""
UniversalFakeDetect Adapter Module
Provides a clean, standardized wrapper around the CVPR 2023 UniversalFakeDetect model.
Follows the DetectorResult schema and isolates model loading and inference logic.
"""

import os
import sys
import time
from typing import Optional, Union, Any
from PIL import Image
import torch

from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)

# Compatibility shim for Python 3.12+ environments where pkg_resources is deprecated/absent
try:
    import pkg_resources
except ImportError:
    try:
        import packaging
        import types
        _pr = types.ModuleType("pkg_resources")
        _pr.packaging = packaging
        sys.modules["pkg_resources"] = _pr
    except Exception:
        pass

# Ensure UniversalFakeDetect repository is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UFD_DIR = os.path.join(BASE_DIR, "UniversalFakeDetect")
if UFD_DIR not in sys.path:
    sys.path.insert(0, UFD_DIR)


class UnivFDAdapter:
    """
    Adapter for UniversalFakeDetect (CLIP:ViT-L/14 + Linear Probe).
    
    Adheres strictly to the official validate.py pipeline (Ojha et al., CVPR 2023):
      - Architecture: CLIP:ViT-L/14
      - Weights: UniversalFakeDetect/pretrained_weights/fc_weights.pth
      - Preprocessing: CenterCrop(224) at native resolution -> ToTensor() -> Normalize(clip stats)
      - Decision Threshold: 0.50 (logit 0.00)
      - Label Mapping: 0 = REAL, 1 = FAKE
    """

    DETECTOR_NAME = "UniversalFakeDetect"
    MODEL_NAME = "CLIP:ViT-L/14 + fc_weights.pth"
    PREPROCESSING_NAME = (
        "CenterCrop(224) [native resolution, validate.py] -> ToTensor() -> "
        "Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])"
    )
    DECISION_THRESHOLD = 0.50
    LIMITATION_WARNING = (
        "Identifies learned patterns associated with synthetic or manipulated images. "
        "Does not guarantee image authenticity."
    )

    def __init__(self, device: Optional[Union[str, torch.device]] = None, lazy_load: bool = True):
        self.device_requested = device
        self.model = None
        self.preprocess = None
        self.device = None
        self.load_error = None
        self.load_time_sec = None

        if not lazy_load:
            self.load_model()

    def load_model(self) -> bool:
        """Loads model backbone, classifier weights, and official validate.py preprocessing."""
        if self.model is not None:
            return True

        start_t = time.perf_counter()
        try:
            from models import get_model
            from torchvision import transforms

            # Determine compute device
            if self.device_requested is not None:
                self.device = torch.device(self.device_requested)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize CLIP backbone
            model = get_model("CLIP:ViT-L/14")

            # Load linear probe checkpoint
            ckpt_path = os.path.join(UFD_DIR, "pretrained_weights", "fc_weights.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Classifier checkpoint not found at: {ckpt_path}")

            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.fc.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()

            self.model = model
            # Official validate.py evaluation transform: direct native CenterCrop(224), no resize
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            self.preprocess = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ])
            self.load_time_sec = time.perf_counter() - start_t
            return True

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            self.preprocess = None
            return False

    def unload_model(self) -> None:
        """
        Explicitly releases the CLIP backbone model and preprocess transform
        to free system and GPU memory.
        """
        self.model = None
        self.preprocess = None
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
            # Check dynamic device placement (ZeroGPU or CPU fallback)
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if next(self.model.parameters()).device != current_device:
                self.model.to(current_device)
                self.device = current_device

            input_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                features = self.model.model.encode_image(input_tensor)
                logit_tensor = self.model.fc(features.float())
                raw_logit = logit_tensor.item()
                prob_fake = torch.sigmoid(logit_tensor).item()

            prob_real = 1.0 - prob_fake
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Output decision and confidence
            is_fake = prob_fake >= self.DECISION_THRESHOLD
            prediction = PREDICTION_FAKE if is_fake else PREDICTION_REAL
            confidence = prob_fake if is_fake else prob_real

            return DetectorResult(
                detector_name=self.DETECTOR_NAME,
                model_name=self.MODEL_NAME,
                prediction=prediction,
                fake_probability=round(prob_fake, 6),
                real_probability=round(prob_real, 6),
                raw_logit=round(raw_logit, 6),
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
