"""
ConvNeXt Deepfake Detector Adapter (Chen et al. 2024 / RAID Constituent)
Provides a clean, standardized wrapper around the chen2024_convnext checkpoint from aimagelab/RAID_ckpt.
Adheres strictly to the DetectorResult schema, supporting lazy loading and robust error handling.
"""

import os
import sys
import time
from typing import Optional, Union, Any, Dict
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.detector_schema import (
    DetectorResult,
    PREDICTION_REAL,
    PREDICTION_FAKE,
    PREDICTION_ERROR,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "models", "chen2024_convnext", "16_acc0.9993.pth"
)


class Chen2024ConvNeXtArchitecture(nn.Module):
    """
    ConvNeXt-Base model with linear embedding and binary classification head,
    matching the exact Chen et al. 2024 / RAID architecture:
      - Backbone: timm convnext_base (stem 128, stage blocks [3, 3, 27, 3], head 1024)
      - Classifier: nn.Linear(1024, 2)
    """

    def __init__(self, num_classes: int = 2, embedding_size: int = 1024):
        super().__init__()
        import timm
        # convnext_base produces 1024-dim embedding through its linear head
        self.model = timm.create_model(
            "convnext_base",
            pretrained=False,
            num_classes=embedding_size,
        )
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model(x)
        return self.fc(feat)


class DRCTNativeCropTransform:
    """
    Official DRCT evaluation transform (--is_crop, Chen et al. ICML 2024):
      - Native-resolution center cropping without prior resizing.
      - If image height or width < 224, pads symmetrically with constant 0 (black).
      - Center crops exactly (224, 224).
      - Normalizes using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
      - Converts to torch float32 tensor of shape [3, 224, 224].
    """

    def __init__(
        self,
        size: int = 224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img_np = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img_np = img[:, :, :3]
            elif img.ndim == 3 and img.shape[2] == 3:
                img_np = img
            else:
                raise ValueError(f"Unsupported numpy image shape: {img.shape}")
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] == 3:
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported tensor shape: {img.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        h, w = img_np.shape[:2]
        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            img_np = cv2.copyMakeBorder(
                img_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            h, w = img_np.shape[:2]

        start_y = (h - self.size) // 2
        start_x = (w - self.size) // 2
        cropped = img_np[start_y:start_y + self.size, start_x:start_x + self.size]

        norm = (cropped.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float()
        return tensor


class ConvNeXtDetectorAdapter:
    """
    Adapter for Chen et al. 2024 ConvNeXt deepfake detector from RAID / DRCT.

    Model Details:
      - Architecture: ConvNeXt-Base + Linear Head (1024 -> 2)
      - Checkpoint: models/chen2024_convnext/16_acc0.9993.pth (aimagelab/RAID_ckpt)
      - Preprocessing: Native-resolution CenterCrop(224) matching official DRCT --is_crop
      - Decision Threshold: 0.50 Fake Probability
      - Label Mapping: Index 0 = Real, Index 1 = Fake
    """

    DETECTOR_NAME = "Chen2024ConvNeXtDetector"
    MODEL_NAME = "chen2024_convnext (ConvNeXt-Base)"
    PREPROCESSING_NAME = (
        "PadIfNeeded(224, border_mode=BORDER_CONSTANT, value=0) -> "
        "CenterCrop(224) [native resolution, no resize] -> ToTensor() -> "
        "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
    )
    DECISION_THRESHOLD = 0.50
    LIMITATION_WARNING = (
        "ConvNeXt detector evaluated on synthetic/generator artifacts from RAID benchmark. "
        "Does not guarantee authenticity."
    )

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        lazy_load: bool = True,
    ):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self.device_requested = device
        self.model = None
        self.device = None
        self.transform = None
        self.real_idx = 0
        self.fake_idx = 1
        self.load_error = None
        self.load_time_sec = None

        if not lazy_load:
            self.load_model()

    def _init_transform(self):
        """Construct the official DRCT evaluation preprocessing transform (--is_crop)."""
        self.transform = DRCTNativeCropTransform(size=224)

    def load_model(self) -> bool:
        """
        Loads the ConvNeXt model architecture and checkpoint weights if not already loaded.
        Ensures lazy loading so module import has zero weight overhead.
        """
        if self.model is not None:
            return True

        start_t = time.perf_counter()
        try:
            # 1. Determine compute device
            if self.device_requested is not None:
                self.device = torch.device(self.device_requested)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 2. Validate checkpoint existence
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at: {self.checkpoint_path}"
                )

            # 3. Instantiate architecture
            net = Chen2024ConvNeXtArchitecture(num_classes=2, embedding_size=1024)

            # 4. Load weights strictly
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = net.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}"
                )

            net.to(self.device)
            net.eval()
            self.model = net

            # 5. Initialize preprocessing pipeline
            self._init_transform()

            self.load_time_sec = time.perf_counter() - start_t
            return True

        except Exception as e:
            self.load_error = str(e)
            self.model = None
            return False

    def unload_model(self) -> None:
        """
        Explicitly releases the neural network model and preprocessing pipeline
        to free system and GPU memory.
        """
        self.model = None
        self.transform = None
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
            # Dynamic device adjustment for ZeroGPU runtime
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if next(self.model.parameters()).device != current_device:
                self.model.to(current_device)
                self.device = current_device

            tensor_input = self.transform(img_rgb).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(tensor_input)  # shape: [1, 2]
                probs = torch.softmax(logits, dim=1).squeeze(0)

            real_prob = float(probs[self.real_idx].item())
            fake_prob = float(probs[self.fake_idx].item())

            real_logit = float(logits[0, self.real_idx].item())
            fake_logit = float(logits[0, self.fake_idx].item())
            relative_logit = fake_logit - real_logit

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Decision rule: index 1 is FAKE, threshold = 0.50
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
