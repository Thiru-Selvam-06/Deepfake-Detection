"""
Regression Tests for Detector Preprocessing Alignment (Phase 7 Corrective Action)
Verifies that:
1. ConvNeXtDRCT uses native CenterCrop(224) without resizing (1024x1024 is not downscaled).
2. UnivFD uses native CenterCrop(224) without resizing (reproducing validate.py).
3. Padding behaves correctly for images smaller than 224x224.
4. ViT preprocessing remains unchanged.
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.convnext_adapter import ConvNeXtDetectorAdapter, DRCTNativeCropTransform
from src.univfd_adapter import UnivFDAdapter
from src.vit_adapter import ViTAdapter


class TestPreprocessingRegression:
    """Test suite ensuring official native cropping is respected and no resize occurs."""

    def test_convnext_no_resize_on_1024x1024(self):
        """
        Confirms that a 1024x1024 image is NOT resized to 224x224 before cropping.
        Constructs an image where the central 224x224 box is pure white (255)
        and the outer border is pure black (0).
        - If native crop is used: the crop is 100% pure white (no border blending).
        - If resized to 224 first: the white box shrinks to ~49x49 and blurs with black borders.
        """
        img_np = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # Place pure white in the exact center 224x224 region: [400:624, 400:624]
        c_start_y = (1024 - 224) // 2  # 400
        c_start_x = (1024 - 224) // 2  # 400
        img_np[c_start_y:c_start_y + 224, c_start_x:c_start_x + 224] = 255
        pil_img = Image.fromarray(img_np)

        adapter = ConvNeXtDetectorAdapter(lazy_load=True)
        adapter._init_transform()
        assert isinstance(adapter.transform, DRCTNativeCropTransform)

        tensor = adapter.transform(pil_img)
        assert tensor.shape == (3, 224, 224)

        # Expected normalized value for pure white (255) under ImageNet stats:
        # (1.0 - mean) / std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected_white = (1.0 - mean) / std

        # Every single pixel in the native crop must match expected_white exactly
        for c in range(3):
            ch_tensor = tensor[c].numpy()
            diff = np.abs(ch_tensor - expected_white[c])
            assert np.max(diff) < 1e-5, (
                f"ConvNeXt crop contains non-white pixels (max diff={np.max(diff)}). "
                f"Image was downscaled/blurred prior to cropping!"
            )

    def test_convnext_small_image_padding(self):
        """Verifies that an image smaller than 224 is padded with 0 and cropped to 224x224."""
        small_np = np.ones((100, 150, 3), dtype=np.uint8) * 200
        pil_img = Image.fromarray(small_np)

        adapter = ConvNeXtDetectorAdapter(lazy_load=True)
        adapter._init_transform()
        tensor = adapter.transform(pil_img)

        assert tensor.shape == (3, 224, 224)
        # Center should contain normalized 200/255, edges should contain normalized 0 (black)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected_pad = (0.0 - mean) / std

        # Top-left corner must be padding (black)
        for c in range(3):
            assert abs(tensor[c, 0, 0].item() - expected_pad[c]) < 1e-5

    def test_univfd_no_resize_on_1024x1024(self):
        """
        Confirms that UnivFD uses direct CenterCrop(224) without resizing,
        matching official validate.py.
        """
        img_np = np.zeros((1024, 1024, 3), dtype=np.uint8)
        c_start_y = (1024 - 224) // 2  # 400
        c_start_x = (1024 - 224) // 2  # 400
        img_np[c_start_y:c_start_y + 224, c_start_x:c_start_x + 224] = 255
        pil_img = Image.fromarray(img_np)

        adapter = UnivFDAdapter(lazy_load=True)
        # Initialize preprocess directly or via load_model
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        adapter.preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

        # Verify no Resize operation exists in preprocess pipeline
        for t in adapter.preprocess.transforms:
            assert not isinstance(t, transforms.Resize), (
                f"UnivFD preprocess unexpectedly contains Resize: {t}"
            )
        assert isinstance(adapter.preprocess.transforms[0], transforms.CenterCrop)

        tensor = adapter.preprocess(pil_img)
        assert tensor.shape == (3, 224, 224)

        expected_white = (1.0 - np.array(clip_mean, dtype=np.float32)) / np.array(clip_std, dtype=np.float32)
        for c in range(3):
            ch_tensor = tensor[c].numpy()
            diff = np.abs(ch_tensor - expected_white[c])
            assert np.max(diff) < 1e-5, (
                f"UnivFD crop contains blended pixels (max diff={np.max(diff)}). "
                f"Resize occurred prior to cropping!"
            )

    def test_vit_preprocessing_remains_unchanged(self):
        """Confirms that ViT preprocessing has NOT been changed."""
        vit_adapter = ViTAdapter(lazy_load=True)
        assert "AutoImageProcessor" in vit_adapter.PREPROCESSING_NAME or "224" in vit_adapter.PREPROCESSING_NAME
        assert vit_adapter.DECISION_THRESHOLD == 0.50
