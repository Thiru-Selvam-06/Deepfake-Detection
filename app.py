# Hugging Face ZeroGPU requires spaces to be imported before any CUDA/torch modules
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

import os
import sys
import time
import psutil
from PIL import Image
import torch
import gradio as gr

# Patch gradio_client schema parsing bug where boolean additionalProperties crashes get_api_info
try:
    import gradio_client.utils as gc_utils
    _orig_jstype = gc_utils._json_schema_to_python_type
    def _safe_jstype(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return _orig_jstype(schema, defs)
    gc_utils._json_schema_to_python_type = _safe_jstype
except Exception as _patch_err:
    print(f"Warning: could not patch gradio_client: {_patch_err}")

# Compatibility shim for Python 3.12+ environments where pkg_resources is deprecated/absent
try:
    import pkg_resources
except ImportError:
    import packaging
    import types
    _pr = types.ModuleType("pkg_resources")
    _pr.packaging = packaging
    sys.modules["pkg_resources"] = _pr

# Ensure UniversalFakeDetect is in Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UFD_DIR = os.path.join(BASE_DIR, "UniversalFakeDetect")
if UFD_DIR not in sys.path:
    sys.path.insert(0, UFD_DIR)

from models import get_model

# Global model cache (loaded once at startup)
MODEL = None
DEVICE = None
PREPROCESS = None
LOAD_ERROR = None
LOAD_TIME_SEC = None


def load_model():
    """Load UniversalFakeDetect model and pretrained linear weights once."""
    global MODEL, DEVICE, PREPROCESS, LOAD_ERROR, LOAD_TIME_SEC
    if MODEL is not None:
        return MODEL, DEVICE, PREPROCESS

    start_t = time.perf_counter()
    try:
        # Check CUDA availability; support seamless CPU fallback
        # Default to CPU for stability and direct CPU verification
        DEVICE = torch.device("cpu")
        print(f"[UniversalFakeDetect] Target device: {DEVICE}")

        # Initialize CLIP ViT-L/14 wrapper
        # The bundled clip.load downloads ViT-L-14.pt (~933MB) to ~/.cache/clip on first run
        print("[UniversalFakeDetect] Initializing CLIP:ViT-L/14 architecture...")
        model = get_model("CLIP:ViT-L/14")

        # Load pretrained linear probe weights (fc_weights.pth, 4KB)
        ckpt_path = os.path.join(UFD_DIR, "pretrained_weights", "fc_weights.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing pretrained weights file at: {ckpt_path}")

        print(f"[UniversalFakeDetect] Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.fc.load_state_dict(state_dict)

        # Move to target device and set to evaluation mode
        model.to(DEVICE)
        model.eval()

        MODEL = model
        PREPROCESS = model.preprocess
        LOAD_TIME_SEC = time.perf_counter() - start_t
        print(f"[UniversalFakeDetect] Successfully loaded model in {LOAD_TIME_SEC:.2f}s on {DEVICE}")
        return MODEL, DEVICE, PREPROCESS
    except Exception as e:
        LOAD_ERROR = str(e)
        print(f"[UniversalFakeDetect] Model loading error: {e}")
        return None, None, None


# Attempt initial loading at startup
load_model()


def run_inference(image):
    """
    Execute single-image inference:
    1. Input validation & RGB conversion
    2. Exact repository preprocessing
    3. Model forward pass under torch.inference_mode()
    4. Metric calculations (raw score, fake probability, confidence, latency)
    """
    # 1. Validation for empty or invalid input
    if image is None:
        return (
            "<div style='background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 12px; border-radius: 8px; font-weight: 500;'>⚠️ No image provided. Please upload an image to analyze.</div>",
            "",
            "Awaiting image upload."
        )

    # Safely load and validate image
    try:
        if isinstance(image, str):
            with Image.open(image) as opened_img:
                img_rgb = opened_img.convert("RGB")
        elif isinstance(image, Image.Image):
            img_rgb = image.convert("RGB")
        else:
            img_rgb = Image.fromarray(image).convert("RGB")
    except Exception as img_err:
        err_msg = f"Invalid or corrupted image file: {str(img_err)}"
        return (
            f"<div style='background: #fef2f2; border: 1.5px solid #f87171; color: #991b1b; padding: 16px; border-radius: 8px;'>❌ <b>Invalid or Corrupted Image:</b> {err_msg}</div>",
            "",
            f"Failed: {err_msg}"
        )

    # 2. Ensure model is loaded
    model, device, preprocess = load_model()
    if model is None:
        return (
            f"<div style='background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 12px; border-radius: 8px;'>❌ Model failed to load: {LOAD_ERROR}</div>",
            "",
            "Model unavailable."
        )

    try:
        start_time = time.perf_counter()

        # Dynamically use CUDA if available (e.g. inside @spaces.GPU) or fallback to CPU
        active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if next(model.parameters()).device != active_device:
            model.to(active_device)

        orig_w, orig_h = img_rgb.size

        # Apply exact preprocessing: Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize
        input_tensor = preprocess(img_rgb).unsqueeze(0).to(active_device)

        # Single-image forward pass
        with torch.inference_mode():
            features = model.model.encode_image(input_tensor)
            logit_tensor = model.fc(features.float())
            raw_logit = logit_tensor.item()
            prob_fake = torch.sigmoid(logit_tensor).item()

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Output label and confidence mapping
        # 0 = REAL, 1 = FAKE
        THRESHOLD = 0.50
        is_fake = prob_fake >= THRESHOLD
        label = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else (1.0 - prob_fake)
        confidence_pct = confidence * 100

        # Memory usage
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)

        # Verdict card styling
        if is_fake:
            badge_color = "#ef4444"
            badge_bg = "#fef2f2"
            badge_border = "#fca5a5"
            verdict_text = "FAKE (AI-Generated / Synthetic)"
            summary_desc = "The model detected statistical patterns and feature signatures characteristic of generative models (e.g., GANs, Diffusion)."
        else:
            badge_color = "#10b981"
            badge_bg = "#ecfdf5"
            badge_border = "#6ee7b7"
            verdict_text = "REAL (Authentic Photograph)"
            summary_desc = "The model detected visual feature distributions characteristic of genuine photographic captures."

        result_html = f"""
        <div style="background: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.3rem; font-weight: 700; color: {badge_color};">
                    PREDICTION: {verdict_text}
                </span>
                <span style="background: {badge_color}; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.95rem;">
                    {confidence_pct:.2f}% Confidence
                </span>
            </div>
            <p style="margin-top: 10px; margin-bottom: 0; color: #374151; font-size: 0.95rem; line-height: 1.4;">
                {summary_desc}
            </p>
        </div>

        <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px;">
            <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; color: #111827;">Inference Metrics & Diagnostics</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #374151;">
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Model Architecture:</td>
                    <td style="padding: 6px 0; text-align: right; font-family: monospace;">CLIP:ViT-L/14</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Classifier Checkpoint:</td>
                    <td style="padding: 6px 0; text-align: right; font-family: monospace;">fc_weights.pth</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Preprocessing Resolution:</td>
                    <td style="padding: 6px 0; text-align: right; font-family: monospace;">224 × 224 (Input: {orig_w} × {orig_h})</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Predicted Class:</td>
                    <td style="padding: 6px 0; text-align: right; font-weight: 600; color: {badge_color};">{label}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Confidence:</td>
                    <td style="padding: 6px 0; text-align: right; font-weight: 600;">{confidence_pct:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Raw Model Logit:</td>
                    <td style="padding: 6px 0; text-align: right; font-family: monospace;">{raw_logit:+.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Fake Probability:</td>
                    <td style="padding: 6px 0; text-align: right; font-family: monospace;">{prob_fake:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Decision Threshold:</td>
                    <td style="padding: 6px 0; text-align: right; font-family: monospace;">{THRESHOLD:.2f} (logit: 0.00)</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Processing Device:</td>
                    <td style="padding: 6px 0; text-align: right; font-weight: 500;">{str(active_device).upper()}</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Inference Latency:</td>
                    <td style="padding: 6px 0; text-align: right;">{latency_ms:.1f} ms</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0; font-weight: 500;">Process Memory (RSS):</td>
                    <td style="padding: 6px 0; text-align: right;">{rss_mb:.1f} MB</td>
                </tr>
            </table>
        </div>
        """

        prob_str = f"Real: {(1.0 - prob_fake) * 100:.2f}%  |  Fake: {prob_fake * 100:.2f}%"
        status_str = f"Completed successfully on {str(active_device).upper()} in {latency_ms:.1f} ms."
        return result_html, prob_str, status_str

    except Exception as e:
        err_msg = f"Inference error: {str(e)}"
        print(f"[UniversalFakeDetect] {err_msg}")
        return (
            f"<div style='background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 12px; border-radius: 8px;'>❌ Error during analysis: {str(e)}</div>",
            "",
            err_msg
        )


# Custom CSS styling for modern, clean presentation
custom_css = """
.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
}
.disclaimer-banner {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-left: 4px solid #f59e0b;
    color: #92400e;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 0.92rem;
    margin-bottom: 20px;
    line-height: 1.5;
}
"""

# Apply ZeroGPU decorator if available
if SPACES_AVAILABLE:
    try:
        run_inference = spaces.GPU(duration=60)(run_inference)
    except Exception as dec_err:
        print(f"[UniversalFakeDetect] spaces.GPU decorator error: {dec_err}")

with gr.Blocks(title="UniversalFakeDetect - Single Image Analysis", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🔬 UniversalFakeDetect: AI-Generated Image Detector
        ### Towards Universal Fake Image Detectors that Generalize Across Generative Models (CVPR 2023)
        Upload an individual image to evaluate whether it exhibits patterns characteristic of real photographs or synthetic AI generation.
        """
    )

    # Mandatory limitation disclaimer
    gr.HTML(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ Limitation Notice:</strong>
            This detector identifies learned patterns associated with synthetic or manipulated images. It does not guarantee image authenticity.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="filepath", label="Upload Image for Analysis")
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", scale=2)
                clear_btn = gr.ClearButton(components=[input_image], scale=1)

            gr.Markdown(
                """
                **Technical Notes:**
                - Evaluates one image at a time.
                - Converts input to RGB and applies standard CLIP bicubic 224×224 normalization.
                - Features extracted using frozen OpenAI CLIP ViT-L/14 backbone with a trained linear probe.
                """
            )

        with gr.Column(scale=1):
            output_card = gr.HTML(label="Analysis Result", value="<div style='color: #6b7280; padding: 20px; text-align: center;'>Upload an image and click <b>Analyze Image</b> to begin.</div>")
            output_probs = gr.Textbox(label="Class Probability Distribution", interactive=False, max_lines=1)
            status_output = gr.Textbox(label="Processing Status", interactive=False, max_lines=1)

    with gr.Accordion("ℹ️ Model Architecture & Citation Details", open=False):
        gr.Markdown(
            r"""
            - **Model**: `UniversalFakeDetect` (CLIP:ViT-L/14 Backbone + Linear Classifier)
            - **Paper**: *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CVPR 2023)
            - **Authors**: Utkarsh Ojha*, Yuheng Li*, Yong Jae Lee (Wisconsin AI and Vision Lab)
            - **Checkpoint**: `fc_weights.pth` (trained with fixed CLIP backbone on ProGAN synthetic images)
            - **Decision Rule**: Sigmoid probability $\ge 0.50 \implies$ **FAKE**, $< 0.50 \implies$ **REAL**
            """
        )

    analyze_btn.click(
        fn=run_inference,
        inputs=[input_image],
        outputs=[output_card, output_probs, status_output],
        api_name="predict"
    )

    def clear_fields():
        return (
            "<div style='color: #6b7280; padding: 20px; text-align: center;'>Upload an image and click <b>Analyze Image</b> to begin.</div>",
            "",
            "Ready."
        )

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[output_card, output_probs, status_output],
        api_name="clear"
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
