import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Add UniversalFakeDetect to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UFD_DIR = os.path.join(BASE_DIR, "UniversalFakeDetect")
if UFD_DIR not in sys.path:
    sys.path.insert(0, UFD_DIR)

from models import get_model

# Constants from validate.py
MEAN_CLIP = [0.48145466, 0.4578275, 0.40821073]
STD_CLIP = [0.26862954, 0.26130258, 0.27577711]

# 1. Pipeline A: validate.py (RealFakeDataset in validate.py)
transform_validate_py = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_CLIP, std=STD_CLIP)
])

# 2. Pipeline B: datasets.py (data_loader in train.py / val_opt with loadSize=256, cropSize=224)
transform_train_val = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_CLIP, std=STD_CLIP)
])

def main():
    print("=================================================================")
    print("TASK 5: Comparing Official Inference Implementations vs Current App")
    print("=================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model exactly as official validate.py
    print("Loading CLIP:ViT-L/14 model...")
    model = get_model("CLIP:ViT-L/14")
    ckpt_path = os.path.join(UFD_DIR, "pretrained_weights", "fc_weights.pth")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Model and weights loaded from {ckpt_path}\n")

    # Pipeline C: clip.load's preprocess (used in current app.py)
    transform_app_py = model.preprocess

    test_images = [
        ("Known real photograph", r"C:\Users\user\.gemini\antigravity-ide\brain\3b54ff09-c09f-4fe6-9276-f4cca9fbc6d7\scratch\real_phone_photo.png", "REAL"),
        ("Known AI-generated image", r"C:\Users\user\.gemini\antigravity-ide\brain\3b54ff09-c09f-4fe6-9276-f4cca9fbc6d7\ai_cyberpunk_city_1789660863192.jpg", "FAKE"),
        ("Known manipulated image", r"C:\Users\user\.gemini\antigravity-ide\brain\3b54ff09-c09f-4fe6-9276-f4cca9fbc6d7\scratch\manipulated_image.png", "EVALUATE")
    ]

    pipelines = [
        ("Current app.py (model.preprocess: Resize(224, bicubic) -> CenterCrop(224))", transform_app_py),
        ("validate.py (Direct CenterCrop(224) without resizing)", transform_validate_py),
        ("train.py/datasets.py (Resize(256, bicubic) -> CenterCrop(224))", transform_train_val)
    ]

    results = []

    for img_name, img_path, expected_label in test_images:
        print(f"-----------------------------------------------------------------")
        print(f"Test Image: {img_name}")
        print(f"Path: {img_path}")
        print(f"Expected / Label: {expected_label}")
        
        with Image.open(img_path) as raw_img:
            img_rgb = raw_img.convert("RGB")
            print(f"Original size: {img_rgb.size} ({img_rgb.width}x{img_rgb.height})")

            for pipe_name, pipe_transform in pipelines:
                # Apply transform
                tensor = pipe_transform(img_rgb).unsqueeze(0).to(device)

                with torch.inference_mode():
                    feat = model.model.encode_image(tensor)
                    logit = model.fc(feat.float()).item()
                    prob_fake = torch.sigmoid(torch.tensor(logit)).item()

                pred_label = "FAKE" if prob_fake >= 0.50 else "REAL"
                conf = prob_fake if pred_label == "FAKE" else (1.0 - prob_fake)

                print(f"  [{pipe_name}]")
                print(f"    Raw Logit:    {logit:+.5f}")
                print(f"    Fake Prob:    {prob_fake:.6f} ({prob_fake*100:.2f}%)")
                print(f"    Prediction:   {pred_label} (Confidence: {conf*100:.2f}%)")
                
                results.append({
                    "image": img_name,
                    "pipeline": pipe_name,
                    "logit": logit,
                    "prob_fake": prob_fake,
                    "prediction": pred_label,
                    "confidence": conf
                })
        print()

    print("=================================================================")
    print("Summary Analysis:")
    print("=================================================================")
    for r in results:
        print(f"{r['image'][:25]:<25} | {r['pipeline'][:35]:<35} | logit: {r['logit']:+8.4f} | prob: {r['prob_fake']*100:6.2f}% | pred: {r['prediction']}")

if __name__ == "__main__":
    main()
