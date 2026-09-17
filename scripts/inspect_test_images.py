import os
from PIL import Image

base = r"C:\Users\user\.gemini\antigravity-ide\brain\3b54ff09-c09f-4fe6-9276-f4cca9fbc6d7"
images = [
    ("Known real photograph", os.path.join(base, "scratch", "real_phone_photo.png"), "Real photograph captured on smartphone camera"),
    ("Known AI-generated image", os.path.join(base, "ai_cyberpunk_city_1789660863192.jpg"), "AI-generated synthetic image created via text-to-image generator (Google Imagen 3)"),
    ("Known manipulated image", os.path.join(base, "scratch", "manipulated_image.png"), "Composite manipulated image created via digital editing / splicing of real and generated elements"),
    ("Invalid / corrupted image", os.path.join(base, "scratch", "corrupted_image.png"), "Corrupted synthetic test file with invalid image header and truncated bytes")
]

for name, path, src_info in images:
    print(f"=== {name} ===")
    print(f"Path: {path}")
    print(f"Test label & source: {src_info}")
    if not os.path.exists(path):
        print("File does not exist!")
        continue
    sz = os.path.getsize(path)
    print(f"File size: {sz} bytes ({sz / (1024*1024):.2f} MB)")
    try:
        with Image.open(path) as img:
            print(f"Format: {img.format}")
            print(f"Dimensions (W x H): {img.width} x {img.height}")
            print(f"Aspect ratio: {img.width / img.height:.4f}")
            print(f"Color mode: {img.mode}")
            has_alpha = ("A" in img.mode) or ("transparency" in img.info)
            print(f"Alpha channel present: {has_alpha}")
            area = img.width * img.height
            print(f"Total pixel count: {area:,} pixels")
            crop_area = 224 * 224
            pct_area = (crop_area / area) * 100
            print(f"Direct CenterCrop(224) coverage: {pct_area:.3f}% of original image area")
    except Exception as e:
        print(f"Cannot open as image: {e}")
    print()
