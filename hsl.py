import os
from PIL import Image, ImageEnhance
import numpy as np


def adjust_hue(image, hue_degrees):
    image_hsv = image.convert("HSV")
    h, s, v = image_hsv.split()
    h = np.array(h, dtype=np.uint8)
    hue_shift = int((hue_degrees / 360) * 255)
    h = (h.astype(int) + hue_shift) % 256
    h = np.clip(h, 0, 255).astype(np.uint8)
    image_hsv = Image.merge("HSV", (Image.fromarray(h), s, v))
    return image_hsv.convert("RGB")


def adjust_saturation(image, saturation_factor):
    return ImageEnhance.Color(image).enhance(saturation_factor)


def adjust_luminosity(image, brightness_factor):
    return ImageEnhance.Brightness(image).enhance(brightness_factor)


def adjust_hsl(image, hue_deg, sat_fac, bright_fac):
    image = adjust_hue(image, hue_deg)
    image = adjust_saturation(image, sat_fac)
    image = adjust_luminosity(image, bright_fac)
    return image


def process_all_subfolders(input_root, output_root, hue_deg, sat_fac, bright_fac):
    for root, _, files in os.walk(input_root):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')):
                continue

            input_path = os.path.join(root, filename)

            # Preserve folder structure
            relative_folder = os.path.relpath(root, input_root)
            output_folder = os.path.join(output_root, relative_folder)
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, filename)

            try:
                img = Image.open(input_path).convert("RGB")
                adjusted_img = adjust_hsl(img, hue_deg, sat_fac, bright_fac)
                adjusted_img.save(output_path)
                print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"❌ Error processing {input_path}: {e}")


if __name__ == "__main__":
    input_root = r"C:\Users\amma\Desktop\Manoj_Research\exp-4\predict224\MP PASTES CASES - output"  # 👈 or your MP PASTES CASES - input
    output_root = r"C:\Users\amma\Desktop\Manoj_Research\exp-4\predict224\hsl_new_prediction_output"

    hue_degrees = 45
    saturation_factor = 1.5
    brightness_factor = 1.2

    process_all_subfolders(input_root, output_root, hue_degrees, saturation_factor, brightness_factor)
    print("\n🎉 All HSL-adjusted images saved successfully!")
