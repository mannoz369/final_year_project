import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import gdown
import os

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"
# Using the direct download link for gdown is more reliable
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b"


# =====================
# DOWNLOAD .PTH MODEL IF NOT EXISTS
# =====================
def download_predictor_model():
    if not os.path.exists(PREDICTOR_PATH):
        with st.spinner("📥 Downloading predictor model from Google Drive..."):
            gdown.download(GOOGLE_DRIVE_URL, PREDICTOR_PATH, quiet=False)
        st.success("✅ Predictor model downloaded successfully!")


# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_visualizer():
    # FIX: Added compile=False to prevent version incompatibility errors when loading the H5 model.
    return load_model(VISUALIZER_PATH, compile=False)

@st.cache_resource
def load_predictor():
    # This class defines a classifier, not an image-to-image model.
    # See the note in Step 2 of the main function.
    class PredictorNet(nn.Module):
        def __init__(self):
            super(PredictorNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # This final layer outputs only 2 values, not an image.
            self.fc = nn.Linear(16 * 112 * 112, 2)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = PredictorNet()
    model.load_state_dict(torch.load(PREDICTOR_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


# =====================
# HSL ADJUSTMENTS FOR DAMAGE QUANTIFICATION
# =====================
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

def quantify_damage(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    damage_score = np.mean(gray) / 255.0
    return round(damage_score * 100, 2)


# =====================
# STREAMLIT APP
# =====================
def main():
    st.title("🔧 Damage Prediction & Visualization App")
    st.write("Upload an image to visualize initial damage, predict future damage, and quantify it.")

    # Ensure predictor model exists
    download_predictor_model()

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
        st.image(image, caption="📌 Uploaded Image", use_container_width=True)

        # Load models
        visualizer = load_visualizer()
        predictor = load_predictor()

        # ========== Step 1: Visualization ==========
        st.subheader("Step 1: Initial Damage Visualization")
        img_resized = image.resize((224, 224))
        input_arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        vis_output = visualizer.predict(input_arr)
        vis_img = Image.fromarray((vis_output[0] * 255).astype(np.uint8))
        # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
        st.image(vis_img, caption="Initial Damage Visualization", use_container_width=True)

        # ========== Step 2: Predictor ==========
        st.subheader("Step 2: Future Damage Prediction")
        img_tensor = torch.tensor(np.array(img_resized).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
        with torch.no_grad():
            pred_output = predictor(img_tensor)

        # IMPORTANT NOTE: The 'PredictorNet' model above outputs a tensor with only 2 values,
        # not an image. The code below will likely fail because it tries to reshape these 2 values
        # into a 224x224 image. You need to either:
        #   1. Change the PredictorNet architecture to an image-to-image model (like a U-Net or Autoencoder).
        #   2. Change the code below to handle the 2-value output (e.g., display it as text).
        
        # This line will likely cause a `ValueError` due to the model's architecture.
        pred_img = pred_output[0].detach().numpy().reshape(224, 224, -1)
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min()) * 255
        pred_img = Image.fromarray(pred_img.astype(np.uint8))
        # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
        st.image(pred_img, caption="Predicted Future Damage", use_container_width=True)

        # ========== Step 3: Quantification ==========
        st.subheader("Step 3: Damage Quantification")
        # Ensure the images are RGB before passing to HSL functions
        hsl_input = image.resize((224, 224)).convert("RGB")
        hsl_output = pred_img.convert("RGB")

        hsl_input_adjusted = adjust_hsl(hsl_input, 45, 1.5, 1.2)
        hsl_output_adjusted = adjust_hsl(hsl_output, 45, 1.5, 1.2)

        damage_in = quantify_damage(hsl_input_adjusted)
        damage_out = quantify_damage(hsl_output_adjusted)

        st.write(f"📊 **Damage in Input Image:** {damage_in}%")
        st.write(f"📊 **Damage in Predicted Image:** {damage_out}%")

        # FIX: Replaced deprecated 'use_column_width' with 'use_container_width'
        st.image([hsl_input_adjusted, hsl_output_adjusted], caption=["HSL Adjusted Input", "HSL Adjusted Prediction"], use_container_width=True)


if __name__ == "__main__":
    main()
