import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import gdown
import os
import io
import functools
import tensorflow as tf
import traceback # Import for detailed error logging

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"
GOOGLE_DRIVE_ID = "https://drive.google.com/file/d/1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b/view?usp=sharing"

# =====================
# PIX2PIX MODEL ARCHITECTURE (Unchanged)
# =====================
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial: use_bias = norm_layer.func == nn.InstanceNorm2d
        else: use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None: input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout: model = down + [submodule] + up + [nn.Dropout(0.5)]
            else: model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    def forward(self, x):
        if self.outermost: return self.model(x)
        else: return torch.cat([x, self.model(x)], 1)

# =====================
# HELPER FUNCTIONS
# =====================

def download_predictor_model():
    if not os.path.exists(PREDICTOR_PATH):
        with st.spinner("📥 Downloading predictor model..."):
            gdown.download(id=GOOGLE_DRIVE_ID, output=PREDICTOR_PATH, quiet=False)
        st.success("✅ Predictor model downloaded!")

@st.cache_resource
def load_visualizer():
    # --- NEW DEBUGGING VERSION ---
    st.warning("Running the NEW debugging version of `load_visualizer`...")
    print("\n--- DEBUG: Attempting to load and build Keras visualizer model ---")
    
    if not os.path.exists(VISUALIZER_PATH):
        print(f"--- DEBUG: Visualizer model file not found at {VISUALIZER_PATH}")
        return None
        
    try:
        print(f"--- DEBUG: Loading model from {VISUALIZER_PATH}...")
        model = load_model(VISUALIZER_PATH, compile=False)
        print("--- DEBUG: Model loaded successfully.")
        
        print("--- DEBUG: Attempting dummy forward pass to build model...")
        dummy_input = tf.zeros((1, 128, 128, 3))
        _ = model(dummy_input)
        print("--- DEBUG: Model built successfully via dummy pass. Printing summary:")
        
        # Print the model summary to the console log
        model.summary(print_fn=lambda x: print(x))
        
        print("--- DEBUG: Keras model is loaded and built. ---")
        return model
        
    except Exception as e:
        print("\n--- DEBUG: CRITICAL ERROR during model loading or building ---")
        # Print the full error traceback to the console
        traceback.print_exc()
        print("--- END OF CRITICAL ERROR ---\n")
        st.error(f"Failed to load or build the Keras model. Check terminal for details. Error: {e}")
        return None
    # --- END OF DEBUGGING VERSION ---

@st.cache_resource
def load_predictor():
    if not os.path.exists(PREDICTOR_PATH):
        return None
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(PREDICTOR_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- HSL and Quantification Functions (Unchanged) ---
def adjust_hsl(image, hue_deg, sat_fac, bright_fac):
    image = adjust_hue(Image.fromarray(np.uint8(image)), hue_deg)
    image = adjust_saturation(image, sat_fac)
    image = adjust_luminosity(image, bright_fac)
    return np.array(image)
def adjust_hue(image, hue_degrees):
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV).astype(float)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_degrees) % 180
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
def adjust_saturation(image, saturation_factor): return ImageEnhance.Color(image).enhance(saturation_factor)
def adjust_luminosity(image, brightness_factor): return ImageEnhance.Brightness(image).enhance(brightness_factor)
def quantify_damage(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return round(np.mean(gray) / 255.0 * 100, 2)


# --- Grad-CAM UTILITIES (Unchanged) ---
def preprocess_for_gradcam(pil_img, model_input_size=(128, 128)):
    original_rgb = np.array(pil_img.convert("RGB"))
    resized = cv2.resize(original_rgb, model_input_size)
    input_tensor = resized.astype(np.float32) / 255.0
    return original_rgb, np.expand_dims(input_tensor, axis=0)
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name and len(layer.output.shape) == 4:
            return layer.name
    return None
def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()
def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)
    heatmap_color_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap_8u, colormap), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 1 - alpha, heatmap_color_rgb, alpha, 0), heatmap_resized

# =====================
# MAIN STREAMLIT APP
# =====================
def main():
    st.set_page_config(page_title="Damage Analysis Tool", layout="wide")
    st.title("🔧 Damage Prediction & Visualization Tool")
    
    download_predictor_model()
    visualizer = load_visualizer()
    predictor = load_predictor()

    last_conv_layer_name = None
    if visualizer:
        last_conv_layer_name = find_last_conv_layer(visualizer)
    
    st.sidebar.header("⚙️ Adjustment Options")
    # ... (sidebar sliders remain the same) ...

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("📌 Uploaded Image")
        # --- NEW STREAMLIT PARAMETER ---
        st.image(image, width='stretch')

        # ... (rest of the main function remains the same, but with the width parameter updated)
        # All instances of use_container_width=True are replaced with width='stretch'
        
        # ... Grad-CAM Step ...
        col1, col2 = st.columns(2)
        col1.image(overlay_pil, caption="Grad-CAM Overlay", width='stretch')
        col2.image(heatmap_vis, caption="Heatmap (Grayscale)", width='stretch')
        
        # ... Prediction Step ...
        st.image(pred_img, caption="Predicted Future Damage", width='stretch')
        
        # ... Quantification Step ...
        col_a, col_b = st.columns(2)
        col_a.image(hsl_input_adjusted, caption="HSL Adjusted Input", width='stretch')
        col_b.image(hsl_output_adjusted, caption="HSL Adjusted Prediction", width='stretch')

# Ensure the main function is called correctly
if __name__ == "__main__":
    main()
