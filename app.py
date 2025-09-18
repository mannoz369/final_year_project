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
import functools  # Required for the UnetGenerator
import tensorflow as tf

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"  # Keras model used both for visualization and Grad-CAM
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b/view?usp=sharing"

# =====================
# PIX2PIX MODEL ARCHITECTURE
# (Pasted from the official repository)
# =====================

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator"""
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection."""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections."""
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


# =====================
# HELPER FUNCTIONS
# =====================

def download_predictor_model():
    if not os.path.exists(PREDICTOR_PATH):
        with st.spinner("📥 Downloading predictor model from Google Drive..."):
            gdown.download(GOOGLE_DRIVE_URL, PREDICTOR_PATH, quiet=False)
        st.success("✅ Predictor model downloaded successfully!")


@st.cache_resource
def load_visualizer():
    # Load Keras model without its training optimizer to prevent version errors
    if not os.path.exists(VISUALIZER_PATH):
        return None
    return load_model(VISUALIZER_PATH, compile=False)


@st.cache_resource
def load_predictor():
    # Initialize the correct UnetGenerator architecture
    if not os.path.exists(PREDICTOR_PATH):
        return None
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(PREDICTOR_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


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

# -----------------------------
# Grad-CAM utilities (adapted)
# -----------------------------
def preprocess_pil_for_keras(pil_img, model_input_size=(128, 128)):
    """Take a PIL RGB image and produce (original_rgb, input_tensor(1,H,W,3))"""
    original = np.array(pil_img.convert("RGB"))
    resized = cv2.resize(original, model_input_size)
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return original, input_tensor

def find_last_conv_layer(model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        # Check for Conv2D or similar conv layers with 4D output
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = None
        if hasattr(layer, "name") and out_shape is not None:
            if isinstance(out_shape, (list, tuple)) and len(out_shape) == 4:
                # Heuristic: conv layers often have 'conv' in name; accept if 4D
                if 'conv' in layer.name or 'Conv' in layer.name:
                    last_conv_layer_name = layer.name
                    break
    return last_conv_layer_name

def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    """
    image_array: (1,H,W,3) float32
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        # If model output is regression scalar, choose predictions[:, 0]
        if len(predictions.shape) == 2 and predictions.shape[1] == 1:
            loss = predictions[:, 0]
        else:
            # fallback: sum of outputs
            loss = tf.reduce_sum(predictions, axis=1)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    denom = tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = heatmap / denom
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)
    heatmap_3ch = cv2.merge([heatmap_8u] * 3)
    # produce color map
    heatmap_color = cv2.applyColorMap(heatmap_8u, colormap)
    # convert heatmap_color from BGR to RGB because applyColorMap outputs BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, heatmap_resized

# =====================
# STREAMLIT APP
# =====================
def main():
    st.set_page_config(page_title="Damage Prediction & Grad-CAM", layout="wide")
    st.title("🔧 Damage Prediction, Visualization & Grad-CAM")
    st.write("Upload an image to visualize initial damage, predict future damage, run Grad-CAM, and quantify damage.")

    download_predictor_model()

    # Load models (Keras visualizer + Grad-CAM model; PyTorch predictor)
    visualizer = load_visualizer()
    predictor = load_predictor()

    if visualizer is None:
        st.warning(f"⚠️ Keras model `{VISUALIZER_PATH}` not found. Grad-CAM and visualization won't work until it's available in the app folder.")
    else:
        # find last conv layer once
        try:
            last_conv_layer_name = find_last_conv_layer(visualizer)
            if last_conv_layer_name is None:
                st.warning("⚠️ Couldn't automatically find a 4D convolutional layer in the Keras model for Grad-CAM.")
        except Exception as e:
            last_conv_layer_name = None
            st.warning(f"⚠️ Error identifying last conv layer: {e}")

    if predictor is None:
        st.warning(f"⚠️ PyTorch predictor `{PREDICTOR_PATH}` not found. Prediction step will be skipped until it's available.")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Sidebar options for HSL adjustments & Grad-CAM
    st.sidebar.header("Options")
    hue_deg = st.sidebar.slider("Hue shift (degrees)", -180, 180, 45)
    sat_fac = st.sidebar.slider("Saturation factor", 0.1, 3.0, 1.5)
    bright_fac = st.sidebar.slider("Brightness factor", 0.1, 3.0, 1.2)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay", value=True)
    gradcam_alpha = st.sidebar.slider("Grad-CAM alpha (overlay opacity)", 0.0, 1.0, 0.4)
    download_overlay = st.sidebar.checkbox("Show 'Download overlay' button", value=True)

    if uploaded_file:
        # read PIL image
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("📌 Uploaded Image")
        st.image(image, use_column_width=True)

        # ========== Visualization (Keras model) ==========
        st.subheader("Step 1: Initial Damage Visualization (Keras visualizer)")
        if visualizer is not None:
            # many Keras models expect 128x128 for your damage_predictor; keep that as default
            img_resized_128 = image.resize((128, 128))
            input_arr = np.expand_dims(np.array(img_resized_128) / 255.0, axis=0).astype(np.float32)
            try:
                vis_output = visualizer.predict(input_arr)
                # Handle outputs that may be same shape as input or different
                vis_img = Image.fromarray((np.clip(vis_output[0], 0, 1) * 255).astype(np.uint8))
                st.image(vis_img, caption="Initial Damage Visualization (from Keras model)", use_column_width=True)
            except Exception as e:
                st.error(f"Error running visualizer model: {e}")
        else:
            st.info("Visualizer model not available. Skipping this step.")

        # ========== Predictor (PyTorch U-Net) ==========
        st.subheader("Step 2: Future Damage Prediction (PyTorch U-Net)")
        if predictor is not None:
            # use 256x256 for U-Net predictor
            img_resized_256 = image.resize((256, 256))
            img_tensor = torch.tensor(np.array(img_resized_256).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            img_tensor = (img_tensor / 127.5) - 1.0
            try:
                with torch.no_grad():
                    pred_output = predictor(img_tensor)
                pred_numpy = pred_output[0].cpu().numpy()
                pred_numpy = (pred_numpy + 1) / 2.0 * 255.0
                pred_numpy = pred_numpy.transpose(1, 2, 0)
                pred_img = Image.fromarray(np.clip(pred_numpy, 0, 255).astype(np.uint8))
                st.image(pred_img, caption="Predicted Future Damage (PyTorch U-Net)", use_column_width=True)
            except Exception as e:
                st.error(f"Error running PyTorch predictor: {e}")
                pred_img = None
        else:
            st.info("PyTorch predictor not available. Skipping prediction step.")
            # still create a fallback 256 image so quantification can run comparably
            img_resized_256 = image.resize((256, 256))
            pred_img = None

        # ========== Grad-CAM (Keras damage_predictor) ==========
        st.subheader("Step 2.1: Grad-CAM (Keras damage_predictor.h5)")
        cam_col1, cam_col2 = st.columns(2)
        cam_displayed = False
        if visualizer is not None and last_conv_layer_name is not None:
            # preprocess original for Grad-CAM
            original_rgb, input_tensor = preprocess_pil_for_keras(image, model_input_size=(128, 128))
            try:
                heatmap = make_gradcam_heatmap(visualizer, input_tensor, last_conv_layer_name)
                overlay_img, heatmap_resized = overlay_heatmap(original_rgb, heatmap, alpha=gradcam_alpha)
                overlay_pil = Image.fromarray(np.uint8(overlay_img))
                heatmap_vis = Image.fromarray(np.uint8(255 * heatmap_resized)).convert("L")

                if show_gradcam:
                    cam_col1.image(overlay_pil, caption="Grad-CAM Overlay", use_column_width=True)
                    cam_col2.image(heatmap_vis, caption="Heatmap (grayscale)", use_column_width=True)
                    cam_displayed = True

                # offer download option
                if download_overlay and show_gradcam:
                    buf = io.BytesIO()
                    overlay_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button("⬇️ Download Grad-CAM overlay", data=byte_im, file_name="gradcam_overlay.png", mime="image/png")
            except Exception as e:
                st.error(f"Error computing Grad-CAM: {e}")
        else:
            st.info("Grad-CAM unavailable: Keras model or last convolutional layer not found.")

        # ========== Step 3: Quantification ==========
        st.subheader("Step 3: Damage Quantification & HSL Adjust")
        # Work on consistent 256x256 images
        if 'img_resized_256' not in locals():
            img_resized_256 = image.resize((256, 256))
        hsl_input = img_resized_256.convert("RGB")
        # If predictor produced pred_img, ensure it's 256x256; else we'll use input as placeholder
        if pred_img is None:
            pred_img_for_quant = hsl_input
        else:
            pred_img_for_quant = pred_img.resize((256, 256)).convert("RGB")

        # Apply HSL adjustments
        hsl_input_adjusted = adjust_hsl(hsl_input, hue_deg, sat_fac, bright_fac)
        hsl_output_adjusted = adjust_hsl(pred_img_for_quant, hue_deg, sat_fac, bright_fac)

        damage_in = quantify_damage(hsl_input_adjusted)
        damage_out = quantify_damage(hsl_output_adjusted)

        st.write(f"📊 **Damage in Input Image (after HSL adj):** {damage_in}%")
        st.write(f"📊 **Damage in Predicted Image (after HSL adj):** {damage_out}%")

        col_a, col_b = st.columns(2)
        col_a.image(hsl_input_adjusted, caption="HSL Adjusted Input", use_column_width=True)
        col_b.image(hsl_output_adjusted, caption="HSL Adjusted Prediction", use_column_width=True)

        # Option to save overlay next to prediction (if overlay exists)
        if cam_displayed and pred_img is not None:
            st.success("✅ Grad-CAM computed and displayed for uploaded image.")

    else:
        st.info("Upload an image to get started. The app will show visualization, prediction, Grad-CAM overlays, and damage quantification.")

if __name__ == "__main__":
    main()

