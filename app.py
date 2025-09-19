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

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"
# Use just the ID for robust downloading
GOOGLE_DRIVE_ID = "https://drive.google.com/file/d/1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b/view?usp=sharing"

# =====================
# PIX2PIX MODEL ARCHITECTURE
# (This section is unchanged)
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
            # Use the 'id' argument for a more reliable download
            gdown.download(id=GOOGLE_DRIVE_ID, output=PREDICTOR_PATH, quiet=False)
        st.success("✅ Predictor model downloaded successfully!")


@st.cache_resource
def load_visualizer():
    # Load Keras model without its training optimizer to prevent version errors
    if not os.path.exists(VISUALIZER_PATH):
        return None
    model = load_model(VISUALIZER_PATH, compile=False)

    # --- THIS IS THE CRITICAL FIX ---
    # Build the model by calling it once with a dummy input.
    # This forces Keras to define the output shapes of all layers.
    try:
        dummy_input = tf.zeros((1, 128, 128, 3))
        _ = model(dummy_input) # The output is ignored.
        print("Keras visualizer model built successfully.")
    except Exception as e:
        st.warning(f"Could not build the Keras model automatically. Grad-CAM might fail. Error: {e}")
    # --- END OF FIX ---

    return model


@st.cache_resource
def load_predictor():
    # Initialize the correct UnetGenerator architecture
    if not os.path.exists(PREDICTOR_PATH):
        return None
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(PREDICTOR_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- HSL and Quantification Functions (Unchanged) ---
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
# Grad-CAM UTILITIES
# -----------------------------
def preprocess_for_gradcam(pil_img, model_input_size=(128, 128)):
    """Converts a PIL image to the format needed for Grad-CAM."""
    original_rgb = np.array(pil_img.convert("RGB"))
    resized = cv2.resize(original_rgb, model_input_size)
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return original_rgb, input_tensor

def find_last_conv_layer(model):
    """Finds the name of the last convolutional layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name and len(layer.output.shape) == 4:
            return layer.name
    return None

def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    """Generates the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
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
    """Overlays the heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8u, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color_rgb, alpha, 0)
    return overlay, heatmap_resized


# =====================
# MAIN STREAMLIT APP
# =====================
def main():
    st.set_page_config(page_title="Damage Analysis Tool", layout="wide")
    st.title("🔧 Damage Prediction & Visualization Tool")
    st.write("Upload an image to visualize damage with Grad-CAM, predict future damage, and quantify the results.")

    download_predictor_model()

    # Load models
    visualizer = load_visualizer()
    predictor = load_predictor()

    last_conv_layer_name = None
    if visualizer:
        last_conv_layer_name = find_last_conv_layer(visualizer)
    
    # Sidebar options
    st.sidebar.header("⚙️ Adjustment Options")
    gradcam_alpha = st.sidebar.slider("Grad-CAM Overlay Opacity", 0.0, 1.0, 0.5)
    hue_deg = st.sidebar.slider("Hue Shift (degrees)", -180, 180, 45)
    sat_fac = st.sidebar.slider("Saturation Factor", 0.1, 3.0, 1.5)
    bright_fac = st.sidebar.slider("Brightness Factor", 0.1, 3.0, 1.2)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("📌 Uploaded Image")
        st.image(image, use_container_width=True) # FIX: use_container_width

        st.markdown("---")

        # ========== STEP 1: Damage Visualization with Grad-CAM ==========
        st.subheader("Step 1: Damage Visualization with Grad-CAM")
        if visualizer is not None and last_conv_layer_name is not None:
            with st.spinner("Generating Grad-CAM visualization..."):
                try:
                    original_rgb, input_tensor = preprocess_for_gradcam(image, model_input_size=(128, 128))
                    heatmap = make_gradcam_heatmap(visualizer, input_tensor, last_conv_layer_name)
                    overlay_img, heatmap_resized = overlay_heatmap(original_rgb, heatmap, alpha=gradcam_alpha)
                    
                    overlay_pil = Image.fromarray(overlay_img)
                    heatmap_vis = Image.fromarray((heatmap_resized * 255).astype(np.uint8))

                    col1, col2 = st.columns(2)
                    col1.image(overlay_pil, caption="Grad-CAM Overlay", use_container_width=True) # FIX
                    col2.image(heatmap_vis, caption="Heatmap (Grayscale)", use_container_width=True) # FIX
                    
                    buf = io.BytesIO()
                    overlay_pil.save(buf, format="PNG")
                    st.download_button("⬇️ Download Grad-CAM Overlay", data=buf.getvalue(), file_name="gradcam_overlay.png", mime="image/png")

                except Exception as e:
                    st.error(f"❌ Error computing Grad-CAM: {e}")
        else:
            st.warning("⚠️ Grad-CAM unavailable: Keras visualizer model or a convolutional layer was not found.")
        
        st.markdown("---")

        # ========== STEP 2: Future Damage Prediction (PyTorch U-Net) ==========
        st.subheader("Step 2: Future Damage Prediction (PyTorch U-Net)")
        pred_img = None
        if predictor is not None:
            with st.spinner("Predicting future damage..."):
                try:
                    img_resized_256 = image.resize((256, 256))
                    img_np = np.array(img_resized_256).transpose(2, 0, 1)
                    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
                    img_tensor = (img_tensor / 127.5) - 1.0

                    with torch.no_grad():
                        pred_output = predictor(img_tensor)
                    
                    pred_numpy = pred_output[0].cpu().numpy()
                    pred_numpy = (pred_numpy + 1) / 2.0 * 255.0
                    pred_numpy = pred_numpy.transpose(1, 2, 0)
                    pred_img = Image.fromarray(np.clip(pred_numpy, 0, 255).astype(np.uint8))
                    
                    st.image(pred_img, caption="Predicted Future Damage", use_container_width=True) # FIX
                except Exception as e:
                    st.error(f"❌ Error running PyTorch predictor: {e}")
        else:
            st.warning("⚠️ PyTorch predictor not available. Skipping prediction step.")

        st.markdown("---")

        # ========== STEP 3: Quantification & HSL Adjustment ==========
        st.subheader("Step 3: Damage Quantification & HSL Adjust")
        img_for_quant = image.resize((256, 256))
        pred_img_for_quant = pred_img.resize((256, 256)) if pred_img else img_for_quant
        
        hsl_input_adjusted = adjust_hsl(img_for_quant, hue_deg, sat_fac, bright_fac)
        hsl_output_adjusted = adjust_hsl(pred_img_for_quant, hue_deg, sat_fac, bright_fac)

        damage_in = quantify_damage(hsl_input_adjusted)
        damage_out = quantify_damage(hsl_output_adjusted)

        st.metric(label="Damage in Input (After HSL)", value=f"{damage_in}%")
        st.metric(label="Damage in Prediction (After HSL)", value=f"{damage_out}%")
        
        col_a, col_b = st.columns(2)
        col_a.image(hsl_input_adjusted, caption="HSL Adjusted Input", use_container_width=True) # FIX
        col_b.image(hsl_output_adjusted, caption="HSL Adjusted Prediction", use_container_width=True) # FIX

    else:
        st.info("👋 Welcome! Please upload an image to begin the analysis.")

if __name__ == "__main__":
    main()

