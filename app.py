import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import gdown
import os
import functools

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"
GOOGLE_DRIVE_ID = "1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b" 

# =====================
# PIX2PIX MODEL ARCHITECTURE
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
            down = [downconv]; up = [uprelu, upconv, nn.Tanh()]; model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]; up = [uprelu, upconv, upnorm]; model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]; up = [uprelu, upconv, upnorm]
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
    # NEW ROBUST METHOD: Load the model and build it immediately with a dummy input.
    # The now-BUILT model is then cached by Streamlit.
    model = load_model(VISUALIZER_PATH, compile=False)
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    model(dummy_input) # This call builds the model graph.
    return model

@st.cache_resource
def load_predictor():
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(PREDICTOR_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    # The model is now guaranteed to be built, so we can create the grad_model directly.
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(original_img_rgb, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8u, colormap)
    overlay = cv2.addWeighted(original_img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def adjust_hsl(image, hue, sat, bright):
    image_hsv = image.convert("HSV")
    h, s, v = image_hsv.split()
    h = np.array(h, dtype=np.uint8); hue_shift = int((hue / 360) * 255); h = (h.astype(int) + hue_shift) % 256
    h = np.clip(h, 0, 255).astype(np.uint8); image_hsv = Image.merge("HSV", (Image.fromarray(h), s, v))
    image = image_hsv.convert("RGB"); image = ImageEnhance.Color(image).enhance(sat)
    image = ImageEnhance.Brightness(image).enhance(bright)
    return image

def quantify_damage(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return round((np.mean(gray) / 255.0) * 100, 2)

# =====================
# STREAMLIT APP
# =====================
def main():
    st.set_page_config(layout="wide")
    st.title("🔧 Damage Prediction & Visualization App")
    st.write("Upload an image to visualize initial damage, predict future damage, and quantify it.")
    download_predictor_model()
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # FIX for deprecation warning: use width='stretch'
        st.image(image, caption="📌 Uploaded Image", width='stretch')

        visualizer = load_visualizer()
        predictor = load_predictor()

        img_resized_128 = image.resize((128, 128))
        img_resized_256 = image.resize((256, 256))

        st.subheader("Step 1: Grad-CAM Damage Visualization")
        input_arr = np.expand_dims(np.array(img_resized_128) / 255.0, axis=0).astype(np.float32)
        
        last_conv_layer_name = next((layer.name for layer in reversed(visualizer.layers) if 'conv' in layer.name and hasattr(layer, 'output') and len(layer.output.shape) == 4), None)
        
        if last_conv_layer_name:
            heatmap = make_gradcam_heatmap(visualizer, input_arr, last_conv_layer_name)
            gradcam_image = overlay_heatmap(np.array(img_resized_128), heatmap)
            st.image(gradcam_image, caption=f"Grad-CAM Visualization (Layer: {last_conv_layer_name})", width='stretch')
        else:
            st.warning("Could not find a suitable convolutional layer for Grad-CAM.")

        st.subheader("Step 2: Future Damage Prediction")
        img_tensor = torch.tensor(np.array(img_resized_256).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        img_tensor = (img_tensor / 127.5) - 1.0

        with torch.no_grad():
            pred_output = predictor(img_tensor)
        pred_numpy = pred_output[0].cpu().numpy(); pred_numpy = (pred_numpy + 1) / 2.0 * 255.0
        pred_numpy = pred_numpy.transpose(1, 2, 0)
        pred_img = Image.fromarray(np.clip(pred_numpy, 0, 255).astype(np.uint8))
        st.image(pred_img, caption="Predicted Future Damage", width='stretch')

        st.subheader("Step 3: Damage Quantification")
        hsl_input_adj = adjust_hsl(img_resized_256, 45, 1.5, 1.2)
        hsl_output_adj = adjust_hsl(pred_img, 45, 1.5, 1.2)

        damage_in = quantify_damage(hsl_input_adj)
        damage_out = quantify_damage(hsl_output_adj)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Initial Damage Score", value=f"{damage_in}%")
            st.image(hsl_input_adj, caption="HSL Adjusted Input")
        with col2:
            delta_val = round(damage_out - damage_in, 2)
            st.metric(label="Predicted Damage Score", value=f"{damage_out}%", delta=f"{delta_val}%")
            st.image(hsl_output_adj, caption="HSL Adjusted Prediction")

if __name__ == "__main__":
    main()
