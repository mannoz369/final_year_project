import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import gdown
import os
import functools # Required for the UnetGenerator

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b"


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
    return load_model(VISUALIZER_PATH, compile=False)


@st.cache_resource
def load_predictor():
    # Initialize the correct UnetGenerator architecture
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


# =====================
# STREAMLIT APP
# =====================
def main():
    st.title("🔧 Damage Prediction & Visualization App")
    st.write("Upload an image to visualize initial damage, predict future damage, and quantify it.")

    download_predictor_model()

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📌 Uploaded Image", width='stretch')

        visualizer = load_visualizer()
        predictor = load_predictor()

        # ========== Create Resized Images for Each Model ==========
        # The Keras 'visualizer' model expects 224x224 images
        img_resized_128 = image.resize((128, 128))
        # The PyTorch 'predictor' U-Net model expects 256x256 images
        img_resized_256 = image.resize((256, 256))

        # ========== Step 1: Visualization (using 224x224 image) ==========
        st.subheader("Step 1: Initial Damage Visualization")
        input_arr = np.expand_dims(np.array(img_resized_128) / 255.0, axis=0).astype(np.float32)
        vis_output = visualizer.predict(input_arr)
        vis_img = Image.fromarray((vis_output[0] * 255).astype(np.uint8))
        st.image(vis_img, caption="Initial Damage Visualization", width='stretch')

        # ========== Step 2: Predictor (using 256x256 image) ==========
        st.subheader("Step 2: Future Damage Prediction")
        
        # Prepare tensor for PyTorch model (C, H, W) and rescale to [-1, 1]
        img_tensor = torch.tensor(np.array(img_resized_256).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        img_tensor = (img_tensor / 127.5) - 1.0

        with torch.no_grad():
            pred_output = predictor(img_tensor)

        # Convert output tensor back to a displayable PIL Image
        pred_numpy = pred_output[0].cpu().numpy()
        pred_numpy = (pred_numpy + 1) / 2.0 * 255.0 # Rescale from [-1, 1] to [0, 255]
        pred_numpy = pred_numpy.transpose(1, 2, 0) # Change from (C, H, W) to (H, W, C)
        pred_img = Image.fromarray(np.clip(pred_numpy, 0, 255).astype(np.uint8))
        
        st.image(pred_img, caption="Predicted Future Damage", width='stretch')

        # ========== Step 3: Quantification ==========
        st.subheader("Step 3: Damage Quantification")

        # Use the correct resized images for consistent comparison
        hsl_input = img_resized_256.convert("RGB")
        hsl_output = pred_img.convert("RGB")

        hsl_input_adjusted = adjust_hsl(hsl_input, 45, 1.5, 1.2)
        hsl_output_adjusted = adjust_hsl(hsl_output, 45, 1.5, 1.2)

        damage_in = quantify_damage(hsl_input_adjusted)
        damage_out = quantify_damage(hsl_output_adjusted)

        st.write(f"📊 **Damage in Input Image:** {damage_in}%")
        st.write(f"📊 **Damage in Predicted Image:** {damage_out}%")

        st.image([hsl_input_adjusted, hsl_output_adjusted], caption=["HSL Adjusted Input", "HSL Adjusted Prediction"], width='stretch')


if __name__ == "__main__":
    main()

