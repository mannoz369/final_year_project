import streamlit as st
import torch
import tensorflow as tf
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import gdown

# --- Page Configuration ---
st.set_page_config(
    page_title="Future Damage Prediction & Visualization",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model & File Management ---

@st.cache_resource
def download_model(gdrive_id, output_path):
    """Downloads a file from Google Drive."""
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading model: {os.path.basename(output_path)}..."):
            gdown.download(id=gdrive_id, output=output_path, quiet=False)
    return output_path

@st.cache_resource
def load_models(pth_path, h5_path):
    """Loads the PyTorch and TensorFlow models into memory."""
    # Load PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Assuming the .pth file is a state_dict
        # If it's a full model, use: model_pth = torch.load(pth_path, map_location=device)
        # --- IMPORTANT: Define your PyTorch model architecture here ---
        # Example: model_pth = YourModelClass()
        # model_pth.load_state_dict(torch.load(pth_path, map_location=device))
        
        # As a placeholder, we'll assume a pre-trained model structure.
        # YOU MUST REPLACE THIS with your actual model loading logic.
        model_pth = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # In a real scenario, you'd load your state dict into your custom model class.
        st.info("Using a placeholder PyTorch model. Replace with your model architecture.", icon="ℹ️")

    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}. Please ensure your model architecture is defined in the script.")
        return None, None
        
    model_pth.to(device)
    model_pth.eval()

    # Load TensorFlow model
    model_tf = tf.keras.models.load_model(h5_path)
    return model_pth, model_tf

# --- Image Processing Functions (from your files) ---

# HSL functions from hsl.py
def adjust_hsl(image, hue_deg, sat_fac, bright_fac):
    """Adjusts Hue, Saturation, and Luminosity of a PIL Image."""
    # Hue
    image_hsv = image.convert("HSV")
    h, s, v = image_hsv.split()
    h = np.array(h, dtype=np.uint8)
    hue_shift = int((hue_deg / 360) * 255)
    h = (h.astype(int) + hue_shift) % 256
    h = Image.fromarray(h.astype(np.uint8))
    image_hsv = Image.merge("HSV", (h, s, v))
    image = image_hsv.convert("RGB")
    
    # Saturation
    image = ImageEnhance.Color(image).enhance(sat_fac)
    
    # Luminosity
    image = ImageEnhance.Brightness(image).enhance(bright_fac)
    return image

# Percentage calculation from percent.py
def quantify_black_percentage(pil_image):
    """Calculates the percentage of black pixels in a PIL image."""
    image_cv = np.array(pil_image.convert('RGB'))
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    _, black_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    black_pixels = np.sum(black_mask == 255)
    total_pixels = image_cv.shape[0] * image_cv.shape[1]
    return (black_pixels / total_pixels) * 100

# Grad-CAM functions from gradcam3.py
def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    """Generates the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        # Assuming regression, use the single output neuron
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
    
    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- Main App ---
st.title("Future Damage Prediction & Analysis Engine 🔬")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"]
)

st.sidebar.header("🎨 HSL Adjustment")
hue_degrees = st.sidebar.slider("Hue (degrees)", 0, 360, 45)
saturation_factor = st.sidebar.slider("Saturation", 0.0, 3.0, 1.5, 0.1)
brightness_factor = st.sidebar.slider("Brightness", 0.0, 3.0, 1.2, 0.1)

# --- Model Loading ---
# !! IMPORTANT !! Replace with your actual Google Drive File IDs
PTH_MODEL_ID = '1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b' 
H5_MODEL_ID = '1yJ88NnHUicxX14tTb6I_L7BZLZF4Oo9m' 

pth_model_path = download_model(PTH_MODEL_ID, "270_net_G.pth")
h5_model_path = download_model(H5_MODEL_ID, "damage_predictor.h5")

model_pth, model_tf = load_models(pth_model_path, h5_model_path)

if model_pth is None or model_tf is None:
    st.error("Models could not be loaded. Please check the GDrive IDs and model files.")
else:
    # --- Main Processing Logic ---
    if uploaded_file is not None:
        # Load and display the original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.header("1. Input Image")
        st.image(original_image, caption="Original Uploaded Image", use_column_width=True)
        
        if st.button("🚀 Analyze Damage", use_container_width=True):
            with st.spinner("Analyzing... This may take a moment."):
                
                # --- 2. Future Damage Prediction (.pth model) ---
                st.header("2. Predicted Future Damage")
                # Define preprocessing transforms for your PyTorch model
                preprocess_pth = transforms.Compose([
                    transforms.Resize((256, 256)), # Match your model's input size
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess_pth(original_image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model_pth(input_tensor)['out'][0]
                # Convert output tensor to a displayable PIL image
                output_predictions = output.argmax(0)
                predicted_mask = output_predictions.byte().cpu().numpy()
                predicted_image = Image.fromarray(predicted_mask).convert("RGB")
                
                st.image(predicted_image, caption="Model Prediction Output", use_column_width=True)

                # --- 3. Grad-CAM Visualization (.h5 model) ---
                st.header("3. Damage Area Visualization (Grad-CAM)")
                
                # Preprocess for TF model
                img_array_tf = np.array(original_image.resize((128, 128))) / 255.0
                img_array_tf = np.expand_dims(img_array_tf, axis=0)
                
                # Find last conv layer name automatically
                last_conv_layer_name = None
                for layer in reversed(model_tf.layers):
                    if 'conv' in layer.name and len(layer.output_shape) == 4:
                        last_conv_layer_name = layer.name
                        break
                
                if last_conv_layer_name:
                    heatmap = make_gradcam_heatmap(model_tf, img_array_tf, last_conv_layer_name)
                    gradcam_image = overlay_heatmap(np.array(original_image), heatmap)
                    st.image(gradcam_image, caption=f"Grad-CAM on layer: '{last_conv_layer_name}'", use_column_width=True)
                else:
                    st.warning("Could not find a suitable convolutional layer for Grad-CAM.")

                # --- 4. HSL and Percentage Analysis ---
                st.header("4. HSL Analysis & Damage Quantification")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original (HSL Adjusted)")
                    original_hsl = adjust_hsl(original_image, hue_degrees, saturation_factor, brightness_factor)
                    st.image(original_hsl, caption="Adjusted Input Image")
                    percent_original = quantify_black_percentage(original_hsl)
                    st.metric(label="Initial Damage Area (%)", value=f"{percent_original:.2f}%")

                with col2:
                    st.subheader("Prediction (HSL Adjusted)")
                    predicted_hsl = adjust_hsl(predicted_image, hue_degrees, saturation_factor, brightness_factor)
                    st.image(predicted_hsl, caption="Adjusted Predicted Image")
                    percent_predicted = quantify_black_percentage(predicted_hsl)
                    delta_change = percent_predicted - percent_original
                    st.metric(
                        label="Predicted Future Damage Area (%)",
                        value=f"{percent_predicted:.2f}%",
                        delta=f"{delta_change:.2f}% (Increase)"
                    )
                st.success("Analysis Complete!")
    else:
        st.info("Please upload an image to begin the analysis.")
