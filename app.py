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
import tempfile
from pathlib import Path

# =====================
# CONFIG
# =====================
PREDICTOR_PTH_PATH = "270_net_G.pth"
GRADCAM_MODEL_PATH = "damage_predictor.h5"
GOOGLE_DRIVE_PTH_URL = None  # Will be set from user input
GOOGLE_DRIVE_H5_URL = None   # Will be set from user input

# =====================
# DOWNLOAD MODELS FROM GOOGLE DRIVE
# =====================
def download_model_from_drive(url, output_path, model_name):
    """Download model from Google Drive if not exists"""
    if not os.path.exists(output_path):
        with st.spinner(f"📥 Downloading {model_name} from Google Drive..."):
            try:
                gdown.download(url, output_path, quiet=False, fuzzy=True)
                st.success(f"✅ {model_name} downloaded successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Error downloading {model_name}: {str(e)}")
                return False
    return True

# =====================
# DEFINE YOUR PYTORCH MODEL ARCHITECTURE
# =====================
class DamagePredictor(nn.Module):
    """
    Define your actual model architecture here.
    This is a placeholder - replace with your actual model structure.
    """
    def __init__(self):
        super(DamagePredictor, self).__init__()
        # Example architecture - REPLACE WITH YOUR ACTUAL MODEL
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 * 128 * 128)  # Output same size as input
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Example forward pass - REPLACE WITH YOUR ACTUAL FORWARD PASS
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to image format
        x = x.view(-1, 3, 128, 128)
        return x

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_pytorch_predictor(model_path):
    """Load PyTorch model for damage prediction"""
    model = DamagePredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_gradcam_model(model_path):
    """Load Keras model for Grad-CAM visualization"""
    return load_model(model_path)

# =====================
# GRAD-CAM FUNCTIONS
# =====================
def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            loss = predictions[:, 0]
        else:
            loss = predictions
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original image"""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)
    
    heatmap_3ch = cv2.merge([heatmap_8u] * 3)
    lut = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), colormap)
    reversed_lut = lut[::-1].copy()
    heatmap_color = cv2.LUT(heatmap_3ch, reversed_lut)
    
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def find_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
            return layer.name
    return None

# =====================
# HSL ADJUSTMENT FUNCTIONS
# =====================
def adjust_hue(image, hue_degrees):
    """Adjust hue of the image"""
    image_hsv = image.convert("HSV")
    h, s, v = image_hsv.split()
    h = np.array(h, dtype=np.uint8)
    hue_shift = int((hue_degrees / 360) * 255)
    h = (h.astype(int) + hue_shift) % 256
    h = np.clip(h, 0, 255).astype(np.uint8)
    image_hsv = Image.merge("HSV", (Image.fromarray(h), s, v))
    return image_hsv.convert("RGB")

def adjust_saturation(image, saturation_factor):
    """Adjust saturation of the image"""
    return ImageEnhance.Color(image).enhance(saturation_factor)

def adjust_luminosity(image, brightness_factor):
    """Adjust luminosity/brightness of the image"""
    return ImageEnhance.Brightness(image).enhance(brightness_factor)

def adjust_hsl(image, hue_deg, sat_fac, bright_fac):
    """Apply HSL adjustments to image"""
    image = adjust_hue(image, hue_deg)
    image = adjust_saturation(image, sat_fac)
    image = adjust_luminosity(image, bright_fac)
    return image

# =====================
# DAMAGE QUANTIFICATION
# =====================
def quantify_black_percentage(image):
    """Calculate percentage of black/dark pixels in image"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Threshold to identify black pixels
    _, black_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate percentage
    black_pixels = np.sum(black_mask == 255)
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    black_percentage = (black_pixels / total_pixels) * 100
    
    return round(black_percentage, 2)

# =====================
# IMAGE PROCESSING PIPELINE
# =====================
def process_image_with_pytorch(image, model):
    """Process image through PyTorch model"""
    # Resize image to model input size
    img_resized = image.resize((128, 128))
    
    # Convert to tensor
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert output back to image
    output_array = output[0].permute(1, 2, 0).cpu().numpy()
    output_array = np.clip(output_array * 255, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_array)
    
    return output_image

def generate_gradcam(image, gradcam_model):
    """Generate Grad-CAM visualization"""
    # Prepare image for Grad-CAM model
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Find last conv layer
    last_conv_layer = find_last_conv_layer(gradcam_model)
    if last_conv_layer is None:
        st.warning("No convolutional layer found for Grad-CAM")
        return None
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(gradcam_model, img_tensor, last_conv_layer)
    
    # Overlay on original image
    overlay = overlay_heatmap(np.array(img_resized), heatmap)
    
    return Image.fromarray(overlay)

# =====================
# STREAMLIT APP MAIN
# =====================
def main():
    st.set_page_config(page_title="Damage Prediction System", layout="wide")
    
    st.title("🔧 Advanced Damage Prediction & Analysis System")
    st.markdown("""
    This application provides:
    - **Future Damage Prediction** using deep learning
    - **Grad-CAM Visualization** for damage localization
    - **HSL Color Analysis** for enhanced damage visibility
    - **Quantitative Damage Assessment** with percentage metrics
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model URLs
        st.subheader("📥 Model Download Links")
        pth_url = st.text_input(
            "PyTorch Model (.pth) Google Drive URL",
            placeholder="https://drive.google.com/file/d/YOUR_FILE_ID/view",
            help="Enter the Google Drive sharing link for your .pth model"
        )
        
        h5_url = st.text_input(
            "Keras Model (.h5) Google Drive URL",
            placeholder="https://drive.google.com/file/d/YOUR_FILE_ID/view",
            help="Enter the Google Drive sharing link for your .h5 model"
        )
        
        # HSL Parameters
        st.subheader("🎨 HSL Adjustment Parameters")
        hue_degrees = st.slider("Hue Shift (degrees)", -180, 180, 45)
        saturation_factor = st.slider("Saturation Factor", 0.0, 3.0, 1.5, 0.1)
        brightness_factor = st.slider("Brightness Factor", 0.0, 3.0, 1.2, 0.1)
        
        # Grad-CAM Parameters
        st.subheader("🔥 Grad-CAM Settings")
        heatmap_alpha = st.slider("Heatmap Transparency", 0.0, 1.0, 0.4, 0.05)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "📤 Upload an image for damage analysis",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload an image to analyze current damage and predict future deterioration"
    )
    
    if uploaded_file is not None:
        # Load uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)
        
        # Download models if URLs provided
        models_ready = True
        if pth_url:
            GOOGLE_DRIVE_PTH_URL = pth_url
            models_ready = download_model_from_drive(GOOGLE_DRIVE_PTH_URL, PREDICTOR_PTH_PATH, "PyTorch Predictor")
        
        if h5_url:
            GOOGLE_DRIVE_H5_URL = h5_url
            models_ready = download_model_from_drive(GOOGLE_DRIVE_H5_URL, GRADCAM_MODEL_PATH, "Grad-CAM Model")
        
        if not models_ready:
            st.error("Please ensure both models are downloaded successfully before proceeding.")
            return
        
        # Check if models exist
        if not os.path.exists(PREDICTOR_PTH_PATH):
            st.warning("⚠️ PyTorch model not found. Please provide the Google Drive URL in the sidebar.")
            return
        
        if not os.path.exists(GRADCAM_MODEL_PATH):
            st.warning("⚠️ Grad-CAM model not found. Please provide the Google Drive URL in the sidebar.")
            return
        
        # Process button
        if st.button("🚀 Analyze Damage", type="primary"):
            with st.spinner("Processing image..."):
                
                # Load models
                predictor_model = load_pytorch_predictor(PREDICTOR_PTH_PATH)
                gradcam_model = load_gradcam_model(GRADCAM_MODEL_PATH)
                
                # Step 1: Predict future damage
                st.header("📊 Analysis Results")
                predicted_image = process_image_with_pytorch(image, predictor_model)
                
                # Step 2: Generate Grad-CAM visualizations
                gradcam_input = generate_gradcam(image, gradcam_model)
                gradcam_output = generate_gradcam(predicted_image, gradcam_model)
                
                # Step 3: Apply HSL adjustments
                hsl_input = adjust_hsl(image, hue_degrees, saturation_factor, brightness_factor)
                hsl_output = adjust_hsl(predicted_image, hue_degrees, saturation_factor, brightness_factor)
                
                # Step 4: Calculate damage percentages
                damage_input = quantify_black_percentage(hsl_input)
                damage_output = quantify_black_percentage(hsl_output)
                damage_increase = damage_output - damage_input
                
                # Display results
                st.subheader("1️⃣ Future Damage Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Current State", use_column_width=True)
                with col2:
                    st.image(predicted_image, caption="Predicted Future Damage", use_column_width=True)
                
                if gradcam_input and gradcam_output:
                    st.subheader("2️⃣ Grad-CAM Damage Localization")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(gradcam_input, caption="Current Damage Heatmap", use_column_width=True)
                    with col2:
                        st.image(gradcam_output, caption="Predicted Damage Heatmap", use_column_width=True)
                
                st.subheader("3️⃣ HSL Enhanced Visualization")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(hsl_input, caption="HSL Enhanced - Current", use_column_width=True)
                with col2:
                    st.image(hsl_output, caption="HSL Enhanced - Predicted", use_column_width=True)
                
                st.subheader("4️⃣ Quantitative Damage Assessment")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Current Damage",
                        value=f"{damage_input}%",
                        delta=None
                    )
                with col2:
                    st.metric(
                        label="Predicted Damage",
                        value=f"{damage_output}%",
                        delta=f"+{damage_increase:.1f}%" if damage_increase > 0 else f"{damage_increase:.1f}%"
                    )
                with col3:
                    severity = "Low" if damage_output < 30 else "Medium" if damage_output < 60 else "High"
                    severity_color = "🟢" if severity == "Low" else "🟡" if severity == "Medium" else "🔴"
                    st.metric(
                        label="Damage Severity",
                        value=f"{severity_color} {severity}",
                        delta=None
                    )
                
                # Summary report
                st.subheader("📝 Analysis Summary")
                summary = f"""
                **Damage Analysis Report:**
                - **Current Damage Level:** {damage_input}%
                - **Predicted Future Damage:** {damage_output}%
                - **Expected Damage Increase:** {damage_increase:.1f}%
                - **Severity Classification:** {severity}
                - **Recommendation:** {"Immediate attention required" if severity == "High" else "Monitor regularly" if severity == "Medium" else "Normal wear, routine maintenance"}
                
                **Technical Parameters Used:**
                - HSL Adjustments: Hue={hue_degrees}°, Saturation={saturation_factor}x, Brightness={brightness_factor}x
                - Grad-CAM Alpha: {heatmap_alpha}
                - Model: PyTorch Damage Predictor
                """
                st.info(summary)
                
                # Download results
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    # Convert predicted image to bytes for download
                    from io import BytesIO
                    buf = BytesIO()
                    predicted_image.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Predicted Image",
                        data=byte_im,
                        file_name="predicted_damage.png",
                        mime="image/png"
                    )
                with col2:
                    # Download analysis report
                    st.download_button(
                        label="Download Analysis Report",
                        data=summary,
                        file_name="damage_analysis_report.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
