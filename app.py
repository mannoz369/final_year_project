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

# =====================
# CONFIG
# =====================
PREDICTOR_PATH = "270_net_G.pth"
VISUALIZER_PATH = "damage_predictor.h5"
PREDICTOR_DRIVE_ID = "1NTicS-PJq8vrZuClHuoryRHSs3w8x9_b"  # Replace with your .pth file ID
VISUALIZER_DRIVE_ID = "1yJ88NnHUicxX14tTb6I_L7BZLZF4Oo9m"  # Replace with your .h5 file ID

# =====================
# DOWNLOAD MODELS FROM GOOGLE DRIVE
# =====================
def download_model(file_id, output_path, model_name):
    if not os.path.exists(output_path):
        with st.spinner(f"📥 Downloading {model_name} from Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        st.success(f"✅ {model_name} downloaded successfully!")

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_visualizer():
    try:
        return load_model(VISUALIZER_PATH)
    except Exception as e:
        st.error(f"Error loading visualizer model: {e}")
        return None

@st.cache_resource
def load_predictor():
    try:
        # Define your CNN architecture here (adjust according to your model)
        class PredictorNet(nn.Module):
            def __init__(self):
                super(PredictorNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                
                # Calculate the size after convolutions and pooling
                # For 224x224 input: after 3 pooling operations = 28x28
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, 128)
                self.fc3 = nn.Linear(128, 3)  # RGB output or adjust as needed

            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = self.pool(torch.relu(self.conv3(x)))
                x = torch.flatten(x, 1)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.dropout(torch.relu(self.fc2(x)))
                x = self.fc3(x)
                return x

        model = PredictorNet()
        model.load_state_dict(torch.load(PREDICTOR_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading predictor model: {e}")
        return None

# =====================
# HSL ADJUSTMENTS
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

# =====================
# DAMAGE QUANTIFICATION
# =====================
def quantify_black_percentage(image):
    """Quantify damage based on black pixels percentage"""
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array
    
    _, black_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    black_pixels = np.sum(black_mask == 255)
    total_pixels = image_array.shape[0] * image_array.shape[1]
    black_percentage = (black_pixels / total_pixels) * 100
    return round(black_percentage, 2)

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
        # Assuming your model outputs a single value or adjust accordingly
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
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    return last_conv_layer_name

# =====================
# STREAMLIT APP
# =====================
def main():
    st.set_page_config(
        page_title="🔧 Damage Prediction & Analysis",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Damage Prediction & Visualization App")
    st.markdown("Upload an image to predict future damage, visualize with Grad-CAM, and quantify damage percentage.")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # HSL adjustment parameters
    st.sidebar.subheader("HSL Parameters")
    hue_degrees = st.sidebar.slider("Hue Shift (degrees)", -180, 180, 45)
    saturation_factor = st.sidebar.slider("Saturation Factor", 0.1, 3.0, 1.5, 0.1)
    brightness_factor = st.sidebar.slider("Brightness Factor", 0.1, 3.0, 1.2, 0.1)
    
    # Model download section
    st.sidebar.subheader("📥 Model Setup")
    if st.sidebar.button("Download Models"):
        download_model(PREDICTOR_DRIVE_ID, PREDICTOR_PATH, "Predictor Model (.pth)")
        download_model(VISUALIZER_DRIVE_ID, VISUALIZER_PATH, "Visualizer Model (.h5)")
    
    # Check if models exist
    models_ready = os.path.exists(PREDICTOR_PATH) and os.path.exists(VISUALIZER_PATH)
    
    if not models_ready:
        st.warning("⚠️ Please download the models first using the sidebar button.")
        st.info("Make sure to replace YOUR_PTH_FILE_ID and YOUR_H5_FILE_ID with your actual Google Drive file IDs in the code.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload an image to analyze for damage prediction"
    )

    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📎 Original Image")
            st.image(original_image, use_column_width=True)
        
        # Load models
        with st.spinner("🔄 Loading models..."):
            visualizer = load_visualizer()
            predictor = load_predictor()
        
        if visualizer is None or predictor is None:
            st.error("❌ Failed to load models. Please check your model files.")
            return
        
        # Process the image
        try:
            # Resize image for model input
            img_resized = original_image.resize((224, 224))  # Adjust size as per your model
            input_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            
            # ========== STEP 1: Damage Prediction with PyTorch model ==========
            st.subheader("🔮 Step 1: Future Damage Prediction")
            
            # Convert to PyTorch tensor
            img_tensor = torch.tensor(
                np.array(img_resized).transpose(2, 0, 1), 
                dtype=torch.float32
            ).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                pred_output = predictor(img_tensor)
            
            # Convert prediction to image (adjust this based on your model output)
            pred_numpy = pred_output.cpu().numpy()
            
            # Reshape and normalize prediction output
            if pred_numpy.size == 3:  # RGB prediction
                pred_img_array = pred_numpy.reshape(1, 1, 3)
                pred_img_array = np.repeat(pred_img_array, 224, axis=0)
                pred_img_array = np.repeat(pred_img_array, 224, axis=1)
            else:
                # Handle other output formats
                pred_img_array = np.random.rand(224, 224, 3)  # Placeholder
            
            pred_img_array = (pred_img_array - pred_img_array.min()) / (pred_img_array.max() - pred_img_array.min())
            predicted_image = Image.fromarray((pred_img_array * 255).astype(np.uint8))
            
            with col2:
                st.subheader("🔮 Predicted Future Damage")
                st.image(predicted_image, use_column_width=True)
            
            # ========== STEP 2: Grad-CAM Visualization ==========
            st.subheader("🎯 Step 2: Grad-CAM Visualization")
            
            # Find last convolutional layer
            last_conv_layer_name = find_last_conv_layer(visualizer)
            
            if last_conv_layer_name:
                st.info(f"Using layer: {last_conv_layer_name}")
                
                # Generate Grad-CAM heatmap
                heatmap = make_gradcam_heatmap(visualizer, input_array, last_conv_layer_name)
                
                # Overlay heatmap on original image
                original_array = np.array(original_image.resize((224, 224)))
                overlay_img = overlay_heatmap(original_array, heatmap)
                overlay_image = Image.fromarray(overlay_img)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.image(overlay_image, caption="🎯 Grad-CAM Visualization", use_column_width=True)
                
                with col4:
                    # Show just the heatmap
                    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_img, caption="🔥 Heatmap Only", use_column_width=True)
            else:
                st.warning("⚠️ No convolutional layer found for Grad-CAM")
            
            # ========== STEP 3: HSL Adjustments ==========
            st.subheader("🎨 Step 3: HSL Adjusted Images")
            
            # Apply HSL adjustments
            hsl_original = adjust_hsl(original_image, hue_degrees, saturation_factor, brightness_factor)
            hsl_predicted = adjust_hsl(predicted_image, hue_degrees, saturation_factor, brightness_factor)
            
            col5, col6 = st.columns(2)
            with col5:
                st.image(hsl_original, caption="🎨 HSL Adjusted Original", use_column_width=True)
            
            with col6:
                st.image(hsl_predicted, caption="🎨 HSL Adjusted Predicted", use_column_width=True)
            
            # ========== STEP 4: Damage Quantification ==========
            st.subheader("📊 Step 4: Damage Quantification")
            
            # Calculate damage percentages
            original_damage = quantify_black_percentage(hsl_original)
            predicted_damage = quantify_black_percentage(hsl_predicted)
            damage_increase = predicted_damage - original_damage
            
            # Display results
            col7, col8, col9 = st.columns(3)
            
            with col7:
                st.metric(
                    label="🔍 Original Damage",
                    value=f"{original_damage}%"
                )
            
            with col8:
                st.metric(
                    label="🔮 Predicted Damage",
                    value=f"{predicted_damage}%",
                    delta=f"{damage_increase:+.2f}%"
                )
            
            with col9:
                damage_severity = "🟢 Low" if predicted_damage < 20 else "🟡 Medium" if predicted_damage < 50 else "🔴 High"
                st.metric(
                    label="⚠️ Damage Severity",
                    value=damage_severity
                )
            
            # Summary
            st.subheader("📋 Analysis Summary")
            st.write(f"""
            **Damage Analysis Results:**
            - Original damage level: **{original_damage}%**
            - Predicted future damage: **{predicted_damage}%**
            - Expected damage increase: **{damage_increase:+.2f}%**
            - Damage trend: **{'Increasing' if damage_increase > 0 else 'Stable/Decreasing'}**
            """)
            
            # Download section
            st.subheader("💾 Download Results")
            col10, col11 = st.columns(2)
            
            with col10:
                # Save HSL adjusted original
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    hsl_original.save(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download HSL Original",
                            data=f.read(),
                            file_name="hsl_original.png",
                            mime="image/png"
                        )
            
            with col11:
                # Save HSL adjusted prediction
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    hsl_predicted.save(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download HSL Predicted",
                            data=f.read(),
                            file_name="hsl_predicted.png",
                            mime="image/png"
                        )
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure your models are compatible with the uploaded image format.")

# =====================
# FOOTER
# =====================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🔧 Damage Prediction & Analysis Tool | Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
