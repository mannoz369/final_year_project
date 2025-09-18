import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# -----------------------------
# Load & Preprocess Image
# -----------------------------
def preprocess_image(path, model_input_size=(128, 128)):
    original = cv2.imread(path)
    if original is None:
        return None, None
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original_rgb, model_input_size)
    input_tensor = resized.astype(np.float32) / 255.0
    return original_rgb, np.expand_dims(input_tensor, axis=0)

# -----------------------------
# Grad-CAM Function
# -----------------------------
def make_gradcam_heatmap(model, image_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]  # Regression scalar

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()

# -----------------------------
# Overlay Heatmap on Original Image
# -----------------------------
def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_8u = np.uint8(255 * heatmap_resized)

    heatmap_3ch = cv2.merge([heatmap_8u] * 3)
    lut = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), colormap)
    reversed_lut = lut[::-1].copy()
    heatmap_color = cv2.LUT(heatmap_3ch, reversed_lut)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -----------------------------
# Load Model
# -----------------------------
model_path = "damage_predictor.h5"
model = load_model(model_path)

# Find last conv layer
last_conv_layer_name = None
for layer in reversed(model.layers):
    if 'conv' in layer.name and len(layer.output_shape) == 4:
        last_conv_layer_name = layer.name
        break
if last_conv_layer_name is None:
    raise ValueError("No 4D convolutional layer found for Grad-CAM.")
print(f"Using last conv layer: {last_conv_layer_name}")

# -----------------------------
# Process Folder of Images
# -----------------------------
input_root = "resize_inputs"   # main input folder
output_root = "GradCam-inputs-resized"             # output folder

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            print(f"Processing: {img_path}")

            original_img, model_input = preprocess_image(img_path)
            if original_img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            heatmap = make_gradcam_heatmap(model, model_input, last_conv_layer_name)
            overlay_img = overlay_heatmap(original_img, heatmap)

            # -----------------------------
            # Preserve Folder Structure in outputs/cam/
            # -----------------------------
            relative_path = os.path.relpath(root, input_root)   # subfolder relative to input
            save_dir = os.path.join(output_root, relative_path) # mirror structure
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{os.path.splitext(file)[0]}.jpg")
            overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, overlay_bgr)

            print(f"Saved Grad-CAM to {save_path}")
