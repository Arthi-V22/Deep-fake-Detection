# tamper_detector.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import os
from PIL import Image

# Load pre-trained Xception model
model = Xception(weights="imagenet", include_top=False, pooling='avg')

def detect_tampered_areas(img_path):
    # --- Validate file ---
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ File not found: {img_path}")
    try:
        Image.open(img_path).verify()
    except Exception as e:
        raise ValueError(f"❌ Not a valid image file: {img_path}\nError: {e}")

    # --- Load image for model ---
    img_model = keras_image.load_img(img_path, target_size=(299, 299))
    img_array = keras_image.img_to_array(img_model)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # --- Grad-CAM ---
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("block14_sepconv2_act").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_preprocessed)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8  # prevent division by zero
    heatmap = heatmap[0]

    # --- Load original image ---
    original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if original_img is None:
        raise ValueError(f"❌ OpenCV failed to read the image: {img_path}")

    # --- Ensure 3 channels ---
    if len(original_img.shape) == 2:  # grayscale
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    elif original_img.shape[2] == 4:  # has alpha channel
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)

    # --- Resize heatmap to match original image ---
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # --- Convert heatmap to color ---
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # --- Ensure type match ---
    if heatmap_color.dtype != original_img.dtype:
        heatmap_color = heatmap_color.astype(original_img.dtype)

    # --- Combine safely ---
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # --- Save output ---
    os.makedirs("static/results", exist_ok=True)
    output_path = os.path.join("static/results", f"tampered_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, overlay)
    return output_path

# --- Test ---
if __name__ == "__main__":
    test_image = r"D:\@CCS_DEMO_2025\NITHYA_IEEE_3\IEEEpro\MY_pro\sample1.jpg"
    result = detect_tampered_areas(test_image)
    print("✅ Tampered image saved at:", result)
