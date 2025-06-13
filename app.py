import streamlit as st
from inference_sdk import InferenceHTTPClient
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# --- Roboflow Setup ---
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="fV5LNBhyGxlZQNNZec6W"
)

MODEL_ID = "my-first-project-mdags/1"

# --- UI Config ---
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Snap or upload a plant image to check for diseases using AI.</p>", unsafe_allow_html=True)

# --- Choose Image Input ---
input_type = st.radio("Choose Image Source:", ["📸 Camera", "🖼️ Upload"], horizontal=True)

img = None

if input_type == "📸 Camera":
    img_file = st.camera_input("Take a picture")
    if img_file:
        img = Image.open(img_file).convert("RGB")
elif input_type == "🖼️ Upload":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")

# --- Process Image ---
if img:
    st.markdown("### 📷 Your Image")
    st.image(img, use_column_width=True)

    # Convert to BGR (OpenCV)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        temp_path = tmp.name

    # --- Roboflow Inference ---
    try:
        prediction = CLIENT.infer(temp_path, model_id=MODEL_ID)

        if not prediction["predictions"]:
            st.warning("😕 No diseases detected. Try a clearer image.")
        else:
            # Draw predictions
            for pred in prediction["predictions"]:
                x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                label = pred["class"]
                conf = pred["confidence"]

                cv2.rectangle(img_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"{label} ({conf:.2f})", (x - w // 2, y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            final_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.markdown("### ✅ Prediction Result")
            st.image(final_img, use_column_width=True)

            st.markdown("### 🔍 Detected Diseases")
            for pred in prediction["predictions"]:
                st.success(f"**{pred['class']}** — {pred['confidence']:.2%} confidence")

    except Exception as e:
        st.error(f"🚨 Error during prediction: {e}")

    # Clean up
    os.remove(temp_path)

