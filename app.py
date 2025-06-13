import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detector")

# Use camera or uploader
use_camera = st.toggle("📷 Use Camera", value=True)

image_data = None
if use_camera:
    image_data = st.camera_input("Take a photo")
else:
    image_data = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_data:
    # Read image
    img = Image.open(image_data)
    st.image(img, caption="Captured Image", use_column_width=True)

    # Convert to OpenCV format (for later processing)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    st.success("Image captured! (Prediction feature will be added next.)")
