import tempfile
import cv2
import numpy as np

from inference import preprocess_image, predict_mask, resize_prediction

import torch
import time

from config import DEVICE, MODEL_PATH
from model import UNet

import streamlit as st
from PIL import Image

@st.cache_resource
def load_segmentation_model():

    model = UNet()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model

st.set_page_config(
    page_title="Industrial Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

model = load_segmentation_model()

st.sidebar.title("Settings")

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.write(f"**Device:** {DEVICE}")

st.sidebar.write("**Model:** U-Net")

st.sidebar.write("**Image Size:** 256 × 256")

st.title("🔍 Industrial Surface Defect Detection")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

st.write(
    "Upload a steel surface image and the trained U-Net model will detect surface defects."
)

left, center, right = st.columns([1,2,1])

with right:

    predict_button = st.button(
        "🔍 Predict Defect",
        use_container_width=True,
        type="primary"
    )

st.markdown("---")

if uploaded_file is not None and predict_button:

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

        tmp.write(uploaded_file.read())

        image_path = tmp.name

    # Preprocess
    image, image_tensor = preprocess_image(
        image_path,
        DEVICE
    )

    # Predict
    start = time.perf_counter()

    predicted_mask = predict_mask(
        model,
        image_tensor,
        threshold
    )

    end = time.perf_counter()

    inference_time = (end - start) * 1000

    # Resize prediction
    predicted_mask = resize_prediction(
        predicted_mask,
        image
    )

    defect_pixels = np.sum(predicted_mask)

    total_pixels = predicted_mask.size

    defect_percentage = (defect_pixels / total_pixels) * 100

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Defect Area",
            f"{defect_percentage:.2f}%"
        )

    with m2:
        st.metric(
            "Inference Time",
            f"{inference_time:.1f} ms"
        )

    with m3:

        if defect_percentage > 1:

            st.error("Defect Detected")

        else:

            st.success("No Defect Detected")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Predicted Mask")

        st.image(
            predicted_mask * 255,
            clamp=True,
            use_container_width=True
        )

    overlay = image.copy()

    overlay[predicted_mask == 1] = [255, 0, 0]

    with col3:

        st.subheader("Overlay")

        st.image(
            overlay,
            use_container_width=True
        )


    st.success("Prediction Completed Successfully!")