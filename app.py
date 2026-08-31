import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease using VGG16.")

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "best_vgg16_model.keras"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Disease classes
# IMPORTANT:
# Replace these with the exact
# class_names from your notebook.
# -----------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___mosaic_virus",
    "Tomato___healthy"
]

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        try:
            model = load_model()

            # Resize image to VGG16 input size
            image_resized = image.resize((224, 224))

            # Convert image to numpy array
            image_array = np.array(image_resized)

            # Normalize pixel values
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Prediction
            predictions = model.predict(image_array)

            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100

            predicted_class = class_names[predicted_index]

            # Display result
            st.success("Prediction completed!")

            st.subheader("🌱 Prediction")
            st.write(f"**Disease:** {predicted_class}")

            st.write(f"**Confidence:** {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error loading or predicting with the model: {e}")
