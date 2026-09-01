```python
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

from tensorflow.keras.applications.vgg16 import preprocess_input


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_vgg16.keras"
    )

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return class_names


model = load_model()
class_names = load_class_names()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and the VGG16 model "
    "will predict the most likely disease."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prediction button
    if st.button(
        "Predict Disease",
        type="primary"
    ):

        with st.spinner("Analyzing image..."):

            # ------------------------------------------------
            # Resize image to VGG16 input size
            # ------------------------------------------------

            image_resized = image.resize(
                (224, 224)
            )

            # ------------------------------------------------
            # Convert PIL image → NumPy array
            # ------------------------------------------------

            image_array = np.array(
                image_resized
            )

            # ------------------------------------------------
            # Add batch dimension
            #
            # Before:
            # (224, 224, 3)
            #
            # After:
            # (1, 224, 224, 3)
            # ------------------------------------------------

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # ------------------------------------------------
            # VGG16 preprocessing
            # ------------------------------------------------

            image_array = preprocess_input(
                image_array.astype(np.float32)
            )

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            predictions = model.predict(
                image_array,
                verbose=0
            )

            # ------------------------------------------------
            # Get predicted class
            # ------------------------------------------------

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = predictions[0][
                predicted_index
            ]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # TOP 5 PREDICTIONS
        # ====================================================

        st.subheader("Top Predictions")

        top_indices = np.argsort(
            predictions[0]
        )[::-1][:5]

        for index in top_indices:

            class_name = class_names[index]

            probability = predictions[0][index]

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Model: VGG16 Transfer Learning | "
    "Dataset: PlantVillage"
)
```
