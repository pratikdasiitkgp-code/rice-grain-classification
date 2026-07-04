import streamlit as st
import cv2
import joblib
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Rice Grain Classification",
    page_icon="🌾",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/rice_svm_model.pkl")
    scaler = joblib.load("models/rice_scaler.pkl")
    return model, scaler


model, scaler = load_model()


# --------------------------------------------------
# Class Names
# IMPORTANT: Must match training class order
# --------------------------------------------------

classes = [
    "Arborio",
    "Basmati",
    "Ipsala",
    "Jasmine",
    "Karacadag"
]


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("🌾 Rice Grain Classification")

st.write(
    """
    Upload an image of a rice grain and the machine learning model
    will predict its variety.
    """
)

st.info(
    "Supported varieties: Arborio, Basmati, Ipsala, Jasmine and Karacadag"
)


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a rice grain image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Rice Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Rice Variety", use_container_width=True):

        # Convert PIL image to NumPy array
        img = np.array(image)

        # Convert RGB to BGR
        # because training images were loaded using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Same preprocessing used during training
        img = cv2.resize(img, (64, 64))

        # Flatten image
        img = img.flatten().reshape(1, -1)

        # Apply saved scaler
        img = scaler.transform(img)

        # Make prediction
        prediction = model.predict(img)

        # Get predicted class
        predicted_class = classes[int(prediction[0])]

        st.success(
            f"🌾 Predicted Rice Variety: **{predicted_class}**"
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning project developed using Python, "
    "OpenCV, Scikit-learn and Streamlit."
)