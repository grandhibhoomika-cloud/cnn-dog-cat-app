import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import tensorflow as tf

# ----------------------------
# Download model from Google Drive
# ----------------------------
file_id = "1-YOBiF94MGL5mRtxHV84l5kt-RqUMMF0"
url = f"https://drive.google.com/uc?id={file_id}"

model_path = "cnn_model.h5"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# ----------------------------
# Load model (SAFE VERSION)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# ----------------------------
# UI
# ----------------------------
st.title("🐶🐱 Dog vs Cat Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] != 3:
        st.error("Please upload RGB image")
    else:
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            st.success("🐶 Dog")
        else:
            st.success("🐱 Cat")

        st.write("Confidence:", float(prediction))
