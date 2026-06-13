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
# Load model
# ----------------------------
model = tf.keras.models.load_model(model_path)

# ----------------------------
# UI
# ----------------------------
st.title("🐶🐱 Dog vs Cat Classifier")

st.write("Upload an image and I will predict if it's a Dog or Cat")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")

    st.write("Confidence:", float(prediction))