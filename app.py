import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Aerial Object Classifier", page_icon="🚁")

st.title("🐦 Aerial Object Classification")
st.subheader("Bird vs Drone Detector")
st.write("Upload an aerial image to classify it as Bird or Drone!")

@st.cache_resource
def load_model():
    model_path = "best_transfer_model.keras"
    if not os.path.exists(model_path):
        with st.spinner("Loading model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1olCLb9E5AsWS3Bl2c6x3RknaLyUOcJKU",
                model_path,
                quiet=False
            )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)
        confidence = prediction[0][0]

    st.markdown("---")
    if confidence > 0.5:
        st.error(f"🚁 DRONE detected!")
        st.metric("Confidence", f"{confidence*100:.2f}%")
    else:
        st.success(f"🐦 BIRD detected!")
        st.metric("Confidence", f"{(1-confidence)*100:.2f}%")

    st.markdown("### Prediction Confidence")
    col1, col2 = st.columns(2)
    with col1:
        st.write("🐦 Bird")
        st.progress(float(1-confidence))
    with col2:
        st.write("🚁 Drone")
        st.progress(float(confidence))

st.markdown("---")
st.markdown("Made with ❤️ by Jeyasri | GUVI Project")
