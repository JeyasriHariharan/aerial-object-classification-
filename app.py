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
    model = tf.keras.models
