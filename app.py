import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

def load_model():
    model_path = "best_transfer_model.keras"
    if not os.path.exists(model_path):
        gdown.download(
            "https://drive.google.com/uc?id=1olCLb9E5AsWS3Bl2c6x3RknaLyUOcJKU",
            model_path,
            quiet=False
        )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

def predict(image):
    img = Image.fromarray(image).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    if confidence > 0.5:
        return {
            "🚁 Drone": float(confidence),
            "🐦 Bird": float(1-confidence)
        }
    else:
        return {
            "🐦 Bird": float(1-confidence),
            "🚁 Drone": float(confidence)
        }

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="🐦 Aerial Object Classification",
    description="Upload an aerial image to classify it as Bird or Drone!",
    examples=[],
    theme="soft"
)

interface.launch()
