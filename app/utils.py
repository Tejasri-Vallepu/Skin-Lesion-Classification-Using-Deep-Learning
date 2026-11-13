
from pathlib import Path
import base64
import requests
import numpy as np
import streamlit as st

from PIL import Image
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_best.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llava:7b"


@st.cache_resource
def load_prediction_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return load_model(MODEL_PATH)


def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image):

    model = load_prediction_model()

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)

    scores = predictions[0]

    if len(scores) != len(CLASS_NAMES):
        raise ValueError("Model output does not match class labels.")

    predicted_index = int(np.argmax(scores))

    return (
        CLASS_NAMES[predicted_index],
        float(scores[predicted_index]),
        scores
    )


def generate_image_description(image):
    """
    Generate a general visual description using Ollama LLaVA.
    This is not a medical diagnosis.
    """

    # Convert image to JPEG bytes
    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="JPEG")

    # Encode image as Base64
    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    prompt = """
    Describe only the visible characteristics of this image
    for an educational computer vision experiment.

    Mention general colors, patterns, texture, and image quality.

    Do not identify a disease, provide a diagnosis, estimate
    medical risk, or recommend treatment.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return result.get("response", "No description generated.")
