import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# ------------------------
# 1. Load Pre-trained Model
# ------------------------
@st.cache_resource
def get_model():
    model = load_model("myModelMnist.keras")
    return model

model = get_model()

# ------------------------
# 2. UI
# ------------------------
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a **handwritten digit (0–9)** image, and the model will try to predict it.")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to 28x28 like MNIST
    img_resized = ImageOps.invert(image).resize((28,28))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.subheader(f"✅ Prediction: {predicted_class}")
    st.bar_chart(prediction[0])
