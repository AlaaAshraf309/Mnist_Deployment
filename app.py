import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical 
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# ------------------------
# 1. Build or Load Model
# ------------------------
@st.cache_resource
def get_model():
    try:
        # لو موجود model مدرب، نعمل load
        model = load_model("mnist_cnn.h5")
    except:
        # لو مش موجود ندرب واحد صغير
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
        model.save("mnist_cnn.h5")
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
