# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load trained model
model = load_model("mnist_cnn.h5")

# UI setup
st.set_page_config(page_title="Digit Recognizer")
st.title("✍️ Handwritten Digit Classifier")
st.markdown("Draw a digit (0–9) below and click **Predict**")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction
if st.button("Predict"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'), mode='L')
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        img = np.array(image).reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)

        st.subheader("🎯 Prediction")
        st.write(f"**It looks like a {predicted_label}**")

        st.subheader("📊 Confidence")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please draw a digit first.")
